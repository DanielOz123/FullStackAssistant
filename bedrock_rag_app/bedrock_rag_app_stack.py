from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_dynamodb as dynamodb,
    aws_s3_notifications as s3n,
    aws_apigateway as apigateway,
    RemovalPolicy,
    Duration,
    Tags
)
from constructs import Construct

class BedrockRagAppStack(Stack):
    """
    AWS CDK Stack for creating a Retrieval Augmented Generation (RAG) application
    using Amazon Bedrock for AI/ML capabilities.
    
    This stack creates a complete serverless architecture for document processing,
    vector storage, and query handling with a REST API interface.
    
    Architecture Components:
    - S3 Bucket for document storage
    - DynamoDB for vector embeddings and metadata
    - Lambda functions for data ingestion and query processing
    - API Gateway for RESTful API interface
    - IAM roles and policies for secure access
    
    Key Features:
    - Automatic document processing when files are uploaded to S3
    - Vector similarity search using DynamoDB
    - AI-powered question answering via Amazon Bedrock
    - CORS-enabled REST API for frontend applications
    """
    
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """
        Initialize the RAG application stack.
        
        Args:
            scope: The parent construct (usually the app)
            construct_id: The unique identifier for this stack
            **kwargs: Additional arguments passed to the Stack base class
        """
        super().__init__(scope, construct_id, **kwargs)

        # ----------------------------------------------------------------------
        # S3 Bucket for Document Storage
        # ----------------------------------------------------------------------
        documents_bucket = s3.Bucket(
            self, "DocumentsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )
        # Purpose: Stores PDF and CSV documents that will be processed by the system
        # Auto-deletion ensures clean cleanup when stack is destroyed
        # Security: Default encryption enabled, public access blocked by default
        # Lifecycle: Objects automatically deleted when stack is destroyed

        # ----------------------------------------------------------------------
        # DynamoDB Table for Vector Embeddings and Metadata
        # ----------------------------------------------------------------------
        embeddings_table = dynamodb.Table(
            self, "EmbeddingsTable",
            partition_key=dynamodb.Attribute(
                name="document_id",           # Unique identifier for each document
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="chunk_id",              # Identifier for text chunks within document
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,  # Pay per read/write
            removal_policy=RemovalPolicy.DESTROY,  # Delete table when stack is destroyed
        )
        # Table Structure:
        # - document_id: UUID for each processed document (Partition Key)
        # - chunk_id: Sequential ID for text chunks (Sort Key)  
        # - content: Text content of the chunk
        # - embedding: Vector embedding from Amazon Titan (stored as JSON string)
        # - source_file: Original S3 file path and name
        # - file_type: PDF or CSV
        # - created_at: Processing timestamp (ISO format)
        # - chunk_size: Length of text content in characters
        #
        # Performance: Pay-per-request billing for cost efficiency with variable workloads
        # Scalability: Automatically scales based on demand without capacity planning

        # ----------------------------------------------------------------------
        # IAM Role for Lambda Functions
        # ----------------------------------------------------------------------
        lambda_role = iam.Role(
            self, "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Execution role for RAG application Lambda functions",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"  # CloudWatch logging permissions
                )
            ]
        )
        # Base permissions include:
        # - logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents
        # - Basic Lambda execution permissions

        # Bedrock Permissions - Allow invoking foundation models
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "bedrock:InvokeModel",           # Permission to invoke Bedrock models
                "bedrock:ListFoundationModels"   # Permission to list available models
            ],
            resources=["*"],  # Access to all Bedrock models
        ))
        # Security Note: Wildcard resource (*) is used for simplicity in development
        # For production, restrict to specific model ARNs for better security posture

        # S3 Permissions - Read/write access to documents bucket
        documents_bucket.grant_read_write(lambda_role)
        # Grants permissions: s3:GetObject, s3:PutObject, s3:ListBucket, s3:DeleteObject
        # Applied to: The specific documents bucket and all objects within it
        
        # DynamoDB Permissions - Full access to embeddings table
        embeddings_table.grant_read_write_data(lambda_role)
        # Grants permissions: dynamodb:GetItem, PutItem, UpdateItem, DeleteItem, Query, Scan
        # Applied to: The specific embeddings table

        # ----------------------------------------------------------------------
        # Data Ingestion Lambda Function
        # ----------------------------------------------------------------------
        data_ingestion_lambda = lambda_.Function(
            self, "DataIngestionFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="lambda_function_v2.lambda_handler",  # Entry point: lambda_function.py
            code=lambda_.Code.from_asset("lambda_functions/data_ingestion"),  # Source code location
            role=lambda_role,                          # IAM role defined above
            timeout=Duration.minutes(15),              # 15-minute timeout for large documents
            memory_size=1024,                          # 1GB memory for PDF processing
            environment={
                "DOCUMENTS_BUCKET": documents_bucket.bucket_name,
                "EMBEDDINGS_TABLE": embeddings_table.table_name,
                "MAX_CHUNKS_PER_FILE": "50"            # Safety limit for chunk processing
            },
        )
        # Functionality: Processes PDF/CSV files from S3, extracts text, generates embeddings,
        # and stores chunks + vectors in DynamoDB
        #
        # Environment Variables:
        # - DOCUMENTS_BUCKET: Name of the S3 bucket for document storage
        # - EMBEDDINGS_TABLE: Name of the DynamoDB table for vector storage  
        # - MAX_CHUNKS_PER_FILE: Safety limit to prevent excessive chunking (50 chunks/file)

        # Configure S3 trigger for automatic processing
        documents_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,               # Trigger when new objects are created
            s3n.LambdaDestination(data_ingestion_lambda),  # Invoke data ingestion Lambda
            s3.NotificationKeyFilter(prefix="uploads/")  # Only process files in uploads/ folder
        )
        # Trigger Behavior:
        # - Activates when any object is created in the bucket
        # - Only processes objects with prefix "uploads/" (folder filtering)
        # - Automatic asynchronous invocation of data_ingestion_lambda
        # - Built-in retry logic for failed invocations

        # ----------------------------------------------------------------------
        # Query Processor Lambda Function  
        # ----------------------------------------------------------------------
        query_processor_lambda = lambda_.Function(
            self, "QueryProcessorFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="lambda_function.lambda_handler",  # Entry point: lambda_function.py
            code=lambda_.Code.from_asset("lambda_functions/query_processor"),  # Source location
            role=lambda_role,                          # Shared IAM role
            timeout=Duration.minutes(1),               # 1-minute timeout for user queries
            memory_size=512,                           # 512MB memory for query processing
            environment={
                "DOCUMENTS_BUCKET": documents_bucket.bucket_name,
                "EMBEDDINGS_TABLE": embeddings_table.table_name
            }
        )
        # Functionality: Handles user queries, performs vector similarity search,
        # and generates responses using Amazon Bedrock models
        #
        # Environment Variables:
        # - DOCUMENTS_BUCKET: Name of the S3 bucket (for reference)
        # - EMBEDDINGS_TABLE: Name of the DynamoDB table for vector search

        # ----------------------------------------------------------------------
        # API Gateway for RESTful Interface
        # ----------------------------------------------------------------------
        api = apigateway.RestApi(
            self, "RagApiGateway",
            rest_api_name="RAG-Query-Service",
            description="REST API for querying processed documents using RAG architecture",
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=apigateway.Cors.ALL_ORIGINS,    # Allow all origins (adjust for production)
                allow_methods=apigateway.Cors.ALL_METHODS,    # Allow all HTTP methods
                allow_headers=apigateway.Cors.DEFAULT_HEADERS  # Allow common headers
            ),
            deploy=True,
            deploy_options=apigateway.StageOptions(
                stage_name="dev",                   # Deployment stage name
                # logging_level=apigateway.MethodLoggingLevel.INFO,  # Enable for debugging
                # data_trace_enabled=True,                           # Enable for detailed tracing
            )
        )
        # API Gateway Configuration:
        # - REST API type: Traditional REST API with resources and methods
        # - CORS enabled: Allows cross-origin requests from web applications
        # - Stage: "dev" for development environment
        # - Logging: Disabled by default (commented out for production use)

        # Lambda Integration for API Gateway
        lambda_integration = apigateway.LambdaIntegration(
            query_processor_lambda,
            proxy=True,  # Pass entire request to Lambda (simpler setup)
            integration_responses=[
                apigateway.IntegrationResponse(
                    status_code="200",
                    response_parameters={
                        "method.response.header.Access-Control-Allow-Origin": "'*'"
                    }
                )
            ]
        )
        # Proxy Integration: 
        # - Forwards all request data (headers, body, query parameters) to Lambda
        # - Lambda responsible for complete response handling
        # - Simplifies API configuration but requires Lambda to handle HTTP details

        # Add API resource and method
        query_resource = api.root.add_resource("query")  # Creates /query endpoint
        method = query_resource.add_method(
            "GET",                                      # HTTP GET method
            lambda_integration,                         # Lambda integration
            authorization_type=apigateway.AuthorizationType.NONE,  # No authentication
            method_responses=[
                apigateway.MethodResponse(
                    status_code="200",
                    response_parameters={
                        "method.response.header.Access-Control-Allow-Origin": True
                    }
                )
            ]
        )
        # API Design:
        # - Endpoint: GET /query
        # - Parameters: Expected via query string parameters (?question=...)
        # - Authentication: None (open API) - add authentication for production
        # - Response: JSON format with answer and sources

        # Grant API Gateway permission to invoke the Lambda function
        query_processor_lambda.grant_invoke(iam.ServicePrincipal("apigateway.amazonaws.com"))
        # Required permission for API Gateway to invoke Lambda function
        # Uses resource-based policy on Lambda function

        # ----------------------------------------------------------------------
        # Resource Tagging for Cost Tracking and Management
        # ----------------------------------------------------------------------
        Tags.of(self).add("Project", "Bedrock-RAG-App")        # Project identifier
        Tags.of(self).add("Environment", "Development")        # Deployment environment
        Tags.of(self).add("Owner", "Danial Ozuna")                 # Resource owner
        Tags.of(self).add("CostCenter", "AI-Research")         # Cost allocation tag
        # Tagging Benefits:
        # - Cost allocation and tracking by project
        # - Resource management and organization
        # - Security and compliance reporting
        # - Automation and resource selection

        # ----------------------------------------------------------------------
        # Stack Outputs for Easy Reference
        # ----------------------------------------------------------------------
        self.output_props = {
            "documents_bucket": documents_bucket,          # S3 bucket for document uploads
            "embeddings_table": embeddings_table,          # DynamoDB table for vectors
            "data_ingestion_lambda": data_ingestion_lambda,  # Data processing function
            "query_processor_lambda": query_processor_lambda,  # Query handling function
            "api_url": api.url  # API Gateway endpoint URL (crucial for frontend integration)
        }
        # Outputs are accessible after deployment via:
        # - AWS CloudFormation console
        # - CDK CLI: `cdk outputs`
        # - Programmatic access in other stacks

    @property
    def outputs(self):
        """
        Property accessor for stack outputs.
        
        Returns:
            Dict: Dictionary containing references to all major stack resources
        """
        return self.output_props