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
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 Bucket para almacenar documentos
        documents_bucket = s3.Bucket(
            self, "DocumentsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True
        )

        # DynamoDB para almacenar embeddings y metadatos
        embeddings_table = dynamodb.Table(
            self, "EmbeddingsTable",
            partition_key=dynamodb.Attribute(
                name="document_id",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="chunk_id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY
        )

        # IAM Role para Lambda functions
        lambda_role = iam.Role(
            self, "LambdaExecutionRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AWSLambdaBasicExecutionRole"
                )
            ]
        )

        # Permisos para Bedrock
        lambda_role.add_to_policy(iam.PolicyStatement(
            actions=[
                "bedrock:InvokeModel",
                "bedrock:ListFoundationModels"
            ],
            resources=["*"]
        ))

        # Permisos para S3
        documents_bucket.grant_read_write(lambda_role)
        
        # Permisos para DynamoDB
        embeddings_table.grant_read_write_data(lambda_role)

        # Lambda para ingesta de datos
        data_ingestion_lambda = lambda_.Function(
            self, "DataIngestionFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="lambda_function.lambda_handler",
            code=lambda_.Code.from_asset("lambda_functions/data_ingestion"),
            role=lambda_role,
            timeout=Duration.minutes(15),
            memory_size=1024,
            environment={
                "DOCUMENTS_BUCKET": documents_bucket.bucket_name,
                "EMBEDDINGS_TABLE": embeddings_table.table_name,
                "MAX_CHUNKS_PER_FILE": "50"
            }
        )

        # Configurar trigger de S3 para la Lambda de ingesta
        documents_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(data_ingestion_lambda),
            s3.NotificationKeyFilter(prefix="uploads/")
        )

        # Lambda para procesamiento de queries
        query_processor_lambda = lambda_.Function(
            self, "QueryProcessorFunction",
            runtime=lambda_.Runtime.PYTHON_3_9,
            handler="lambda_function.lambda_handler",
            code=lambda_.Code.from_asset("lambda_functions/query_processor"),
            role=lambda_role,
            timeout=Duration.minutes(1),
            memory_size=512,
            environment={
                "DOCUMENTS_BUCKET": documents_bucket.bucket_name,
                "EMBEDDINGS_TABLE": embeddings_table.table_name
            }
        )

        # Crear API Gateway - CONFIGURACIÓN CORREGIDA
        api = apigateway.RestApi(
            self, "RagApiGateway",
            rest_api_name="RAG-Query-Service",
            description="API for RAG query processing",
            default_cors_preflight_options=apigateway.CorsOptions(
                allow_origins=apigateway.Cors.ALL_ORIGINS,
                allow_methods=apigateway.Cors.ALL_METHODS,
                allow_headers=apigateway.Cors.DEFAULT_HEADERS
            ),
            deploy=True,
            deploy_options=apigateway.StageOptions(
                stage_name="dev",
                # logging_level=apigateway.MethodLoggingLevel.INFO,
                # data_trace_enabled=True
            )
        )

        # Integración Lambda con API Gateway
        lambda_integration = apigateway.LambdaIntegration(
            query_processor_lambda,
            proxy=True,
            integration_responses=[
                apigateway.IntegrationResponse(
                    status_code="200",
                    response_parameters={
                        "method.response.header.Access-Control-Allow-Origin": "'*'"
                    }
                )
            ]
        )

        # Añadir recurso y método
        query_resource = api.root.add_resource("query")
        method = query_resource.add_method(
            "GET",
            lambda_integration,
            authorization_type=apigateway.AuthorizationType.NONE,
            method_responses=[
                apigateway.MethodResponse(
                    status_code="200",
                    response_parameters={
                        "method.response.header.Access-Control-Allow-Origin": True
                    }
                )
            ]
        )

        # Dar permisos a API Gateway para invocar la Lambda
        query_processor_lambda.grant_invoke(iam.ServicePrincipal("apigateway.amazonaws.com"))

        # Agregar tags
        Tags.of(self).add("Project", "Bedrock-RAG-App")
        Tags.of(self).add("Environment", "Development")
        Tags.of(self).add("Owner", "TuNombre")
        Tags.of(self).add("CostCenter", "AI-Research")

        # Outputs
        self.output_props = {
            "documents_bucket": documents_bucket,
            "embeddings_table": embeddings_table,
            "data_ingestion_lambda": data_ingestion_lambda,
            "query_processor_lambda": query_processor_lambda,
            "api_url": api.url  # Este output es crucial
        }

    @property
    def outputs(self):
        return self.output_props