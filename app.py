import aws_cdk as cdk
from bedrock_rag_app.bedrock_rag_app_stack import BedrockRagAppStack
from cloudwatch_setup_stack.cloudwatch_setup_stack import CloudWatchSetupStack

# Initialize the CDK Application
app = cdk.App()
"""
The CDK Application is the root construct that represents your AWS CloudFormation
application. It manages the synthesis of CloudFormation templates from your CDK stacks.

Key Responsibilities:
- Acts as the container for all stacks in your application
- Manages context and environment configuration
- Handles synthesis of CloudFormation templates
- Manages cross-stack references and dependencies
"""

# ------------------------------------------------------------------------------
# CloudWatch Setup Stack - Prerequisite Infrastructure
# ------------------------------------------------------------------------------
cloudwatch_stack = CloudWatchSetupStack(app, "CloudWatchSetupStack")
"""
Creates the CloudWatch setup stack that must be deployed BEFORE the main RAG stack.

Purpose:
- Creates IAM role for API Gateway CloudWatch Logs integration
- Solves the error: "CloudWatch Logs role ARN must be set in account settings"
- Provides necessary permissions for API Gateway to write logs to CloudWatch

Why deploy first:
1. API Gateway requires the CloudWatch role ARN to be configured at the account level
2. The main RAG stack will fail if API Gateway cannot find the required IAM role
3. This stack creates the foundational logging infrastructure

Stack Outputs:
- ApiGatewayCloudWatchLogsRoleArn: ARN of the IAM role for CloudWatch logging
"""

# ------------------------------------------------------------------------------
# Main RAG Application Stack - Core Infrastructure
# ------------------------------------------------------------------------------
rag_stack = BedrockRagAppStack(app, "BedrockRagAppStack",
    description="RAG Application with Bedrock and API Gateway"
)
"""
Creates the main Retrieval Augmented Generation (RAG) application stack.

Components:
- S3 Bucket: For storing PDF/CSV documents
- DynamoDB Table: For storing document chunks and vector embeddings
- Lambda Functions: 
  • Data Ingestion: Processes uploaded documents, creates embeddings
  • Query Processor: Handles user queries, performs semantic search
- API Gateway: REST API endpoint for querying documents
- IAM Roles: Permissions for Lambda functions to access AWS services

Dependencies:
- Requires the CloudWatch setup stack to be deployed first
- Depends on the IAM role created by CloudWatchSetupStack

Features:
- Automatic document processing when files are uploaded to S3
- Vector similarity search using Amazon Titan embeddings
- Natural language responses using Anthropic Claude 3
- CORS-enabled REST API for web frontend integration
"""

# ------------------------------------------------------------------------------
# Explicit Stack Dependencies
# ------------------------------------------------------------------------------
rag_stack.add_dependency(cloudwatch_stack)
"""
Establishes a hard dependency between stacks to ensure proper deployment order.

Why this dependency is critical:
1. CloudWatchSetupStack creates the IAM role that API Gateway requires
2. BedrockRagAppStack will fail during deployment if the IAM role doesn't exist
3. CloudFormation will automatically deploy stacks in the correct order

Deployment Order:
1. CloudWatchSetupStack (prerequisite infrastructure)
2. BedrockRagAppStack (main application infrastructure)

Without this dependency, CDK might attempt to deploy stacks in parallel or
incorrect order, causing deployment failures.
"""

# ------------------------------------------------------------------------------
# Application-wide Tagging
# ------------------------------------------------------------------------------
cdk.Tags.of(app).add("Project", "Bedrock-RAG-App")
cdk.Tags.of(app).add("Environment", "Development")
"""
Applies tags to ALL resources within the application for better management.

Benefits of tagging:
1. Cost Allocation: Track costs by project and environment
2. Resource Management: Filter and organize resources in AWS Console
3. Security: Implement tag-based access control policies
4. Automation: Use tags for automated operations and cleanup

Tag Schema:
- Project: Identifies the project name ("Bedrock-RAG-App")
- Environment: Indicates the deployment stage ("Development")

Additional recommended tags for production:
- Owner: Team or individual responsible
- CostCenter: Accounting cost center code  
- DataClassification: Security classification (Public, Internal, Confidential)
- Compliance: Regulatory compliance requirements (GDPR, HIPAA, etc.)
- Version: Application version number
"""

# ------------------------------------------------------------------------------
# CloudFormation Template Synthesis
# ------------------------------------------------------------------------------
app.synth()
"""
Synthesizes the CDK application into AWS CloudFormation templates.

What happens during synthesis:
1. CDK constructs are converted to CloudFormation resources
2. Cross-stack references are resolved
3. Templates are validated for correctness
4. Assets (Lambda code, etc.) are prepared for packaging
5. Output files are written to the 'cdk.out' directory

Output Files:
- cdk.out/CloudWatchSetupStack.template.json
- cdk.out/BedrockRagAppStack.template.json
- cdk.out/manifest.json (metadata about the synthesis)
- cdk.out/asset.* (packaged Lambda function code)

After synthesis, you can deploy using:
- `cdk deploy --all` (deploy all stacks)
- `cdk deploy CloudWatchSetupStack` (deploy specific stack)
- `cdk deploy BedrockRagAppStack` (deploy specific stack)
"""
