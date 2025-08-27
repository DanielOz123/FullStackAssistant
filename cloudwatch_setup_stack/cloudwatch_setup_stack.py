from aws_cdk import (
    Stack,
    aws_iam as iam,
    CfnOutput
)
from constructs import Construct

class CloudWatchSetupStack(Stack):
    """
    AWS CDK Stack for setting up CloudWatch Logs integration for API Gateway.
    
    This stack creates the necessary IAM role and configuration to enable 
    CloudWatch Logs for API Gateway, which is a prerequisite for detailed
    logging and monitoring of API requests and responses.
    
    Purpose:
    - Creates IAM role with permissions for API Gateway to write to CloudWatch Logs
    - Provides the role ARN as a stack output for easy reference
    - Solves the common error: "CloudWatch Logs role ARN must be set in account settings"
    
    Prerequisites:
    - This stack should be deployed BEFORE any API Gateway stacks
    - The role ARN should be configured in API Gateway account settings
    
    Architecture:
    [API Gateway] → [CloudWatchSetupStack IAM Role] → [CloudWatch Logs]
    """
    
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """
        Initialize the CloudWatch setup stack.
        
        Args:
            scope: The parent construct (usually the app)
            construct_id: The unique identifier for this stack
            **kwargs: Additional arguments passed to the Stack base class
        """
        super().__init__(scope, construct_id, **kwargs)

        # ----------------------------------------------------------------------
        # IAM Role for API Gateway CloudWatch Logs
        # ----------------------------------------------------------------------
        api_gw_logs_role = iam.Role(
            self, "ApiGatewayCloudWatchLogsRole",
            assumed_by=iam.ServicePrincipal("apigateway.amazonaws.com"),
            role_name="APIGatewayCloudWatchLogsRole",
            description="IAM role that allows API Gateway to write logs to CloudWatch",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonAPIGatewayPushToCloudWatchLogs"
                )
            ],
        )
        
        # ----------------------------------------------------------------------
        # CloudFormation Output for Role ARN
        # ----------------------------------------------------------------------
        CfnOutput(
            self, "ApiGatewayLogsRoleArn",
            value=api_gw_logs_role.role_arn,
            description="ARN of the IAM role for API Gateway CloudWatch Logs integration. "
                       "Use this ARN to configure API Gateway account settings.",
            export_name="ApiGatewayCloudWatchLogsRoleArn"  # Enables cross-stack reference
        )


