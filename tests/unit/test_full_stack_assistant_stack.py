import aws_cdk as core
import aws_cdk.assertions as assertions

from full_stack_assistant.full_stack_assistant_stack import FullStackAssistantStack

# example tests. To run these tests, uncomment this file along with the example
# resource in full_stack_assistant/full_stack_assistant_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = FullStackAssistantStack(app, "full-stack-assistant")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
