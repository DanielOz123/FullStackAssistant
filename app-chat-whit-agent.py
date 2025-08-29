import streamlit as st
import requests
import json
import re
from datetime import datetime
from typing import Dict, Any

class RagClient:
    """
    Client class for communicating with the RAG API Gateway endpoint.
    
    Handles all HTTP requests to the backend API and manages connection state.
    Provides robust error handling and logging for API interactions.
    """
    
    def __init__(self, api_url: str):
        """
        Initialize the RAG client with the API Gateway URL.
        
        Args:
            api_url (str): The base URL of the API Gateway endpoint
            
        Example:
            client = RagClient("https://abc123.execute-api.us-east-1.amazonaws.com/prod/")
        """
        # Ensure URL ends with trailing slash for proper endpoint construction
        if api_url and not api_url.endswith('/'):
            api_url += '/'
        self.api_url = api_url
        print(f"Initializing RAG client with URL: {api_url}")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Send a question to the RAG API and return the response.
        
        Args:
            question (str): The user's question to send to the API
            
        Returns:
            Dict: Response containing answer, sources, and metadata
            
        Example:
            response = client.ask_question("What are the company policies?")
            print(response["answer"])
        """
        try:
            # Validate API URL is configured
            if not self.api_url or self.api_url == "":
                return {"answer": "⚠️ Please configure the API Gateway URL first", "sources": []}
            
            # Construct the full query URL
            query_url = f"{self.api_url}query"
            print(f"Sending request to: {query_url}")
            print(f"Question: {question}")
            
            # Send GET request to API Gateway
            response = requests.get(
                query_url,
                params={"question": question},
                timeout=60  # Increased timeout for complex queries
            )
            
            print(f"Response status: {response.status_code}")
            
            # Handle successful response
            if response.status_code == 200:
                return response.json()
            else:
                # Handle server errors
                return {"answer": f"❌ Server error: {response.status_code} - {response.text}", "sources": []}
                
        except requests.exceptions.RequestException as e:
            # Handle connection errors
            print(f"Request error: {str(e)}")
            return {"answer": f"❌ Connection error: Please verify the URL is correct and the API is deployed. Error: {str(e)}", "sources": []}
        except json.JSONDecodeError as e:
            # Handle invalid JSON responses
            print(f"JSON decode error: {str(e)}")
            return {"answer": "❌ Error processing server response", "sources": []}

def is_valid_api_url(url: str) -> bool:
    """
    Validate if the URL matches the expected API Gateway pattern.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if URL matches API Gateway pattern, False otherwise
        
    Example:
        is_valid_api_url("https://abc123.execute-api.us-east-1.amazonaws.com/prod/")
        True
    """
    if not url:
        return False
    # Regex pattern for API Gateway URLs
    pattern = r'^https://[a-z0-9]+\.execute-api\.[a-z0-9-]+\.amazonaws\.com/.*$'
    return re.match(pattern, url) is not None

def init_session_state():
    """
    Initialize all session state variables for the Streamlit app.
    
    Session state persists across reruns and maintains app state.
    """
    if "rag_client" not in st.session_state:
        st.session_state.rag_client = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_url" not in st.session_state:
        st.session_state.api_url = ""
    if "connection_tested" not in st.session_state:
        st.session_state.connection_tested = False

def setup_sidebar() -> str:
    """
    Create and configure the sidebar with connection settings.
    
    Returns:
        str: The API URL entered by the user
    """
    st.sidebar.header("⚙️ Configuration")
    
    # Help information for users
    st.sidebar.info("""
        **How to get the URL:**
        1. Run `cdk deploy`
        2. Look for output: `BedrockRagAppStack.RagApiGatewayEndpoint`
        3. Copy that URL and paste it here
    """)
    
    # API URL input field
    api_url = st.sidebar.text_input(
        "🔗 API Gateway URL",
        value=st.session_state.get("api_url", ""),
        placeholder="https://abc123.execute-api.us-east-1.amazonaws.com/prod/",
        help="URL obtained from CDK deployment outputs"
    )
    
    # URL validation
    if api_url and not is_valid_api_url(api_url):
        st.sidebar.warning("⚠️ The URL doesn't appear to be a valid API Gateway URL")
    
    st.sidebar.markdown("---")
    
    # Action buttons in two columns
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        # Connect button
        if st.button("🔗 Connect", use_container_width=True, help="Establish connection to RAG API"):
            if api_url and is_valid_api_url(api_url):
                st.session_state.rag_client = RagClient(api_url)
                st.session_state.api_url = api_url
                st.session_state.connection_tested = False
                st.sidebar.success("✅ Connected to RAG API!")
            else:
                st.sidebar.error("❌ Invalid URL")
    
    with col2:
        # Test connection button
        if st.button("🔄 Test Connection", use_container_width=True, help="Test API connectivity") and st.session_state.rag_client:
            with st.sidebar:
                with st.spinner("Testing connection..."):
                    test_response = st.session_state.rag_client.ask_question("Hello")
                    if "error" not in test_response.get("answer", "").lower():
                        st.success("✅ Connection successful!")
                        st.session_state.connection_tested = True
                    else:
                        st.error("❌ Connection failed")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 System Status")
    
    # Display connection status
    if st.session_state.rag_client and st.session_state.connection_tested:
        st.sidebar.success("**Status:** ✅ Connected")
        st.sidebar.info(f"**URL:**\n`{st.session_state.api_url}`")
    elif st.session_state.rag_client:
        st.sidebar.warning("**Status:** ⚠️ Connected (unverified)")
    else:
        st.sidebar.error("**Status:** ❌ Disconnected")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("❓ How to Use?")
    
    # Usage instructions
    st.sidebar.info("""
        1. ⚡ Deploy infrastructure with CDK
        2. 🔗 Get API Gateway URL from outputs
        3. 📤 Upload PDF/CSV documents to S3 bucket
        4. 💬 Start chatting with your documents!
    """)
    
    return api_url

def display_chat_history():
    """
    Display the entire chat history with proper formatting.
    
    Shows user questions and assistant responses with sources and metadata.
    """
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message(chat["role"]):
            if chat["role"] == "assistant":
                # Display assistant response
                st.markdown(chat["content"])
                
                # Show sources if available
                if "sources" in chat and chat["sources"]:
                    with st.expander("📁 Reference Documents"):
                        st.write("**Sources used:**")
                        for source in chat["sources"]:
                            st.write(f"• 📄 {source}")
                
                # Show statistics if available
                if "stats" in chat:
                    st.caption(f"📊 {chat['stats']}")
            else:
                # Display user question
                st.markdown(chat["content"])

def display_connection_help():
    """
    Display help message when not connected to API.
    
    Provides deployment instructions and code examples for users.
    """
    st.warning("""
    ## ⚡ To get started:
    
    1. **Deploy infrastructure** with CDK:
    ```bash
    cdk deploy
    ```
    
    2. **Get API Gateway URL** from the outputs
    
    3. **Paste the URL** in the sidebar
    
    4. **Upload documents** to S3 bucket in the `uploads/` folder
    
    5. **Start chatting!** 🎉
    """)
    
    # Code examples
    st.code("""
        # Commands to deploy:
        cdk synth
        cdk deploy

        # Look for in outputs:
        MyBedrockRagAppStack.RagApiGatewayEndpoint = https://...
    """)

def main():
    """
    Main Streamlit application for RAG Chat Interface.
    
    Creates a user-friendly interface for querying documents using
    Amazon Bedrock and displaying results in a chat-like format.
    """
    # Page configuration
    st.set_page_config(
        page_title="🤖 FS Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS styling
    st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stChatMessage [data-testid="stMarkdownContainer"] {
        font-size: 16px;
        line-height: 1.6;
    }
    .stButton button {
        width: 100%;
        border-radius: 6px;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Application header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 FS Chat Assistant")
        st.markdown("### 💬 Chat with your documents using Amazon Bedrock")
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    
    st.markdown("---")
    
    # Initialize session state
    init_session_state()
    
    # Setup sidebar and get API URL
    api_url = setup_sidebar()
    
    # Main chat interface
    if not st.session_state.rag_client:
        display_connection_help()
    else:
        # Display chat history
        display_chat_history()
        
        # Chat input
        if prompt := st.chat_input("💭 Haz una pregunta sobre tus documentos..."):
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user", 
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching through documents..."):
                    response = st.session_state.rag_client.ask_question(prompt)
                
                if "answer" in response:
                    # Display the answer
                    st.markdown(response["answer"])
                    
                    # Prepare statistics for display
                    stats_parts = []
                    if response.get("documents_used"):
                        stats_parts.append(f"📊 {response['documents_used']} documents")
                    if response.get("context_chunks"):
                        stats_parts.append(f"{response['context_chunks']} chunks")
                    
                    stats_text = " • ".join(stats_parts) if stats_parts else ""
                    
                    # Display sources if available
                    if response.get("sources"):
                        with st.expander("📁 Reference Documents", expanded=False):
                            st.write("**Sources used in this response:**")
                            for source in response["sources"]:
                                st.write(f"• 📄 {source}")
                    
                    # Add to chat history with metadata
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                        "stats": stats_text,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    # Handle API errors
                    st.error("❌ Error getting response from API")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "❌ Error getting response",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Rerun to update the chat history display
            st.rerun()
    
    # Footer with chat controls
    st.markdown("---")
    
    if st.session_state.chat_history:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info(f"💬 {len(st.session_state.chat_history)} messages in conversation")
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True, help="Clear conversation history"):
                st.session_state.chat_history = []
                st.rerun()
        
        with col3:
            if st.button("📥 Export Conversation", use_container_width=True, help="Download chat as JSON"):
                chat_data = {
                    "export_date": datetime.now().isoformat(),
                    "api_url": st.session_state.api_url,
                    "messages": st.session_state.chat_history
                }
                
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json.dumps(chat_data, indent=2, ensure_ascii=False),
                    file_name=f"rag_chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()