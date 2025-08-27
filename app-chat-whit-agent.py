import streamlit as st
import requests
import json
import re
from datetime import datetime

class RagClient:
    def __init__(self, api_url):
        # Asegurarse de que la URL tenga el formato correcto
        if api_url and not api_url.endswith('/'):
            api_url += '/'
        self.api_url = api_url
        print(f"Initializing RAG client with URL: {api_url}")
    
    def ask_question(self, question):
        """Send question to RAG API"""
        try:
            if not self.api_url or self.api_url == "https://your-api-url.execute-api.region.amazonaws.com/prod/":
                return {"answer": "⚠️ Por favor configura la URL de API Gateway primero", "sources": []}
            
            # Construir la URL correctamente
            query_url = f"{self.api_url}query"
            print(f"Sending request to: {query_url}")
            print(f"Question: {question}")
            
            response = requests.get(
                query_url,
                params={"question": question},
                timeout=60  # Aumentar timeout para respuestas largas
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"answer": f"❌ Error del servidor: {response.status_code} - {response.text}", "sources": []}
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
            return {"answer": f"❌ Error de conexión: Verifica que la URL sea correcta y que el API esté desplegado. Error: {str(e)}", "sources": []}
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            return {"answer": "❌ Error procesando la respuesta del servidor", "sources": []}

def is_valid_api_url(url):
    """Validate if the URL looks like a valid API Gateway URL"""
    if not url:
        return False
    # Patrón para URLs de API Gateway
    pattern = r'^https://[a-z0-9]+\.execute-api\.[a-z0-9-]+\.amazonaws\.com/.*$'
    return re.match(pattern, url) is not None

def init_session_state():
    """Initialize session state variables"""
    if "rag_client" not in st.session_state:
        st.session_state.rag_client = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_url" not in st.session_state:
        st.session_state.api_url = ""
    if "connection_tested" not in st.session_state:
        st.session_state.connection_tested = False

def setup_sidebar():
    """Setup the sidebar configuration"""
    st.sidebar.header("⚙️ Configuración")
    
    st.sidebar.info("""
    **Cómo obtener la URL:**
    1. Ejecuta `cdk deploy`
    2. Busca en los outputs: `MyBedrockRagAppStack.RagApiGatewayEndpoint`
    3. Copia esa URL y pégala aquí
    """)
    
    api_url = st.sidebar.text_input(
        "🔗 API Gateway URL",
        value=st.session_state.get("api_url", ""),
        placeholder="https://abc123.execute-api.us-east-1.amazonaws.com/prod/",
        help="URL obtenida del despliegue de CDK"
    )
    
    # Validar URL
    if api_url and not is_valid_api_url(api_url):
        st.sidebar.warning("⚠️ La URL no parece ser una URL válida de API Gateway")
    
    st.sidebar.markdown("---")
    
    # Botones de acción
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔗 Conectar", use_container_width=True):
            if api_url and is_valid_api_url(api_url):
                st.session_state.rag_client = RagClient(api_url)
                st.session_state.api_url = api_url
                st.session_state.connection_tested = False
                st.sidebar.success("✅ ¡Conectado al RAG API!")
            else:
                st.sidebar.error("❌ URL inválida")
    
    with col2:
        if st.button("🔄 Probar Conexión", use_container_width=True) and st.session_state.rag_client:
            with st.sidebar:
                with st.spinner("Probando conexión..."):
                    test_response = st.session_state.rag_client.ask_question("Hola")
                    if "error" not in test_response.get("answer", "").lower():
                        st.success("✅ ¡Conexión exitosa!")
                        st.session_state.connection_tested = True
                    else:
                        st.error("❌ Error de conexión")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Estado del Sistema")
    
    # Mostrar estado de conexión
    if st.session_state.rag_client and st.session_state.connection_tested:
        st.sidebar.success("**Estado:** ✅ Conectado")
        st.sidebar.info(f"**URL:**\n`{st.session_state.api_url}`")
    elif st.session_state.rag_client:
        st.sidebar.warning("**Estado:** ⚠️ Conectado (no verificado)")
    else:
        st.sidebar.error("**Estado:** ❌ Desconectado")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("❓ ¿Cómo usar?")
    
    st.sidebar.info("""
    1. ⚡ Despliega la infraestructura con CDK
    2. 🔗 Obtén la URL de API Gateway
    3. 📤 Sube documentos PDF/CSV al bucket S3
    4. 💬 ¡Comienza a chatear con tus documentos!
    """)
    
    return api_url

def display_chat_history():
    """Display the chat history"""
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message(chat["role"]):
            if chat["role"] == "assistant":
                # Mostrar respuesta del asistente
                st.markdown(chat["content"])
                
                # Mostrar fuentes y estadísticas si están disponibles
                if "sources" in chat and chat["sources"]:
                    with st.expander("📁 Documentos de Referencia"):
                        st.write("**Fuentes utilizadas:**")
                        for source in chat["sources"]:
                            st.write(f"• 📄 {source}")
                
                if "stats" in chat:
                    st.caption(f"📊 {chat['stats']}")
            else:
                # Mostrar pregunta del usuario
                st.markdown(chat["content"])

def display_connection_help():
    """Display help message when not connected"""
    st.warning("""
    ## ⚡ Para comenzar:
    
    1. **Despliega la infraestructura** con CDK:
    ```bash
    cdk deploy
    ```
    
    2. **Obtén la URL de API Gateway** de los outputs
    
    3. **Pega la URL** en el sidebar
    
    4. **Sube documentos** al bucket S3 en la carpeta `uploads/`
    
    5. **¡Comienza a chatear!** 🎉
    """)
    
    st.code("""
# Comandos para desplegar:
cdk synth
cdk deploy

# Busca en los outputs:
MyBedrockRagAppStack.RagApiGatewayEndpoint = https://...
    """)

def main():
    """
    RAG Chat Interface - Query your documents using Amazon Bedrock
    """
    st.set_page_config(
        page_title="🤖 RAG Chat Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] {
        font-size: 16px;
    }
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🤖 RAG Chat Assistant")
        st.markdown("### 💬 Chatea con tus documentos usando Amazon Bedrock")
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    
    st.markdown("---")
    
    # Initialize session state
    init_session_state()
    
    # Setup sidebar
    api_url = setup_sidebar()
    
    # Main chat area
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
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get response from RAG API
            with st.chat_message("assistant"):
                with st.spinner("🔍 Buscando en los documentos..."):
                    response = st.session_state.rag_client.ask_question(prompt)
                
                if "answer" in response:
                    # Display answer
                    st.markdown(response["answer"])
                    
                    # Prepare statistics
                    stats_parts = []
                    if response.get("documents_used"):
                        stats_parts.append(f"📊 {response['documents_used']} documentos")
                    if response.get("context_chunks"):
                        stats_parts.append(f"{response['context_chunks']} fragmentos")
                    
                    stats_text = " • ".join(stats_parts) if stats_parts else ""
                    
                    # Display sources if available
                    if response.get("sources"):
                        with st.expander("📁 Documentos de Referencia"):
                            st.write("**Fuentes utilizadas:**")
                            for source in response["sources"]:
                                st.write(f"• 📄 {source}")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", []),
                        "stats": stats_text,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    st.error("❌ Error obteniendo respuesta del API")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "❌ Error obteniendo respuesta",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Rerun to update chat history
            st.rerun()
    
    # Footer and controls
    st.markdown("---")
    
    if st.session_state.chat_history:
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info(f"💬 {len(st.session_state.chat_history)} mensajes en la conversación")
        
        with col2:
            if st.button("🗑️ Limpiar Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with col3:
            if st.button("📥 Exportar Conversación", use_container_width=True):
                chat_data = {
                    "export_date": datetime.now().isoformat(),
                    "api_url": st.session_state.api_url,
                    "messages": st.session_state.chat_history
                }
                
                st.download_button(
                    label="⬇️ Descargar JSON",
                    data=json.dumps(chat_data, indent=2, ensure_ascii=False),
                    file_name=f"rag_chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

if __name__ == "__main__":
    main()