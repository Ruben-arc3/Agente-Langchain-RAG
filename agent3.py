import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.agents import initialize_agent, AgentType
import requests

# ===============================
# Configuración
# ===============================
CHROMA_DIR = "chroma_db"
GLOBAL_LLM = None  # para post-procesado de idioma

st.set_page_config(
    page_title="📄 RAG Assistant (Español)",
    page_icon="🤖",
    layout="wide"
)

# ===============================
# TOOL: Buscar en PDFs
# ===============================
def search_pdfs(query: str) -> str:
    embeddings = OllamaEmbeddings(model="embeddinggemma:300m")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    docs = vectordb.similarity_search(query, k=4)

    if not docs:
        return "No se encontró información relevante en los documentos."

    context = ""
    sources = []
    for d in docs:
        source = d.metadata.get("source_file", "Desconocido")
        page = d.metadata.get("page", "N/A")
        sources.append(f"{source} (pág. {page})")
        context += d.page_content + "\n"

    return f"""Información encontrada en los documentos:

{context}

Fuentes: {', '.join(set(sources))}
"""

pdf_tool = Tool(
    name="Buscar_en_PDFs",
    func=search_pdfs,
    description="Usa esta herramienta para buscar información dentro de documentos PDF indexados. Siempre úsala cuando te pregunten sobre el contenido de documentos."
)

# ===============================
# TOOL: Wikipedia
# ===============================
wikipedia_wrapper = WikipediaAPIWrapper(
    lang="es",
    top_k_results=2,
    doc_content_chars_max=2000
)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# Renombrar para consistencia
wikipedia_tool.name = "Buscar_en_Wikipedia"
wikipedia_tool.description = "Usa esta herramienta para buscar información general en Wikipedia en español. Útil para consultas sobre personas, lugares, conceptos, historia, etc. Entrada: término o pregunta a buscar."

# ===============================
# TOOL: Clima (One Call API 3.0)
# ===============================
def consultar_clima(ciudad: str) -> str:
    """Consulta el clima actual usando la API 2.5 gratuita de OpenWeatherMap"""
    try:
        # API key de OpenWeatherMap
        api_key = "6acd66c9486dec5b473367fb2fb34143"
        
        # Limpiar el nombre de la ciudad
        ciudad = ciudad.strip()
        
        # Paso 1: Obtener ID de la ciudad usando Geocoding API
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={ciudad}&limit=1&appid={api_key}"
        geo_response = requests.get(geo_url, timeout=10)
        
        if geo_response.status_code != 200:
            return f"Error de conexion al buscar {ciudad}. Codigo: {geo_response.status_code}"
        
        geo_data = geo_response.json()
        if not geo_data:
            return f"No se encontro la ciudad {ciudad}. Intenta con otro nombre."
        
        # Obtener coordenadas y nombre
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        nombre_ciudad = geo_data[0].get('local_names', {}).get('es', geo_data[0]['name'])
        
        # Paso 2: Obtener clima actual con weather endpoint (más simple)
        weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&lang=es&appid={api_key}"
        weather_response = requests.get(weather_url, timeout=10)
        
        if weather_response.status_code == 200:
            data = weather_response.json()
            
            temp = data['main']['temp']
            sensacion = data['main']['feels_like']
            descripcion = data['weather'][0]['description']
            humedad = data['main']['humidity']
            viento = data['wind']['speed']
            presion = data['main']['pressure']
            
            # Paso 3: Obtener pronóstico usando forecast endpoint
            forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&lang=es&appid={api_key}"
            forecast_response = requests.get(forecast_url, timeout=10)
            
            pronostico_texto = ""
            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                # Obtener próximas 3 horas
                if 'list' in forecast_data and len(forecast_data['list']) > 0:
                    proxima = forecast_data['list'][0]
                    temp_prox = proxima['main']['temp']
                    desc_prox = proxima['weather'][0]['description']
                    pronostico_texto = f", pronostico proximas horas: {temp_prox}C con {desc_prox}"
            
            return f"Clima en {nombre_ciudad}: temperatura {temp}C (sensacion {sensacion}C), {descripcion}, humedad {humedad}%, viento {viento} m/s, presion {presion} hPa{pronostico_texto}"
        elif weather_response.status_code == 401:
            return "La API Key es invalida. Verifica tu clave de OpenWeatherMap."
        elif weather_response.status_code == 429:
            return "Limite de solicitudes excedido. Intenta en unos minutos."
        else:
            return f"Error al obtener clima. Codigo HTTP: {weather_response.status_code}"
            
    except requests.exceptions.Timeout:
        return "Timeout: el servicio de clima tardo demasiado en responder."
    except requests.exceptions.ConnectionError:
        return "Error de conexion: no se pudo conectar al servicio de clima."
    except KeyError as e:
        return f"Error en datos de respuesta: falta campo {str(e)}"
    except Exception as e:
        return f"Error inesperado: {str(e)}"

clima_tool = Tool(
    name="Consultar_Clima",
    func=consultar_clima,
    description="Usa esta herramienta para consultar el clima actual de cualquier ciudad del mundo. Entrada: nombre de la ciudad (ej: 'Madrid', 'Bogotá', 'Ciudad de México')"
)

# ===============================
# Construir el agente (cache)
# ===============================
@st.cache_resource
def build_agent():
    global GLOBAL_LLM

    # LLM configurado para español
    GLOBAL_LLM = ChatOllama(
        model="llama3",
        temperature=0.1
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Prompt del agente en español
    prefix = """Eres un asistente útil que SIEMPRE responde en ESPAÑOL.

INSTRUCCIONES IMPORTANTES:
1. NUNCA respondas en inglés, solo en español
2. Cuando el usuario pregunte sobre documentos, DEBES usar la herramienta "Buscar_en_PDFs"
3. Para información general, usa "Buscar_en_Wikipedia"
4. Para el clima, usa "Consultar_Clima"
5. Si la herramienta devuelve información, basa tu respuesta en esa información
6. Si no hay información en los documentos, dilo claramente
7. Todas tus respuestas deben ser en español, sin excepción

Tienes acceso a las siguientes herramientas:"""

    suffix = """¡Comencemos!

Historial de conversación:
{chat_history}

Pregunta del usuario: {input}
{agent_scratchpad}

Recuerda: Tu respuesta DEBE estar COMPLETAMENTE en español."""

    agent = initialize_agent(
        tools=[pdf_tool, wikipedia_tool, clima_tool],
        llm=GLOBAL_LLM,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate",
        agent_kwargs={
            "prefix": prefix,
            "suffix": suffix
        }
    )
    return agent

agent = build_agent()

# ===============================
# Post-procesador de idioma
# ===============================
def forzar_espanol(texto: str) -> str:
    """Traduce cualquier respuesta al español si detecta inglés"""
    global GLOBAL_LLM

    # Detectar si hay palabras en inglés
    palabras_ingles = [
        "the", "and", "is", "are", "of", "to", "in", "that", 
        "for", "with", "on", "at", "this", "from", "by", "was", "were"
    ]
    
    texto_lower = texto.lower()
    tiene_ingles = any(f" {palabra} " in f" {texto_lower} " for palabra in palabras_ingles)
    
    # Si no detecta inglés, devolver tal cual
    if not tiene_ingles:
        return texto
    
    # Si detecta inglés, forzar traducción
    prompt = f"""Traduce el siguiente texto al español. Si ya está en español, déjalo igual.
IMPORTANTE: Tu respuesta debe contener SOLO el texto traducido, nada más.

Texto:
{texto}

Traducción al español:"""
    
    try:
        resp = GLOBAL_LLM.invoke(prompt)
        resultado = resp.content if hasattr(resp, "content") else str(resp)
        return resultado.strip()
    except:
        return texto

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.title("🤖 RAG Assistant")
    st.markdown("---")
    st.markdown("**Estado:** ✅ Conectado")
    st.markdown("**Vector DB:** Chroma")
    st.markdown("**Modelo:** Llama3 (Ollama)")
    st.markdown("**Idioma:** 🇪🇸 Español")
    
    st.markdown("---")
    st.markdown("### 🛠️ Herramientas disponibles")
    st.markdown("📄 **Buscar en PDFs** - Busca en documentos")
    st.markdown("📚 **Wikipedia** - Información general")
    st.markdown("🌤️ **Clima** - Consulta el tiempo ✅")
    
    st.markdown("---")
    st.markdown("### ℹ️ Estado de APIs")
    st.success("✅ One Call API 3.0 configurada")

    if st.button("🗑️ Limpiar historial"):
        if "history" in st.session_state:
            st.session_state.history = []
        st.rerun()

# ===============================
# UI principal
# ===============================
st.title("📄 Asistente de Documentos")
st.markdown("*Todas las respuestas en español* 🇪🇸")

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("### 💬 Haz una pregunta")
st.markdown("*Puedo buscar en tus documentos, consultar Wikipedia o decirte el clima* 🔍")

user_input = st.text_area(
    "Pregunta:",
    placeholder="Ej: ¿Qué dice el documento sobre...? / ¿Quién fue Albert Einstein? / ¿Cómo está el clima en Madrid?",
    height=120,
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 4])
with col1:
    send_btn = st.button("🚀 Enviar")
with col2:
    if st.button("🔄 Limpiar campo"):
        st.rerun()

# ===============================
# Procesar
# ===============================
if send_btn and user_input.strip():
    with st.spinner("🔍 Procesando tu consulta..."):
        try:
            raw = agent.run(user_input)
            respuesta = forzar_espanol(raw)
        except Exception as e:
            respuesta = f"Error al procesar la consulta: {str(e)}"

    st.session_state.history.append({
        "question": user_input,
        "answer": respuesta
    })

# ===============================
# Historial
# ===============================
if st.session_state.history:
    st.markdown("---")
    st.markdown("#Historial de conversación")

    for i, chat in enumerate(reversed(st.session_state.history)):
        with st.container(border=True):
            st.markdown(f"**Pregunta:** {chat['question']}")
            st.markdown("**Respuesta:**")
            st.success(chat['answer'])
else:
    st.info("👋 Haz una pregunta para comenzar")
    
    # Sugerencias de ejemplo
    st.markdown("**💡 Prueba preguntas como:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("¿Qué dice el documento sobre...?*")
    with col2:
        st.markdown("¿Quién fue Gabriel García Márquez?*")
    with col3:
        st.markdown("¿Cómo está el clima en Barcelona?*")