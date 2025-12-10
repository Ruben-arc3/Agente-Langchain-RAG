import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate


CHROMA_DIR = "chroma_db"


# ================================
# SISTEMA RAG
# ================================
@st.cache_resource
def load_qa_chain():
    embeddings = OllamaEmbeddings(model="embeddinggemma:300m")

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOllama(
        model="llama3",
        temperature=0.1
    )

    template = """
Eres un asistente que responde usando EXCLUSIVAMENTE el contexto proporcionado.
Si la respuesta no está en el contexto, di claramente que no aparece en los documentos.

---------------- CONTEXTO ----------------
{context}
-----------------------------------------

Pregunta del usuario: {question}

Respuesta en español, clara y concisa:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # ✅ Cadena compatible con LangChain 1.x
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


# ================================
# STREAMLIT APP
# ================================
def main():
    st.set_page_config(
        page_title="PDF Assistant | RAG + Ollama",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ===== SIDEBAR =====
    with st.sidebar:
        st.title("📄 PDF Assistant")
        st.markdown("---")

        st.subheader("ℹ️ Sistema RAG")
        st.markdown("""
        - Búsqueda semántica en PDFs
        - Respuestas basadas en contexto
        - Fuentes documentadas
        """)

        st.markdown("---")

        k_value = st.slider("Fragmentos a recuperar", 1, 10, 4)
        temperature = st.slider("Creatividad", 0.0, 1.0, 0.1, 0.05)

        st.markdown("---")

        if "history" in st.session_state:
            st.metric("Preguntas realizadas", len(st.session_state.history))

        if st.button("🗑️ Limpiar historial", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        st.markdown("---")
        st.caption("Powered by Ollama + LangChain + ChromaDB")

    # ===== MAIN =====
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# 🔍 Consulta tus Documentos")
        st.markdown("Haz preguntas sobre tus PDFs indexados.")
    with col2:
        st.metric("Estado", "✅ Conectado")

    st.markdown("---")

    # Cargar QA
    with st.spinner("🔄 Inicializando sistema..."):
        qa_chain = load_qa_chain()

    # ===== CHAT =====
    st.markdown("### 💬 Nueva Consulta")

    with st.container(border=True):
        pregunta = st.text_area(
            "Escribe tu pregunta:",
            placeholder="Ej: ¿Cuál es el resumen del documento?",
            height=120,
            label_visibility="collapsed"
        )

        col_btn1, col_btn2, _ = st.columns([1, 1, 2])
        with col_btn1:
            submit_btn = st.button("🚀 Enviar", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🔄 Reiniciar", use_container_width=True):
                st.rerun()

        if submit_btn and pregunta.strip():
            with st.spinner("🔍 Buscando en documentos..."):

                # ✅ Uso correcto para RetrievalQA
                result = qa_chain({"query": pregunta})

                answer = result["result"]
                sources = result.get("source_documents", [])

                # Inicializar historial
                if "history" not in st.session_state:
                    st.session_state.history = []

                # Guardar historial
                st.session_state.history.append({
                    "question": pregunta,
                    "answer": answer,
                    "sources": [
                        {
                            "file": doc.metadata.get("source_file", "Desconocido"),
                            "page": doc.metadata.get("page", "N/A"),
                        }
                        for doc in sources
                    ],
                })

                # Mostrar respuesta
                st.markdown("---")
                st.markdown("### 📋 Respuesta")

                with st.container(border=True):
                    st.markdown(f"**Pregunta:** {pregunta}")
                    st.markdown("---")
                    st.markdown("**Respuesta:**")
                    st.success(answer)

                    if sources:
                        with st.expander(f"📚 Ver {len(sources)} fuentes"):
                            for i, doc in enumerate(sources, 1):
                                file = doc.metadata.get("source_file", "Desconocido")
                                page = doc.metadata.get("page", "N/A")
                                st.markdown(f"**{i}.** `{file}` - Página {page}")

    # ===== HISTORIAL =====
    if "history" in st.session_state and st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Historial de Consultas")

        for idx, item in enumerate(reversed(st.session_state.history), 1):
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"#### Consulta #{len(st.session_state.history) - idx + 1}")
                with col2:
                    st.caption(f"📄 {len(item['sources'])} fuentes")

                st.markdown(f"**❓ Pregunta:** {item['question']}")

                with st.expander("Ver respuesta completa", expanded=(idx == 1)):
                    st.info(item['answer'])

                    if item["sources"]:
                        st.markdown("**📖 Fuentes:**")
                        for j, src in enumerate(item["sources"], 1):
                            st.code(f"{src['file']} - Página {src['page']}", language="text")

    # ===== FOOTER =====
    st.markdown("---")
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        st.caption("💾 Base vectorial: ChromaDB")
    with colf2:
        st.caption("🤖 Modelo: Llama 3")
    with colf3:
        st.caption("🔧 Embeddings: Gemma 300M")


# ================================
if __name__ == "__main__":
    main()
