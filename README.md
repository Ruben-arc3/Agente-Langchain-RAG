# Agente LangChain RAG — Español

Proyecto Streamlit que implementa un asistente RAG (Retrieval-Augmented Generation) usando Ollama, LangChain y ChromaDB. Permite:

- Buscar información dentro de PDFs indexados.
- Responder usando contexto recuperado mediante RAG.
- Consultar Wikipedia en español.
- Obtener clima por ciudad usando OpenWeatherMap.
- Interfaz web con historial, memoria y herramientas.

---

## Estructura del repositorio

RAG/
├─ app.py
├─ qa_app.py
├─ ingest.py
├─ requirements.txt
├─ pdfs/
├─ chroma_db/
├─ .gitignore
└─ README.md

---

## Requisitos

- Python 3.9+
- Ollama instalado localmente
- Dependencias en `requirements.txt`

Ejemplo:

streamlit  
requests  
langchain  
langchain-community  
langchain-classic  
chromadb  
ollama  
PyPDF2  
tqdm  

Instalación:

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
