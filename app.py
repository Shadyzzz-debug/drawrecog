import streamlit as st
import base64
import json
import requests
import time
import io
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import numpy as np

# --- Configuraciones del LLM para el entorno ---
GEMINI_CHAT_MODEL = "gemini-2.5-flash-preview-09-2025" 

# --- CSS BASE (Estilo Limpio para Dibujo) ---
base_css = """
<style>
/* Reset básico para Streamlit */
.stApp {
    background-color: #F8F8F8;
    color: #333333;
    font-family: Arial, sans-serif;
}

h1 {
    color: #2C3E50;
    text-align: center;
    border-bottom: 2px solid #3498DB;
    padding-bottom: 10px;
    margin-bottom: 20px;
}

h3 {
    color: #34495E;
}

/* Contenedores de entrada/salida */
div[data-testid="stTextInput"], div[data-testid="stTextarea"] {
    background-color: #ECF0F1;
    border-radius: 5px;
    padding: 10px;
}

/* Botones (Estilo Moderno) */
.stButton>button {
    background-color: #3498DB; /* Azul Brillante */
    color: white;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
    border-radius: 5px;
    transition: all 0.2s;
    box-shadow: 0 4px #2980B9;
}

.stButton>button:hover {
    background-color: #2980B9;
    box-shadow: 0 6px #1F618D;
    transform: translateY(-2px);
}

.stButton>button:active {
    box-shadow: 0 2px #1F618D;
    transform: translateY(2px);
}

/* Placeholder para la respuesta */
div[data-testid="stMarkdownContainer"] {
    background-color: #FFFFFF;
    padding: 20px;
    border: 1px solid #BDC3C7;
    border-radius: 5px;
    margin-top: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
"""
st.markdown(base_css, unsafe_allow_html=True)


# --- Funciones de Utilidad (Uso de 'requests' para la API de Gemini) ---

def safe_fetch_request(url, api_key, method='POST', headers=None, body=None, max_retries=3, delay=1):
    """Realiza llamadas a la API con reintentos y retroceso exponencial usando 'requests'."""
    if headers is None:
        headers = {'Content-Type': 'application/json'}
    
    # Agregar la clave API a la URL
    url_with_key = f"{url}?key={api_key}"
    
    for attempt in range(max_retries):
        try:
            response = requests.request(method, url_with_key, headers=headers, data=body, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
                continue
            else:
                error_detail = response.text if response.text else f"Código de estado: {response.status_code}"
                raise Exception(f"Fallo en la llamada a la API ({response.status_code}). {error_detail}")
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
                continue
            raise Exception(f"Error de red/conexión: {e}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))
                continue
            raise e
    raise Exception("Llamada a la API fallida después de múltiples reintentos.")


def get_gemini_vision_answer(base64_image: str, mime_type: str, user_prompt: str, api_key: str) -> str:
    """Invoca la API de Gemini para análisis de visión."""
    
    # Construcción del payload
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_image
                        }
                    }
                ]
            }
        ]
    }
    
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_CHAT_MODEL}:generateContent"

    response_data = safe_fetch_request(apiUrl, api_key, body=json.dumps(payload))
    
    # Manejo de la respuesta
    candidate = response_data.get('candidates', [{}])[0]
    text = candidate.get('content', {}).get('parts', [{}])[0].get('text', None)

    if text:
        return text
    
    # Revisar si hay un mensaje de error explícito de la API
    error_message = response_data.get('error', {}).get('message', 'Respuesta incompleta o vacía del modelo.')
    raise Exception(f"Fallo en la Visión: {error_message}")


# --- Streamlit App Setup ---
st.set_page_config(page_title='Tablero Inteligente', layout="centered")
st.title('Tablero Inteligente con Visión Gemini')

# --- Sidebar para Controles ---
with st.sidebar:
    st.subheader("Acerca de:")
    st.markdown("Esta aplicación interpreta bocetos dibujados en el panel usando la **Gemini API**.")
    st.markdown("---")
    
    st.subheader("Configuración del Boceto")
    drawing_mode = "freedraw"
    stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 5)

# --- Canvas Principal ---
st.subheader("Dibuja el boceto en el panel y presiona el botón para analizarlo")

# Canvas Parameters
stroke_color = "#000000"  # Línea negra
bg_color = '#FFFFFF'     # Fondo blanco

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.0)",  # Sin relleno
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=300,
    width=400,
    drawing_mode=drawing_mode,
    key="canvas_intelligent",
)

# --- Controles de la API y Análisis ---
ke = st.text_input('Ingresa tu Clave API (Gemini Key)', type="password")

additional_details = st.text_area(
    "Instrucción de Análisis Adicional:",
    placeholder="Ej: Describe brevemente la imagen o clasifica el objeto dibujado.",
    value="Describe en español y de forma concisa el objeto o concepto que has identificado en el boceto."
)

analyze_button = st.button("Analiza la Imagen", type="primary")


# --- Lógica de Análisis ---
if canvas_result.image_data is not None and analyze_button:
    
    # 1. Validación
    if not ke:
        st.warning("Por favor, ingresa tu Clave API de Gemini.")
        st.stop()
        
    # Verificar si el dibujo está vacío
    if not canvas_result.image_data.any():
        st.warning("El lienzo está en blanco. Por favor, dibuja un boceto para analizar.")
        st.stop()

    with st.spinner("Analizando el boceto con Visión Gemini..."):
        try:
            # 2. Preparar la Imagen (Codificación Base64)
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA').convert('RGB')
            
            # Guardar en memoria como PNG para Base64
            buf = io.BytesIO()
            input_image.save(buf, format='PNG')
            file_bytes = buf.getvalue()
            
            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            mime_type = 'image/png'

            # 3. Construir el Prompt
            prompt_text = additional_details
            
            # 4. Invocar la Visión (Usando la función con requests)
            response = get_gemini_vision_answer(base64_image, mime_type, prompt_text, ke)
            
            # 5. Mostrar la Respuesta
            st.markdown("### Resultado del Análisis:")
            st.markdown(response)
            
        except Exception as e:
            st.error(f"Error durante la Invocación a la API: {e}")
            
elif analyze_button:
    st.warning("Asegúrate de haber dibujado algo y de haber ingresado tu Clave API.")
