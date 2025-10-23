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

# --- CSS MÍSTICO / ENIGMÁTICO (Azul Profundo y Oro) ---
base_css = """
<style>
/* Reset básico para Streamlit */
.stApp {
    background-color: #0E1A2B; /* Azul profundo / medianoche */
    color: #E6E6E6; /* Tono pergamino o plata */
    font-family: 'Georgia', serif; 
}

h1 {
    color: #FFD700; /* Oro */
    text-align: center;
    border-bottom: 3px solid #34495E; /* Borde de acero */
    padding-bottom: 10px;
    margin-bottom: 30px;
    font-size: 2.5em;
    letter-spacing: 1.5px;
    text-shadow: 2px 2px 4px #000000;
}

h3 {
    color: #C0C0C0; /* Plata */
    margin-top: 25px;
    font-weight: normal;
}

/* Contenedores de entrada/salida (Pergamino) */
div[data-testid="stTextInput"], div[data-testid="stTextarea"] {
    background-color: #1A2C3E; /* Azul oscuro acentuado */
    border: 1px solid #FFD700;
    border-radius: 5px;
    padding: 10px;
    color: #F0F0F0;
}

/* Sidebar */
.css-1d3w5ta, .css-1lcbmhc {
    background-color: #152438;
    color: #C0C0C0;
}

/* Botones (Sello de Convocación) */
.stButton>button {
    background-color: #34495E; /* Acero/Pizarra */
    color: #FFD700; /* Texto Dorado */
    border: 2px solid #FFD700; /* Borde Dorado */
    padding: 10px 25px;
    font-weight: bold;
    border-radius: 10px;
    transition: all 0.3s;
    box-shadow: 0 5px #1A2C3E;
    letter-spacing: 1px;
}

.stButton>button:hover {
    background-color: #4A637F; /* Ligeramente más claro */
    box-shadow: 0 8px #0E1A2B;
    transform: translateY(-2px);
}

.stButton>button:active {
    box-shadow: 0 2px #0E1A2B;
    transform: translateY(2px);
}

/* Placeholder para la respuesta (Piedra Revelada) */
div[data-testid="stMarkdownContainer"] {
    background-color: #1A2C3E;
    padding: 20px;
    border: 2px solid #FFD700;
    border-radius: 8px;
    margin-top: 25px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    color: #E6E6E6;
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
st.title('Tablero Místico de la Clarividencia')

# --- Sidebar para Controles ---
with st.sidebar:
    st.subheader("El Scriptorium")
    st.markdown("Este antiguo Tablero Místico invoca la **Visión de Géminis** para interpretar tus trazos. Cada línea es un conjuro.")
    st.markdown("---")
    
    st.subheader("Trazo del Augurio")
    drawing_mode = "freedraw"
    stroke_width = st.slider('Define la Potencia del Trazo', 1, 30, 5)

# --- Canvas Principal ---
st.subheader("Traza el Símbolo o Visión en el Lienzo")

# Canvas Parameters
stroke_color = "#000000"  # Tinta Negra
bg_color = '#FFFFFF'     # Fondo Pergamino Blanco

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.0)",  # Sin relleno
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=350, # Aumento de altura
    width=500,  # Aumento de ancho
    drawing_mode=drawing_mode,
    key="canvas_intelligent",
)

# --- Controles de la API y Análisis ---
ke = st.text_input('Ingresa la Llave Arcaica (Gemini Key)', type="password")

additional_details = st.text_area(
    "Fórmula de Invocación (Instrucción de Análisis):",
    placeholder="Ej: Describe la arquitectura de este diseño. O, ¿Qué criatura he dibujado?",
    value="Describe en español y con un tono formal y solemne el objeto, concepto o símbolo que has identificado en este trazo místico. Usa lenguaje evocador."
)

analyze_button = st.button("Revela el Significado del Símbolo", type="primary")


# --- Lógica de Análisis ---
if canvas_result.image_data is not None and analyze_button:
    
    # 1. Validación
    if not ke:
        st.error("La Llave Arcaica (API Key) es necesaria para desvelar el significado.")
        st.stop()
        
    # Verificar si el dibujo está vacío
    # Verifica si todos los valores de los canales R, G, B son el color de fondo (blanco en este caso)
    # st_canvas retorna una imagen RGBA
    image_array = np.array(canvas_result.image_data)
    # Comprobar si hay algo que no sea blanco puro (255, 255, 255, 255)
    # Se ignora el canal alfa ya que puede variar en las áreas no dibujadas, 
    # pero nos enfocamos en que los RGB no sean todos 255 (blanco).
    is_blank = np.all(image_array[:, :, :3] == 255)
    if is_blank:
        st.warning("El Lienzo del Augurio está vacío. Por favor, traza una visión.")
        st.stop()


    with st.spinner("El Tablero Místico está procesando la visión..."):
        try:
            # 2. Preparar la Imagen (Codificación Base64)
            # El array RGBA debe ser convertido a RGB antes de guardar/enviar, 
            # ya que el fondo del canvas es blanco y el trazo es negro.
            input_numpy_array = np.array(canvas_result.image_data)
            # Convertir a RGB, ya que el modelo Gemini generalmente funciona mejor con RGB.
            # Esto ignora el canal Alfa que es 255 en áreas no dibujadas y 255 en el trazo negro (opaco).
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
            st.markdown("### 🔮 El Verbo de Géminis:")
            st.markdown(response)
            
        except Exception as e:
            st.error(f"Error durante la Invocación. La Visión fue bloqueada: {e}")
            
elif analyze_button:
    # Se cubre si se presiona sin clave o sin dibujo. La clave se verifica antes con un st.error.
    st.info("A la espera del trazo y la Llave Arcaica para la revelación.")
