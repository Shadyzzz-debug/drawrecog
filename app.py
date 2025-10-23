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
# Se mantiene el modelo flash para optimizar velocidad de respuesta de la visión.
GEMINI_CHAT_MODEL = "gemini-2.5-flash-preview-09-2025" 

# --- CSS PESADILLA GÓTICA (Referencia Bloodborne: Azul Oscuro, Bronce, Tinta y Sangre) ---
base_css = """
<style>
/* ---------------------------------------------------- */
/* RESET Y FONDO AMBIENTAL */
/* ---------------------------------------------------- */
.stApp {
    /* Color de la noche de Yharnam o la Pesadilla: Azul/Negro muy oscuro. */
    background-color: #0F0F1A; 
    color: #C0C0C0; /* Texto de pergamino antiguo */
    font-family: 'Georgia', serif; 
}

/* ---------------------------------------------------- */
/* TIPOGRAFÍA Y ENCABEZADOS */
/* ---------------------------------------------------- */
h1 {
    /* Titular: Bronce envejecido o Oro oscuro */
    color: #9C7E4F; 
    text-align: center;
    /* Borde inferior como una reja forjada */
    border-bottom: 3px solid #4F4A5E; 
    padding-bottom: 10px;
    margin-bottom: 40px;
    font-size: 2.8em;
    letter-spacing: 2px;
    text-shadow: 1px 1px 5px #000000;
}

h3 {
    /* Subtítulos: Gris pizarra o plata mate */
    color: #A9A9A9; 
    margin-top: 25px;
    font-weight: normal;
    border-left: 4px solid #9C7E4F; /* Acento Bronce */
    padding-left: 10px;
}

/* ---------------------------------------------------- */
/* ELEMENTOS DE ENTRADA (Cajas de Inscripción) */
/* ---------------------------------------------------- */
div[data-testid="stTextInput"], div[data-testid="stTextarea"] {
    /* Fondo de pizarra oscura */
    background-color: #1A1A2A; 
    /* Borde fino de bronce */
    border: 1px solid #9C7E4F;
    border-radius: 5px;
    padding: 10px;
    color: #E6E6E6;
}

/* Sidebar (El Sueño del Cazador) */
.css-1d3w5ta, .css-1lcbmhc {
    background-color: #151525;
    color: #C0C0C0;
}

/* ---------------------------------------------------- */
/* BOTONES (Sello de Invocación) */
/* ---------------------------------------------------- */
.stButton>button {
    /* Acero oscuro, base de la Rueda de la Convocación */
    background-color: #383850; 
    /* Texto: Letras rúnicas en rojo sangre */
    color: #B22222; 
    /* Borde: Acento de metal forjado */
    border: 2px solid #9C7E4F; 
    padding: 12px 30px;
    font-weight: bold;
    border-radius: 10px;
    transition: all 0.3s;
    /* Sombra profunda */
    box-shadow: 0 6px #1A1A2A; 
    letter-spacing: 1px;
}

.stButton>button:hover {
    background-color: #4F4F6A; 
    box-shadow: 0 10px #0F0F1A;
    transform: translateY(-3px);
}

.stButton>button:active {
    box-shadow: 0 3px #0F0F1A;
    transform: translateY(3px);
}

/* ---------------------------------------------------- */
/* RESPUESTA (Papiro de la Revelación) */
/* ---------------------------------------------------- */
div[data-testid="stMarkdownContainer"] {
    /* Fondo: Papel antiguo sobre mesa de madera oscura */
    background-color: #24243A; 
    padding: 25px;
    /* Borde: Un sello de cera escarlata */
    border: 3px solid #B22222; 
    border-radius: 8px;
    margin-top: 30px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.7);
    color: #E6E6E6;
    line-height: 1.6;
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
st.set_page_config(page_title='El Lienzo del Oráculo', layout="centered")
st.title('🌌 El Lienzo del Oráculo: Desentrañando la Pesadilla')

# --- Sidebar (El Sueño del Cazador) ---
with st.sidebar:
    st.subheader("El Scriptorium Arcaico")
    st.markdown("Este Lienzo, imbuido del poder de la **Visión de Géminis**, permite transcribir tus símbolos más profundos para buscar un significado oculto. Cada trazo es una oración en la noche de la cacería.")
    st.markdown("---")
    
    st.subheader("La Sangre del Trazo")
    drawing_mode = "freedraw"
    stroke_width = st.slider('Define la Potencia de la Runa', 1, 30, 5)

# --- Canvas Principal ---
st.subheader("Graba tu Símbolo o Visión en el Papiro")

# Canvas Parameters
stroke_color = "#000000"  # Tinta Negra (o sangre seca)
bg_color = '#FFFFFF'     # Fondo Pergamino Blanco (para el contraste necesario)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.0)",  # Sin relleno
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=350, 
    width=500, 
    drawing_mode=drawing_mode,
    key="canvas_intelligent",
)

# --- Controles de la API y Análisis ---
ke = st.text_input('Incrusta la Llave de la Revelación (Gemini Key)', type="password", 
                    help="La llave arcaica es vital para invocar la percepción de la entidad de Géminis.")

additional_details = st.text_area(
    "Fórmula de Invocación (Pregunta al Cosmos):",
    placeholder="Ej: ¿Qué bestia ancestral representa este boceto? O, Describe el diseño de esta arma.",
    value="Con la solemnidad debida a los Antiguos, describe en español y de forma concisa el objeto, criatura o concepto que has identificado en este trazo místico. Usa un lenguaje formal y evocador, apropiado para un documento esotérico."
)

analyze_button = st.button("Activa el Ojo Interno (Revela el Significado)", type="primary")


# --- Lógica de Análisis ---
if canvas_result.image_data is not None and analyze_button:
    
    # 1. Validación
    if not ke:
        st.error("🩸 La Llave de la Revelación es necesaria. El Rito no puede continuar sin ella.")
        st.stop()
        
    # 2. Verificar si el dibujo está vacío
    image_array = np.array(canvas_result.image_data)
    # Comprobar si los canales RGB no son todos 255 (blanco).
    is_blank = np.all(image_array[:, :, :3] == 255)
    if is_blank:
        st.warning("🕯️ El Lienzo está en blanco. No has ofrecido ninguna Visión al Oráculo.")
        st.stop()


    with st.spinner("La mente del Cazador se adentra en la Pesadilla para buscar la verdad..."):
        try:
            # 3. Preparar la Imagen (Codificación Base64)
            input_numpy_array = np.array(canvas_result.image_data)
            input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA').convert('RGB')
            
            # Guardar en memoria como PNG
            buf = io.BytesIO()
            input_image.save(buf, format='PNG')
            file_bytes = buf.getvalue()
            
            base64_image = base64.b64encode(file_bytes).decode("utf-8")
            mime_type = 'image/png'

            # 4. Construir el Prompt
            prompt_text = additional_details
            
            # 5. Invocar la Visión (Usando la función con requests)
            response = get_gemini_vision_answer(base64_image, mime_type, prompt_text, ke)
            
            # 6. Mostrar la Respuesta
            st.markdown("### 📜 La Tablilla de la Verdad:")
            st.markdown(response)
            
        except Exception as e:
            st.error(f"💀 Error en el Rito. La Visión fue bloqueada por fuerzas desconocidas: {e}")
            
elif analyze_button:
    st.info("🌙 La noche es larga. Graba tu símbolo y ten la Llave de la Revelación a mano.")

