import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# ⚠️ ¡CRUCIAL! Esto permite que tu página de GitHub Pages hable con este servidor de Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción puedes cambiar esto por tu link de GitHub Pages
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializamos el cliente de Cerebras usando variables de entorno por seguridad
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
client = OpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=CEREBRAS_API_KEY
)

# Definimos qué datos esperamos recibir desde el HTML
class DramaInput(BaseModel):
    relacion: str
    historia: str

@app.get("/")
def home():
    return {"status": "Tribunal de la IA Online"}

@app.post("/api/veredicto")
def obtener_veredicto(data: DramaInput):
    if not CEREBRAS_API_KEY:
        raise HTTPException(status_code=500, detail="Falta la clave API en el servidor")

    # El System Prompt define la personalidad del juez de IA
    system_prompt = (
        f"Sos un juez de un tribunal humorístico e implacable llamado 'queboludo?'. "
        f"Tu trabajo es analizar una discusión entre el usuario y su {data.relacion}. "
        f"Sé extremadamente sarcástico, divertido, informal (usá jerga argentina o neutra divertida) "
        f"y dictaminá un veredicto definitivo de quién es el culpable y 'le bolude' de la situación. "
        f"Tu respuesta DEBE ser corta (máximo 3 líneas) y muy fulminante."
    )

    try:
        respuesta = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.historia}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return {"veredicto": respuesta.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
