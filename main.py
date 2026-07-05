import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# ⚠️ ¡CRUCIAL! Esto permite que tu página de GitHub Pages hable con este servidor de Python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción podés cambiar esto por tu link de GitHub Pages
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


# --- MODELOS DE ENTRADA ---
class DramaInput(BaseModel):
    relacion: str
    historia: str


class ConsejoInput(BaseModel):
    relacion: str
    historia: str
    veredicto: str
    culpable: str  # "usuario" o "otro"


@app.get("/")
def home():
    return {"status": "Tribunal de la IA Online"}


@app.post("/api/veredicto")
def obtener_veredicto(data: DramaInput):
    if not CEREBRAS_API_KEY:
        raise HTTPException(status_code=500, detail="Falta la clave API en el servidor")

    # El System Prompt define la personalidad del juez de IA.
    # Le pedimos JSON estructurado para saber con certeza quién es el "culpable",
    # en vez de adivinarlo después buscando palabras clave en el texto.
    system_prompt = (
        f"Sos un juez de un tribunal humorístico e implacable llamado 'queboludo?'. "
        f"Tu trabajo es analizar una discusión entre el usuario y su {data.relacion}. "
        f"Sé extremadamente sarcástico, divertido, informal (usá jerga argentina o neutra divertida) "
        f"y dictaminá un veredicto definitivo de quién es el culpable y 'le bolude' de la situación. "
        f"El veredicto debe ser corto (máximo 3 líneas), muy fulminante, sin explicar razonamiento paso a paso. "
        f"Respondé EXCLUSIVAMENTE con un objeto JSON, sin texto antes ni después, con esta forma exacta:\n"
        f'{{"veredicto": "<el veredicto fulminante>", "culpable": "usuario"}}\n'
        f"El valor de \"culpable\" debe ser el string \"usuario\" si el usuario tiene la culpa, "
        f"o \"otro\" si la culpa es de su {data.relacion}."
    )

    try:
        respuesta = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.historia}
            ],
            max_tokens=1024,
            temperature=0.7,
            reasoning_effort="low",
            response_format={"type": "json_object"}
        )

        mensaje = respuesta.choices[0].message
        raw_content = (getattr(mensaje, "content", None) or "").strip()

        if not raw_content:
            raise HTTPException(
                status_code=502,
                detail="La IA no generó un veredicto esta vez (respuesta vacía). Probá de nuevo."
            )

        try:
            parsed = json.loads(raw_content)
            veredicto_texto = (parsed.get("veredicto") or "").strip()
            culpable = (parsed.get("culpable") or "").strip().lower()
        except (json.JSONDecodeError, AttributeError):
            # Fallback defensivo: si por algún motivo no vino JSON válido,
            # usamos el texto crudo como veredicto y marcamos el culpable como desconocido
            veredicto_texto = raw_content
            culpable = ""

        if not veredicto_texto:
            raise HTTPException(
                status_code=502,
                detail="La IA no generó un veredicto válido esta vez. Probá de nuevo."
            )

        if culpable not in ("usuario", "otro"):
            culpable = "desconocido"

        return {"veredicto": veredicto_texto, "culpable": culpable}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/consejo")
def obtener_consejo(data: ConsejoInput):
    if not CEREBRAS_API_KEY:
        raise HTTPException(status_code=500, detail="Falta la clave API en el servidor")

    if data.culpable == "usuario":
        instruccion = (
            f"El usuario fue declarado culpable en la discusión con su {data.relacion}. "
            f"Dale 3 o 4 sugerencias concretas y breves de qué actitud tomar ahora: cómo reconocer "
            f"el error, qué decir para reparar el vínculo, y cómo evitar repetir el problema."
        )
    else:
        instruccion = (
            f"El/la {data.relacion} del usuario fue declarado culpable en la discusión. "
            f"Dale 3 o 4 sugerencias concretas y breves de cómo manejar la situación de forma civilizada: "
            f"qué decir, cómo poner un límite sin escalar el conflicto, y cómo cuidar el vínculo si vale la pena."
        )

    system_prompt = (
        f"Sos un consejero cercano y con sentido del humor, parte del tribunal 'queboludo?'. "
        f"Ya se dictó este veredicto sobre la discusión: \"{data.veredicto}\". "
        f"{instruccion} "
        f"Respondé en formato de lista corta con viñetas, sin rodeos, máximo 6 líneas en total. "
        f"Tono cálido, con un toque de humor, sin ser condescendiente ni alentar el conflicto."
    )

    try:
        respuesta = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.historia}
            ],
            max_tokens=1024,
            temperature=0.7,
            reasoning_effort="low"
        )

        consejo_texto = (respuesta.choices[0].message.content or "").strip()

        if not consejo_texto:
            raise HTTPException(
                status_code=502,
                detail="La IA no generó un consejo esta vez (respuesta vacía). Probá de nuevo."
            )

        return {"consejo": consejo_texto}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
