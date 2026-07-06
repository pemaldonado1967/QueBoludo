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
    return {"status": "Queboludo! operativo"}


@app.post("/api/veredicto")
def obtener_veredicto(data: DramaInput):
    if not CEREBRAS_API_KEY:
        raise HTTPException(status_code=500, detail="Falta la clave API en el servidor")

    # El System Prompt define la personalidad del "boludómetro".
    # Le pedimos JSON estructurado para saber con certeza quién es el "culpable",
    # en vez de adivinarlo después buscando palabras clave en el texto.
    # También le pedimos que fundamente el veredicto con detalles concretos del relato,
    # para que no quede corto ni genérico.
    system_prompt = (
        f"Sos el Boludómetro, la voz del sentido común del argentino de a pie: directo, "
        f"cargador, con jerga bien argentina (che, boludo, la posta, zarpado, etc.), "
        f"pero con calle y picardía, no con soberbia de juez. "
        f"Tu laburo es leer la discusión que el usuario tuvo con su {data.relacion} y confirmar "
        f"(o desmentir) lo que el usuario probablemente ya piensa: quién es el/la boludo/a de la historia. "
        f"El veredicto tiene que sonar a lo que te diría un amigo copado en el bar, no a una sentencia judicial: "
        f"nada de 'culpable', 'inapelable' ni vocabulario de tribunal. "
        f"IMPORTANTE: el veredicto debe tener sustancia, no ser una frase suelta. Escribí entre 4 y 6 líneas, "
        f"retomando al menos dos detalles concretos que la persona haya contado (nombres de situaciones, frases, "
        f"actitudes puntuales), para que se sienta que de verdad lo leíste y no que tiraste una genérica. "
        f"Mantené el filo, el sarcasmo y el humor en todo momento. "
        f"Respondé EXCLUSIVAMENTE con un objeto JSON, sin texto antes ni después, con esta forma exacta:\n"
        f'{{"veredicto": "<el veredicto con sustancia, 4 a 6 líneas>", "culpable": "usuario"}}\n'
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
            max_tokens=1200,
            temperature=0.75,
            reasoning_effort="low",
            response_format={"type": "json_object"}
        )

        mensaje = respuesta.choices[0].message
        raw_content = (getattr(mensaje, "content", None) or "").strip()

        if not raw_content:
            raise HTTPException(
                status_code=502,
                detail="El Boludómetro no generó un veredicto esta vez (respuesta vacía). Probá de nuevo."
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
                detail="El Boludómetro no generó un veredicto válido esta vez. Probá de nuevo."
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
            f"El usuario fue el boludo de esta historia con su {data.relacion}. "
            f"Dale 4 consejos concretos, prácticos y con onda de qué actitud tomar ahora: cómo reconocer "
            f"el error sin humillarse, qué decir para reparar el vínculo, y cómo no repetirla la próxima vez."
        )
    else:
        instruccion = (
            f"El/la {data.relacion} del usuario fue el/la boludo/a de esta historia. "
            f"Dale 4 consejos concretos, prácticos y con onda de cómo tratar a esa persona sin escalar el conflicto: "
            f"qué decir, cómo poner un límite con clase, y cómo cuidar el vínculo si vale la pena cuidarlo."
        )

    system_prompt = (
        f"Sos el mismo Boludómetro de siempre: un amigo con calle, gracioso, que ya dictaminó este veredicto "
        f"sobre la historia: \"{data.veredicto}\". "
        f"{instruccion} "
        f"Cada consejo tiene un título corto y picante (3 a 5 palabras, como un título de sección, no una oración larga) "
        f"y un texto de desarrollo de 1 a 2 frases, con algún ejemplo concreto de qué decir si aplica. "
        f"Nada de sonar a manual de autoayuda genérico: mantené la jerga argentina y el sentido del humor. "
        f"Respondé EXCLUSIVAMENTE con un objeto JSON, sin texto antes ni después, con esta forma exacta:\n"
        f'{{"consejos": [{{"titulo": "<título corto>", "texto": "<desarrollo breve>"}}, ...]}}\n'
        f"El array \"consejos\" debe tener exactamente 4 elementos."
    )

    try:
        respuesta = client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.historia}
            ],
            max_tokens=1200,
            temperature=0.75,
            reasoning_effort="low",
            response_format={"type": "json_object"}
        )

        mensaje = respuesta.choices[0].message
        raw_content = (getattr(mensaje, "content", None) or "").strip()

        if not raw_content:
            raise HTTPException(
                status_code=502,
                detail="El Boludómetro no generó un consejo esta vez (respuesta vacía). Probá de nuevo."
            )

        try:
            parsed = json.loads(raw_content)
            consejos = parsed.get("consejos") or []
            # Nos quedamos solo con items bien formados (con título y texto)
            consejos_limpios = [
                {"titulo": (c.get("titulo") or "").strip(), "texto": (c.get("texto") or "").strip()}
                for c in consejos
                if isinstance(c, dict) and (c.get("titulo") or c.get("texto"))
            ]
        except (json.JSONDecodeError, AttributeError):
            consejos_limpios = []

        if not consejos_limpios:
            raise HTTPException(
                status_code=502,
                detail="El Boludómetro no generó consejos válidos esta vez. Probá de nuevo."
            )

        return {"consejos": consejos_limpios}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
