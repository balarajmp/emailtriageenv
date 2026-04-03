"""FastAPI app wiring for EmailTriageEnv."""

from openenv.core.env_server.http_server import create_app

# ✅ Use ONLY absolute imports (Docker-safe)
from env import EmailTriageEnvironment
from models import EmailTriageAction, EmailTriageObservation


app = create_app(
    EmailTriageEnvironment,
    EmailTriageAction,
    EmailTriageObservation,
    env_name="EmailTriageEnv",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()