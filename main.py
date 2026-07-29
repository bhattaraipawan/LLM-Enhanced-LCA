"""Development entry point.

Run ``python main.py`` and open http://127.0.0.1:8000/.
"""

from app import create_app
from app.config import get_settings

app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.app_host, port=settings.app_port, reload=False)
