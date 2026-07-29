from app.routes.bom import router as bom_router
from app.routes.material import router as material_router
from app.routes.system import router as system_router
from app.routes.ui import router as ui_router

__all__ = ["bom_router", "material_router", "system_router", "ui_router"]
