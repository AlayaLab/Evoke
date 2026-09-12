import os
from .remote_app import app
from .frame_hub import serve


if __name__ == '__main__':
    import uvicorn
    listener = serve(os.environ['EVOKE_FRAME_SOCKET'])
    try:
        uvicorn.run(app, host=os.environ.get('EVOKE_UI_HOST', '127.0.0.1'),
                    port=int(os.environ.get('EVOKE_UI_PORT', '7861')))
    finally:
        listener.close()
