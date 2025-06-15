from fastapi import FastAPI
import uvicorn
from gradio_ui import build_gradio_ui

app = FastAPI()

demo = build_gradio_ui()
app = demo.launch(share=False, inline=True, prevent_thread_lock=True)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)
