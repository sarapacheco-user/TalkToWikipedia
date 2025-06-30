import os
import json
import io
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = io.BytesIO()

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "audio_chunk":
                chunk = base64.b64decode(message["audio"])
                audio_buffer.write(chunk)
                
            elif message.get("type") == "end_of_stream":
                audio_buffer.seek(0)
                audio_buffer.name = "audio.wav"  # Required by genai.upload_file

                try:
                    # Upload to Gemini
                    audio_file = genai.upload_file(
                        audio_buffer,
                        mime_type="audio/wav"
                    )

                    # Wait for file processing
                    while audio_file.state.name == "PROCESSING":
                        audio_file = genai.get_file(audio_file.name)

                    if audio_file.state.name == "FAILED":
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "text": "Audio processing failed"
                        }))
                        continue

                    # Generate response
                    response = await model.generate_content_async([
                    { "text": "You are a knowledgeable assistant. First transcribe the user's audio question. Then answer it factually as if from Wikipedia." },
                    audio_file
                    ])

                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "text": response.text
                    }))

                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "text": f"Gemini API error: {e}"
                    }))

                # Reset buffer for next session
                audio_buffer.seek(0)
                audio_buffer.truncate()

    except WebSocketDisconnect:
        print("Client disconnected")

    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "text": str(e)
        }))

    finally:
        audio_buffer.close()

# Serve frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
