import logging
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
import librosa
import requests
import numpy as np
from scipy.spatial.distance import cdist
import yaml
import os
import soundfile as sf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your front-end URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_embeddings(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

file_path = 'data/speakers_embeddings.yaml'
known_speakers_embeddings = load_embeddings(file_path)

def get_embedding_via_api(audio_path):
    with open(audio_path, 'rb') as f:
        response = requests.post("http://localhost:8000/compute_embedding/", files={"file": f})
    response.raise_for_status()
    json_response = response.json()
    if "embedding" not in json_response:
        raise ValueError("Response JSON does not contain 'embedding' key.")
    return np.array(json_response["embedding"])

def compute_distance(embedding1, embedding2):
    embedding1 = np.reshape(embedding1, (1, -1))
    embedding2 = np.reshape(embedding2, (1, -1))
    return cdist(embedding1, embedding2, metric="cosine")[0, 0]

def recognize_voice(audio_file_path):
    try:
        data, samplerate = librosa.load(audio_file_path, sr=None)
        sf.write(audio_file_path, data, samplerate)
        unknown_embedding = get_embedding_via_api(audio_file_path)
        os.remove(audio_file_path)

        min_distance = float('inf')
        second_min_distance = float('inf')
        closest_speaker = None
        second_closest_speaker = None

        for speaker, embedding in known_speakers_embeddings.items():
            distance = compute_distance(unknown_embedding, embedding)
            logging.debug(f"Speaker: {speaker}, Distance: {distance}")

            if distance < min_distance:
                second_min_distance = min_distance
                second_closest_speaker = closest_speaker
                min_distance = distance
                closest_speaker = speaker
            elif distance < second_min_distance:
                second_min_distance = distance
                second_closest_speaker = speaker

        return {
            "speaker": closest_speaker,
            "distance": min_distance,
            "second_speaker": second_closest_speaker,
            "second_distance": second_min_distance,
        }

    except Exception as e:
        logging.error(f"Error in recognizing the voice: {str(e)}")
        raise HTTPException(status_code=500, detail="Error while recognizing user")

@app.post('/identify_speaker')
async def recognize_voice_route(audio: UploadFile = File(...)):
    file_path = os.path.join(os.getcwd(), "uploaded_audio.wav")

    with open(file_path, "wb") as f:
        content = await audio.read()
        f.write(content)

    result = recognize_voice(file_path)
    return result

@app.post('/register_speaker')
async def register_speaker(audio: UploadFile = File(...), speaker: str = Form(...)):
    try:
        file_path = os.path.join(os.getcwd(), "uploaded_audio.wav")
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        # Check if user already exists
        if speaker in known_speakers_embeddings:
            os.remove(file_path)
            return {"response": f"User '{speaker}' already exists."}

        # Load and save audio
        data, samplerate = librosa.load(file_path, sr=None)
        sf.write(file_path, data, samplerate)
        
        # Get embedding from API
        embedding = get_embedding_via_api(file_path)
        os.remove(file_path)
        
        if embedding.size == 0:
            return {"response": "Failed to extract embedding from the provided audio."}

        # Save embedding
        known_speakers_embeddings[speaker] = embedding.tolist()
        with open(file_path, 'w') as f:
            yaml.dump(known_speakers_embeddings, f, default_flow_style=False)
        
        return {"response": f"User '{speaker}' added successfully."}
        
    except Exception as e:
        logging.error(f"Error in registering the embedding: {str(e)}")
        return {"response": "Internal Server Error"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
