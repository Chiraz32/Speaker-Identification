from fastapi import FastAPI, UploadFile, File, HTTPException
from pyannote.audio import Model, Inference
import numpy as np
import uvicorn
import soundfile as sf
import io
import torch

app = FastAPI()

# Load the pre-trained model
# model = Model.from_pretrained("pytorch_model.bin")
model = Model.from_pretrained("pyannote/embedding", 
                              use_auth_token="AUTH_Token")
# Perform inference to get embeddings
inference = Inference(model, window="whole")

@app.post("/compute_embedding/")
async def compute_embedding(file: UploadFile = File(...)):
    try:
        # Read the uploaded audio file
        audio_data, sample_rate = sf.read(io.BytesIO(await file.read()))
        if len(audio_data.shape) == 1:  # Mono
            audio_data = np.expand_dims(audio_data, axis=0)
        
        # Convert numpy array to PyTorch tensor
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32)
        
        # Run inference
        embedding = inference({'waveform': audio_tensor, 'sample_rate': sample_rate})
        response = {"embedding": embedding.tolist()}
        return response
    except Exception as e:
        print(f"Error: {e}")  # Log the error to the console
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
