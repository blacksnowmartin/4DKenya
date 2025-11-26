from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pinn_engine import get_4d_state
import math

app = FastAPI(title="4DKenya Prototyping Engine")

# Allow the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "system": "4DKenya Frontier Engine"}

@app.get("/simulate")
def simulate(time: float):
    """
    Returns the AI-predicted 4D coordinates for a given time step.
    In a real scenario, this could be fetching cybersecurity threat vectors 
    or logistics optimization paths encoded in 4D.
    """
    # Get AI prediction (Physics-Informed)
    # We wrap time to keep it looping for the demo
    looped_time = (time % 10) / 10.0 
    coords = get_4d_state(looped_time)
    
    return {
        "time": time,
        "coords": {
            "x": coords[0],
            "y": coords[1],
            "z": coords[2],
            "w": coords[3]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
