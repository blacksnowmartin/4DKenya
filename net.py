import torch
import math
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Allow browser connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE ---
# This dictionary holds the state of our simulation
system_state = {
    "attack_mode": False,   # Is the system under attack?
    "defense_mode": False,  # Is the AI fixing it?
    "entropy": 0.0          # How much "chaos" is in the 4th dimension
}

# --- THE AI ENGINE ---
def calculate_4d_physics(time_step):
    """
    Uses PyTorch to calculate the stability of the 4D Tensor.
    """
    # Base movement (The heartbeat of the system)
    base_tensor = torch.tensor([math.sin(time_step), math.cos(time_step)])
    
    noise_level = 0.0
    
    if system_state["attack_mode"]:
        # If under attack, inject massive entropy
        noise_level = 0.8
        
    if system_state["defense_mode"]:
        # AI Counter-measure reduces noise using a decay function
        noise_level = noise_level * 0.1
    
    # Store the calculated entropy to send to frontend
    system_state["entropy"] = noise_level
    
    # Return data for the frontend
    return {
        "status": "CRITICAL" if noise_level > 0.4 else "SECURE",
        "color": "#ff0000" if noise_level > 0.4 else "#00ffcc",
        "distortion": noise_level, # How much the Tesseract should glitch
        "defense_active": system_state["defense_mode"]
    }

# --- API ENDPOINTS ---

@app.get("/telemetry")
def get_telemetry(time: float):
    # This is called 60 times a second by the dashboard
    return calculate_4d_physics(time)

@app.post("/trigger_attack")
def trigger_attack():
    system_state["attack_mode"] = True
    system_state["defense_mode"] = False # Disable defense if new attack starts
    return {"msg": "INTRUSION DETECTED"}

@app.post("/activate_defense")
def activate_defense():
    system_state["defense_mode"] = True
    return {"msg": "NEURAL SHIELD ACTIVE"}

@app.post("/reset_system")
def reset_system():
    system_state["attack_mode"] = False
    system_state["defense_mode"] = False
    return {"msg": "SYSTEM REBOOTED"}

if __name__ == "__main__":
    print("--- 4DKenya Quantum Engine ONLINE ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
