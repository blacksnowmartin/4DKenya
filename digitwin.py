import torch
import torch.nn as nn
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIG ---
GRID_SIZE = 24 # Bigger city
# We have 2 Channels: [0] = Building Height, [1] = Data Traffic Intensity
city_state = torch.zeros(1, 2, GRID_SIZE, GRID_SIZE)

# --- ADVANCED PHYSICS ENGINE ---
# Filter 1: Diffusion (Spreads data smoothly like water)
diffusion = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
with torch.no_grad():
    kernel = torch.tensor([[0.05, 0.2, 0.05],
                           [0.2,  0.0, 0.2 ],
                           [0.05, 0.2, 0.05]])
    diffusion.weight.copy_(kernel.unsqueeze(0).unsqueeze(0))

def evolve_system():
    global city_state
    
    # 1. SEPARATE CHANNELS
    infrastructure = city_state[:, 0:1, :, :]
    traffic = city_state[:, 1:2, :, :]
    
    # 2. EVOLVE TRAFFIC (Fast Physics)
    # Traffic spreads to neighbors
    traffic_spread = diffusion(traffic)
    # Traffic decays fast (0.8) but picks up noise
    traffic = (traffic * 0.7) + (traffic_spread * 0.3)
    
    # 3. RANDOM INJECTIONS (Simulate active users)
    if np.random.rand() > 0.5:
        # Create 4 random hotspots
        for _ in range(4):
            rx, ry = np.random.randint(0, GRID_SIZE, 2)
            # Inject traffic
            traffic[0, 0, rx, ry] += np.random.uniform(0.5, 1.0)
            # Slightly boost infrastructure there too (Growth follows traffic)
            infrastructure[0, 0, rx, ry] += 0.05

    # 4. EVOLVE INFRASTRUCTURE (Slow Physics)
    # Buildings decay slowly if no traffic
    infrastructure *= 0.995 
    
    # 5. RECOMBINE & CLAMP
    city_state = torch.cat([infrastructure, traffic], dim=1)
    city_state = torch.clamp(city_state, 0.0, 1.0)

@app.get("/nexus_pulse")
def get_pulse():
    evolve_system()
    return {
        "buildings": city_state[0, 0].flatten().tolist(), # Height map
        "traffic": city_state[0, 1].flatten().tolist(),   # Heat map (Glow)
        "dim": GRID_SIZE
    }

@app.post("/hack_event")
def hack_event(type: str):
    global city_state
    if type == "ddos":
        # Max out traffic channel only
        city_state[:, 1, :, :] = 1.0 
    elif type == "wipe":
        city_state *= 0.0
    elif type == "boost":
        city_state += 0.3
    return {"status": "ACK"}

if __name__ == "__main__":
    print("--- NAIROBI NEURAL NEXUS ONLINE ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)