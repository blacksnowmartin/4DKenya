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

# --- THE CITY BRAIN (NEURAL PHYSICS) ---
# A 16x16 Grid (Batch=1, Channels=1, Height=16, Width=16)
# This represents 256 city sectors.
city_grid = torch.zeros(1, 1, 16, 16)

# We use a Convolutional Filter to simulate "Diffusion"
# (e.g., if one sector has high energy, it spreads to neighbors)
diffusion_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

# Set weights manually to create a "Game of Life" style spread
# Center = positive (growth), Neighbors = varying influence
with torch.no_grad():
    kernel = torch.tensor([[0.1, 0.2, 0.1],
                           [0.2, 0.8, 0.2],
                           [0.1, 0.2, 0.1]])
    diffusion_filter.weight.copy_(kernel.unsqueeze(0).unsqueeze(0))

def evolve_city():
    global city_grid
    
    # 1. Apply the Neural Filter (The "Spread")
    # This calculates the new state based on neighbors
    new_state = diffusion_filter(city_grid)
    
    # 2. Apply Activation Function (Non-linearity)
    # Using Sigmoid keeps values between 0 and 1 (Organic growth)
    city_grid = torch.sigmoid(new_state)
    
    # 3. Random "Sparks" (Innovation/Traffic)
    # Occasionally add random energy to keep the city alive
    if np.random.rand() > 0.8:
        x, y = np.random.randint(0, 16, 2)
        city_grid[0, 0, x, y] += 1.0

    # Clamp values to prevent explosion
    city_grid = torch.clamp(city_grid, 0, 1)

# --- API ENDPOINTS ---

@app.get("/city_pulse")
def get_pulse():
    evolve_city()
    # Flatten the 16x16 grid to a simple list of 256 numbers
    data = city_grid.squeeze().detach().numpy().flatten().tolist()
    return {"grid": data}

@app.post("/trigger_event")
def trigger_event(type: str):
    global city_grid
    with torch.no_grad():
        if type == "boost":
            # Investing in infrastructure (Boost all sectors)
            city_grid += 0.3
        elif type == "blackout":
            # Power failure (Kill grid)
            city_grid *= 0.1
        elif type == "reset":
            city_grid = torch.rand(1, 1, 16, 16) # Random start
    return {"status": f"Event {type} processed"}

if __name__ == "__main__":
    # Initialize with random noise
    city_grid = torch.rand(1, 1, 16, 16)
    print("--- NAIROBI 4.0 NEURAL TWIN ONLINE ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
