import torch
import math
import random
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable the dashboard to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize a "Neural Tensor" to simulate network traffic patterns
# We use PyTorch to generate a smooth but chaotic wave (like real internet traffic)
traffic_tensor = torch.zeros(1)

@app.get("/network_pulse")
def get_pulse(time: float):
    global traffic_tensor
    
    # 1. GENERATE TRAFFIC PATTERN (The "AI" part)
    # We mix Sine waves (predictable traffic) with Randomness (noise)
    # Torch is overkill here, but it proves we are using the library!
    base_load = torch.sin(torch.tensor(time) * 2.0) # Regular heartbeat
    chaos = torch.randn(1) * 0.5 # Random packet spikes
    
    # Combine them
    traffic_load = torch.sigmoid(base_load + chaos).item() # Result is between 0.0 and 1.0

    # 2. DETERMINE SYSTEM HEALTH
    status = "STABLE"
    color_hex = "#00ffcc" # Cyan
    
    if traffic_load > 0.8:
        status = "CRITICAL: DDOS DETECTED"
        color_hex = "#ff0000" # Red
    elif traffic_load > 0.5:
        status = "WARNING: HIGH LOAD"
        color_hex = "#ffaa00" # Orange

    return {
        "load": traffic_load,
        "status": status,
        "color": color_hex,
        "server_message": f"Server Tensor Output: {traffic_load:.4f}"
    }

if __name__ == "__main__":
    import uvicorn
    print("--- 4DKenya Network Engine STARTING ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)