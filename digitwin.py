def get_pulse():
def trigger_event(type: str):
import time
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict
from collections import deque
from fastapi import FastAPI, HTTPException
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
GRID_SIZE = 16
TOTAL_CELLS = GRID_SIZE * GRID_SIZE

# --- CITY STATE ---
city_grid = torch.rand(1, 1, GRID_SIZE, GRID_SIZE) * 0.2
tick = 0

# --- DIFFUSION FILTER (SIMPLIFIED PHYSICS) ---
diffusion_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
with torch.no_grad():
    kernel = torch.tensor([[0.05, 0.1, 0.05],
                           [0.1, 0.6, 0.1],
                           [0.05, 0.1, 0.05]])
    diffusion_filter.weight.copy_(kernel.unsqueeze(0).unsqueeze(0))


@dataclass
class ScheduledEvent:
    id: int
    type: str
    execute_at: int
    params: Dict = None

# Event queue & history
event_counter = 0
event_queue: List[ScheduledEvent] = []
history = deque(maxlen=300)


# --- PREDICTION MODEL (tiny MLP) ---
class CityPredictor(nn.Module):
    def __init__(self, input_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

predictor = CityPredictor()


def apply_diffusion(grid: torch.Tensor) -> torch.Tensor:
    """Apply convolutional diffusion and non-linearity."""
    with torch.no_grad():
        new_state = diffusion_filter(grid)
        new_state = torch.sigmoid(new_state)
    return torch.clamp(new_state, 0.0, 1.0)


def evolve_city():
    global city_grid, tick
    # 1. Diffuse
    city_grid = apply_diffusion(city_grid)

    # 2. Random innovations (sparks)
    if np.random.rand() > 0.85:
        x, y = np.random.randint(0, GRID_SIZE, 2)
        city_grid[0, 0, x, y] = min(1.0, city_grid[0, 0, x, y] + np.random.rand() * 0.8)

    # 3. Decay small values slowly to simulate consumption
    city_grid *= 0.995

    # 4. Clamp
    city_grid = torch.clamp(city_grid, 0.0, 1.0)

    tick += 1


def compute_metrics(grid: torch.Tensor) -> Dict:
    arr = grid.squeeze().detach().numpy()
    avg = float(arr.mean())
    hotspots = []
    flat = arr.flatten()
    top_idx = np.argsort(flat)[-6:][::-1]
    for idx in top_idx:
        x = int(idx // GRID_SIZE)
        y = int(idx % GRID_SIZE)
        hotspots.append({"x": x, "y": y, "value": float(flat[idx])})

    # Simulate transport and energy load derived from activity
    transport_load = float(min(1.0, avg * 2.5 + np.random.randn() * 0.02))
    energy_status = float(max(0.0, min(1.0, 1.0 - avg * 0.3 + np.random.randn() * 0.01)))

    return {
        "avg_activity": avg,
        "hotspots": hotspots,
        "transport_load": transport_load,
        "energy_status": energy_status
    }


def process_scheduled_events():
    global event_queue, city_grid
    due = [e for e in event_queue if e.execute_at <= tick]
    remaining = [e for e in event_queue if e.execute_at > tick]
    event_queue = remaining
    for e in due:
        apply_event(e.type, e.params or {})


def apply_event(event_type: str, params: Dict):
    global city_grid
    with torch.no_grad():
        if event_type == "boost":
            # localized infrastructure boost if coords provided
            cx = params.get("x")
            cy = params.get("y")
            radius = params.get("radius", 2)
            if cx is not None and cy is not None:
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        dist = np.hypot(i - cx, j - cy)
                        if dist <= radius:
                            city_grid[0, 0, i, j] = min(1.0, city_grid[0, 0, i, j] + (1 - dist / (radius + 0.1)) * 0.6)
            else:
                city_grid += 0.25
        elif event_type == "blackout":
            # Drop activity sharply in an area or whole grid
            cx = params.get("x")
            cy = params.get("y")
            if cx is not None and cy is not None:
                for i in range(GRID_SIZE):
                    for j in range(GRID_SIZE):
                        dist = np.hypot(i - cx, j - cy)
                        if dist <= params.get("radius", 3):
                            city_grid[0, 0, i, j] *= 0.05
            else:
                city_grid *= 0.05
        elif event_type == "reset":
            city_grid = torch.rand(1, 1, GRID_SIZE, GRID_SIZE) * 0.2


# --- API ENDPOINTS ---
@app.get("/city_pulse")
def get_pulse(sim_time: float = None):
    """Advance the city one tick and return the grid + summary metrics.
    The endpoint also processes scheduled events due on this tick.
    """
    evolve_city()
    process_scheduled_events()

    metrics = compute_metrics(city_grid)
    snapshot = {
        "tick": tick,
        "metrics": metrics,
        "grid": city_grid.squeeze().detach().numpy().flatten().tolist()
    }
    history.append(snapshot)

    # Prediction: simple features -> predicted avg next-step activity
    features = torch.tensor([[
        metrics["avg_activity"],
        metrics["transport_load"],
        metrics["energy_status"],
        len(metrics["hotspots"]) / 10.0
    ]], dtype=torch.float32)
    with torch.no_grad():
        pred = predictor(features).item()

    return {"tick": tick, "grid": snapshot["grid"], "metrics": metrics, "prediction": pred}


@app.post("/trigger_event")
def trigger_event(type: str = "boost", x: int = None, y: int = None, radius: int = 2):
    params = {"x": x, "y": y, "radius": radius}
    apply_event(type, params)
    return {"status": f"Event {type} applied", "tick": tick, "params": params}


@app.post("/schedule_event")
def schedule_event(type: str, delay: int = 1, x: int = None, y: int = None, radius: int = 2):
    global event_counter, event_queue
    if delay < 0:
        raise HTTPException(status_code=400, detail="Delay must be >= 0")
    event_counter += 1
    ev = ScheduledEvent(id=event_counter, type=type, execute_at=tick + delay, params={"x": x, "y": y, "radius": radius})
    event_queue.append(ev)
    return {"scheduled": asdict(ev)}


@app.get("/city_status")
def city_status():
    metrics = compute_metrics(city_grid)
    return {
        "tick": tick,
        "metrics": metrics,
        "queued_events": [asdict(e) for e in event_queue]
    }


@app.get("/history")
def get_history(limit: int = 50):
    return list(history)[-limit:]


if __name__ == "__main__":
    # Warm start
    city_grid = torch.rand(1, 1, GRID_SIZE, GRID_SIZE) * 0.2
    print("--- NAIROBI 4.0 NEURAL TWIN ONLINE ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
