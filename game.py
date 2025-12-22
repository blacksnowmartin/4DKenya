import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GAME CONFIG ---
GRID_SIZE = 15
CENTER = GRID_SIZE // 2

# Game State
state = {
    "grid": np.zeros((GRID_SIZE, GRID_SIZE)), # 0=Empty, 1=Tower, 2=Core
    "enemies": [], # List of {x, y, hp}
    "projectiles": [], # Visual only
    "credits": 100,
    "score": 0,
    "game_over": False,
    "wave": 1
}

# Initialize Core
state["grid"][CENTER, CENTER] = 2

def spawn_enemy():
    # Spawn at random edge
    side = np.random.randint(0, 4)
    if side == 0: x, y = 0, np.random.randint(0, GRID_SIZE)
    elif side == 1: x, y = GRID_SIZE-1, np.random.randint(0, GRID_SIZE)
    elif side == 2: x, y = np.random.randint(0, GRID_SIZE), 0
    else: x, y = np.random.randint(0, GRID_SIZE), GRID_SIZE-1
    
    state["enemies"].append({"x": x, "y": y, "hp": 3})

def update_logic():
    if state["game_over"]: return

    # 1. SPAWN LOGIC (Random chance based on wave)
    if np.random.rand() < (0.05 + state["wave"] * 0.01):
        spawn_enemy()

    # 2. TOWER COMBAT (Simple distance check)
    # Find all towers
    towers = np.argwhere(state["grid"] == 1)
    
    for t_pos in towers:
        tx, ty = t_pos
        # Find closest enemy
        for enemy in state["enemies"]:
            dist = math.sqrt((tx - enemy["x"])**2 + (ty - enemy["y"])**2)
            if dist < 3.5: # Range
                # Shoot!
                enemy["hp"] -= 1
                # Visual laser data could be stored here
                break # One shot per tick

    # 3. MOVE ENEMIES (Pathfind to center)
    surviving_enemies = []
    for enemy in state["enemies"]:
        if enemy["hp"] <= 0:
            state["credits"] += 5
            state["score"] += 10
            continue # Enemy dead
            
        # Move towards center
        dx = CENTER - enemy["x"]
        dy = CENTER - enemy["y"]
        
        # Simple movement: Move along axis with biggest distance
        if abs(dx) > abs(dy):
            step_x = 1 if dx > 0 else -1
            enemy["x"] += step_x * 0.2 # Speed
        else:
            step_y = 1 if dy > 0 else -1
            enemy["y"] += step_y * 0.2
            
        # Check Collision with Core
        dist_to_core = math.sqrt((CENTER - enemy["x"])**2 + (CENTER - enemy["y"])**2)
        if dist_to_core < 1.0:
            state["game_over"] = True
        else:
            surviving_enemies.append(enemy)
            
    state["enemies"] = surviving_enemies

# --- ENDPOINTS ---

@app.get("/gameloop")
def gameloop():
    update_logic()
    # Return everything needed to render
    return {
        "grid": state["grid"].flatten().tolist(),
        "enemies": state["enemies"],
        "stats": {"credits": state["credits"], "score": state["score"], "game_over": state["game_over"]}
    }

@app.post("/action")
def action(x: int, y: int, type: str):
    if state["game_over"]: return {"status": "GAMEOVER"}
    
    if type == "build":
        if state["credits"] >= 20 and state["grid"][x, y] == 0:
            state["grid"][x, y] = 1 # Build Tower
            state["credits"] -= 20
            return {"status": "BUILT"}
            
    elif type == "reset":
        state["grid"] = np.zeros((GRID_SIZE, GRID_SIZE))
        state["grid"][CENTER, CENTER] = 2
        state["enemies"] = []
        state["credits"] = 100
        state["score"] = 0
        state["game_over"] = False
        
    return {"status": "OK"}

if __name__ == "__main__":
    print("--- CYBER DEFENSE PROTOCOL ACTIVE ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)