from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import List

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
TICK = 0

@dataclass
class Tower:
    x: int
    y: int
    level: int = 1
    energy: float = 100.0
    cooldown: int = 0
    
    def get_range(self):
        return 3.5 + self.level * 0.5
    
    def get_damage(self):
        return 1.5 + self.level * 0.5
    
    def get_cost(self):
        return 20 + self.level * 15

@dataclass
class Enemy:
    x: float
    y: float
    hp: float
    max_hp: float
    wave: int
    enemy_type: str = "basic"  # basic, elite, swarm
    speed: float = 0.15
    
    def get_color(self):
        if self.enemy_type == "elite":
            return {"r": 1.0, "g": 0.2, "b": 1.0}  # Magenta
        elif self.enemy_type == "swarm":
            return {"r": 1.0, "g": 0.5, "b": 0.0}  # Orange
        else:
            return {"r": 1.0, "g": 0.0, "b": 0.0}  # Red

@dataclass
class Projectile:
    sx: float
    sy: float
    ex: float
    ey: float
    life: float = 10.0
    damage: float = 1.0
    
# Game State
state = {
    "towers": {},  # {(x,y): Tower}
    "enemies": [],  # List[Enemy]
    "projectiles": [],  # List[Projectile]
    "credits": 150,
    "score": 0,
    "lives": 10,
    "game_over": False,
    "wave": 1,
    "enemy_queue": deque(),
    "tick": 0,
    "core_hp": 100.0
}

# Initialize Core
state["towers"][(CENTER, CENTER)] = Tower(CENTER, CENTER, level=0)  # Core is a special tower

def get_wave_config(wave):
    """Exponential difficulty scaling"""
    base_enemies = 3 + wave * 2
    elite_ratio = 0.1 + wave * 0.05
    swarm_ratio = 0.05 + wave * 0.03
    
    enemies_to_spawn = []
    for _ in range(base_enemies):
        rand = random.random()
        if rand < swarm_ratio:
            enemies_to_spawn.append("swarm")
        elif rand < elite_ratio + swarm_ratio:
            enemies_to_spawn.append("elite")
        else:
            enemies_to_spawn.append("basic")
    return enemies_to_spawn

def spawn_enemy_wave():
    """Spawn wave of enemies"""
    enemies = get_wave_config(state["wave"])
    for enemy_type in enemies:
        side = random.randint(0, 3)
        if side == 0:
            x, y = 0, random.randint(0, GRID_SIZE-1)
        elif side == 1:
            x, y = GRID_SIZE-1, random.randint(0, GRID_SIZE-1)
        elif side == 2:
            x, y = random.randint(0, GRID_SIZE-1), 0
        else:
            x, y = random.randint(0, GRID_SIZE-1), GRID_SIZE-1
        
        hp_multi = 1.0
        speed = 0.15
        if enemy_type == "elite":
            hp_multi = 2.5
            speed = 0.12
        elif enemy_type == "swarm":
            hp_multi = 0.6
            speed = 0.25
        
        state["enemies"].append(Enemy(
            x=float(x), y=float(y),
            hp=3.0 * hp_multi * (1 + state["wave"] * 0.3),
            max_hp=3.0 * hp_multi * (1 + state["wave"] * 0.3),
            wave=state["wave"],
            enemy_type=enemy_type,
            speed=speed
        ))

def update_logic():
    global TICK
    if state["game_over"]:
        return
    
    TICK += 1
    state["tick"] = TICK
    
    # 1. SPAWN WAVE (Every 300 ticks and if all dead)
    if (TICK % 300 == 0 and len(state["enemies"]) == 0) or (len(state["enemies"]) == 0 and TICK % 100 == 0):
        state["wave"] += 1
        spawn_enemy_wave()
    
    # 2. TOWER COMBAT (Advanced targeting)
    towers = list(state["towers"].values())
    new_projectiles = []
    
    for tower in towers:
        if tower.x == CENTER and tower.y == CENTER:  # Core doesn't shoot
            continue
        
        tower.cooldown = max(0, tower.cooldown - 1)
        tower.energy = min(100.0, tower.energy + 0.5)  # Regenerate energy
        
        if tower.cooldown == 0 and tower.energy >= 20:
            # Find closest enemy in range
            closest_enemy = None
            closest_dist = tower.get_range() + 1
            
            for enemy in state["enemies"]:
                dist = math.sqrt((tower.x - enemy.x)**2 + (tower.y - enemy.y)**2)
                if dist < closest_dist:
                    closest_dist = dist
                    closest_enemy = enemy
            
            if closest_enemy:
                # Shoot!
                tower.cooldown = 10
                tower.energy -= 20
                damage = tower.get_damage()
                new_projectiles.append(Projectile(
                    sx=float(tower.x), sy=float(tower.y),
                    ex=closest_enemy.x, ey=closest_enemy.y,
                    damage=damage
                ))
    
    # 3. PROJECTILE COLLISIONS
    surviving_projectiles = []
    for proj in state["projectiles"] + new_projectiles:
        proj.life -= 1
        
        hit_enemy = None
        for i, enemy in enumerate(state["enemies"]):
            dist_to_enemy = math.sqrt((proj.ex - enemy.x)**2 + (proj.ey - enemy.y)**2)
            if dist_to_enemy < 0.5:
                enemy.hp -= proj.damage
                hit_enemy = i
                break
        
        if hit_enemy is not None or proj.life <= 0:
            continue
        surviving_projectiles.append(proj)
    
    state["projectiles"] = surviving_projectiles
    
    # 4. MOVE ENEMIES (Pathfinding towards center)
    surviving_enemies = []
    for enemy in state["enemies"]:
        if enemy.hp <= 0:
            # Award credits based on type
            reward = 5
            if enemy.enemy_type == "elite":
                reward = 25
            elif enemy.enemy_type == "swarm":
                reward = 2
            state["credits"] += reward
            state["score"] += int(reward * 10)
            continue
        
        # Move towards center with randomized paths
        dx = CENTER - enemy.x
        dy = CENTER - enemy.y
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist > 0.1:
            # Diagonal movement with slight randomization
            nx = (dx / dist) * enemy.speed
            ny = (dy / dist) * enemy.speed
            
            # Add small random deviation for more natural movement
            nx += (random.random() - 0.5) * 0.02
            ny += (random.random() - 0.5) * 0.02
            
            enemy.x += nx
            enemy.y += ny
        
        # Check collision with core
        dist_to_core = math.sqrt((CENTER - enemy.x)**2 + (CENTER - enemy.y)**2)
        if dist_to_core < 0.8:
            state["lives"] -= 1
            if state["lives"] <= 0:
                state["game_over"] = True
        else:
            surviving_enemies.append(enemy)
    
    state["enemies"] = surviving_enemies

# --- ENDPOINTS ---

@app.get("/gameloop")
def gameloop():
    update_logic()
    
    # Flatten tower data
    tower_data = {}
    for (x, y), tower in state["towers"].items():
        tower_data[f"{x},{y}"] = {
            "x": tower.x, "y": tower.y,
            "level": tower.level,
            "energy": tower.energy,
            "range": tower.get_range()
        }
    
    # Enemy data with color
    enemy_data = []
    for enemy in state["enemies"]:
        color = enemy.get_color()
        enemy_data.append({
            "x": enemy.x, "y": enemy.y,
            "hp": enemy.hp, "max_hp": enemy.max_hp,
            "type": enemy.enemy_type,
            "color": color
        })
    
    # Projectile data
    proj_data = [{"sx": p.sx, "sy": p.sy, "ex": p.ex, "ey": p.ey} for p in state["projectiles"]]
    
    return {
        "towers": tower_data,
        "enemies": enemy_data,
        "projectiles": proj_data,
        "stats": {
            "credits": state["credits"],
            "score": state["score"],
            "lives": state["lives"],
            "game_over": state["game_over"],
            "wave": state["wave"],
            "tick": state["tick"],
            "core_hp": state["core_hp"]
        }
    }

@app.post("/action")
def action(x: int, y: int, type: str):
    if state["game_over"]:
        return {"status": "GAMEOVER"}
    
    if type == "build":
        if state["credits"] >= 20 and (x, y) not in state["towers"]:
            state["towers"][(x, y)] = Tower(x, y, level=1)
            state["credits"] -= 20
            return {"status": "BUILT"}
        return {"status": "INVALID"}
    
    elif type == "upgrade":
        if (x, y) in state["towers"]:
            tower = state["towers"][(x, y)]
            cost = tower.get_cost()
            if state["credits"] >= cost and tower.level < 5:
                state["credits"] -= cost
                tower.level += 1
                return {"status": "UPGRADED"}
        return {"status": "INVALID"}
    
    elif type == "sell":
        if (x, y) in state["towers"] and (x, y) != (CENTER, CENTER):
            tower = state["towers"][(x, y)]
            refund = int(tower.get_cost() * 0.7)
            state["credits"] += refund
            del state["towers"][(x, y)]
            return {"status": "SOLD"}
        return {"status": "INVALID"}
    
    elif type == "reset":
        state["towers"] = {(CENTER, CENTER): Tower(CENTER, CENTER, level=0)}
        state["enemies"] = []
        state["projectiles"] = []
        state["credits"] = 150
        state["score"] = 0
        state["lives"] = 10
        state["game_over"] = False
        state["wave"] = 1
        TICK = 0
        return {"status": "RESET"}
    
    return {"status": "OK"}

if __name__ == "__main__":
    print("╔═══════════════════════════════════════╗")
    print("║  NAIROBI PROTOCOL: DEFENSE ACTIVE    ║")
    print("║  Multi-Tier Threat Detection         ║")
    print("║  Tower Defense Engine v2.0           ║")
    print("╚═══════════════════════════════════════╝")
    uvicorn.run(app, host="0.0.0.0", port=8000)