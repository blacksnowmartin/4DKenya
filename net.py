import torch
import torch.nn as nn
import math
import random
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from collections import deque

app = FastAPI()

# Allow browser connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NEURAL NETWORK FOR THREAT DETECTION ---
class ThreatDetectionNet(nn.Module):
    """Neural network to predict and detect threats"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc3(x))

threat_net = ThreatDetectionNet()

# --- ATTACK TYPE DEFINITIONS ---
@dataclass
class AttackVector:
    name: str
    intensity: float
    frequency: float
    pattern: str  # "sine", "spike", "random", "cascade"

# --- GLOBAL STATE & HISTORY ---
system_state = {
    "attack_mode": False,
    "defense_mode": False,
    "adaptive_defense": False,
    "entropy": 0.0,
    "threat_level": 0.0,
    "anomaly_score": 0.0,
    "energy_level": 100.0,
    "attack_vectors": [],
    "defense_efficiency": 1.0,
    "learning_mode": True
}

# History tracking for anomaly detection
telemetry_history = deque(maxlen=100)

# Attack patterns
attack_patterns = {
    "ddos": AttackVector("DDoS", 0.9, 50.0, "spike"),
    "ransomware": AttackVector("Ransomware", 0.7, 2.0, "cascade"),
    "exploit": AttackVector("Exploit", 0.6, 10.0, "sine"),
    "probe": AttackVector("Probe", 0.3, 5.0, "random")
}

# --- ADVANCED 4D PHYSICS ENGINE ---
class QuantumPhysicsEngine:
    def __init__(self):
        self.resonance_frequency = 1.0
        self.dimensional_stability = 1.0
        self.quantum_coherence = 0.95
        
    def calculate_4d_tensor(self, time_step: float, entropy: float) -> torch.Tensor:
        """Calculate 4D tensor dynamics with quantum properties"""
        base = torch.tensor([
            math.sin(time_step),
            math.cos(time_step),
            math.sin(time_step * 0.7),
            math.cos(time_step * 0.3)
        ])
        
        # Apply entropy as decoherence
        noise = torch.randn(4) * entropy * 0.5
        result = base + noise
        
        return result
    
    def compute_stability(self, tensor: torch.Tensor) -> float:
        """Compute system stability from 4D tensor"""
        magnitude = torch.norm(tensor).item()
        stability = 1.0 / (1.0 + magnitude * 0.5)
        return max(0.0, min(1.0, stability))

physics_engine = QuantumPhysicsEngine()

# --- ANOMALY DETECTION ---
def detect_anomalies(time_step: float) -> float:
    """Detect anomalies using statistical analysis of history"""
    if len(telemetry_history) < 10:
        return 0.0
    
    recent_entropy = [t["distortion"] for t in list(telemetry_history)[-10:]]
    mean_entropy = np.mean(recent_entropy)
    std_entropy = np.std(recent_entropy)
    
    current_entropy = system_state["entropy"]
    anomaly = abs(current_entropy - mean_entropy) / (std_entropy + 1e-6)
    
    return min(1.0, anomaly / 10.0)

# --- ADAPTIVE DEFENSE ALGORITHM ---
def adaptive_defense_ai(time_step: float, entropy: float, anomaly: float) -> float:
    """Advanced AI defense that adapts to threats"""
    if not system_state["defense_mode"]:
        return 1.0
    
    # Prepare threat input features
    threat_input = torch.tensor([
        entropy,
        anomaly,
        system_state["threat_level"],
        len(system_state["attack_vectors"]),
        system_state["defense_efficiency"],
        math.sin(time_step),
        math.cos(time_step),
        system_state["energy_level"] / 100.0
    ], dtype=torch.float32)
    
    # Run threat detection network
    with torch.no_grad():
        defense_scores = threat_net(threat_input.unsqueeze(0))
    
    # Calculate mitigation factor
    mitigation = defense_scores[0][0].item()
    
    # Energy-based defense scaling
    energy_factor = system_state["energy_level"] / 100.0
    final_mitigation = mitigation * energy_factor * system_state["defense_efficiency"]
    
    # Update energy consumption
    system_state["energy_level"] = max(0, system_state["energy_level"] - 0.5)
    
    return final_mitigation

# --- LEARNING & REINFORCEMENT ---
def update_defense_efficiency(success: bool):
    """Learn from defensive outcomes"""
    if system_state["learning_mode"]:
        adjustment = 1.02 if success else 0.98
        system_state["defense_efficiency"] = max(0.5, min(2.0, 
            system_state["defense_efficiency"] * adjustment))

# --- MAIN PHYSICS CALCULATION ---
def calculate_4d_physics(time_step: float) -> Dict:
    """Main calculation with multiple threat vectors"""
    
    # Calculate base tensor
    tensor_4d = physics_engine.calculate_4d_tensor(time_step, system_state["entropy"])
    
    # Simulate attack vectors
    noise_level = 0.0
    threat_contribution = 0.0
    
    if system_state["attack_mode"]:
        for vector in system_state["attack_vectors"]:
            # Different attack patterns
            if vector.pattern == "spike":
                pattern_value = 1.0 if (time_step * vector.frequency) % 1.0 < 0.1 else 0.0
            elif vector.pattern == "sine":
                pattern_value = (math.sin(time_step * vector.frequency) + 1) / 2
            elif vector.pattern == "cascade":
                pattern_value = min(1.0, time_step * vector.frequency * 0.1)
            else:  # random
                pattern_value = random.random()
            
            threat_contribution += vector.intensity * pattern_value
    
    noise_level = min(1.0, threat_contribution / len(system_state["attack_vectors"]) if system_state["attack_vectors"] else 0.0)
    
    # Apply adaptive defense
    anomaly_score = detect_anomalies(time_step)
    system_state["anomaly_score"] = anomaly_score
    
    if system_state["defense_mode"]:
        defense_factor = adaptive_defense_ai(time_step, noise_level, anomaly_score)
        noise_level = noise_level * (1.0 - defense_factor * 0.9)
    
    # Compute stability
    stability = physics_engine.compute_stability(tensor_4d)
    system_state["entropy"] = noise_level
    system_state["threat_level"] = threat_contribution
    
    # Determine status
    if noise_level > 0.6:
        status = "CRITICAL"
        color = "#ff0000"
    elif noise_level > 0.3:
        status = "WARNING"
        color = "#ffaa00"
    else:
        status = "SECURE"
        color = "#00ffcc"
    
    # Store telemetry
    telemetry_point = {
        "distortion": noise_level,
        "status": status,
        "time": time_step
    }
    telemetry_history.append(telemetry_point)
    
    return {
        "status": status,
        "color": color,
        "distortion": noise_level,
        "defense_active": system_state["defense_mode"],
        "anomaly_score": anomaly_score,
        "threat_level": threat_contribution,
        "stability": stability,
        "energy": system_state["energy_level"],
        "efficiency": system_state["defense_efficiency"]
    }

# --- API ENDPOINTS ---

@app.get("/telemetry")
def get_telemetry(time: float):
    """Get real-time telemetry data"""
    return calculate_4d_physics(time)

@app.post("/trigger_attack/{attack_type}")
def trigger_attack(attack_type: str = "ddos"):
    """Trigger specific attack vector"""
    if attack_type not in attack_patterns:
        raise HTTPException(status_code=400, detail="Unknown attack type")
    
    system_state["attack_mode"] = True
    system_state["defense_mode"] = False
    vector = attack_patterns[attack_type]
    system_state["attack_vectors"].append(vector)
    
    return {
        "msg": f"INTRUSION DETECTED: {attack_type.upper()}",
        "timestamp": datetime.now().isoformat(),
        "attack_vectors": len(system_state["attack_vectors"])
    }

@app.post("/activate_defense")
def activate_defense():
    """Activate advanced adaptive defense"""
    system_state["defense_mode"] = True
    system_state["adaptive_defense"] = True
    update_defense_efficiency(True)
    
    return {
        "msg": "ADAPTIVE NEURAL SHIELD DEPLOYED",
        "efficiency": system_state["defense_efficiency"]
    }

@app.post("/toggle_learning")
def toggle_learning():
    """Toggle machine learning mode"""
    system_state["learning_mode"] = not system_state["learning_mode"]
    return {"learning_mode": system_state["learning_mode"]}

@app.post("/reset_system")
def reset_system():
    """Reset system to initial state"""
    system_state["attack_mode"] = False
    system_state["defense_mode"] = False
    system_state["attack_vectors"] = []
    system_state["energy_level"] = 100.0
    system_state["adaptive_defense"] = False
    telemetry_history.clear()
    
    return {"msg": "SYSTEM REBOOTED", "timestamp": datetime.now().isoformat()}

@app.get("/system_status")
def get_system_status():
    """Get comprehensive system status"""
    return {
        "attack_active": system_state["attack_mode"],
        "defense_active": system_state["defense_mode"],
        "entropy": system_state["entropy"],
        "threat_level": system_state["threat_level"],
        "anomaly_score": system_state["anomaly_score"],
        "energy_level": system_state["energy_level"],
        "defense_efficiency": system_state["defense_efficiency"],
        "attack_count": len(system_state["attack_vectors"]),
        "learning_mode": system_state["learning_mode"]
    }

@app.get("/history")
def get_telemetry_history():
    """Get recent telemetry history for analysis"""
    return list(telemetry_history)

if __name__ == "__main__":
    print("--- 4DKenya Quantum Engine ONLINE ---")
    print("Attack Types: " + ", ".join(attack_patterns.keys()))
    print("Neural Defense Network Initialized")
    uvicorn.run(app, host="0.0.0.0", port=8000)
