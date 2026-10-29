import torch
import torch.nn as nn
import math
import random
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime
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

# --- NETWORK PROTOCOL DEFINITIONS ---
@dataclass
class NetworkProtocol:
    name: str
    port: int
    traffic_weight: float
    latency: float
    packet_loss: float

@dataclass
class PacketFlow:
    source: str
    destination: str
    protocol: str
    size: int
    timestamp: float
    priority: int = 0
    status: str = "ACTIVE"

# --- ADVANCED NEURAL NETWORK FOR TRAFFIC PREDICTION ---
class TrafficPredictionNet(nn.Module):
    """LSTM-inspired traffic prediction network"""
    def __init__(self, input_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 5)  # Predict 5 steps ahead
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

traffic_net = TrafficPredictionNet()

# --- NETWORK PROTOCOLS ---
protocols = {
    "HTTP": NetworkProtocol("HTTP", 80, 0.4, 0.02, 0.001),
    "HTTPS": NetworkProtocol("HTTPS", 443, 0.35, 0.03, 0.0005),
    "DNS": NetworkProtocol("DNS", 53, 0.15, 0.01, 0.002),
    "SSH": NetworkProtocol("SSH", 22, 0.05, 0.05, 0.0001),
    "FTP": NetworkProtocol("FTP", 21, 0.05, 0.04, 0.003)
}

# --- GLOBAL STATE ---
network_state = {
    "total_bandwidth": 0.0,
    "active_connections": 0,
    "packet_queue": deque(maxlen=1000),
    "anomaly_score": 0.0,
    "ddos_active": False,
    "congestion_level": 0.0,
    "protocol_distribution": {},
    "network_health": 100.0,
    "jitter": 0.0,
    "packet_loss_rate": 0.0,
    "threat_vectors": []
}

# History for trend analysis
traffic_history = deque(maxlen=200)
health_history = deque(maxlen=200)

# --- ADVANCED TRAFFIC ANALYSIS ENGINE ---
class NetworkAnalyticsEngine:
    def __init__(self):
        self.buffer_capacity = 1000.0
        self.queue_length = 0
        self.throughput = 0.0
        self.latency_ms = 0.0
        
    def simulate_packet_flow(self, time_step: float, ddos_active: bool) -> List[PacketFlow]:
        """Simulate realistic network packet flow"""
        flows = []
        
        for proto_name, proto in protocols.items():
            # Base traffic pattern
            packet_count = int(20 * proto.traffic_weight * (math.sin(time_step * 2) + 1))
            
            if ddos_active:
                if proto_name == "HTTP" or proto_name == "HTTPS":
                    packet_count = int(packet_count * 10)  # DDoS amplification
            
            for _ in range(packet_count):
                flow = PacketFlow(
                    source=f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                    destination="10.0.0.1",
                    protocol=proto_name,
                    size=random.randint(64, 1500),
                    timestamp=time_step,
                    priority=random.randint(0, 7),
                    status="ACTIVE"
                )
                flows.append(flow)
        
        return flows
    
    def calculate_metrics(self, flows: List[PacketFlow]) -> Dict:
        """Calculate network metrics from packet flows"""
        if not flows:
            return {"throughput": 0, "latency": 0, "jitter": 0, "loss_rate": 0}
        
        total_bytes = sum(f.size for f in flows)
        throughput = total_bytes / 1024.0  # KB/s
        
        # Simulate latency with variations
        latency = np.random.normal(25, 5)  # Mean 25ms, std 5ms
        jitter = np.random.normal(2, 1)   # Mean 2ms, std 1ms
        loss_rate = sum(1 for f in flows if random.random() < 0.005) / len(flows) if flows else 0
        
        return {
            "throughput": max(0, throughput),
            "latency": max(0, latency),
            "jitter": max(0, jitter),
            "loss_rate": min(1, max(0, loss_rate))
        }
    
    def compute_congestion(self, flows: List[PacketFlow]) -> float:
        """Estimate network congestion level"""
        queue_depth = len(flows) / 100.0
        return min(1.0, queue_depth)

analytics = NetworkAnalyticsEngine()

# --- ANOMALY DETECTION ---
def detect_anomalies(flows: List[PacketFlow], metrics: Dict) -> float:
    """Detect network anomalies using statistical methods"""
    if len(traffic_history) < 20:
        return 0.0
    
    recent_throughputs = [h["throughput"] for h in list(traffic_history)[-20:]]
    mean_tp = np.mean(recent_throughputs)
    std_tp = np.std(recent_throughputs)
    
    current_tp = metrics["throughput"]
    z_score = abs(current_tp - mean_tp) / (std_tp + 1e-6)
    
    # Check for protocol-based anomalies
    http_flows = len([f for f in flows if f.protocol in ["HTTP", "HTTPS"]])
    total_flows = len(flows)
    http_ratio = http_flows / total_flows if total_flows > 0 else 0
    
    # HTTP floods are anomalies
    http_anomaly = 1.0 if http_ratio > 0.8 else 0.0
    
    anomaly_score = (min(z_score / 10.0, 1.0) + http_anomaly) / 2.0
    return min(1.0, anomaly_score)

# --- THREAT DETECTION ---
def detect_threats(flows: List[PacketFlow], anomaly: float) -> List[str]:
    """Identify specific network threats"""
    threats = []
    
    # DDoS detection
    http_flows = len([f for f in flows if f.protocol in ["HTTP", "HTTPS"]])
    if http_flows > 200:
        threats.append("DDOS_ATTACK")
        network_state["ddos_active"] = True
    else:
        network_state["ddos_active"] = False
    
    # Port scanning
    unique_ports = len(set(p.port for p in protocols.values()))
    if len(flows) > 500 and anomaly > 0.7:
        threats.append("PORT_SCAN")
    
    # Data exfiltration
    ssh_flows = len([f for f in flows if f.protocol == "SSH"])
    if ssh_flows > 50:
        threats.append("SUSPICIOUS_SSH")
    
    return threats

# --- NEURAL PREDICTION ---
def predict_traffic(metrics: Dict) -> torch.Tensor:
    """Use neural network to predict next traffic patterns"""
    input_features = torch.tensor([
        metrics["throughput"] / 100.0,
        metrics["latency"] / 50.0,
        metrics["jitter"] / 10.0,
        metrics["loss_rate"],
        network_state["congestion_level"],
        len(network_state["threat_vectors"]) / 5.0,
        network_state["anomaly_score"],
        1.0 if network_state["ddos_active"] else 0.0,
        network_state["network_health"] / 100.0,
        network_state["active_connections"] / 100.0
    ], dtype=torch.float32)
    
    with torch.no_grad():
        predictions = traffic_net(input_features.unsqueeze(0))
    
    return predictions

# --- CALCULATE NETWORK HEALTH ---
def calculate_health(metrics: Dict, anomaly: float, threats: List[str]) -> float:
    """Calculate overall network health percentage"""
    health = 100.0
    
    # Latency impact
    latency_penalty = (metrics["latency"] / 50.0) * 20  # Max 20 points
    health -= latency_penalty
    
    # Packet loss impact
    loss_penalty = metrics["loss_rate"] * 30  # Max 30 points
    health -= loss_penalty
    
    # Anomaly impact
    anomaly_penalty = anomaly * 25  # Max 25 points
    health -= anomaly_penalty
    
    # Threat impact
    threat_penalty = len(threats) * 10  # 10 points per threat
    health -= threat_penalty
    
    return max(0, min(100, health))

# --- MAIN PULSE ENDPOINT ---
@app.get("/network_pulse")
def get_pulse(time: float):
    """Main network monitoring endpoint"""
    
    # 1. Generate packet flows
    flows = analytics.simulate_packet_flow(time, network_state["ddos_active"])
    
    # 2. Calculate metrics
    metrics = analytics.calculate_metrics(flows)
    network_state["total_bandwidth"] = metrics["throughput"]
    network_state["active_connections"] = len(flows)
    network_state["jitter"] = metrics["jitter"]
    network_state["packet_loss_rate"] = metrics["loss_rate"]
    
    # 3. Compute congestion
    congestion = analytics.compute_congestion(flows)
    network_state["congestion_level"] = congestion
    
    # 4. Detect anomalies
    anomaly = detect_anomalies(flows, metrics)
    network_state["anomaly_score"] = anomaly
    
    # 5. Detect threats
    threats = detect_threats(flows, anomaly)
    network_state["threat_vectors"] = threats
    
    # 6. Calculate health
    health = calculate_health(metrics, anomaly, threats)
    network_state["network_health"] = health
    
    # 7. Protocol distribution
    protocol_dist = {}
    for flow in flows:
        protocol_dist[flow.protocol] = protocol_dist.get(flow.protocol, 0) + 1
    network_state["protocol_distribution"] = protocol_dist
    
    # 8. Predict future traffic
    predictions = predict_traffic(metrics)
    
    # Store history
    traffic_history.append(metrics)
    health_history.append({"health": health, "anomaly": anomaly, "time": time})
    
    # Determine status and color
    if len(threats) > 0:
        status = f"ALERT: {', '.join(threats[:2])}"
        color = "#ff0000"  # Red
    elif health < 50:
        status = "CRITICAL: NETWORK DEGRADATION"
        color = "#ff6600"  # Orange
    elif anomaly > 0.6:
        status = "WARNING: ANOMALY DETECTED"
        color = "#ffaa00"  # Yellow
    elif congestion > 0.7:
        status = "CONGESTION: HIGH LOAD"
        color = "#ffaa00"
    else:
        status = "HEALTHY"
        color = "#00ffcc"  # Cyan
    
    return {
        "load": metrics["throughput"] / 100.0,  # Normalized
        "status": status,
        "color": color,
        "server_message": f"Health: {health:.1f}% | Latency: {metrics['latency']:.1f}ms | Loss: {metrics['loss_rate']*100:.2f}%",
        "health": health,
        "throughput": metrics["throughput"],
        "latency": metrics["latency"],
        "jitter": metrics["jitter"],
        "loss_rate": metrics["loss_rate"],
        "anomaly_score": anomaly,
        "active_connections": len(flows),
        "threat_count": len(threats),
        "congestion": congestion,
        "protocol_distribution": protocol_dist
    }

# --- ANALYTICS ENDPOINTS ---
@app.get("/network_stats")
def get_stats():
    """Get comprehensive network statistics"""
    return {
        "ddos_active": network_state["ddos_active"],
        "health": network_state["network_health"],
        "anomaly_score": network_state["anomaly_score"],
        "threat_vectors": network_state["threat_vectors"],
        "protocol_distribution": network_state["protocol_distribution"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/traffic_history")
def get_history():
    """Get recent traffic history"""
    return {
        "traffic_history": list(traffic_history)[-50:],
        "health_history": list(health_history)[-50:]
    }

@app.post("/trigger_ddos")
def trigger_ddos():
    """Simulate DDoS attack"""
    network_state["ddos_active"] = True
    return {"msg": "DDoS simulation started"}

@app.post("/stop_attack")
def stop_attack():
    """Stop attack simulation"""
    network_state["ddos_active"] = False
    network_state["threat_vectors"] = []
    return {"msg": "Attack simulation stopped"}

if __name__ == "__main__":
    print("--- 4DKenya Advanced Network Monitor STARTING ---")
    print("Protocols: " + ", ".join(protocols.keys()))
    print("Neural traffic prediction enabled")
    uvicorn.run(app, host="0.0.0.0", port=8000)
