# 4DKenya

A collection of experimental 4D, AI, cyber-defense, and smart-city simulation prototypes built around a shared vision of next-generation systems design.

## Repository structure

```text
4DKenya/
├── README.md
├── index.html                  # root landing/demo entry
├── apps/
│   ├── 4d-core/
│   │   └── server.py
│   ├── defense-game/
│   │   ├── game.py
│   │   └── game.html
│   ├── city-nexus/
│   │   ├── nexus.py
│   │   └── nexus.html
│   ├── network-monitor/
│   │   ├── net.py
│   │   └── net.html
│   ├── network-analytics/
│   │   ├── net_monitor.py
│   │   └── net_monitor.html
│   ├── digital-twin/
│   │   ├── digital_twin.py
│   │   └── digital_twin.html
│   ├── experiments/
│   │   └── color_pattern.py
│   └── README.md
├── docs/
│   └── ideas.md
├── archive/
│   └── 4D-codes.zip
├── requirements.txt
├── .gitignore
└── .git/
```

## Quick start

```bash
cd 4DKenya
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## Run each project

### 4D core

```bash
cd apps/4d-core
python server.py
```

### Defense game

```bash
cd apps/defense-game
python game.py
```

### City nexus

```bash
cd apps/city-nexus
python nexus.py
```

### Network monitor

```bash
cd apps/network-monitor
python net.py
```

### Network analytics

```bash
cd apps/network-analytics
python net_monitor.py
```

### Digital twin

```bash
cd apps/digital-twin
python digital_twin.py
```

## Notes

- The repo was reorganized into feature-based folders so the experiments are easier to navigate.
- HTML demos are intended to be opened in a browser while their matching Python service runs locally.
- Archived or experimental content lives under the archive and experiments folders instead of polluting the main workspace.

Developed with curiosity, creativity, and community-driven innovation.

---

Martin Kitonga