# 4DKenya
“This project creates the first Kenyan-built 4D computational prototyping engine — a tool that allows scientists, engineers, and innovators to simulate interactions in higher-dimensional spaces. This opens unexplored frontiers for future technologies such as teleportation, ultra-efficient routing, quantum architectures, and advanced materials.”

Setting up 
```bash
git clone https://github.com/blacksnowmartin/4DKenya.hit
```
Make sure you have this installed in your machine

```bash
sudo apt update
sudo apt install python3-full python3-venv
```

Inside the 4DKenya folder make sure you run
```bash
python3 -m venv venv
```
(This creates a folder named venv that will store your libraries.)

To activate environment 
```bash
source venv/bin/activate
```

To install the packages safely you run
```bash
pip install torch numpy fastapi uvicorn
```

TO RUN THE MAIN PROJECT
```bash
cd ~/4DKenya
source venv/bin/activate
python server.py
```
