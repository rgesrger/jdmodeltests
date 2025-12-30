# setup for distilgpt2
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv

python3.11 -m venv distgpt2venv
source distgpt2venv/bin/activate
pip install -r distilgpt2req.txt