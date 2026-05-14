# 🐘 EleGuard: Multimodal Elephant Detection

EleGuard is an intelligent wildlife surveillance system designed for the 24/7 monitoring and detection of elephant activity in natural habitats. This project is a submission for the [Gemma 4 Good Hackathon](https://www.kaggle.com/competitions/gemma-4-good-hackathon/overview) and focuses on local, offline deployment to mitigate human-elephant conflict in rural communities.

EleGuard is a model trained on a dataset based on Gemma 4 E2B.

## Features
* Multimodal Detection: Integrated visual (Infrared/Daytime) and acoustic analysis for robust monitoring.
* Reasoning Distillation: Utilizes expert logic derived from Gemini 3.1 Flash to provide contextual safety alerts.
* Edge-Optimized: Built for offline, CPU-only execution on standard hardware using llama.cpp.
* Low-Connectivity Ready: Designed for resource-constrained environments with no cloud dependency.

## Technical Architecture
The system uses a Teacher-Student Knowledge Distillation framework to achieve high intelligence in a small footprint.

1. The Teacher Model: Gemini 3.1 Flash analyzed 2,600 samples to generate detailed "thought blocks" explaining visual and acoustic features.
2. The Student Model (EleGuard): Based on Gemma 4 E2B, fine-tuned using the Unsloth framework to internalize this expert reasoning.

## EleGuard Dashboard
The dashboard provides a real-time interface built with Streamlit.
* Multi-Camera Support: Simultaneous monitoring of multiple video feeds.
* Motion Filtering: OpenCV-based tracking and camera shake suppression.
* Local Inference: Connects to a local llama.cpp server for offline processing.

## How It Works
```text
Camera / Audio Input
        ↓
OpenCV Motion + Audio Filtering
        ↓
Event Trigger
        ↓
EleGuard (Gemma 4) Multimodal Reasoning
        ↓
SAFE / ALERT Output
        ↓
Local Dashboard Warning
```

##Installation and Usage

1. Clone the repository:
```Bash
git clone [https://github.com/MalithaBandara/EleGuard.git](https://github.com/MalithaBandara/EleGuard.git)
cd EleGuard
```

2. Install dependencies:
```Bash
pip install opencv-python streamlit librosa numpy moviepy requests
```

3. Start the Local Inference Server:
Run the llama.cpp server on port 8080 with your local model paths:
```bash
C:\EleGuard\llama-b9090-bin-win-cpu-x64\llama-server.exe -m C:\EleGuard\gemma-4-e2b-it.Q4_K_M.gguf --mmproj C:\EleGuard\gemma-4-e2b-it.F16-mmproj.gguf -c 512 -ngl 0 --host 127.0.0.1 --port 8080
```

4. Launch the Dashboard:
```Bash
python -m streamlit run app.py
```
## Resources
* **Model Weights (GGUF)**: Fine-tuned weights and multimodal projectors are hosted on [Hugging Face](https://huggingface.co/MalithaBandara/EleGuard).
* **Dataset**: The model was trained on the [EleGuard Dataset](https://www.kaggle.com/datasets/malithabandara/eleguard-dataset), which contains 2,600 curated samples of infrared/daytime imagery and bioacoustic spectrograms.

## Legal and Trademarks
* Gemma is a trademark of Google LLC.
* EleGuard is a model trained on a dataset based on Gemma 4 E2B.
* This project is independently developed and is not an official Google release.
