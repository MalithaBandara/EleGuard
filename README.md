# EleGuard

EleGuard is an offline AI-powered elephant early warning prototype built with **Gemma 4**, **OpenCV**, and **llama.cpp**.

It is designed for rural communities affected by human-elephant conflict and focuses on **local, CPU-only, low-connectivity deployment**.

---

## Overview

EleGuard monitors camera feeds and audio-based inputs to detect possible elephant activity and generate contextual safety alerts.

The system is built to run **fully offline** on ordinary laptops without requiring cloud infrastructure or dedicated GPUs.

It combines:

- lightweight motion filtering
- audio spectrogram analysis
- multimodal Gemma 4 reasoning
- SAFE / ALERT warning output

---

## Features

- Offline multimodal inference with Gemma 4
- CPU-only deployment using llama.cpp
- OpenCV-based motion filtering
- Camera shake suppression
- Audio spectrogram support
- Event-triggered alert generation
- Local dashboard for monitoring
- Designed for rural, low-resource environments

---

## How It Works

```text
Camera / Audio Input
        ↓
OpenCV Motion + Audio Filtering
        ↓
Event Trigger
        ↓
Gemma 4 Multimodal Reasoning
        ↓
SAFE / ALERT Output
        ↓
Local Warning / Dashboard Alert
