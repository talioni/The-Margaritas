# Margaritas - Smart Trash Can

An automated waste sorting system that uses image recognition to identify trash and sort it into the correct bin — no manual selection needed.

## What It Does

Hold an item in front of the camera. The system classifies it (plastic, metal, paper, or general waste), rotates the lid to the right bin, opens it, and resets after you drop the item. If it's unsure, it asks you to try again.

## Tech Stack

- Raspberry Pi 5 — runs a Python-based classifier using OpenCV for image capture and a rule-based model for categorization
- Microcontroller — handles real-time motor control (DC motor + servo), limit switches, and a serial command protocol (`BIN:X` / `DONE:X`)
- Webcam (OV2640) — captures items via I2C interface
- Servo (SG90) — opens the lid in an elevator-door style

## First Prototype Step: Live Color Recognition (Webcam)

This first step is a **simple color-recognition prototype** in Python.  
It is **not true material recognition** yet (for example: plastic vs. metal). It only detects dominant color in a center region of the webcam image.

### Features

- Opens the default webcam
- Shows live webcam preview
- Draws a center ROI rectangle
- Converts ROI from BGR to HSV
- Uses simple HSV threshold masks for:
  - red
  - green
  - blue
  - yellow
  - brown
  - white
  - black
  - unknown
- Chooses dominant color by mask pixel count
- Overlays detected label on-screen
- Quits with `q`

### Environment

- Linux
- Python 3
- OpenCV
- NumPy

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python3 color_test.py
```

### Notes

- HSV ranges are in `COLOR_RANGES` inside `color_test.py` so you can quickly tune them for your camera and lighting.
- This ROI pipeline is a useful base that can later evolve into **surface/material classification** by adding extra features or an ML model.
