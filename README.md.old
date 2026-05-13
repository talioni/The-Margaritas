Margaritas - Smart Trash Can
An automated waste sorting system that uses image recognition to identify trash and sort it into the correct bin — no manual selection needed.
What It Does
Hold an item in front of the camera. The system classifies it (plastic, metal, paper, or general waste), rotates the lid to the right bin, opens it, and resets after you drop the item. If it's unsure, it asks you to try again.
Tech Stack

Raspberry Pi 5 — runs a Python-based classifier using OpenCV for image capture and a rule-based model for categorization
Microcontroller — handles real-time motor control (DC motor + servo), limit switches, and a serial command protocol (BIN:X / DONE:X)
Webcam (OV2640) — captures items via I2C interface
Servo (SG90) — opens the lid in an elevator-door style
