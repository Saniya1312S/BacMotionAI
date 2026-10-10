# BacMotionAI: Dashboard User Guide

Welcome to the BacMotionAI interactive dashboard. This guide will help you navigate the simulator and analyze real-time bacterial motility.

## 1. Launching the Dashboard
To start the dashboard locally:
1.  Navigate to the project directory in your terminal.
2.  Run a local Python server: `python -m http.server`
3.  Open your web browser and go to `http://localhost:8000` (or directly double-click the `index.html` file, though a local server is recommended for full feature support).

## 2. Hero Section
Upon loading, the Hero Section provides a high-level overview of the BacMotionAI project. It introduces the core concept of combining physics-based run-and-tumble mechanics with machine learning classification in microfluidic environments. 

## 3. Pipeline Flowchart
Below the hero section, you will find an interactive flowchart representing the system architecture. You can hover over or click on individual nodes (like "Experiment Setup", "Feature Extraction", "AI/ML Modelling") to view detailed information about that specific step in the data pipeline.

## 4. Simulator Controls
The core of the dashboard is the real-time simulation canvas. You can control the simulation parameters using the side panel:
*   **Channel Width:** Adjust the physical confinement of the microfluidic channel. Narrower channels lead to increased hydrodynamic wall interactions.
*   **Number of Bacteria:** Increase or decrease the population size in the simulation.
*   **Environment Presets:** Quickly load pre-configured scenarios (e.g., "Narrow Channel", "Open Space").
*   **Chemical Stimuli:** Apply different chemical gradients:
    *   *None:* Baseline motility.
    *   *Attractant:* Induces positive chemotaxis.
    *   *Repellent:* Induces negative chemotaxis.
    *   *Antibiotic:* Reduces motility and alters swimming dynamics.

## 5. Analysis Charts
As the simulation runs, data is extracted in real-time and displayed across 8 dynamic charts:
1.  **Trajectory Plot:** A 2D spatial view of the bacterial paths.
2.  **Velocity Distribution:** Histogram showing the current speed of the population.
3.  **Wall Interaction Time:** A gauge or bar chart showing the percentage of time bacteria spend trapped near the walls.
4.  **Tumble Frequency:** Tracks the rate of directional changes over time.
5.  **Mean Squared Displacement (MSD):** Indicates the diffusive nature of the transport.
6.  **Orientation Angle:** A polar plot showing the heading direction of the bacteria.
7.  **Chemotactic Index:** Tracks the efficiency of motion toward or away from a stimulus.
8.  **Feature Importance (Static):** Displays the relative importance of the 14 features used by the underlying Python ML model.

## 6. Reading Prediction Results
A dedicated panel displays the real-time motility regime prediction based on the current simulation state. The classifier will output one of the 5 distinct regimes (e.g., Baseline, Chemotactic, Trapped, etc.). 

> **⚠️ IMPORTANT NOTE:** 
> The live predictions shown in this browser-based simulator are generated using a lightweight, **JavaScript rule-based approximation**. This allows for real-time visualization without server latency. It is *not* the actual heavy Random Forest model trained in Python, which is reserved for the full dataset analysis documented in the project report.
