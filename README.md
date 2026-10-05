<div align="center">

# Physics Projects

**Scientific computing and AI work in physics: a Schrödinger equation simulator and an explainable AI research presentation.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

</div>

---

## Overview

This repository groups two physics projects:

1. **Schrödinger equation visualisation:** a Streamlit app that solves the Schrödinger equation numerically and shows the results as scientific plots and as art. This is the first prototype of **SchrödArt**; the full version, with a second solver and a live demo, is in **[Alyaa203/P2i](https://github.com/Alyaa203/P2i)**.
2. **Explainable AI (XAI) for pigment classification:** the slides from a one-month research internship (2025) at the Laboratoire de Chimie Physique – Matière et Rayonnement (CNRS / Sorbonne Université), presented on 25 June 2025.

**Why it matters:** together they show numerical simulation in Python and applied machine learning on real scientific data.

---

## Features

### Schrödinger simulator (`streamlit_app.py`)
- **2D stationary regime:** builds the Hamiltonian with finite differences and computes the lowest-energy eigenstates with sparse diagonalisation (ARPACK)
- **1D time-dependent regime:** evolves the wave function with modal decomposition and shows the probability density over time, including a 3D surface
- **Quantum art:** turns the computed wave functions and eigenvalues into images
- All physical parameters adjustable live with sliders

### XAI research presentation (`presentation.pdf`)
- Goal: understand **which parts of the spectrum** CNN and DNN models rely on when they classify art pigments from hyperspectral data (785 spectral bands)
- Methods: **SHAP** (DeepSHAP) and **LIME**, applied to spectral zones defined from known pigment features
- Results: models reach up to 99% accuracy; SHAP and LIME highlight the useful spectral zones and also show the limits of global explanations

---

## Tech stack

| Area | Tools |
| --- | --- |
| Simulation | Python, NumPy, SciPy (sparse matrices, `eigsh`, `eigh_tridiagonal`) |
| Visualisation | Matplotlib, Pillow |
| Web interface | Streamlit |
| XAI research | Deep learning (CNN, DNN), SHAP, LIME, hyperspectral data |

---

## Getting started

Requires Python 3.9 or later.

```bash
git clone https://github.com/Alyaa203/physics-projects.git
cd physics-projects
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open http://localhost:8501. The XAI presentation is in [`presentation.pdf`](presentation.pdf).

### Project structure

```
├── streamlit_app.py   # Schrödinger web interface
├── simulation.py      # Numerical solver (2D stationary + 1D time-dependent)
├── visualisation.py   # Artistic rendering
├── presentation.pdf   # XAI internship presentation
└── requirements.txt
```

---

**Author:** Alyaa Saab, engineering student at ENSC (Bordeaux INP)
