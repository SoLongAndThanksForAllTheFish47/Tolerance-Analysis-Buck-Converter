# Tolerance-Analysis-Buck-Converter

Copyright 2026 shersoli

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![Python](https://github.com/shersoli/Tolerance-Analysis-Buck-Converter/actions/workflows/python-app.yml/badge.svg)

This project is released under a MIT LICENSE

## Project Scope
The goal of this project is to provide a robust reliability analysis tool for power electronics engineers. In hardware design, components like resistors or voltage sources are never perfect—they have tolerances (e.g., ±10%). 

This tool uses a **Monte Carlo Simulation** to model how these component variations propagate through a standard Buck Converter circuit. By simulating thousands of units, we can predict the manufacturing **Yield** (the percentage of units that will function within the desired voltage range) before a single PCB is ever manufactured.

## Purpose of Functions
The core logic is contained within `Worst_Case_Buck_Converter.py` and consists of the following four functions:

* **`calculate_v_out`**: The physical engine of the script. It implements the fundamental transfer function of a step-down converter ($V_{out} = V_{in} \cdot D \cdot \eta$). It includes input validation to ensure duty cycles remain within physical limits (0 to 1).
* **`run_monte_carlo`**: The simulation driver. It generates a normal (Gaussian) distribution of input voltages based on a defined tolerance and runs the converter model for a specified number of iterations. It uses a fixed seed to ensure reproducible results during testing.
* **`get_statistics`**: An analysis utility that processes the simulation results to provide the mean ($\mu$) and standard deviation ($\sigma$), helping engineers understand the "spread" of the output voltage.
* **`calculate_yield`**: The decision-making function. It compares the simulation results against a target voltage and an allowed error margin to calculate the final manufacturing success rate.

## AI Disclosure
I used **Gemini (AI)** as a collaborative partner to brainstorm the project scope and to assist in structuring the `pytest` suite, the markdown formatting of the README and the GitHub Actions. The logic of `pytest`  was manually verified for accuracy. The functions were taken from a real life project and adapted to fit the scope of this assignment.
