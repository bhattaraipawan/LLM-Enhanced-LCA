# AI-Enhanced Life Cycle Assessment

## An Integrated Framework for Automated Upfront Embodied Carbon Assessment of Building Materials

This repository contains the source code, sample data, and supporting documentation for the research study:

> **AI-Enhanced Life Cycle Assessment: An Integrated Framework for Automated Upfront Embodied Carbon Assessment of Building Materials**

The proposed framework integrates a locally deployable large language model, openLCA, the European Reference Life Cycle Database, and a FastAPI-based interface to automate upfront embodied carbon assessment for building materials and whole-building Bills of Materials.

The framework is designed to reduce the manual effort associated with material interpretation, environmental process matching, emission-factor retrieval, unit conversion, carbon calculation, and result visualization.

---

## Overview

Whole-building life cycle assessment commonly requires practitioners to manually:

- interpret inconsistent material descriptions;
- normalize quantities and units;
- search environmental databases;
- select representative processes;
- document proxy assumptions;
- retrieve emission factors; and
- calculate and aggregate environmental impacts.

The framework presented in this repository automates these tasks for product-stage life cycle modules **A1-A3**, including:

- A1: raw-material supply;
- A2: transportation to manufacturing;
- A3: material manufacturing.

Users can evaluate either a single construction material or upload a complete Bill of Materials for automated whole-building assessment.

---

## Main Features

- Automated processing of Bill of Materials spreadsheets
- Single-material embodied carbon estimation
- Material-name interpretation using a large language model
- Fuzzy matching with openLCA process records
- Automated openLCA API queries
- ELCD-based emission-factor retrieval
- Automated unit conversion and quantity processing
- Material-level Global Warming Potential calculation
- Whole-building A1-A3 embodied carbon calculation
- Material hotspot identification
- Excel report generation
- Graphical material-contribution summaries
- Transparent source classification for every inventory item

Each material is classified using one of the following source categories:

1. **Direct database match**
2. **Documented database proxy**
3. **LLM-supported estimate**

This classification allows users to distinguish database-supported results from proxy assignments and knowledge-based estimates.

---

## Framework Architecture

```text
Bill of Materials
        |
        v
Input Validation and Preprocessing
        |
        v
Material Dictionary and Fuzzy Search
        |
        v
LLM-Assisted Material Interpretation
        |
        v
openLCA Process Matching
        |
        v
ELCD Emission-Factor Retrieval
        |
        v
Material-Level GWP Calculation
        |
        v
Whole-Building GWP Aggregation
        |
        v
Excel Reports and Visualizations
