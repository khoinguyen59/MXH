# Visualization & Interactive Demo Service (`src/visualization/`)

This package provides static graph rendering and the interactive Streamlit web demonstrator for GNN explanation subgraphs.

## Design Specification
- Visual design system, color palette, node sizing, and interactive layout standards are governed by [docs/design/DESIGN.md](file:///c:/Users/Nguyen%20Trong%20Khoi/Downloads/MXH/docs/design/DESIGN.md).

## Key Modules
- `graph_plotter.py`: High-resolution Matplotlib/NetworkX rendering highlighting salient explanation edges (crimson red) and target nodes (amber gold) over background neighborhoods.
- `app_demo.py`: Streamlit interactive dashboard with dual-panel layout (graph canvas on left, metrics & grounded NLE narrative on right).
