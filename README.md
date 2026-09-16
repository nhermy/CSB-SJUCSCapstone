# Tactical Analysis Using Graph Neural Networks in Professional Soccer

A Spatiotemporal Graph Neural Network (STGNN) that represents professional soccer matches as time-evolving graphs — players as nodes, spatial and tactical relationships as edges — to perform team formation recognition and player-level tactical analysis.

**91.14% validation accuracy** on 4-class formation recognition using StatsBomb 360 tracking data.

## Overview

Traditional soccer analytics (xG, pass completion, shot maps) score discrete events but miss the relational structure that actually drives the game — a through ball is a spatial interaction between a passer, a runner, a defensive line, and the lane between them. This project models soccer the way it's actually played: as a relational system.

The core architecture combines:
- **Graph Convolutional Network (GCN)** layers for spatial message passing within a single frame
- **Gated Recurrent Unit (GRU)** for temporal aggregation across a 15-frame window (τ = 15)
- **Sum-based aggregation**, identified through benchmarking as optimal for formation recognition due to its sensitivity to player density and neighborhood size

## Key Results

| Formation | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| 4-3-3     | 0.89      | 0.94   | 0.91     |
| 4-4-2     | 0.92      | 0.86   | 0.89     |
| 4-2-3-1   | 0.88      | 0.88   | 0.88     |
| 3-5-2     | 0.96      | 0.97   | 0.97     |
| **Overall Validation Accuracy** | | | **91.14%** |

The 3-5-2 is the most cleanly separable formation in the model's latent space (t-SNE), consistent with its geometrically distinct five-midfielder structure. All other formations show substantial overlap — which isn't a model failure, but reflects that formations aren't discrete tactical identities in practice; two teams labeled 4-3-3 can play very differently depending on compactness and pressing triggers.

Applied to the **2022 FIFA World Cup Final**, the model produced a frame-by-frame tactical narrative of Argentina's shift from an attacking 4-3-3 to a defensive 3-5-2 as the match progressed, correctly flagging the tactical instability during France's late equalizing goals.

## What I Explored

- **Graph construction**: benchmarked Euclidean threshold, Delaunay triangulation, and Voronoi adjacency as edge-construction strategies
- **Aggregation functions**: compared sum, mean, max, and attention aggregation across formation recognition and player-level tactical analysis tasks
- **Static GNN → STGNN**: showed that single-frame GNNs can't distinguish a temporary attacking transition from a sustained defensive structure, motivating the move to a temporal architecture
- **Ground truth vs. game reality**: analyzed cases where the model's prediction diverged from the human-annotated formation label but was visually more tactically accurate — highlighting a limitation of supervised learning against static labels for a continuous, fluid sport
- **Set-piece robustness**: examined how the model maintains classification confidence during corner kicks and free kicks (via temporal memory) even though the spatial graph itself becomes uninformative during high-density events

## Tech Stack

- Python, PyTorch
- Graph Convolutional Networks (GCN), Gated Recurrent Units (GRU)
- StatsBomb 360 tracking data
- t-SNE for latent space visualization
- Delaunay triangulation for graph construction

## Repository Structure

```
.
├── data/               # Data loading & preprocessing (StatsBomb 360 — not included, see Data below)
├── graph_construction/ # Edge-building strategies (Euclidean, Delaunay, Voronoi)
├── models/             # GCN + GRU (STGNN) architecture, aggregation functions
├── training/           # Training loop, class-imbalance handling
├── analysis/           # t-SNE visualization, tactical flow plots, World Cup case study
├── figures/            # Generated figures from the paper
├── paper/              # Full write-up (PDF)
└── README.md
```

*(Adjust this to match your actual folder layout before pushing.)*

## Data

This project uses [StatsBomb 360](https://statsbomb.com/) tracking data, which is subject to StatsBomb's own licensing terms. Raw tracking data is **not included** in this repository — see StatsBomb's data access program if you want to reproduce results.

## Future Directions

- Move from post-match analysis toward real-time tactical feedback
- Explore unsupervised/self-supervised formation clustering to avoid mismatches with static human-annotated labels
- Extend beyond formation recognition into individual player credit (e.g. GoalNet-style expected threat models) alongside xG
- Build gender-specific models rather than assuming architectures trained on men's data transfer to women's soccer

## Author

**Noah Hermanson** — M.S. Applied Data Science, Clarkson University
[LinkedIn](https://linkedin.com/in/noah-hermanson)

Full write-up with methodology, figures, and references available in [`/paper`](./SOFTv4).
