```mermaid
flowchart LR
    imgt-1[image t-1] --> flow(optical flow)
    imgt[image t] --> flow
    flow --> foe(FoE based\n moving pixel detection)
    imgt --> seg(segmentation)
    foe --> mov[moving object region]
    seg --> mov

    style flow fill:#3390FF
    style foe fill:#098739
    style seg fill:#FF9633
```