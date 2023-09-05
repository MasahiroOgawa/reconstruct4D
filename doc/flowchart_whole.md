```mermaid
flowchart LR
    imgt-1[image t-1] --> flow(optical flow)
    imgt[image t] --> flow
    flow --> foe(FoE)
    foe --> cam(camera motion)
    cam --> movpix(moving pixel)
    imgt --> seg(segmentation)
    movpix --> mov[moving object region]
    seg --> mov

    style flow fill:#3390FF
    style foe fill:#098739
    style cam fill: #8EA928
    style movpix fill: #3DA98C
    style seg fill:#FF9633
```