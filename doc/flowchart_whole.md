```mermaid
flowchart LR
    imgt-1[image t-1] --> flow(optical flow)
    imgt[image t] --> flow
    imgt --> seg(segmentation)
    flow --> foe(FoE)
    seg --> foe
    foe --> cam(camera motion)
    cam --> movpix(moving pixel probability)

    style flow fill:#3390FF
    style foe fill:#098739
    style cam fill: #8EA928
    style movpix fill: #3DA98C
    style seg fill:#FF9633
```