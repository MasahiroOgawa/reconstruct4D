```mermaid
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'fontSize': '28pix',
      'curve': 'linear'
    }
  }
}%%

flowchart TB
    D(segment):::seg --> |segmented mask| P(remove sky area):::seg
    P --> |object mask| R{Does ground/building exist?}:::seg
    R --> |Yes| M{flow existing rate in static area > thre ?}:::cam
    R --> |No| N
    A(compute flows) -->|optical flow| M
    M --> |No| N(camera is stopping):::cam
    M --> |Yes| O(camera is moving):::cam
    O --> B(select 2 points randomly from granound/building):::foe
    B --> |flow_x| C(compute previous position x_t-1):::foe
    C --> |x_t-1| E(compute flow arrow):::foe
    E --> |2 flow arrows| G(compute FoE as cross point of 2 arrows):::foe
    G --> H(update FoE by RANSAC):::foe
    H --> |FoE, inliers, outliers, no flow| J(compute moving pixel probability; outliers have higher probability):::movobj
    N --> Q(compute moving pixel probability from flow length):::movobj
    Q --> U(multiply moving pixel probability and prior probabilit from segmentation):::movobj
    J --> U
    U --> |moving pixel probability|S[check intersection Instance segmentation rate in moving pixel connected region]:::movobj
    S --> T[extract moving object]:::movobj

    style A fill:#3390FF
    classDef seg fill:#FF9633 
    classDef foe fill:#098739
    classDef cam fill: #8EA928
    classDef movobj fill: #3DA98C