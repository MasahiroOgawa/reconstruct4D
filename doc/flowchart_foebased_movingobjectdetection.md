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
    R --> |Yes| M{flow existing area in ground/building > thre ?}:::cam
    R --> |No| N
    A(compute flows) -->|optical flow| M
    M --> |No| N(camera is stopping):::cam
    M --> |Yes| O(camera is moving):::cam
    O --> B(select points randomly from granound/building):::foe
    B --> |flow_x| C(compute previous position x_t-1):::foe
    C --> |x_t-1| E(compute flow arrow):::foe
    E --> F(select 2 arrows from distanced region):::foe
    F --> |2 flow arrows| G(compute FoE as cross point of 2 arrows):::foe
    H --> F
    G --> H(update FoE by RANSAC):::foe
    H --> |FoE, inliers, outliers, no flow|J(extract outliers as high probability moving pixels):::movobj
    N --> Q(extract flow existing pixels as high probability moving pixels):::movobj
    Q --> U(multiply moving pixel probability with prior probability):::movobj
    J --> U
    U --> V[moving pixel probability]:::movobj
    V --> S[check intersection of semantic segmentation and Instance segmentation]:::movobj
    S --> T[moving object detection]:::movobj

    style A fill:#3390FF
    classDef seg fill:#FF9633 
    classDef foe fill:#098739
    classDef cam fill: #8EA928
    classDef movobj fill: #3DA98C