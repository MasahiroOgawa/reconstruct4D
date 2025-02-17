```mermaid
%%{
  init: {
    'theme': 'dark',
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
    A(compute flows):::flow -->|optical flow| M
    M --> |No| N(camera is stopping):::cam
    M --> |Yes| O(camera is moving):::cam
    O --> B(select 2 points randomly from granound/building):::foe
    B --> |flow_x| C(compute previous position x_t-1):::foe
    C --> |x_t-1| E(compute flow arrow):::foe
    E --> |2 flow arrows| G(compute FoE as cross point of 2 arrows):::foe
    G --> H(update FoE by RANSAC):::foe
    H --> |FoE, inliers, outliers, no flow| J(compute moving pixel probability from flow length and angle difference):::movpix
    N --> Q(compute moving pixel probability from flow length):::movpix
    Q --> U(multiply moving pixel probability and prior probabilit from segmentation):::movpix
    J --> U
    U --> |moving pixel probability|S[check intersection Instance segmentation rate in moving pixel connected region]:::objref
    S --> T[extract moving object]:::objref

    classDef flow fill: #0650AC
    classDef seg fill: #C74606
    classDef cam fill: #6C0372
    classDef foe fill: #8B0000
    classDef movpix fill: #5A7203
    classDef objref fill: #0C6A51
