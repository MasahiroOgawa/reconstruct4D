```mermaid
flowchart TD
    A(compute flows) -->|optical flow| B(select points randomly)
    B --> |flow_x| C(compute previous position x_t-1)
    C --> |x_t-1| E(compute flow arrow)
    E --> F(select 2 arrows from distanced region)
    F --> |2 flow arrows| G(compute FoE as cross point of 2 arrows)
    G --> |iteratively compute FoE| H(update FoE by RANSAC)
    H --> |FoE, inliers, outliers, unkown| I{inlier > 90% of valid points?}
    I --> |Yes| J(extract outliers as moving points)
    I --> |No| K(compute dominant flow)
    K --> L(extract undominant flow as moving objects)
```