Copy and paste the following code to [Mermaid Live Editor](https://mermaid.live/) to save the diagram as an image, vector graphics, or PDF:

```mermaid
flowchart TD
  Q1[1: The intervention is targeted towards an adult population in any employment or volunteer sector including academia, industry, law, government, education, business, and STEM.]
  Q2[2: Is the study a relevant study design?]
  Q3[3: Does this study examine any gender equity outcomes?]
  Exclude
  Include

  Q1 -->|Yes| Q2
  Q1 -->|Maybe/Unclear| Q2
  Q1 -->|No| Exclude

  Q2 -->|Yes| Q3
  Q2 -->|Maybe/Unclear| Q3
  Q2 -->|No| Exclude

  Q3 -->|Yes| Include
  Q3 -->|Maybe/Unclear| Include
  Q3 -->|No| Exclude
```
