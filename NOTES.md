# Findings so far

- Burst detection (global baseline) correctly flags Laxmii (bomb) and Dil Bechara (inflation).
- Burst reviewers are mostly one-shot accounts (~84% for bomb days vs ~39% elsewhere).
- Text model on weak labels scored high, but the result is confounded: matched
  genuine reviews were few, and topic/reviewer type separate the groups.
  Text alone cannot reliably tell bombing from ordinary negative reviews.
- Plan: adjusted rating from burst + reviewer evidence, reported as a range.
