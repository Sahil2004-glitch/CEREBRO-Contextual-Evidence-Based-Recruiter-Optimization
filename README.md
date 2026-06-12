# CEREBRO: Contextual Evidence-Based Recruiter Optimization

A hybrid multi-signal candidate ranking system that ranks candidates the way a great recruiter would — by understanding *who fits the role*, not just matching keywords.

## Architecture Overview

```
100K Candidates
      │
      ▼
Phase 1: Fast Pre-Filter (heuristics, ~6s)
  ├─ Disqualify non-tech roles
  ├─ AI keyword presence scoring
  ├─ Basic YOE & activity filter
  └─ Top 5,000 shortlist
      │
      ▼
Phase 2: TF-IDF Semantic Similarity (~2s)
  └─ JD vs. candidate full text (title + summary + career + skills)
      │
      ▼
Phase 3: Multi-Signal Scoring (~2s per 5K)
  ├─ Hard Skill Coverage (6 groups: embeddings, vector DBs, eval, LLM, MLOps, Python)
  ├─ Career Quality (product company vs consulting, progression, tenure)
  ├─ Experience Fit (sweet spot: 6–8yr, JD range 5–9yr)
  ├─ Behavioral Signals (activity, response rate, notice period, GitHub)
  ├─ Location Fit (Pune/Noida > other JD cities > India > elsewhere)
  └─ Anti-Pattern Detection (honeypots, keyword stuffers, consulting-only, CV-only)
      │
      ▼
Weighted Ensemble → Top 100 → submission.csv
```

## Key Design Decisions

### Why NOT pure keyword matching
The JD explicitly warns: "The right answer is not finding candidates whose skills section contains the most AI keywords." CEREBRO reads *career descriptions*, not just skill lists. A candidate who built a recommendation system at a product company without listing "RAG" gets ranked higher than a title-match with hollow skill tags.

### Anti-pattern detection (honeypot safety)
CEREBRO detects and penalizes:
- **Honeypots**: Impossible timelines (claimed 8yr exp at 3yr-old company), expert skills with 0 months usage
- **Keyword stuffers**: Many AI skill tags but no matching technical depth in career descriptions
- **Consulting-only**: Entire career at TCS/Infosys/etc — explicitly flagged in JD
- **Pure researchers**: Heavy academic signals with no production deployment evidence
- **Title chasers**: Short tenures (< 12mo) across last 3 roles
- **CV/NLP mismatch**: Computer vision specialists without NLP/IR background

### Behavioral signals as availability multiplier
A perfect-on-paper candidate who hasn't logged in for 6 months is, for hiring purposes, not actually available. CEREBRO weights:
- Recency of last login (25%)
- Open-to-work flag (20%)
- Recruiter response rate (20%)
- Notice period (15%) — JD says sub-30 days preferred
- Interview completion rate (10%)
- GitHub activity (10%)

### Two-phase architecture for <5 min CPU constraint
Full TF-IDF on 100K docs is slow. CEREBRO uses a fast heuristic pre-filter (pure Python, no ML) to narrow to 5K candidates, then applies the heavy scoring only on the shortlist. Result: **~17 seconds** on a standard CPU.

## Quick Start

### Requirements
```bash
pip install numpy pandas scikit-learn scipy tqdm
```

### Produce submission CSV
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

### Options
```
--candidates  Path to candidates JSONL file (default: candidates.jsonl)
--out         Output CSV path (default: submission.csv)
--top_n       Number of top candidates (default: 100)
```

### Expected runtime
| Phase | Time |
|-------|------|
| Load 100K candidates | ~7s |
| Pre-filter (heuristics) | ~6s |
| TF-IDF on 5K shortlist | ~2s |
| Full scoring | ~2s |
| **Total** | **~17s** |

## Scoring Weights

| Component | Weight | Rationale |
|-----------|--------|-----------|
| TF-IDF Semantic Match | 22% | Captures JD–profile alignment beyond keywords |
| Hard Skills Coverage | 20% | Must-haves: embeddings, vector DBs, eval, Python |
| Career Quality | 18% | Product company > consulting, progression, tenure |
| Behavioral Signals | 15% | Availability, engagement, responsiveness |
| Anti-Pattern Penalty | 9% | Inverted: deducts for honeypots/mismatches |
| Experience Fit | 10% | 6–8yr sweet spot, 5–9yr acceptable range |
| Location Fit | 6% | Pune/Noida > other India cities > willing to relocate |

## File Structure
```
cerebro/
├── rank.py                    # Main ranker (single-command execution)
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── submission_metadata.yaml   # Hackathon metadata
└── submission.csv             # Generated output (top 100 ranked candidates)
```

## Compute Constraints Compliance
- ✅ No hosted LLM APIs called during ranking
- ✅ CPU-only (no GPU required)
- ✅ <5 minutes runtime (~17 seconds actual)
- ✅ <16GB RAM (uses ~1–2GB peak for TF-IDF matrix)
- ✅ Single command produces submission CSV
