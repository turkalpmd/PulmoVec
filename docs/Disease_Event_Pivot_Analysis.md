# Disease vs Event Type Analysis

## Summary

- **Total Events**: 24,808
- **Unique Diseases**: 16
- **Unique Event Types**: 8

## Pivot Table: Absolute Counts

| Disease | Coarse Crackle | Fine Crackle | No Event | Normal | Rhonchi | Stridor | Wheeze | Wheeze+Crackle | TOTAL |
|-|-|-|-|-|-|-|-|-|-|
| Pneumonia (non-severe) | 50 | 1,341 | 146 | 8,821 | 104 | 2 | 569 | 34 | **11,067** |
| Unknown | 65 | 1,838 | 3 | 2,367 | 9 | 8 | 262 | 261 | **4,813** |
| Bronchitis | 18 | 141 | 6 | 1,906 | 53 | 15 | 183 | 0 | **2,322** |
| Control Group | 10 | 2 | 27 | 1,932 | 34 | 23 | 16 | 0 | **2,044** |
| Asthma | 1 | 28 | 15 | 1,379 | 0 | 7 | 162 | 0 | **1,592** |
| Pneumonia (severe) | 32 | 117 | 9 | 952 | 8 | 0 | 146 | 7 | **1,271** |
| Other respiratory diseases | 0 | 2 | 2 | 365 | 2 | 19 | 20 | 1 | **411** |
| Bronchiectasia | 0 | 41 | 3 | 258 | 7 | 0 | 7 | 0 | **316** |
| Bronchiolitis | 0 | 17 | 0 | 157 | 0 | 0 | 131 | 0 | **305** |
| Acute upper respiratory infection | 0 | 2 | 1 | 233 | 0 | 0 | 0 | 0 | **236** |
| Hemoptysis | 0 | 1 | 15 | 218 | 0 | 0 | 0 | 0 | **234** |
| Pulmonary hemosiderosis | 0 | 0 | 2 | 70 | 0 | 0 | 0 | 0 | **72** |
| Chronic cough | 1 | 0 | 1 | 53 | 0 | 0 | 0 | 0 | **55** |
| Airway foreign body | 0 | 0 | 0 | 38 | 0 | 0 | 0 | 0 | **38** |
| Protracted bacterial bronchitis | 0 | 0 | 0 | 21 | 0 | 0 | 0 | 0 | **21** |
| Kawasaki disease | 0 | 0 | 0 | 2 | 0 | 0 | 9 | 0 | **11** |

## Pivot Table: Percentage Distribution

Percentage of each event type within each disease (rows sum to 100%)

| Disease | Coarse Crackle | Fine Crackle | No Event | Normal | Rhonchi | Stridor | Wheeze | Wheeze+Crackle |
|-|-|-|-|-|-|-|-|-|
| Pneumonia (non-severe) | 0.5% | 12.1% | 1.3% | 79.7% | 0.9% | 0.0% | 5.1% | 0.3% |
| Unknown | 1.4% | 38.2% | 0.1% | 49.2% | 0.2% | 0.2% | 5.4% | 5.4% |
| Bronchitis | 0.8% | 6.1% | 0.3% | 82.1% | 2.3% | 0.6% | 7.9% | 0.0% |
| Control Group | 0.5% | 0.1% | 1.3% | 94.5% | 1.7% | 1.1% | 0.8% | 0.0% |
| Asthma | 0.1% | 1.8% | 0.9% | 86.6% | 0.0% | 0.4% | 10.2% | 0.0% |
| Pneumonia (severe) | 2.5% | 9.2% | 0.7% | 74.9% | 0.6% | 0.0% | 11.5% | 0.6% |
| Other respiratory diseases | 0.0% | 0.5% | 0.5% | 88.8% | 0.5% | 4.6% | 4.9% | 0.2% |
| Bronchiectasia | 0.0% | 13.0% | 0.9% | 81.6% | 2.2% | 0.0% | 2.2% | 0.0% |
| Bronchiolitis | 0.0% | 5.6% | 0.0% | 51.5% | 0.0% | 0.0% | 43.0% | 0.0% |
| Acute upper respiratory infection | 0.0% | 0.8% | 0.4% | 98.7% | 0.0% | 0.0% | 0.0% | 0.0% |
| Hemoptysis | 0.0% | 0.4% | 6.4% | 93.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| Pulmonary hemosiderosis | 0.0% | 0.0% | 2.8% | 97.2% | 0.0% | 0.0% | 0.0% | 0.0% |
| Chronic cough | 1.8% | 0.0% | 1.8% | 96.4% | 0.0% | 0.0% | 0.0% | 0.0% |
| Airway foreign body | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Protracted bacterial bronchitis | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Kawasaki disease | 0.0% | 0.0% | 0.0% | 18.2% | 0.0% | 0.0% | 81.8% | 0.0% |

## Key Findings

### Diseases Most Associated with Each Event Type

**Coarse Crackle**:
- Pneumonia (severe): 2.5% (32 events)
- Chronic cough: 1.8% (1 events)
- Unknown: 1.4% (65 events)

**Fine Crackle**:
- Unknown: 38.2% (1,838 events)
- Bronchiectasia: 13.0% (41 events)
- Pneumonia (non-severe): 12.1% (1,341 events)

**No Event**:
- Hemoptysis: 6.4% (15 events)
- Pulmonary hemosiderosis: 2.8% (2 events)
- Chronic cough: 1.8% (1 events)

**Normal**:
- Airway foreign body: 100.0% (38 events)
- Protracted bacterial bronchitis: 100.0% (21 events)
- Acute upper respiratory infection: 98.7% (233 events)

**Rhonchi**:
- Bronchitis: 2.3% (53 events)
- Bronchiectasia: 2.2% (7 events)
- Control Group: 1.7% (34 events)

**Stridor**:
- Other respiratory diseases: 4.6% (19 events)
- Control Group: 1.1% (23 events)
- Bronchitis: 0.6% (15 events)

**Wheeze**:
- Kawasaki disease: 81.8% (9 events)
- Bronchiolitis: 43.0% (131 events)
- Pneumonia (severe): 11.5% (146 events)

**Wheeze+Crackle**:
- Unknown: 5.4% (261 events)
- Pneumonia (severe): 0.6% (7 events)
- Pneumonia (non-severe): 0.3% (34 events)

### Disease Acoustic Profiles

**Pneumonia (non-severe)** (11,067 events):
- Main characteristics: Fine Crackle (12.1%), Normal (79.7%), Wheeze (5.1%)

**Unknown** (4,813 events):
- Main characteristics: Fine Crackle (38.2%), Normal (49.2%), Wheeze (5.4%), Wheeze+Crackle (5.4%)

**Bronchitis** (2,322 events):
- Main characteristics: Fine Crackle (6.1%), Normal (82.1%), Wheeze (7.9%)

**Control Group** (2,044 events):
- Main characteristics: Normal (94.5%)

**Asthma** (1,592 events):
- Main characteristics: Normal (86.6%), Wheeze (10.2%)

**Pneumonia (severe)** (1,271 events):
- Main characteristics: Fine Crackle (9.2%), Normal (74.9%), Wheeze (11.5%)

**Other respiratory diseases** (411 events):
- Main characteristics: Normal (88.8%)

**Bronchiectasia** (316 events):
- Main characteristics: Fine Crackle (13.0%), Normal (81.6%)

**Bronchiolitis** (305 events):
- Main characteristics: Fine Crackle (5.6%), Normal (51.5%), Wheeze (43.0%)

**Acute upper respiratory infection** (236 events):
- Main characteristics: Normal (98.7%)

