"""
Analyze Disease vs Event Type relationship.
Creates pivot table and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
print("Loading SPRSound Event Level Dataset...")
df = pd.read_csv('SPRSound_Event_Level_Dataset_CLEAN.csv')

print(f"Total events: {len(df):,}")
print(f"Unique diseases: {df['disease'].nunique()}")
print(f"Unique event types: {df['event_type'].nunique()}")

# Create pivot table: Disease x Event Type
print("\n" + "="*100)
print("PIVOT TABLE: Disease vs Event Type")
print("="*100)

pivot = pd.pivot_table(
    df,
    values='filename',  # Count events
    index='disease',
    columns='event_type',
    aggfunc='count',
    fill_value=0
)

# Sort by total events
pivot['TOTAL'] = pivot.sum(axis=1)
pivot = pivot.sort_values('TOTAL', ascending=False)

# Display the pivot table
print("\nAbsolute counts:")
print(pivot.to_string())

# Calculate percentages (row-wise)
print("\n" + "="*100)
print("PERCENTAGE DISTRIBUTION (By Disease)")
print("="*100)

pivot_pct = pivot.copy()
# Don't include TOTAL in percentage calculation
event_cols = [col for col in pivot.columns if col != 'TOTAL']
pivot_pct[event_cols] = pivot_pct[event_cols].div(pivot_pct['TOTAL'], axis=0) * 100

print("\nPercentage (each row sums to 100%):")
print(pivot_pct.to_string())

# Save to CSV
pivot.to_csv('disease_event_pivot_counts.csv')
pivot_pct.to_csv('disease_event_pivot_percentages.csv')
print("\n✓ Saved to: disease_event_pivot_counts.csv")
print("✓ Saved to: disease_event_pivot_percentages.csv")

# Create detailed markdown report
print("\nCreating detailed report...")

with open('Disease_Event_Pivot_Analysis.md', 'w', encoding='utf-8') as f:
    f.write("# Disease vs Event Type Analysis\n\n")
    f.write("## Summary\n\n")
    f.write(f"- **Total Events**: {len(df):,}\n")
    f.write(f"- **Unique Diseases**: {df['disease'].nunique()}\n")
    f.write(f"- **Unique Event Types**: {df['event_type'].nunique()}\n\n")
    
    f.write("## Pivot Table: Absolute Counts\n\n")
    f.write("| Disease | " + " | ".join(event_cols) + " | TOTAL |\n")
    f.write("|" + "-|" * (len(event_cols) + 2) + "\n")
    
    for disease, row in pivot.iterrows():
        f.write(f"| {disease} |")
        for col in event_cols:
            f.write(f" {int(row[col]):,} |")
        f.write(f" **{int(row['TOTAL']):,}** |\n")
    
    f.write("\n## Pivot Table: Percentage Distribution\n\n")
    f.write("Percentage of each event type within each disease (rows sum to 100%)\n\n")
    f.write("| Disease | " + " | ".join(event_cols) + " |\n")
    f.write("|" + "-|" * (len(event_cols) + 1) + "\n")
    
    for disease, row in pivot_pct.iterrows():
        f.write(f"| {disease} |")
        for col in event_cols:
            f.write(f" {row[col]:.1f}% |")
        f.write("\n")
    
    f.write("\n## Key Findings\n\n")
    
    # Find diseases with highest percentage of each event type
    f.write("### Diseases Most Associated with Each Event Type\n\n")
    for event_type in event_cols:
        top_diseases = pivot_pct.nlargest(3, event_type)
        f.write(f"**{event_type}**:\n")
        for disease, row in top_diseases.iterrows():
            if row[event_type] > 0:
                f.write(f"- {disease}: {row[event_type]:.1f}% ({int(pivot.loc[disease, event_type]):,} events)\n")
        f.write("\n")
    
    # Disease profiles
    f.write("### Disease Acoustic Profiles\n\n")
    for disease, row in pivot_pct.head(10).iterrows():
        total_events = int(pivot.loc[disease, 'TOTAL'])
        f.write(f"**{disease}** ({total_events:,} events):\n")
        
        # Get top 3 event types for this disease
        event_profile = []
        for col in event_cols:
            if row[col] > 5:  # Only show if >5%
                event_profile.append(f"{col} ({row[col]:.1f}%)")
        
        if event_profile:
            f.write(f"- Main characteristics: {', '.join(event_profile)}\n")
        f.write("\n")

print("✓ Saved to: Disease_Event_Pivot_Analysis.md")

# Create visualization
print("\nCreating heatmap visualization...")

# Select top 15 diseases by event count
top_diseases = pivot.nlargest(15, 'TOTAL').index
pivot_top = pivot_pct.loc[top_diseases, event_cols]

plt.figure(figsize=(14, 10))
sns.heatmap(
    pivot_top,
    annot=True,
    fmt='.1f',
    cmap='YlOrRd',
    cbar_kws={'label': 'Percentage (%)'},
    linewidths=0.5
)
plt.title('Disease vs Event Type Heatmap (Top 15 Diseases)\nPercentage Distribution', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Event Type', fontsize=12, fontweight='bold')
plt.ylabel('Disease', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('disease_event_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved to: disease_event_heatmap.png")

print("\n" + "="*100)
print("ANALYSIS COMPLETE!")
print("="*100)
print("\nGenerated files:")
print("  1. disease_event_pivot_counts.csv - Absolute event counts")
print("  2. disease_event_pivot_percentages.csv - Percentage distribution")
print("  3. Disease_Event_Pivot_Analysis.md - Detailed markdown report")
print("  4. disease_event_heatmap.png - Visual heatmap")
