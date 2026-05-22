"""
Compute retention rate from Decagon polypharmacy file after
STITCH-to-DrugBank entity alignment.
"""

import pandas as pd

# Paths (relative to project root)
COMBO_FILE = "Data/bio-decagon-combo.csv"
MAPPING_FILE = "Data/drug-mappings.tsv"

print("Loading drug mapping file...")
mapping_df = pd.read_csv(MAPPING_FILE, sep='\t').dropna(
    subset=['drugbankId', 'stitch_id']
)
stitch_to_drugbank = pd.Series(
    mapping_df.drugbankId.str.strip().values,
    index=mapping_df.stitch_id.str.strip()
).to_dict()
print(f"  Total STITCH->DrugBank mappings available: {len(stitch_to_drugbank)}")

print("\nLoading Decagon polypharmacy file...")
combo_df = pd.read_csv(COMBO_FILE)
print(f"  Total Decagon triples (drug, drug, side_effect): {len(combo_df)}")
print(f"  Decagon columns: {list(combo_df.columns)}")

# Count unique drugs in Decagon
unique_stitch_in_decagon = pd.unique(
    combo_df[['STITCH 1', 'STITCH 2']].values.ravel()
)
print(f"\n  Unique drugs in Decagon (STITCH IDs): {len(unique_stitch_in_decagon)}")

# Count how many unique drugs got mapped
mapped_drugs = [s for s in unique_stitch_in_decagon
                if str(s).strip() in stitch_to_drugbank]
print(f"  Of these, successfully mapped to DrugBank: {len(mapped_drugs)}")
print(f"  Drug-level retention: "
      f"{100*len(mapped_drugs)/len(unique_stitch_in_decagon):.2f}%")

# Count how many triples survive (both drugs must be mappable)
combo_df['drug_1_id'] = combo_df['STITCH 1'].astype(str).str.strip().map(stitch_to_drugbank)
combo_df['drug_2_id'] = combo_df['STITCH 2'].astype(str).str.strip().map(stitch_to_drugbank)
mappable = combo_df.dropna(subset=['drug_1_id', 'drug_2_id'])
print(f"\n  Triples where BOTH drugs map to DrugBank: {len(mappable)}")
print(f"  Triple-level retention: {100*len(mappable)/len(combo_df):.2f}%")

# Count unique side effects in original vs after mapping
unique_se_original = combo_df['Side Effect Name'].nunique()
unique_se_after = mappable['Side Effect Name'].nunique()
print(f"\n  Unique side effects in Decagon (original): {unique_se_original}")
print(f"  Unique side effects after mapping: {unique_se_after}")

# Investigate why 1309 causes_* edge types
print(f"\n  After alphanumeric-only normalization:")
clean_se = mappable['Side Effect Name'].apply(
    lambda s: ''.join(e for e in str(s) if e.isalnum() or e == '_')
)
print(f"  Unique normalized side effect strings: {clean_se.nunique()}")

# Count how many positive examples for thrombocytopenia specifically
thrombo_mask = mappable['Side Effect Name'].str.lower().str.contains(
    'thrombocytopenia', na=False
)
print(f"\n  Decagon triples mentioning 'thrombocytopenia': {thrombo_mask.sum()}")
print(f"  Distinct side effect names containing 'thrombocytopenia':")
for name in mappable.loc[thrombo_mask, 'Side Effect Name'].unique():
    count = (mappable['Side Effect Name'] == name).sum()
    print(f"    '{name}': {count} drug pairs")