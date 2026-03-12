#
# do łatwiejszego testowania, uruchamia generacje obrazów bezpośrednio z pliku znormalizowanego
#


import os
import pandas as pd
from annotateData import annotateData
from generatePathwayImages import generate_kegg_pathway_images

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

data_dir = os.path.abspath(os.path.join(BASE_DIR, "../../data"))
raw_dir = os.path.join(data_dir, "raw")
output_dir = os.path.join(data_dir, "output")
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

normalized_path = os.path.join(BASE_DIR, "normalized_counts.tsv")
counts_raw_path = os.path.join(raw_dir, "counts_raw.csv")

print(normalized_path)
# ---------- WARUNEK: jeśli już istnieje znormalizowany plik ----------
if os.path.exists(normalized_path):
    print("Loading already normalized data...")
    data_filtered = pd.read_csv(normalized_path, sep="\t", index_col=0)

# --------------------------
# 2. Annotate
# --------------------------
gtf_file = os.path.join(raw_dir, "gencode.v19.annotation.gtf")
print("Annotating data...")
annotation_result = annotateData(gtf_file, data_filtered)
data_filtered_annotated = annotation_result["dataFiltered"]

# --------------------------
# 3. KEGG images
# --------------------------
kegg_json = os.path.join(script_dir, "kegg_pathways.json")
matrix_path = generate_kegg_pathway_images(
    path=data_dir,
    data_filtered=data_filtered_annotated,
    json_path=kegg_json,
    sample_groups=None,
    max_images=100
)