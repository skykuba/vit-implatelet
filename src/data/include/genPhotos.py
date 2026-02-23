import os
import gzip
import shutil
import requests
import pandas as pd

from normalize_deseq import normalize_deseq2_no_report
from annotateData import annotateData
from generatePathwayImages import generate_kegg_pathway_images


# ==============================
# Paths
# ==============================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

data_dir = os.path.abspath(os.path.join(BASE_DIR, "../../data"))
raw_dir = os.path.join(data_dir, "raw")
output_dir = os.path.join(data_dir, "output")
script_dir = os.path.dirname(os.path.abspath(__file__))


os.makedirs(raw_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# ==============================
# Files
# ==============================

counts_file_csv = "counts_raw.csv"
#counts_file_tsv = os.path.join(raw_dir, "counts_raw.tsv")
#counts_file = counts_file_csv if os.path.exists(counts_file_csv) else counts_file_tsv
counts_file = counts_file_csv

gtf_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.chr_patch_hapl_scaff.annotation.gtf.gz"
gtf_gz = os.path.join(raw_dir, "gencode.v19.annotation.gtf.gz")
gtf_file = os.path.join(raw_dir, "gencode.v19.annotation.gtf")

kegg_json = os.path.join(script_dir, "kegg_pathways.json")

# ==============================
# Download GTF if needed
# ==============================

if not os.path.exists(gtf_file):
    print("Downloading GTF...")
    if not os.path.exists(gtf_gz):
        response = requests.get(gtf_url, stream=True)
        with open(gtf_gz, "wb") as f:
            shutil.copyfileobj(response.raw, f)

    print("Extracting GTF...")
    with gzip.open(gtf_gz, "rb") as f_in:
        with open(gtf_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(gtf_gz)
else:
    print("GTF file already exists, skipping download.")

# ==============================
# Load raw counts
# ==============================

if not os.path.exists(counts_file):
    raise FileNotFoundError("Raw counts file not found.")

print(f"Loading raw counts from: {counts_file}")

if counts_file.endswith(".csv"):
    counts_raw = pd.read_csv(counts_file, index_col=0)
else:
    counts_raw = pd.read_csv(counts_file, sep="\t", index_col=0)

counts_raw = counts_raw.apply(pd.to_numeric, errors="coerce").dropna()

# ==============================
# Normalize
# ==============================

print("Normalizing data...")
data_filtered = normalize_deseq2_no_report(counts_raw, fast=False)
print("Normalization completed.")

# ==============================
# Annotate
# ==============================

print("Annotating data...")
annotation_result = annotateData(gtf_file, data_filtered)
data_filtered_annotated = annotation_result["dataFiltered"]
print("Annotation completed.")

# ==============================
# Sample groups (optional)
# ==============================

sample_groups = None
sample_info_path = os.path.join(raw_dir, "sample_info.csv")

if os.path.exists(sample_info_path):
    sample_info = pd.read_csv(sample_info_path)
    if "Group" in sample_info.columns:
        if len(sample_info) == data_filtered_annotated.shape[1]:
            sample_groups = sample_info["Group"].tolist()
        else:
            print("Warning: sample_info length mismatch – ignoring groups.")
else:
    print("No sample_info.csv found – files will be named by sample only.")

# ==============================
# Generate KEGG images
# ==============================

print("Starting generation of KEGG pathway images...")

matrix_path = generate_kegg_pathway_images(
    path=data_dir,
    data_filtered=data_filtered_annotated,
    json_path=kegg_json,
    sample_groups=sample_groups
)

print(f"Completed.")
print(f"Matrices saved to: {matrix_path}")
print(f"Images saved to: {matrix_path.replace('Matrices', 'Images')}")