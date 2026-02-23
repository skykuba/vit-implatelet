import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# Helper: create directory
# ==============================

def create_dir_if_not_exists(main_dir, sub_dir):
    path = os.path.join(main_dir, sub_dir)
    os.makedirs(path, exist_ok=True)
    return path


# ==============================
# Build pathway matrix
# ==============================

def build_expression_pathway_matrix(sig_sym, met_sym, cancer_sym,
                                    data, col_idx, pic_width, pic_height):

    output = np.zeros((pic_height, pic_width))
    k = 0

    for sym_list in [sig_sym, met_sym, cancer_sym]:
        for pathway in sym_list:
            symbols = [s for s in pathway if s is not None]

            matched = [s for s in symbols if s in data.index]

            if len(matched) > 0:
                values = data.loc[matched, data.columns[col_idx]].values
                output[k, 0:len(values)] = values

            k += 1

    return output


# ==============================
# Main function
# ==============================

def generate_kegg_pathway_images(path,
                                 data_filtered,
                                 json_path,
                                 pathways_folder="KEGG_Pathway_Image",
                                 sample_groups=None):

    # ---- Load JSON of KEGG ----
    with open(json_path, "r", encoding="utf-8") as f:
        kegg_data = json.load(f)

    cancer_sym = list(kegg_data["cancer"].values())
    sig_sym = list(kegg_data["signaling"].values())
    met_sym = list(kegg_data["metabolism"].values())

    # ---- Calculate picWidth ----
    def count_matched(sym_list):
        counts = []
        for pathway in sym_list:
            pathway = [s for s in pathway if s is not None]
            counts.append(sum(s in data_filtered.index for s in pathway))
        return counts

    all_counts = (
        count_matched(sig_sym)
        + count_matched(met_sym)
        + count_matched(cancer_sym)
    )

    pic_width = max(all_counts) + 10
    pic_height = len(sig_sym) + len(met_sym) + len(cancer_sym)

    print(f"Maksymalna liczba genów w szlaku: {max(all_counts)} -> picWidth: {pic_width}")
    print(f"Liczba szlaków KEGG: {pic_height}")

    # ---- Create directories ----
    pathway_path = create_dir_if_not_exists(path, pathways_folder)
    matrix_path = create_dir_if_not_exists(pathway_path, "Matrices")
    image_path = create_dir_if_not_exists(pathway_path, "Images")

    # ---- File labels ----
    sample_names = list(data_filtered.columns)

    if sample_groups and len(sample_groups) == len(sample_names):
        file_labels = [
            f"{sample_groups[i]}_{sample_names[i]}"
            for i in range(len(sample_names))
        ]
    else:
        file_labels = sample_names

    print(f"Generowanie obrazów dla {len(sample_names)} próbek...")

    # ---- Generate for each sample ----
    for col_idx in range(len(sample_names)):

        output = build_expression_pathway_matrix(
            sig_sym, met_sym, cancer_sym,
            data_filtered,
            col_idx,
            pic_width,
            pic_height
        )

        # Save matrix
        np.savetxt(
            os.path.join(matrix_path, file_labels[col_idx] + ".txt"),
            output,
            fmt="%.5f"
        )

        # Prepare image (only positive values -> red)
        pos_output = np.maximum(output, 0)
        max_val = np.max(pos_output)

        if max_val <= 0 or np.isnan(max_val):
            max_val = 1

        plt.figure(figsize=(pic_width / 100, pic_height / 100))
        plt.imshow(
            pos_output.T,
            cmap="Reds",
            vmin=0,
            vmax=max_val,
            aspect="auto"
        )
        plt.axis("off")

        plt.savefig(
            os.path.join(image_path, file_labels[col_idx] + ".png"),
            dpi=100,
            bbox_inches="tight",
            pad_inches=0
        )
        plt.close()

    print("KEGG pathway images prepared")
    return matrix_path