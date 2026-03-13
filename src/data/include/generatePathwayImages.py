import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
    # Inicjalizacja macierzy zerami (czarne tło)
    output = np.zeros((pic_height, pic_width))
    k = 0
    for sym_list in [sig_sym, met_sym, cancer_sym]:
        for pathway in sym_list:
            symbols = [s for s in pathway if s is not None]
            matched = [s for s in symbols if s in data.index]
            if len(matched) > 0:
                values = data.loc[matched, data.columns[col_idx]].values
                # Wpisujemy wartości od lewej strony wiersza
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
                                 sample_groups=None,
                                 max_images=None):

    # ---- Load JSON of KEGG ----
    with open(json_path, "r", encoding="utf-8") as f:
        kegg_data = json.load(f)

    cancer_sym = list(kegg_data["cancer"].values())
    sig_sym = list(kegg_data["signaling"].values())
    met_sym = list(kegg_data["metabolism"].values())

    # Obliczanie wymiarów
    def count_matched(sym_list):
        counts = []
        for pathway in sym_list:
            pathway = [s for s in pathway if s is not None]
            counts.append(sum(s in data_filtered.index for s in pathway))
        return counts

    all_counts = count_matched(sig_sym) + count_matched(met_sym) + count_matched(cancer_sym)
    pic_width = max(all_counts) + 10
    pic_height = len(sig_sym) + len(met_sym) + len(cancer_sym)

    # ---- Create directories ----
    pathway_path = create_dir_if_not_exists(path, pathways_folder)
    image_path = create_dir_if_not_exists(pathway_path, "Images")
    matrix_path = create_dir_if_not_exists(pathway_path, "Matrices")

    sample_names = list(data_filtered.columns)

    if sample_groups is not None and len(sample_groups) == len(sample_names):
        file_labels = [f"{g}_{n}" for g, n in zip(sample_groups, sample_names)]
    else:
        file_labels = sample_names

    total_samples = min(max_images, len(sample_names)) if max_images else len(sample_names)

    # PALETA: od black do red, 256 kolorów
    cmap_kegg = LinearSegmentedColormap.from_list("KeggRed", ["black", "red"], N=256)

    print(f"Generowanie obrazów ({pic_width}x{pic_height}) dla {total_samples} próbek...")

    for col_idx in range(total_samples):
        # 1. Pobranie danych
        output = build_expression_pathway_matrix(
            sig_sym, met_sym, cancer_sym,
            data_filtered, col_idx, pic_width, pic_height
        )
        matrix_save_name = os.path.join(matrix_path, f"{file_labels[col_idx]}.txt")
        np.savetxt(matrix_save_name, output, fmt='%.6f', delimiter=' ')        


        # 2. PRZETWARZANIE: Wszystko ujemne staje się 0 (idealna czerń)
        processed_data = np.maximum(output, 0)
        
        max_val = np.max(processed_data)
        if max_val <= 0: max_val = 1 # Uniknięcie błędu przy pustych danych

        # 3. RYSOWANIE
        fig = plt.figure(figsize=(pic_width / 100, pic_height / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]) # Brak marginesów
        
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Wyświetlamy macierz. origin='lower' naprawia odwrócenie do góry nogami.
        ax.imshow(
            processed_data, 
            cmap=cmap_kegg, 
            vmin=0, 
            vmax=max_val, 
            aspect="auto",
            interpolation='nearest',
            origin='lower' 
        )

        ax.axis("off")

        # 4. ZAPIS
        save_name = os.path.join(image_path, f"{sample_names[col_idx]}.png")
        plt.savefig(
            save_name,
            dpi=100,
            facecolor='black',
            edgecolor='none',
            pad_inches=0
        )
        plt.close(fig)

    print("KEGG pathway images prepared.")
    return pathway_path