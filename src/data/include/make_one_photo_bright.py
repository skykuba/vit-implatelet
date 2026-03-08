import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ==============================
# Konfiguracja
# ==============================
# Upewnij się, że ten plik jest w tym samym folderze co skrypt
INPUT_FILE = "Malignant_AMC-Chol-002-TR2072.txt"
OUTPUT_FILE = "Malignant_AMC-Chol-002-TR2072_rotated_fixed.png"

# Stałe wymiary z Twojego skryptu R
PIC_WIDTH = 345
PIC_HEIGHT = 243

def generate_kegg_rotated_script(input_path, output_path):
    print(f"--- ROZPOCZĘCIE GENEROWANIA ---")
    
    # 1. Wczytanie danych
    try:
        raw_data = np.loadtxt(input_path)
        print(f"Wczytano macierz: {raw_data.shape}")
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        return

    # 2. Logika przetwarzania wartości (Shift + Mask tła)
    # Znajdujemy biologiczne tło (najmniejsza wartość, np. -2.37)
    min_val = np.min(raw_data)
    
    # Przesuwamy wszystko tak, by -2.37 stało się 0 (czarny)
    processed_data = raw_data - min_val
    
    # Maska dla technicznego tła (zera na końcu wierszy)
    # Wymuszamy, by oryginalne 0 pozostały czarne.
    zero_mask = (raw_data == 0)
    processed_data[zero_mask] = 0
    
    # Upewniamy się, że nie mamy wartości ujemnych
    processed_data = np.maximum(processed_data, 0)
    
    # 3. Przygotowanie wizualizacji
    max_val = np.max(processed_data)
    if max_val <= 0: max_val = 1
    
    # Paleta: czarny -> czerwony -> czerwony (zgodnie z R: c("black", "red", "red"))
    cmap_colors = ["black", "red", "red"]
    cmap_kegg = LinearSegmentedColormap.from_list("KeggPalette", cmap_colors, N=25)

    # 4. Generowanie obrazu (Dokładnie wymiary 345x243)
    # figsize w calach, dpi=100 daje 1 piksel na 1 wartość macierzy
    fig = plt.figure(figsize=(PIC_WIDTH/100, PIC_HEIGHT/100), dpi=100)
    
    # add_axes([0,0,1,1]) usuwa wszystkie marginesy (odpowiednik par(mar=c(0,0,0,0)))
    ax = fig.add_axes([0, 0, 1, 1])
    
    # --- KLUCZOWA ZMIANA TUTAJ ---
    # Zmieniamy origin z 'upper' na 'lower'.
    # To sprawia, że wiersz 0 (dół danych) jest na dole obrazu, 
    # a ostatni wiersz (góra danych) jest na górze obrazu.
    # To poprawnie obraca Twój szlak "do góry nogami".
    ax.imshow(
        processed_data, 
        cmap=cmap_kegg, 
        vmin=0, 
        vmax=max_val, 
        aspect='auto', 
        interpolation='nearest',
        origin='lower' # <--- OBRÓT O 180 STOPNI (W PIONIE)
    )
    
    # Wyłączenie osi i ustawienie czarnego tła figury
    ax.axis('off')
    fig.patch.set_facecolor('black')

    # 5. Zapis do pliku z wymuszeniem czarnego tła (facecolor)
    plt.savefig(
        output_path, 
        dpi=100, 
        facecolor='black', 
        edgecolor='none', 
        pad_inches=0
    )
    plt.close(fig)
    
    print(f"--- GOTOWE ---")
    print(f"Obraz poprawnie obrócony i zapisany jako: {output_path}")

if __name__ == "__main__":
    # Sprawdzenie czy plik istnieje w bieżącym folderze
    if os.path.exists(INPUT_FILE):
        generate_kegg_rotated_script(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"Błąd: Nie znaleziono pliku {INPUT_FILE} w folderze skryptu.")
        print(f"Upewnij się, że plik wejściowy znajduje się w tym samym katalogu co ten skrypt Python.")