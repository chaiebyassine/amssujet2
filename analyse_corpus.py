from collections import Counter
import string
import csv

def load_data(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f if line.strip()]

data = load_data("train.txt")

# 1) Taille corpus
total = len(data)

# 2) Longueurs
lengths = [len(p) for p in data]
avg_len = sum(lengths) / total if total else 0
min_len = min(lengths) if lengths else 0
max_len = max(lengths) if lengths else 0

length_dist = Counter(lengths)

# 3) Types de caractères
only_letters = 0
only_digits = 0
only_special = 0
letters_digits = 0
letters_special = 0
digits_special = 0
letters_digits_special = 0

def has_letter(s): return any(c.isalpha() for c in s)
def has_digit(s):  return any(c.isdigit() for c in s)
def has_spec(s):   return any((not c.isalnum()) for c in s)

for pwd in data:
    L = has_letter(pwd)
    D = has_digit(pwd)
    S = has_spec(pwd)

    if L and not D and not S:
        only_letters += 1
    elif D and not L and not S:
        only_digits += 1
    elif S and not L and not D:
        only_special += 1
    elif L and D and not S:
        letters_digits += 1
    elif L and S and not D:
        letters_special += 1
    elif D and S and not L:
        digits_special += 1
    elif L and D and S:
        letters_digits_special += 1

# 4) Redondance
unique = len(set(data))
duplicates = total - unique

# 5) Export pour graphique (CSV)
# colonne 1 = longueur, colonne 2 = nombre
with open("length_distribution.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["length", "count"])
    for k in sorted(length_dist.keys()):
        w.writerow([k, length_dist[k]])

# Affichage résumé
print("===== Analyse du corpus (train) =====")
print("Nombre total de mots :", total)
print("Nombre de mots uniques :", unique)
print("Nombre de doublons :", duplicates)
print()
print("Longueur moyenne :", avg_len)
print("Longueur min :", min_len)
print("Longueur max :", max_len)
print()
print("===== Types de mots =====")
print("Uniquement lettres :", only_letters)
print("Uniquement chiffres :", only_digits)
print("Uniquement spéciaux :", only_special)
print("Lettres + chiffres :", letters_digits)
print("Lettres + spéciaux :", letters_special)
print("Chiffres + spéciaux :", digits_special)
print("Lettres + chiffres + spéciaux :", letters_digits_special)
print()
print("Fichier exporté pour le graphique : length_distribution.csv")
