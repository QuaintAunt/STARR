import requests
import os

output_file = "wikipedia_dpr_passages.txt"

# Download the official DPR Wikipedia TSV archive.
url = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
gz_file = "psgs_w100.tsv.gz"
tsv_file = "psgs_w100.tsv"

# Download the compressed archive.
if not os.path.exists(gz_file):
    print("Downloading...")
    response = requests.get(url, stream=True)
    with open(gz_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete")

# Extract the archive.
if not os.path.exists(tsv_file):
    print("Extracting...")
    import gzip
    import shutil

    with gzip.open(gz_file, "rb") as f_in:
        with open(tsv_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Extraction complete")

import csv

count = 0
with open(tsv_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    reader = csv.reader(fin, delimiter="\t")
    next(reader)  # Skip the header.
    for parts in reader:
        if len(parts) >= 2:
            text = parts[1].strip().replace("\n", " ")
            fout.write(text + "\n")
            count += 1
            if count % 100000 == 0:
                print(f"Processed {count} passages...")

print(f"Done: wrote {count} passages")
