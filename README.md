# Imprints

1. Download the data from the LC
2. Run the following command to generate local files:

```zsh
python data_collection.py \
    --input_dir "/path/to/xmlgz" \
    --output_dir "/path/to/output_pickles" \
    --class_range PR
```

python data_collection.py \
    --input_dir "/Users/erik/Documents/Corpora/Library of Congress Books All MARC records/2020445551_2019" \
    --output_dir "../../data" \
    --class_range F

Clean local data:

```zsh
python data_cleaning.py \
    --input_dir "../../data/PS/" \
    --output_csv "../../data/PS/data.csv" \
    --class_range PS
```
