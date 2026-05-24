import zipfile
import csv
import io

try:
    with zipfile.ZipFile('data/raw/nwl3_mailer_processed_view.csv.zip') as z:
        filename = z.namelist()[0]
        print(f"Reading {filename}...")
        with z.open(filename) as f:
            # Use TextIOWrapper to handle bytes as text
            wrapper = io.TextIOWrapper(f, encoding='utf-8-sig') 
            reader = csv.reader(wrapper)
            headers = next(reader)
            print("Columns found:")
            with open('data/raw/cols.txt', 'w', encoding='utf-8') as out:
                for col in headers:
                    out.write(col + '\n')
            print("Columns written to data/raw/cols.txt")
except Exception as e:
    print(f"Error: {e}")
