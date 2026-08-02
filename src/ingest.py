from bs4 import BeautifulSoup
import os

filings_dir = "data/filings"

def table_to_text(table):
    """Convert an HTML table into readable row-by-row text."""
    rows_text = []
    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all(["td", "th"])
        cell_texts = [cell.get_text(strip=True) for cell in cells]
        # Skip empty rows (common in these heavily-styled SEC tables)
        cell_texts = [c for c in cell_texts if c]
        if cell_texts:
            rows_text.append(" | ".join(cell_texts))
    return "\n".join(rows_text)

for filename in os.listdir(filings_dir):
    if filename.endswith(".html"):
        filepath = os.path.join(filings_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        # Replace each table with a placeholder-preserved text block
        for table in soup.find_all("table"):
            table_text = table_to_text(table)
            table.replace_with("\n[TABLE]\n" + table_text + "\n[/TABLE]\n")

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        clean_text = "\n".join(lines)

        print(f"{filename}: {len(clean_text):,} characters extracted")

        output_path = os.path.join(filings_dir, filename.replace(".html", "_extracted.txt"))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text)