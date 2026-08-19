import json
import os
import pandas as pd
import requests

url = os.environ.get("PEAKFLO_SOURCE_URL")
output = "data/data.json"

if not url:
    raise SystemExit(
        "Set PEAKFLO_SOURCE_URL to a current, private download URL before running."
    )

try:
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    raise SystemExit(f"Download failed: {e}")

text = response.text.strip()
if not text:
    raise SystemExit("Downloaded file is empty. The link may be expired.")

try:
    json.loads(text)
except json.JSONDecodeError as e:
    raise SystemExit(f"Downloaded content is not valid JSON: {e}")

with open(output, "w", encoding="utf-8") as f:
    f.write(text)

df = pd.read_json(output)
df.to_csv("data/dataset.csv", index=False)
print("Download and CSV conversion complete.")