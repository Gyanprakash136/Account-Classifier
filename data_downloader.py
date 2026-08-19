import json
import pandas as pd
import requests as re

url = "https://file.notion.com/f/f/a9150c0a-2325-4d8c-8637-a3bf9e71030e/0c36f913-f959-418f-9705-07ea3c6e1578/accounts-bills.json?table=block&id=30b582ce-c790-80ea-9144-f34cf0b1dd8f&spaceId=a9150c0a-2325-4d8c-8637-a3bf9e71030e&expirationTimestamp=1786932000000&signature=2_M8tzHvieoEQGixQaRTb1JH8US25tS3rNQKf9WQOVw&downloadName=accounts-bills.json"
output = "data/data.json"

try:
    response = re.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
except re.exceptions.RequestException as e:
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