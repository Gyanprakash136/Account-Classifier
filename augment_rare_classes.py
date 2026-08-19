import pandas as pd
import numpy as np
import json, re, time, requests

DATA_PATH       = "data/dataset.csv"
OUTPUT_PATH     = "data/augmented_dataset.csv"
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_MODEL    = "llama3.2"
TARGET_MIN      = 15
RARE_THRESHOLD  = 10


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df['itemDescription'] = df['itemDescription'].fillna(df['itemName'])
vc = df['accountName'].value_counts()
rare_classes = vc[vc < RARE_THRESHOLD].index.tolist()
total_needed = sum(max(0, TARGET_MIN - vc.get(c, 0)) for c in rare_classes)
print(f"Rare classes : {len(rare_classes)}")
print(f"Samples needed: {total_needed}")


def call_ollama(prompt: str) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model" : OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }, timeout=120)
    return resp.json().get("response", "")


def generate_samples(account_name: str, real_examples: list, n: int = 5) -> list:
    examples_str = "\n".join([
        f"  - \"{ex['itemName']}\" / \"{ex['itemDescription']}\""
        for ex in real_examples[:3]
    ])
    prompt = f"""You are a finance assistant. Generate exactly {n} realistic invoice line items for account: "{account_name}"

Real examples:
{examples_str}

Rules:
- Include date prefixes like "0125", "0226", "1224" etc.
- Vary vendors, descriptions, and time periods
- Keep it realistic (real business expenses)

Respond ONLY with a valid JSON array, no explanation:
[
  {{"itemName": "...", "itemDescription": "..."}},
  {{"itemName": "...", "itemDescription": "..."}}
]"""

    try:
        response = call_ollama(prompt)
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            items = json.loads(match.group())
            return [
                {"itemName": str(i.get("itemName","")).strip(),
                 "itemDescription": str(i.get("itemDescription","")).strip()}
                for i in items if i.get("itemName")
            ][:n]
    except Exception as e:
        print(f"    parse error: {e}")
    return []



synthetic_rows = []
failed = []

for idx, account_name in enumerate(rare_classes):
    current = vc.get(account_name, 0)
    needed  = max(0, TARGET_MIN - current)
    if needed == 0:
        continue

    real_examples = df[df['accountName'] == account_name][['itemName','itemDescription']].to_dict('records')
    print(f"\n[{idx+1}/{len(rare_classes)}] {account_name}")
    print(f"  {current} real → need {needed} synthetic")

    generated = []
    attempts  = 0
    while len(generated) < needed and attempts < 6:
        batch = generate_samples(account_name, real_examples, n=min(5, needed - len(generated)))
        generated.extend(batch)
        attempts += 1
        time.sleep(0.2)

    if not generated:
        print(f"Failed")
        failed.append(account_name)
        continue

    generated = generated[:needed]
    print(f"{len(generated)} samples generated")

    for item in generated:
        synthetic_rows.append({
            "_id"            : f"synthetic_{len(synthetic_rows)}",
            "vendorId"       : f"SYNTH_{re.sub(r'[^a-zA-Z0-9]','_',account_name[:12])}",
            "itemName"       : item["itemName"],
            "itemDescription": item["itemDescription"],
            "accountId"      : f"SYNTH_ACC",
            "accountName"    : account_name,
            "itemTotalAmount": round(np.random.uniform(100, 5000), 2),
            "is_synthetic"   : True
        })


print(f"\n{'='*50}")
print(f"DONE — {len(synthetic_rows)} synthetic rows generated")
df["is_synthetic"] = False
augmented_df = pd.concat([df, pd.DataFrame(synthetic_rows)], ignore_index=True)
augmented_df.to_csv(OUTPUT_PATH, index=False)
print(f"Saved → {OUTPUT_PATH}  ({len(augmented_df)} total rows)")
new_vc = augmented_df['accountName'].value_counts()
print(f"Classes still < {TARGET_MIN} samples: {(new_vc < TARGET_MIN).sum()}")
if failed:
    print(f"Failed classes: {failed}")
