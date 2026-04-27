import re, json, urllib.request, time
API = "http://127.0.0.1:8000"

with open(r"C:\龙虾\tetramem_data_v2\tetramem_export.md", 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

pattern = r'## \[([a-f0-9]+)\] \(w=([\d.]+) a=([\d.]+)\) \[([^\]]+)\]\n(.+?)(?=\n## \[|$)'
matches = re.findall(pattern, content, re.DOTALL)

memories = []
for mid, weight, activation, labels_str, text in matches:
    labels = [l.strip() for l in labels_str.split(',')]
    text = text.strip()
    if text:
        memories.append({"content": text, "labels": labels, "weight": max(0.1, float(weight))})

print(f"Total: {len(memories)}", flush=True)

imported = 0
errors = 0
i = 0

while i < len(memories):
    m = memories[i]
    try:
        payload = json.dumps({"content": m["content"], "labels": m["labels"], "weight": m["weight"]}).encode("utf-8")
        req = urllib.request.Request(f"{API}/api/v1/store", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            json.loads(resp.read().decode("utf-8"))
        imported += 1
        if (i+1) % 50 == 0:
            print(f"+50 (total:{imported})", flush=True)
        i += 1
        time.sleep(0.08)
    except Exception as e:
        print(f"Error at {i}: {e}", flush=True)
        time.sleep(3)
        errors += 1
        if errors > 20:
            print("Too many errors, stopping", flush=True)
            break

print(f"Done: {imported}/{len(memories)}", flush=True)
