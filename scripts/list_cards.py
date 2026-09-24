with open("src/dashboard/static/index.html", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, l in enumerate(lines, 1):
    if "<section" in l or ("<div" in l and ("id=" in l or "class=\"panel-card" in l or "class=\"precision" in l)):
        if "id=" in l or "class=" in l:
            # print up to 80 chars
            print(f"L{i}: {l.strip()[:80]}")
