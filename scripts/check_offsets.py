with open("src/dashboard/static/index.html", "r", encoding="utf-8") as f:
    html = f.read()

prec_start = html.find("<!-- Multi-Precision Engine Section -->")
chunk = html[prec_start:prec_start+2500]
print(chunk[chunk.find("<!--"):chunk.find("-->", chunk.find("<!--"))+3])
idx = chunk.find("tseSection")
print("tseSection offset in chunk:", idx)
print(chunk[idx-100:idx+100])
