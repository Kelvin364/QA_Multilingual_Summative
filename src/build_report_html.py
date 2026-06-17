"""
build_report_html.py — render reports/REPORT.md to a self-contained reports/REPORT.html with
all figures embedded as base64 (open in a browser, Print -> Save as PDF). Run:
  python src/build_report_html.py
"""
import markdown, base64, os, re
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
RP = os.path.join(ROOT, "reports")

md_text = open(os.path.join(RP, "REPORT.md")).read()
html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

def embed(m):
    path = os.path.join(RP, m.group(1))
    if os.path.exists(path):
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        return f'src="data:image/png;base64,{b64}"'
    return m.group(0)

html_body = re.sub(r'src="(figures/[^"]+)"', embed, html_body)
css = ("<style>body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:820px;"
       "margin:40px auto;line-height:1.5;color:#222;padding:0 20px}h1{border-bottom:3px solid "
       "#1f77b4;padding-bottom:6px}h2{border-bottom:1px solid #ddd;margin-top:28px;color:#1a3a5c}"
       "table{border-collapse:collapse;margin:10px 0;font-size:14px}th,td{border:1px solid #ccc;"
       "padding:5px 9px}th{background:#f0f4f8}code{background:#f5f5f5;padding:1px 4px;border-radius:3px}"
       "pre{background:#f5f5f5;padding:10px;overflow:auto}img{max-width:680px;display:block;"
       "margin:14px auto;border:1px solid #eee;border-radius:4px}</style>")
out = os.path.join(RP, "REPORT.html")
open(out, "w").write('<html><head><meta charset="utf-8">' + css + "</head><body>" + html_body + "</body></html>")
print(f"wrote {out} ({os.path.getsize(out)//1024} KB, {html_body.count('data:image/png;base64')} figures embedded)")
