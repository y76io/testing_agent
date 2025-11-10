import csv, os, json
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from .config import REPORTS_DIR

def export_csv(run_id: str, metric_results: dict, overall: float):
    out_dir = os.path.join(REPORTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "report.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric_code","value","pass"])
        for k, v in metric_results.items():
            if "mean" in v:
                val = v["mean"]
            elif "success_rate" in v:
                val = v["success_rate"]
            elif "p95" in v:
                val = v["p95"]
            else:
                val = json.dumps(v)
            w.writerow([k, val, v.get("pass", False)])
        w.writerow(["overall_score", overall, ""])
    return csv_path

def export_pdf(run_id: str, metric_results: dict, overall: float):
    out_dir = os.path.join(REPORTS_DIR, run_id)
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Testing Agent Report — Run {run_id}")
    y -= 30
    c.setFont("Helvetica", 11)
    for k, v in metric_results.items():
        line = f"{k}: " + (f"mean={v['mean']}" if "mean" in v else f"p95={v.get('p95')}" if "p95" in v else f"success_rate={v.get('success_rate')}")
        line += f" | pass={v.get('pass', False)}"
        c.drawString(50, y, line)
        y -= 18
        if y < 60:
            c.showPage(); y = height - 50
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, f"Overall score: {overall:.3f}")
    c.save()
    return pdf_path