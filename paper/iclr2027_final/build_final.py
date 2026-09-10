"""Build and inspect the final paper; software/PDF checks are not acceptance claims."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    compiler = args.compiler.resolve()
    command = [str(compiler), "--keep-logs", "--keep-intermediates", "cerbench_final.tex"]
    with (root / "build_stdout.log").open("w") as out, (root / "build_stderr.log").open("w") as err:
        result = subprocess.run(command, cwd=root, stdout=out, stderr=err, check=False)
    if result.returncode:
        raise RuntimeError("Compiler failed; inspect retained build logs")
    import fitz
    from PIL import Image, ImageDraw
    pdf = root / "cerbench_final.pdf"
    doc = fitz.open(pdf)
    text = "\n".join(page.get_text() for page in doc)
    aux = (root / "cerbench_final.aux").read_text()
    marker = re.search(r"\\newlabel\{page:mainend\}\{\{[^}]*\}\{(\d+)\}", aux)
    main_end = int(marker.group(1)) if marker else None
    latexlog = (root / "cerbench_final.log").read_text(errors="replace")
    bad = re.findall(r"[^\n]*(?:undefined citations|undefined references|Citation .* undefined|Reference .* undefined|multiply defined|Overfull \\[hv]box|^!)[^\n]*", latexlog, re.M)
    fonts, unembedded = {}, []
    for page in doc:
        for font in page.get_fonts(full=True):
            xref = font[0]
            if xref in fonts:
                continue
            extracted = doc.extract_font(xref)
            embedded = bool(extracted[3]) or font[2] == "Type3"
            fonts[xref] = {"basefont": font[3], "type": font[2], "embedded": embedded}
            if not embedded:
                unembedded.append(font[3])
    forbidden = [v for v in ["/Users/", "aayambansal", "SHARED_OPENAI", "sk-proj-", "sk-ant-", "prj_2db"] if v.lower() in text.lower()]
    pages = [{"page": i + 1, "width": float(p.rect.width), "height": float(p.rect.height), "characters": len(p.get_text())} for i, p in enumerate(doc)]
    render = root / "pdf_review"
    render.mkdir(exist_ok=True)
    thumbs = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=fitz.Matrix(0.6, 0.6), alpha=False)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        canvas = Image.new("RGB", (image.width, image.height + 24), "#dddddd")
        canvas.paste(image, (0, 24))
        ImageDraw.Draw(canvas).text((8, 5), f"Page {i + 1}", fill="black")
        thumbs.append(canvas)
        if i == 0 or "Controlled nested disclosure of the existing" in page.get_text():
            page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(str(render / f"page_{i + 1}.png"))
    columns = 3
    width, height = max(t.width for t in thumbs), max(t.height for t in thumbs)
    sheet = Image.new("RGB", (columns * width, ((len(thumbs) + columns - 1) // columns) * height), "#bbbbbb")
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % columns) * width, (i // columns) * height))
    sheet.save(render / "contact_sheet.png")
    checks = {"build_returncode": result.returncode, "main_text_end_page": main_end,
              "within_nine_page_main_limit": main_end is not None and main_end <= 9,
              "total_pages": len(doc), "pages": pages, "metadata": doc.metadata,
              "all_fonts_embedded": not unembedded, "fonts": fonts,
              "latex_issues": bad, "private_text_matches": forbidden,
              "encrypted": bool(doc.is_encrypted), "pdf_sha256": sha(pdf),
              "tex_sha256": sha(root / "cerbench_final.tex"), "compiler_sha256": sha(compiler),
              "note": "Formatting checks do not establish human relevance validation, model experiments or acceptance."}
    with (root / "pdf_checks.json").open("w") as handle:
        json.dump(checks, handle, indent=2)
        handle.write("\n")
    (root / "pdf_extracted_text.txt").write_text(text)
    print(json.dumps({k: checks[k] for k in ["main_text_end_page", "total_pages", "within_nine_page_main_limit", "all_fonts_embedded", "latex_issues", "private_text_matches", "pdf_sha256"]}, indent=2))
    if not checks["within_nine_page_main_limit"] or unembedded or forbidden or bad:
        raise RuntimeError("PDF requires revision; inspect pdf_checks.json and page renders")


if __name__ == "__main__":
    main()
