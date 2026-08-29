import zipfile, re, sys, os

def text(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf8", "ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    xml = re.sub(r"<w:br[^>]*/>", "\n", xml)
    xml = re.sub(r"<[^>]+>", "", xml)
    xml = xml.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return "\n".join(l.rstrip() for l in xml.split("\n"))

for p in sys.argv[1:]:
    t = text(p)
    paras = [l for l in t.split("\n") if l.strip()]
    print(f"\n{'='*78}\n{os.path.basename(p)}  |  {len(paras)} paragraphs, {len(t.split())} words\n{'='*78}")
    with open(os.path.basename(p) + ".txt", "w", encoding="utf8") as fh:
        fh.write(t)
