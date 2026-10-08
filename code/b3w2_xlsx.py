"""Minimal dependency-free .xlsx reader (values only): read_xlsx(path) -> {sheet_name: [[cell, ...], ...]}."""
import zipfile, re
import xml.etree.ElementTree as ET
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def _col(ref):
    s = re.match(r'([A-Z]+)', ref).group(1); n = 0
    for ch in s: n = n * 26 + ord(ch) - 64
    return n - 1
def read_xlsx(path):
    z = zipfile.ZipFile(path)
    ss = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS):
            ss.append(''.join(t.text or '' for t in si.iter('{%s}t' % NS['m'])))
    wb = ET.fromstring(z.read('xl/workbook.xml'))
    rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    rmap = {r.get('Id'): r.get('Target') for r in rels}
    out = {}
    for sh in wb.find('m:sheets', NS):
        tgt = rmap[sh.get('{%s}id' % NS['r'])]; tgt = tgt.lstrip('/'); tgt = tgt if tgt.startswith('xl/') else 'xl/' + tgt
        rows = []
        for row in ET.fromstring(z.read(tgt)).iter('{%s}row' % NS['m']):
            cells = {}
            for c in row.findall('m:c', NS):
                t = c.get('t'); v = c.find('m:v', NS); val = None
                if t == 's' and v is not None: val = ss[int(v.text)]
                elif t == 'inlineStr': val = ''.join(x.text or '' for x in c.iter('{%s}t' % NS['m']))
                elif v is not None:
                    val = v.text
                    try: val = float(val); val = int(val) if val.is_integer() else val
                    except Exception: pass
                cells[_col(c.get('r'))] = val
            rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
        out[sh.get('name')] = rows
    return out
