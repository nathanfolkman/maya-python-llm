import requests
from bs4 import BeautifulSoup
import json
import time
import re

# Target: Maya 2026 Python API 2.0 Reference
BASE_URL = "https://help.autodesk.com/cloudhelp/2026/CHS/MAYA-API-REF/py_ref/"
CLASSES_INDEX = f"{BASE_URL}annotated.html"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def infer_module(url: str) -> str:
    if "open_maya_anim" in url:
        return "OpenMayaAnim"
    if "open_maya_f_x" in url:
        return "OpenMayaFX"
    if "open_maya_render" in url:
        return "OpenMayaRender"
    if "open_maya_u_i" in url:
        return "OpenMayaUI"
    if "open_maya" in url:
        return "OpenMaya"
    return "Unknown"


def parse_description(soup: BeautifulSoup) -> str:
    """
    Extract the class description. Autodesk's docs put it as the first line
    of the pre.fragment inside div.textblock.
    """
    textblock = soup.find("div", class_="textblock")
    if textblock:
        pre = textblock.find("pre", class_="fragment")
        if pre:
            text = pre.get_text(strip=True)
            # Take first line only — subsequent lines may be constructor overloads
            first_line = text.split("\n")[0].strip()
            if first_line:
                return first_line
    return "No description available."


def get_memname_text(memitem: BeautifulSoup) -> str:
    """
    Extract the full qualified name from a div.memitem.
    Handles both plain table.memname and static methods wrapped in table.mlabels.
    """
    # Static methods use table.mlabels > table.memname
    mlabels = memitem.find("div", class_="memproto")
    if not mlabels:
        return ""

    td = mlabels.find("td", class_="memname")
    return td.get_text(strip=True) if td else ""


def is_static_method(memitem: BeautifulSoup) -> bool:
    """Check for 'static' span label in memproto."""
    memproto = memitem.find("div", class_="memproto")
    if not memproto:
        return False
    label = memproto.find("span", class_="mlabel")
    return label is not None and "static" in label.get_text(strip=True).lower()


def parse_memdoc_kv(memdoc: BeautifulSoup) -> dict:
    """
    Parse the unlabeled border=0 table inside memdoc.
    Each row has format: <td><b>Key:</b></td> <td>value</td>
    Special case: Parameters value may contain a nested Name/Type/Description table.
    Returns a dict with lowercase keys.
    """
    result: dict = {}
    table = memdoc.find("table", attrs={"border": "0"})
    if not table:
        return result

    for row in table.find_all("tr", recursive=False):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        b = cells[0].find("b")
        if not b:
            continue
        key = b.get_text(strip=True).rstrip(":").lower()
        val_cell = cells[1]

        if key == "parameters":
            nested = val_cell.find("table")
            if nested:
                params = []
                rows = nested.find_all("tr")
                for pr in rows[1:]:  # skip header row
                    pcells = pr.find_all("td")
                    if len(pcells) >= 3:
                        params.append({
                            "name": pcells[0].get_text(strip=True),
                            "type": pcells[1].get_text(separator=" ", strip=True),
                            "description": pcells[2].get_text(separator=" ", strip=True),
                        })
                    elif len(pcells) == 2:
                        params.append({
                            "name": pcells[0].get_text(strip=True),
                            "type": None,
                            "description": pcells[1].get_text(separator=" ", strip=True),
                        })
                result["parameters"] = params
            else:
                # Simple text form: "name - Type\nname2 - Type2"
                raw = val_cell.get_text(separator="\n", strip=True)
                result["parameters_raw"] = raw
        else:
            result[key] = val_cell.get_text(separator=" ", strip=True)

    return result


def classify_memname(full_name: str) -> tuple[str, str, str | None]:
    """
    Returns (item_type, method_name, constant_value).
    item_type: 'constant' | 'property' | 'method'
    """
    if "= property(" in full_name:
        base = full_name.split(" = ")[0].strip()
        method_name = base.split(".")[-1] if "." in base else base
        return "property", method_name, None

    if " = " in full_name and "def " not in full_name:
        base, _, val = full_name.partition(" = ")
        method_name = base.strip().split(".")[-1] if "." in base else base.strip()
        return "constant", method_name, val.strip()

    # Regular method — may have "def " prefix
    clean = full_name.removeprefix("def ").strip()
    method_name = clean.split(".")[-1] if "." in clean else clean
    return "method", method_name, None


def parse_memitem(memitem: BeautifulSoup) -> dict | None:
    """Parse one div.memitem into a structured dict."""
    full_name = get_memname_text(memitem)
    if not full_name:
        return None

    item_type, method_name, const_value = classify_memname(full_name)
    if not method_name:
        return None

    method_is_static = is_static_method(memitem)

    memdoc = memitem.find("div", class_="memdoc")
    if not memdoc:
        return None

    pre_frag = memdoc.find("pre", class_="fragment")
    pre_text = pre_frag.get_text(strip=True) if pre_frag else ""

    kv = parse_memdoc_kv(memdoc)

    base: dict = {
        "name": method_name,
        "type": item_type,
        "is_static": method_is_static,
    }

    if item_type == "constant":
        base["value"] = const_value
        base["value_type"] = kv.get("type")
        base["description"] = kv.get("description") or pre_text or None
        return base

    if item_type == "property":
        base["value_type"] = kv.get("type")
        base["access"] = kv.get("access")
        base["description"] = kv.get("description") or pre_text or None
        return base

    # ── method ──

    # Signature: prefer table, then pre_text first line if it looks like a method call
    # (must start with the method name to avoid using generic Python docstrings)
    signature = kv.get("signature") or kv.get("name")
    if not signature and pre_text:
        first_line = pre_text.split("\n")[0].strip()
        if "(" in first_line:
            name_part = first_line.split("(")[0].strip()
            # Accept if name_part is method_name or qualified (ends with .method_name)
            if name_part == method_name or name_part.endswith("." + method_name):
                signature = first_line
    base["signature"] = signature

    # Return type: from table "returns" key, then from "->" in signature/pre first line
    return_type = kv.get("returns")
    if not return_type and signature:
        m = re.search(r"->\s*(.+)$", signature)
        if m:
            return_type = m.group(1).strip()
            # Remove arrow from signature string
            base["signature"] = signature[: signature.rfind("->")].strip()
    if not return_type and pre_text:
        first_line = pre_text.split("\n")[0]
        m = re.search(r"->\s*(.+)$", first_line)
        if m:
            return_type = m.group(1).strip()
    base["return_type"] = return_type

    # Parameters
    params = kv.get("parameters")
    if params is None:
        # Fall back to raw text parsing: "name - Type" per line
        params = []
        raw = kv.get("parameters_raw", "")
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            if " - " in line:
                name, _, type_str = line.partition(" - ")
                params.append({"name": name.strip(), "type": type_str.strip(), "description": ""})
            else:
                params.append({"name": line, "type": None, "description": ""})
    base["parameters"] = params

    # Description: prefer table, then pre_text (skip first line if it IS the signature)
    description = kv.get("description")
    if not description and pre_text:
        lines = pre_text.split("\n")
        first = lines[0].strip()
        # If first line looks like a signature call for this method, skip it
        if "(" in first and first.split("(")[0].strip().split(".")[-1] == method_name:
            description = "\n".join(lines[1:]).strip() or None
        else:
            description = pre_text
    base["description"] = description

    return base


def scrape_class_page(class_name: str, class_url: str) -> dict | None:
    try:
        res = requests.get(class_url, headers=HEADERS, timeout=15)
        res.raise_for_status()
    except Exception as e:
        print(f"  [skip] {class_name}: {e}")
        return None

    soup = BeautifulSoup(res.text, "html.parser")
    description = parse_description(soup)

    methods = []
    for memitem_div in soup.find_all("div", class_="memitem"):
        entry = parse_memitem(memitem_div)
        if entry:
            methods.append(entry)

    return {
        "class": class_name,
        "module": infer_module(class_url),
        "description": description,
        "methods": methods,
        "source": class_url,
    }


def scrape_maya_api():
    try:
        response = requests.get(CLASSES_INDEX, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch index: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    class_links = soup.select("a.el")
    print(f"Found {len(class_links)} classes. Starting crawl...")

    api_data = []
    skipped = 0

    for i, link in enumerate(class_links, 1):
        class_name = link.text.strip()
        href = link.get("href", "")
        if not href:
            continue
        class_url = BASE_URL + href

        print(f"[{i}/{len(class_links)}] {class_name}...", end=" ", flush=True)
        result = scrape_class_page(class_name, class_url)

        if result:
            method_count = len(result["methods"])
            has_desc = result["description"] != "No description available."
            with_params = sum(1 for m in result["methods"] if m.get("parameters"))
            print(f"{method_count} entries, {with_params} w/params, desc={'yes' if has_desc else 'no'}")
            api_data.append(result)
        else:
            skipped += 1

        time.sleep(0.3)

    with open("maya_api_2_raw.json", "w", encoding="utf-8") as f:
        json.dump(api_data, f, indent=2, ensure_ascii=False)

    total_methods = sum(len(c["methods"]) for c in api_data)
    with_desc = sum(1 for c in api_data if c["description"] != "No description available.")
    total_with_params = sum(
        1 for c in api_data for m in c["methods"]
        if m.get("type") == "method" and m.get("parameters")
    )
    total_methods_only = sum(
        1 for c in api_data for m in c["methods"] if m.get("type") == "method"
    )

    print(f"\nDone!")
    print(f"  Classes scraped  : {len(api_data)} (skipped {skipped})")
    print(f"  Total entries    : {total_methods}")
    print(f"  Methods only     : {total_methods_only}")
    print(f"  Methods w/ params: {total_with_params}/{total_methods_only}")
    print(f"  With description : {with_desc}/{len(api_data)}")
    print(f"  Saved to maya_api_2_raw.json")


if __name__ == "__main__":
    scrape_maya_api()
