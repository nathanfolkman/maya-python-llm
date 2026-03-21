import requests
from bs4 import BeautifulSoup
import json
import time

# Target: Maya 2026 Python API 2.0 Reference
BASE_URL = "https://help.autodesk.com/cloudhelp/2026/CHS/MAYA-API-REF/py_ref/"
CLASSES_INDEX = f"{BASE_URL}annotated.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def scrape_maya_api():
    try:
        response = requests.get(CLASSES_INDEX, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch index: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    class_links = soup.select('a.el')
    api_data = []

    print(f"Found {len(class_links)} classes. Starting crawl...")

    for link in class_links:
        class_name = link.text.strip()
        class_url = BASE_URL + link['href']
        
        try:
            class_res = requests.get(class_url, headers=HEADERS, timeout=15)
            class_res.raise_for_status()
            class_soup = BeautifulSoup(class_res.text, 'html.parser')
            
            # 1. Extract Description
            content_div = class_soup.find('div', class_='contents')
            description = "No description available."
            
            if content_div:
                p_tag = content_div.find('p')
                brief_div = class_soup.find('div', class_='briefit')
                
                if p_tag and p_tag.text.strip():
                    description = p_tag.text.strip()
                elif brief_div:
                    description = brief_div.text.strip()

            # 2. Extract Methods
            methods = []
            method_rows = class_soup.select('tr[class^="memitem"]')
            for row in method_rows:
                clean_method = " ".join(row.text.split())
                if clean_method:
                    methods.append(clean_method)
            
            api_data.append({
                "class": class_name,
                "description": description,
                "methods": methods,
                "source": class_url
            })
            print(f"Scraped: {class_name}")
            time.sleep(0.1) 

        except Exception as e:
            print(f"Skipping {class_name} due to error: {e}")

    # Save to JSON
    with open("maya_api_2_raw.json", "w", encoding="utf-8") as f:
        json.dump(api_data, f, indent=4)
    
    print(f"\nDone! Saved {len(api_data)} classes to maya_api_2_raw.json")

if __name__ == "__main__":
    scrape_maya_api()
