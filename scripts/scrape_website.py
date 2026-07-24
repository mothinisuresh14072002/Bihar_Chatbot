import sys
import os
import json
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BOCW_BASE_URL, SCRAPE_PAGES, RAW_DATA_DIR, DATA_DIR
from playwright.async_api import async_playwright

async def scrape_page(page, url):
    print(f"Scraping {url}...")
    try:
        await page.goto(url, wait_until="networkidle")
        # Wait a bit more for Angular to settle
        await page.wait_for_timeout(3000)
        
        # Extract text content
        text_content = await page.evaluate('document.body.innerText')
        return text_content
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

async def main():
    metadata = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        for page_info in SCRAPE_PAGES:
            url = BOCW_BASE_URL + page_info["path"]
            text = await scrape_page(page, url)
            
            if text:
                file_path = RAW_DATA_DIR / f"{page_info['name']}.txt"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                metadata.append({
                    "url": url,
                    "name": page_info["name"],
                    "label": page_info["label"],
                    "file_path": str(file_path),
                    "status": "success"
                })
            else:
                metadata.append({
                    "url": url,
                    "name": page_info["name"],
                    "label": page_info["label"],
                    "status": "failed"
                })
                
        await browser.close()
        
    # Save metadata
    metadata_path = DATA_DIR / "scraped_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    # Print summary
    success_count = sum(1 for m in metadata if m["status"] == "success")
    print(f"\nScraping complete!")
    print(f"Successfully scraped: {success_count}/{len(SCRAPE_PAGES)} pages.")
    print(f"Data saved to: {RAW_DATA_DIR}")
    print(f"Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    asyncio.run(main())
