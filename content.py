#!/usr/bin/env python3
"""
Automated Canvas Course Content Downloader

This script downloads all course materials from both current and past Canvas courses 
using the Canvas API. It supports all file types, including `.pdf`, `.R`, `.Rmd`, 
`.csv`, `.ipynb`, and others. Files embedded inside pages, modules, or viewer-style 
links are automatically identified and downloaded.

Setup and Usage:
1. Install Required Python Packages:
   - requests
   - beautifulsoup4
   - pdfkit

2. Install wkhtmltopdf:
   - This tool is required by pdfkit to convert HTML content (pages, assignments, 
     modules) into PDF files.
   - Download and install it from: https://wkhtmltopdf.org/downloads.html
   - Default path after installation will work.

3. Set Environment Variables:
   - Log into Canvas, go to Account > Settings > Approved Integrations, and 
     generate a new token.
   - Set CANVAS_API_TOKEN environment variable with your API token.
   - Set CANVAS_DOMAIN environment variable with your Canvas domain 
     (e.g., 'https://canvas.pitt.edu').
   - For GitHub Actions, add these as secrets in your repository settings.

4. Run the Script:
   - python canvas_course_downloader.py
   - It will retrieve both active and completed Canvas courses
   - Download all available files, linked content, and assignment submissions
   - Convert Canvas-hosted HTML content into PDFs (no `.html` files are saved)
   - Save everything to a structured local folder organized by course

Output:
All downloaded files are saved to a local directory named `canvas_all_content`, 
with one subfolder per course. Original filenames and extensions are preserved.

Folder Structure:
canvas_all_content/
├── Course Name A/
│   ├── lecture1.pdf
│   ├── page - Syllabus.pdf
│   ├── assignment - Essay.pdf
│   ├── module - Week 1 Overview.pdf
│   └── submission - final_essay.pdf
├── Course Name B/
│   └── ...

Notes:
- Most module and assignment PDFs may appear blank. This is expected behavior:
  - Modules are often used as containers for linked content rather than 
    standalone descriptions.
  - Assignment pages are also frequently blank unless the instructor specifically 
    writes assignment details in the Canvas page itself.
  - These files are still processed because they often contain embedded links 
    to downloadable materials.
"""

import os
import re
import platform
import shutil
import requests
import pdfkit
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# Configuration for wkhtmltopdf (platform-agnostic)
# Try to find wkhtmltopdf automatically
wkhtmltopdf_path = shutil.which('wkhtmltopdf')
if wkhtmltopdf_path:
    pdfkit_config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
else:
    # Fallback paths for different platforms
    if platform.system() == 'Windows':
        pdfkit_config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")
    elif platform.system() == 'Darwin':  # macOS
        # Common macOS installation paths
        possible_paths = [
            '/usr/local/bin/wkhtmltopdf',
            '/opt/homebrew/bin/wkhtmltopdf',
            '/usr/bin/wkhtmltopdf'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                pdfkit_config = pdfkit.configuration(wkhtmltopdf=path)
                break
        else:
            # If not found, let pdfkit try to find it automatically
            pdfkit_config = pdfkit.configuration()
    else:
        # Linux or other Unix-like systems
        pdfkit_config = pdfkit.configuration()


# Get Canvas API token and domain from environment variables
CANVAS_API_TOKEN = os.getenv('CANVAS_API_TOKEN')
CANVAS_DOMAIN = os.getenv('CANVAS_DOMAIN')

# Validate that required environment variables are set
if not CANVAS_API_TOKEN:
    raise ValueError(
        "CANVAS_API_TOKEN environment variable is not set. "
        "Please set it before running the script."
    )

if not CANVAS_DOMAIN:
    raise ValueError(
        "CANVAS_DOMAIN environment variable is not set. "
        "Please set it before running the script (e.g., 'https://canvas.pitt.edu')."
    )

BASE_API_URL = f'{CANVAS_DOMAIN}/api/v1'
DOWNLOADS_BASE = os.path.join(os.path.expanduser("~"), "Downloads", "canvas_all_content")
HEADERS = {'Authorization': f'Bearer {CANVAS_API_TOKEN}'}

downloaded_file_urls = set()


def make_safe(name):
    """Sanitize filename by removing invalid characters."""
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip()


def safe_paginate(url):
    """Safely paginate through API results."""
    results = []
    try:
        while url:
            r = requests.get(url, headers=HEADERS)
            if r.status_code in [403, 404]:
                print(f"    Skipping ({r.status_code} error): {url}")
                return []
            r.raise_for_status()
            results.extend(r.json())
            url = r.links.get('next', {}).get('url')
        return results
    except Exception as e:
        print(f"    Error during pagination: {e}")
        return []


def save_html_as_pdf(folder, name, html_content):
    """Convert HTML content to PDF and save it."""
    safe_name = make_safe(name)
    pdf_path = os.path.join(folder, f"{safe_name}.pdf")
    try:
        pdfkit.from_string(html_content, pdf_path, configuration=pdfkit_config)
        print(f"    Saved PDF: {safe_name}.pdf")
    except Exception as e:
        print(f"    Error converting {safe_name} to PDF: {e}")


def download_canvas_file_by_id(file_id, course_folder):
    """Download a Canvas file by its file ID."""
    try:
        meta = requests.get(f"{BASE_API_URL}/files/{file_id}", headers=HEADERS)
        meta.raise_for_status()
        file_data = meta.json()
        download_url = file_data['url']
        filename = make_safe(file_data['display_name'])

        if download_url in downloaded_file_urls:
            return

        r = requests.get(download_url, headers=HEADERS)
        r.raise_for_status()
        with open(os.path.join(course_folder, filename), 'wb') as f:
            f.write(r.content)

        downloaded_file_urls.add(download_url)
        print(f"    ✅ Downloaded file from API: {filename}")
    except Exception as e:
        print(f"    ❌ Error downloading file ID {file_id}: {e}")


def extract_and_download_linked_files(html, course_folder):
    """Extract file IDs from HTML and download the associated files."""
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup.find_all(['a', 'iframe'], href=True) + soup.find_all(['a', 'iframe'], src=True):
        href = tag.get('href') or tag.get('src')
        if href:
            match = re.search(r'/files/(\d+)', href)
            if match:
                file_id = match.group(1)
                download_canvas_file_by_id(file_id, course_folder)

    for script in soup.find_all('script'):
        if script.string:
            matches = re.findall(r'/files/(\d+)', script.string)
            for file_id in set(matches):
                download_canvas_file_by_id(file_id, course_folder)


def main():
    """Main workflow to download all course content."""
    print("Fetching your Canvas courses...")

    current_courses = safe_paginate(f"{BASE_API_URL}/courses?per_page=100&enrollment_state=active")
    completed_courses = safe_paginate(f"{BASE_API_URL}/courses?per_page=100&enrollment_state=completed")

    courses = current_courses + completed_courses

    for course in courses:
        course_id = course['id']
        course_name = make_safe(course.get('name') or f"course_{course_id}")
        print(f"\nCourse: {course_name}")
        course_folder = os.path.join(DOWNLOADS_BASE, course_name)
        os.makedirs(course_folder, exist_ok=True)

        print("  Downloading files...")
        for file in safe_paginate(f"{BASE_API_URL}/courses/{course_id}/files?per_page=100"):
            try:
                file_url = file['url']
                if file_url in downloaded_file_urls:
                    continue
                r = requests.get(file_url, headers=HEADERS)
                r.raise_for_status()
                file_path = os.path.join(course_folder, make_safe(file['filename']))
                with open(file_path, 'wb') as f:
                    f.write(r.content)
                downloaded_file_urls.add(file_url)
                print(f"    ✅ Downloaded file: {make_safe(file['filename'])}")
            except Exception as e:
                print(f"    Error downloading {file.get('filename', 'unknown')}: {e}")

        print("  Downloading pages...")
        for page in safe_paginate(f"{BASE_API_URL}/courses/{course_id}/pages?per_page=100"):
            try:
                detail = requests.get(f"{BASE_API_URL}/courses/{course_id}/pages/{page['url']}", headers=HEADERS)
                if detail.status_code in [403, 404]:
                    continue
                detail.raise_for_status()
                body = detail.json().get('body', '')
                name = f"page - {page['title']}"
                extract_and_download_linked_files(body, course_folder)
                save_html_as_pdf(course_folder, name, body)
            except Exception as e:
                print(f"    Error handling page {page['title']}: {e}")

        print("  Downloading assignments...")
        for assignment in safe_paginate(f"{BASE_API_URL}/courses/{course_id}/assignments?per_page=100"):
            try:
                description_html = assignment.get('description', '')
                name = f"assignment - {assignment['name']}"
                extract_and_download_linked_files(description_html, course_folder)
                html = f"<h1>{assignment['name']}</h1><p>{description_html}</p>"
                save_html_as_pdf(course_folder, name, html)
            except Exception as e:
                print(f"    Error handling assignment {assignment['name']}: {e}")

        print("  Downloading modules...")
        for module in safe_paginate(f"{BASE_API_URL}/courses/{course_id}/modules?per_page=100"):
            try:
                content = f"<h1>{module['name']}</h1><ul>"
                items = safe_paginate(f"{BASE_API_URL}/courses/{course_id}/modules/{module['id']}/items?per_page=100")
                for item in items:
                    content += f"<li>{item['title']} ({item['type']})</li>"

                    if item['type'] == 'File' and 'content_id' in item:
                        download_canvas_file_by_id(item['content_id'], course_folder)

                    elif 'html_url' in item:
                        html_url = item['html_url']
                        item_resp = requests.get(html_url, headers=HEADERS)
                        if item_resp.ok:
                            extract_and_download_linked_files(item_resp.text, course_folder)

                    elif item.get('type') == 'Page' and 'page_url' in item:
                        page_api_url = f"{BASE_API_URL}/courses/{course_id}/pages/{item['page_url']}"
                        page_resp = requests.get(page_api_url, headers=HEADERS)
                        if page_resp.ok:
                            body = page_resp.json().get('body', '')
                            extract_and_download_linked_files(body, course_folder)

                content += "</ul>"
                name = f"module - {module['name']}"
                save_html_as_pdf(course_folder, name, content)
            except Exception as e:
                print(f"    Error saving module {module['name']}: {e}")

        print("  Downloading your submissions...")
        submissions = safe_paginate(f"{BASE_API_URL}/courses/{course_id}/students/submissions?per_page=100")
        for sub in submissions:
            for attachment in sub.get("attachments", []):
                try:
                    file_url = attachment['url']
                    if file_url in downloaded_file_urls:
                        continue
                    filename = make_safe(f"submission - {attachment['filename']}")
                    r = requests.get(file_url, headers=HEADERS)
                    r.raise_for_status()
                    with open(os.path.join(course_folder, filename), 'wb') as f:
                        f.write(r.content)
                    downloaded_file_urls.add(file_url)
                    print(f"    ✅ Downloaded submission: {filename}")
                except Exception as e:
                    print(f"    Error downloading submission file: {e}")

    print("\n✅ All course content downloaded to your Downloads/canvas_all_content folder.")


if __name__ == "__main__":
    main()
