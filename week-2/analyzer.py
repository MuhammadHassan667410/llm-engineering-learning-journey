import json
import os
import mimetypes
from typing import Dict, Any

def fetch_and_analyze_file(file_path: str) -> Dict[str, Any]:
    """
    Universal file ingestion utility for LLM-based analysis.

    Parameters:
        file_path (str): Path to the file provided by user.

    Returns:
        dict with:
            - filename
            - extension
            - mime_type
            - size_bytes
            - extracted_content
            - structure (if applicable)
            - notes
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    result = {
        "filename": os.path.basename(file_path),
        "extension": os.path.splitext(file_path)[1].lower(),
        "mime_type": mimetypes.guess_type(file_path)[0],
        "size_bytes": os.path.getsize(file_path),
        "extracted_content": None,
        "structure": None,
        "notes": ""
    }

    ext = result["extension"]

    try:
        # -----------------------
        # TEXT-BASED FILES
        # -----------------------
        if ext in [".py", ".txt", ".md", ".yaml", ".yml", ".json", ".csv"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            result["extracted_content"] = content

            if ext == ".json":
                result["structure"] = json.loads(content)

        # -----------------------
        # JUPYTER NOTEBOOK
        # -----------------------
        elif ext == ".ipynb":
            with open(file_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            cells_summary = []

            for i, cell in enumerate(nb.get("cells", [])):
                cells_summary.append({
                    "cell_index": i,
                    "cell_type": cell.get("cell_type"),
                    "source": "".join(cell.get("source", []))
                })

            result["structure"] = {
                "nbformat": nb.get("nbformat"),
                "total_cells": len(cells_summary),
                "cells": cells_summary
            }

            result["extracted_content"] = "\n\n".join(
                c["source"] for c in cells_summary
            )

        # -----------------------
        # PDF FILES
        # -----------------------
        elif ext == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(file_path)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text())
                result["extracted_content"] = "\n".join(text)
            except Exception as e:
                result["notes"] = f"PDF parsing failed: {str(e)}"

        # -----------------------
        # DOCX FILES
        # -----------------------
        elif ext == ".docx":
            try:
                from docx import Document
                doc = Document(file_path)
                result["extracted_content"] = "\n".join(
                    p.text for p in doc.paragraphs
                )
            except Exception as e:
                result["notes"] = f"DOCX parsing failed: {str(e)}"

        # -----------------------
        # UNKNOWN / BINARY FILES
        # -----------------------
        else:
            with open(file_path, "rb") as f:
                raw = f.read(1024)

            result["notes"] = (
                "Binary or unsupported file type. "
                "Only first 1KB read for inspection."
            )
            result["extracted_content"] = raw.hex()

    except Exception as e:
        result["notes"] = f"Error during processing: {str(e)}"

    return result
