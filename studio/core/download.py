"""
Model download utilities for Civitai and HuggingFace.
"""
from __future__ import annotations

from pathlib import Path

import requests
from tqdm import tqdm

from ..schema.errors import DownloadError
from .logging_utils import stealth_print

__all__ = [
    "download_file_civitai",
    "download_file_huggingface",
]


def download_file_civitai(url: str, destination: Path) -> None:
    """
    Download model from Civitai with progress bar.
    """
    try:
        stealth_print(f"Downloading from Civitai: {url}", "progress")
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        stealth_print(f"Downloaded to {destination}", "success")
        
    except Exception as e:
        raise DownloadError(f"Civitai download failed: {e}") from e


def download_file_huggingface(url: str, destination: Path) -> None:
    """
    Download model from HuggingFace with progress bar.
    """
    try:
        stealth_print(f"Downloading from HuggingFace: {url}", "progress")
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        with open(destination, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        stealth_print(f"Downloaded to {destination}", "success")
        
    except Exception as e:
        raise DownloadError(f"HuggingFace download failed: {e}") from e
