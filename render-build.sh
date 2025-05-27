#!/usr/bin/env bash

set -e

# Update package lists and install Tesseract OCR engine
apt-get update
apt-get install -y tesseract-ocr
