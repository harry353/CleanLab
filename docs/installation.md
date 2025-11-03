# 🧰 Installation Guide

This guide explains how to install and set up **CleanLab**.

---

## 📦 Requirements

CleanLab requires **Python 3.9 or later** and the following dependencies:

- `numpy`
- `scipy`
- `matplotlib`
- `astropy`
- `tqdm`
- `tabulate`

---

## 🚀 Quick Installation

### 1. Clone the repository
`bash`
`git clone https://github.com/<your-username>/CleanLab.git`
`cd CleanLab`

### 2. Install dependencies
`pip install -r requirements.txt`

### 3. Verifying the Installation

To verify CleanLab is working, run:

`./CLEAN --help`

You should see the command-line help message printed in your terminal.

## Notes

- The repository includes example FITS files under images/ for quick testing.
- Output FITS files (`clean_image.fits`, `residual_image.fits`, etc.) are ignored by Git but generated automatically when you run the pipeline.
