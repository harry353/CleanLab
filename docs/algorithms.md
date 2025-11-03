# ⚙️ CLEAN Algorithms in CleanLab

This document provides a detailed overview of the different CLEAN algorithm implementations available in **CleanLab**, explaining their purpose, design, and when to use each one.

---

## 🧩 Overview

The CLEAN algorithm is an iterative deconvolution technique used in radio interferometry to reconstruct the true sky brightness distribution from a dirty image. CleanLab provides multiple variants of CLEAN, each optimized for specific use cases and performance.

At the core, every CLEAN method in CleanLab follows the same fundamental steps:

1. Identify the strongest peak in the residual image.
2. Subtract a scaled version of the PSF (dirty beam) from the image at that location.
3. Add the corresponding flux component to the clean model image.
4. Repeat until the noise threshold or iteration limit is reached.

---

## 🔹 Clark CLEAN (Standard)

**Module:** `clean/clark.py`

The classic Clark CLEAN implementation, optimized for speed and stability. It operates in major and minor cycles:

* **Minor cycles:** Clean small residuals using a truncated PSF.
* **Major cycles:** Recalculate the full residual from the updated model.

**Use when:** You need a fast, stable CLEAN suitable for most images, or you want to test different cleaning parameters in an image.

**Advantages:**

* Efficient.
* Well-tested and reliable.

---

## 🔹 Sinc CLEAN

**Module:** `clean/sinc_clean.py`

The Sinc CLEAN uses a sinc kernel as a clean component to more accurately locate and subtract sources that fall between pixels.

**Use when:** High-precision imaging is required or when sources appear slightly smeared due to pixel alignment.

**Advantages:**

* Improved source localization accuracy.

**Limitations:**

* Slower due to convolution overhead.
* Sensitive to PSF truncation settings.

---

## 🔹 Cluster CLEAN

**Module:** `clean/cluster_clean.py`

This method groups nearby detected peaks into clusters and cleans them collectively. It improves stability in crowded fields and reduces false positives from noise fluctuations.

**Use when:** The field contains many closely spaced sources.

**Advantages:**

* Handles crowded source regions better compared to the previous two algorithms.

**Limitations:**

* Slightly more computational overhead due to clustering operations.

---

## 🔹 Multi-Peak CLEAN (Parallel)

**Module:** `clean/multi_peak_clean.py`

The Multi-Peak CLEAN detects several bright peaks per iteration and processes them in parallel using multiple threads. This should, in theory, reduce total runtime significantly.

**Use when:** Working with large images where many sources exist and computing power is available.

**Advantages:**

* Much faster convergence (in theory).
* Utilizes multi-core CPUs.

**Limitations:**

* Requires careful tuning of `multi_N` (number of peaks per iteration).
* Performance gains questionable.

---

## 🔹 Spectral CLEAN (Cube Deconvolution)

**Module:** `clean/core_spectral.py`

The Spectral CLEAN extends the deconvolution process into the third dimension (frequency or velocity) to clean full spectral cubes channel by channel.

**Use when:** Deconvolving spectral line cubes (e.g., CO, HCN, HCO⁺) where CLEAN must be applied per velocity channel.

**Advantages:**

* Automates multi-channel CLEANing.
* Saves output cubes directly to FITS.

**Limitations:**

* Computationally heavy for large cubes.

---

## 🔹 Gain Functions

**Module:** `clean_utils/gain_function.py`

The gain controls how much of the detected peak flux is subtracted each iteration. CleanLab supports:

* Constant gain (`gain=0.1`, standard CLEAN)
* Adaptive gain (variable per iteration)
* Custom functional gains (user-defined)

**Tip:** Lower gains yield smoother convergence; higher gains accelerate but risk oversubtraction.

---

## 🔹 Masking Strategies

**Modules:** `clean_utils/apply_mask.py`, `clean_utils/detect_peak.py`

Masking determines which regions of the image are eligible for CLEANing.

Available options:

* **`none`** – No restriction; CLEAN entire image.
* **`manual`** – Interactive mask selection via matplotlib.
* **`bgs`** – Background-subtracted automatic mask.

**Recommendation:** Always inspect automatic masks visually before running CLEAN, especially on faint or extended sources.

---

## 🔹 Exit Conditions

**Module:** `clean_utils/exit_conditions.py`

Exit conditions determine when CLEAN stops:

* When peak residual < noise threshold.
* When maximum iterations are reached.

The standard stopping condition ensures convergence without over-cleaning.

---

### 🔹 Peak Detection Algorithms  
**Module:** `clean_utils/detect_peak.py`

The peak detection module controls how CleanLab identifies the brightest points in the residual image that are potential source locations.
It provides several algorithms optimized for different use cases and signal conditions.

**Available methods:**
- **`regular`** — Locates the single global maximum in the residual map (default). Ideal for sparse fields and low-noise images.
- **`multi`** — Finds multiple bright peaks per iteration. Used by the Multi-Peak CLEAN for parallel subtraction.
- **`matched_filter`** — Applies PSF-shaped filtering before detection, improving sensitivity to faint sources. Best for low-S/N regions.

---

### 🧩 Updated Summary Table  

| Algorithm / Module | Best for | Key Advantage | Module Path |
|--------------------|-----------|----------------|--------------|
| **Clark** | General imaging | Stable and fast | `clean/clark.py` |
| **Sinc** | High-precision imaging | Sub-pixel accuracy | `clean/sinc_clean.py` |
| **Cluster** | Crowded fields | Reduces noise false positives | `clean/cluster_clean.py` |
| **Multi-Peak** | Large images, parallel runs | Fastest convergence | `clean/multi_peak_clean.py` |
| **Spectral** | 3D spectral cubes | Channel-by-channel cleaning | `clean/core_spectral.py` |
| **Gain Function** | Tuning convergence | Adaptive or fixed gain | `clean_utils/gain_function.py` |
| **Masking** | Region control | Manual or automatic masks | `clean_utils/apply_mask.py` |
| **Peak Detection** | Source identification | Multiple or matched detection modes | `clean_utils/detect_peak.py` |
| **Exit Conditions** | Stopping criteria | Automatic convergence detection | `clean_utils/exit_conditions.py` |


