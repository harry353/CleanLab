# 🚀 Usage Guide

This document shows how to use **CleanLab** both from the **command line (CLI)** and as a **Python module**, covering 2D and spectral (cube) CLEAN modes.

---

## ⚙️ Command-Line Interface (CLI)

After installation, the simplest way to run CleanLab is via the `CLEAN` executable script.

### 🧩 Example 1: Standard 2D Clark CLEAN

```bash
./CLEAN --dirty_image images/dense.fits \
        --psf wsclean-psf.fits \
        --threshold 2 \
        --max_iter 500 \
        --mode clark \
        --show_plots
```

**What happens:**

* Loads your dirty image and PSF (the above example uses the pre-existing images in the `images` directory).
* Runs the classic Clark CLEAN algorithm.
* Prints iteration statistics.
* Optionally displays plots of the clean and residual images.

---

### 🧩 Example 2: Sinc CLEAN

```bash
./CLEAN --dirty_image images/dense.fits \
        --psf wsclean-psf.fits \
        --threshold 2 \
        --max_iter 800 \
        --mode sinc \
        --gain 0.25 \
        --show_plots
```

**Description:**
The Sinc CLEAN uses a sinc kernel as clean components to better estimate sub-pixel source positions. This allows improved accuracy for images where point sources do not align exactly with pixel centers, though at a cost in performance.

---

### 🧩 Example 3: Cluster CLEAN

```bash
./CLEAN --dirty_image images/dense.fits \
        --psf wsclean-psf.fits \
        --threshold 2 \
        --max_iter 1000 \
        --mode cluster \
        --gain 0.2 \
        --mask none
```

**Description:**
The Cluster CLEAN is the Sinc CLEAN, with the added functionality of grouping closely located point sources and cleaning them together, improving the positional accuracy further.

---

### 🧩 Example 4: Multi-Peak CLEAN (Parallel)

```bash
./CLEAN --dirty_image images/dense.fits \
        --psf wsclean-psf.fits \
        --threshold 2 \
        --max_iter 500 \
        --mode multi \
        --peak_detection multi \
        --show_plots
```

**Description:**
Finds and subtracts multiple bright peaks per iteration using multithreading, making CLEAN much faster (in theory).

---

### 🧩 Example 5: Spectral Cube CLEAN

```bash
./CLEAN --dirty_image images/cube.fits \
        --psf wsclean-psf.fits \
        --threshold 1 \
        --max_iter 1000 \
        --mode spectral
```

**Description:**
Runs CLEAN along the spectral axis for each channel in a 3D FITS cube, producing `images/clean_cube.fits` and `images/residual_cube.fits`.

---

## 🧠 Flags

| Flag               | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `--dirty_image`    | Path to the dirty image or cube FITS file                         |
| `--psf`            | Path to the PSF (dirty beam) FITS file                            |
| `--threshold`      | Stopping noise threshold in σ                                     |
| `--max_iter`       | Maximum number of iterations                                      |
| `--mode`           | CLEAN algorithm (`clark`, `sinc`, `cluster`, `multi`, `spectral`) |
| `--mask`           | Masking mode (`none`, `manual`, `bgs`)                            |
| `--iter_per_cycle` | Number of iterations per major cycle (default: 100)               |
| `--show_plots`     | Display images after cleaning                                     |
| `--print_results`  | Print summary statistics                                          |
| `--gain`           | Loop gain (or function) controlling subtraction strength          |
| `--peak_detection` | Peak-finding method (`regular`, `matched_filter`, `multi`)        |
| `--debug_results`  | Enable verbose debugging output                                   |

---

## 💾 Output Files

Depending on the mode, CleanLab will generate:

| Output                       | Description              |
| ---------------------------- | ------------------------ |
| `images/clean_image.fits`    | Cleaned 2D image         |
| `images/residual_image.fits` | Residual (uncleaned) map |
| `images/clean_cube.fits`     | Cleaned 3D spectral cube |
| `images/residual_cube.fits`  | Residual 3D cube         |

---

## 🧩 Tips

* Use `--mask manual` to interactively select circular regions to CLEAN, or `--mask bgs` to set a rectangular region as background noise.
* Use a smaller `--gain` (e.g. `0.1`) for more stable convergence.
* The spectral mode can take time, start with a small cube for testing.
* You can monitor iteration progress with the live progress bar (powered by `tqdm`).
