<p align="center">
  <img src="docs/CleanLab_logo.png" alt="CleanLab Logo" width="400"/>
</p>


# CleanLab
*A modular laboratory for radio interferometric deconvolution.*

CleanLab is a flexible Python code implementing multiple variants of the CLEAN algorithm, from classic Clark CLEAN to spectral and parallel multi-peak approaches.
It allows experimenting with peak detection, gain functions, and residual update strategies for both 2D images and 3D spectral cubes.

## Example Usage

After installing the required dependencies:

```bash
pip install numpy scipy matplotlib astropy tqdm tabulate

python run_clean.py \
  --dirty_image images/wsclean-dirty.fits \
  --psf wsclean-psf.fits \
  --threshold 2.0 \
  --max_iter 1000 \
  --mode clark \
  --mask none \
  --iter_per_cycle 100 \
  --show_plots True \
  --print_results True \
  --gain 0.2 \
  --peak_detection regular \
  --debug_results False

---

### 📚 Documentation

For full details, visit the [docs](docs/) directory:

- [Overview](docs/overview.md)
- [Installation Guide](docs/installation.md)
- [Usage Examples](docs/usage.md)
- [Algorithms](docs/algorithms.md)
- [API Reference](docs/api_reference.md)

