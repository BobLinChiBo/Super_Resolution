# Document Image Super Resolution

A powerful image enhancement tool for improving the quality of scanned documents, old texts, and photographs using state-of-the-art super-resolution AI models. Particularly effective for historical documents, manuscripts, and degraded images.

## Features

- **Multiple AI Models**: Support for Real-ESRGAN, SwinIR, and BSRGAN models
- **Document Enhancement**: Optimized preprocessing and postprocessing for text clarity
- **Batch Processing**: Process entire folders of images with progress tracking
- **Memory Management**: Automatic image tiling for large documents to prevent memory issues
- **Background Cleaning**: Remove yellowing, stains, and artifacts from old documents
- **Two-Stage Processing**: Advanced upscaling with better quality preservation
- **Content Detection**: Automatically detect content type and select optimal model
- **Resume Support**: Continue processing from where you left off

## Quick Start

### Prerequisites

- Python 3.8 - 3.9 (PyTorch compatibility)
- CUDA-capable GPU (recommended) or CPU
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA 11.7 (for GPU acceleration)

### Installation with uv

1. Install uv if you haven't already:
```bash
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/Super_Resolution.git
cd Super_Resolution
```

3. Configure uv for Windows (optional, prevents hardlink warnings):
```bash
# On Windows PowerShell
$env:UV_LINK_MODE="copy"

# Or set it permanently in your system environment variables
# This prevents the "Failed to hardlink files" warning on Windows
```

4. Create virtual environment and install dependencies:
```bash
# Create and activate virtual environment
uv venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# Install the project and dependencies
uv pip install -e . --link-mode=copy

# For GPU support (NVIDIA CUDA 11.7):
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117 --link-mode=copy

# For CPU-only (no GPU required):
# uv pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --index-url https://download.pytorch.org/whl/cpu --link-mode=copy
```

5. Download model weights (automatic on first run):
The models will be automatically downloaded when you first run the enhancement script.

## Usage

### Basic Usage

Process all images in the input folder:
```bash
# Using GPU (default)
uv run python src/enhance.py --input-dir data/input --output-dir data/output

# Using CPU only (no GPU required)
uv run python src/enhance.py --input-dir data/input --output-dir data/output --device cpu

# Or if you've activated the virtual environment:
python src/enhance.py --input-dir data/input --output-dir data/output --device cpu
```

### Common Use Cases

#### Old Document Restoration
```bash
# Test without preprocessing/postprocessing first
uv run python src/enhance.py --scale 2 --clean-background

# Then try with preprocessing/postprocessing to compare results
uv run python src/enhance.py --scale 2 --preprocess --postprocess --clean-background --enhance-text
```

#### High-Quality Book Scanning
```bash
uv run python src/enhance.py --scale 4 --two-stage --target-dpi 600 --quality 95
```

#### Photograph Enhancement
```bash
uv run python src/enhance.py --model-variant RealESRGAN_x4plus --scale 4
```

#### Handwritten Text / Calligraphy
```bash
uv run python src/enhance.py --model-variant realesrgan-animevideo-xsx2 --enhance-text
```

#### Large Format Documents
```bash
uv run python src/enhance.py --tile-size 1024 --max-memory-gb 8
```

### Command Line Arguments

#### Input/Output
- `--input-dir`: Input directory containing images (default: `data/input`)
- `--output-dir`: Output directory for enhanced images (default: `data/output`)

#### Model Selection
- `--model`: Model type to use (`realesrgan`, `swinir`, `bsrgan`)
- `--model-variant`: Specific model variant
  - `realesr-general-x4v3`: Best for documents and text (default)
  - `realesr-general-wdn-x4v3`: With denoising capability
  - `RealESRGAN_x4plus`: General purpose, natural images
  - `RealESRGAN_x2plus`: 2x upscaling only
  - `realesrgan-animevideo-xsx2`: Good for line art and strokes
  - `RealESRGAN_x4plus_anime_6B`: Illustration and artwork

#### Processing Options
- `--scale`: Upscaling factor (2 or 4, default: 2)
- `--preprocess`: Apply preprocessing for old/degraded documents (Note: may not always improve results - test first)
- `--postprocess`: Apply postprocessing for better clarity (Note: may not always improve results - test first)
- `--enhance-text`: Apply text-specific sharpening
- `--clean-background`: Remove background stains and discoloration
- `--background-threshold`: Threshold for background cleaning (0-255, default: 240)
- `--two-stage`: Use two-stage processing for better quality
- `--target-dpi`: Target DPI for output (e.g., 300, 600)
- `--auto-detect`: Auto-detect content type and select best model

#### Performance
- `--device`: Device to use (`cuda` for GPU or `cpu` for CPU-only processing, default: `cuda`)
  - Use `cuda` for faster processing with NVIDIA GPU
  - Use `cpu` if no GPU is available (slower but works everywhere)
- `--batch-size`: Number of images to process in parallel
- `--tile-size`: Tile size for large images (default: 1024, reduce for less memory usage)
- `--max-memory-gb`: Maximum GPU memory to use (default: 6.0)

#### Output
- `--quality`: JPEG quality (1-100, default: 95)
- `--resume`: Resume from previous run

## Project Structure

```
Super_Resolution/
├── data/
│   ├── input/           # Place your source images here
│   └── output/          # Enhanced images will be saved here
├── models/              # Downloaded model weights (auto-created)
├── src/
│   ├── enhance.py       # Main enhancement script
│   ├── models.py        # Model implementations
│   └── utils.py         # Utility functions
├── pyproject.toml       # Project configuration
├── uv.lock             # Dependency lock file
└── README.md           # This file
```

## Development

### Setting up development environment

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint code
flake8 src/
```

### Adding new dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Update all dependencies
uv sync
```

## Tips for Best Results

### Document Types

1. **Printed Text Documents**
   - Use `--scale 2` for most cases (readable text without excessive file size)
   - Test with and without `--preprocess --postprocess` to see which gives better results
   - Add `--enhance-text` for additional text sharpening if needed
   - For yellowed/aged paper: `--clean-background --background-threshold 230`

2. **Handwritten Documents**
   - Use `--model-variant realesrgan-animevideo-xsx2` (better for strokes)
   - Keep `--scale 2` to preserve natural writing characteristics
   - Avoid aggressive postprocessing

3. **Mixed Content (Text + Images)**
   - Use `--auto-detect` to automatically switch models
   - Or use `--model-variant RealESRGAN_x4plus` for balanced results

4. **Historical Photographs**
   - Use `--model-variant RealESRGAN_x4plus --scale 4`
   - Skip text-specific options (`--enhance-text`)

5. **Technical Drawings / Diagrams**
   - Use `--model-variant realesr-general-x4v3`
   - Test `--preprocess --postprocess` to see if it improves line clarity

### Quality vs Speed Trade-offs

- **Maximum Quality (GPU)**: `--two-stage --scale 4 --quality 95`
- **Balanced (GPU)**: `--scale 2` (test with/without `--preprocess --postprocess`)
- **Fast Processing (GPU)**: `--scale 2` with no additional options
- **CPU Only Processing**: `--device cpu --tile-size 512`
  - Note: CPU processing is 5-10x slower than GPU
  - Reduce tile-size to 256 if running out of RAM

## System Requirements

### Minimum
- 8GB RAM
- 4GB VRAM (for GPU processing)
- 10GB free disk space
- Python 3.8+

### Recommended
- 16GB RAM
- 8GB+ VRAM (NVIDIA GPU)
- 20GB free disk space
- NVIDIA GPU with CUDA 11.0+

## Troubleshooting

### Package Compatibility Issues

If you encounter `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`:
```bash
# This error occurs with newer PyTorch versions. The project requires specific versions:
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
uv pip install "numpy>=1.24.0,<2.0"
```

### CUDA not available
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If false, either:
# 1. Install CUDA toolkit matching your PyTorch version
# 2. Use CPU processing: --device cpu
```

### Out of memory errors
```bash
# Reduce tile size
python src/enhance.py --tile-size 512

# Limit GPU memory usage
python src/enhance.py --max-memory-gb 4

# Process one image at a time
python src/enhance.py --batch-size 1
```

### Poor quality results
- Try different model variants for your content type
- Adjust preprocessing: experiment with `--background-threshold`
- For noisy images: use `--model-variant realesr-general-wdn-x4v3`
- For maximum detail: use `--two-stage --scale 4`

### Installation issues with uv

#### Hardlink warning on Windows
If you see "Failed to hardlink files; falling back to full copy":
```bash
# Option 1: Set environment variable (temporary)
$env:UV_LINK_MODE="copy"

# Option 2: Use --link-mode flag
uv pip install -e . --link-mode=copy

# Option 3: Set permanently in Windows
# Go to System Properties → Environment Variables → Add UV_LINK_MODE = copy
```

#### Other installation issues
```bash
# Clear uv cache
uv cache clean

# Reinstall with specific Python version
uv venv --python 3.9
uv pip install -e . --link-mode=copy
```

## Performance Benchmarks

Typical processing times (NVIDIA RTX 3070, 8GB VRAM):
- 1024x768 image, 2x scale: ~2-3 seconds
- 1024x768 image, 4x scale: ~5-8 seconds
- 4000x3000 image, 2x scale (with tiling): ~15-20 seconds
- CPU processing: 5-10x slower than GPU

## License

This project is for educational and research purposes. The underlying models have their own licenses:
- Real-ESRGAN: BSD 3-Clause License
- BasicSR: Apache 2.0 License

## Acknowledgments

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by xinntao
- [BasicSR](https://github.com/XPixelGroup/BasicSR) by XPixelGroup
- [SwinIR](https://github.com/JingyunLiang/SwinIR) by JingyunLiang
- [BSRGAN](https://github.com/cszn/BSRGAN) by cszn

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Citation

If you use this tool in your research, please cite the underlying models you use:

```bibtex
@InProceedings{wang2021realesrgan,
    author    = {Xintao Wang and Liangbin Xie and Chao Dong and Ying Shan},
    title     = {Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data},
    booktitle = {International Conference on Computer Vision Workshops (ICCVW)},
    year      = {2021}
}
```