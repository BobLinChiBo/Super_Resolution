# Super Resolution for Old Chinese Documents

This project provides a super resolution system specifically optimized for enhancing old Chinese book pages and historical documents. It supports multiple state-of-the-art models and includes preprocessing options tailored for aged paper and traditional Chinese text.

## Features

- **Multiple Models**: Support for Real-ESRGAN, SwinIR, and BSRGAN
- **Batch Processing**: Process hundreds of images automatically with progress tracking
- **Document Optimization**: Special preprocessing for old documents including:
  - Background cleaning for aged paper
  - Contrast enhancement for faded text
  - Text-specific sharpening
- **Memory Management**: Automatic tiling for large images to prevent OOM errors
- **Resume Capability**: Continue processing from where you left off
- **GPU Acceleration**: Full CUDA support for fast processing

## Installation

### Option 1: Using UV (Recommended - Fast & Modern)

1. Clone the repository:
```bash
cd D:\GitHub\Super_Resolution
```

2. Install UV if you haven't already:
```bash
# On Windows
pip install uv

# Or using pipx
pipx install uv
```

3. Create virtual environment and install dependencies:
```bash
# Create virtual environment
uv venv

# Activate the environment
# On Windows (PowerShell)
.venv\Scripts\activate

# On Windows (Command Prompt)
.venv\Scripts\activate.bat

# Install all dependencies (including Real-ESRGAN and BasicSR)
uv pip install -r requirements.txt
```

### Option 2: Using Traditional pip

1. Clone the repository:
```bash
cd D:\GitHub\Super_Resolution
```

2. Create virtual environment (optional but recommended):
```bash
python -m venv venv

# Activate the environment
# On Windows (PowerShell)
venv\Scripts\activate

# On Windows (Command Prompt)
venv\Scripts\activate.bat
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Verifying Installation

After installation, verify everything is working:
```bash
# Quick import test
python -c "import torch; import torchvision; print('Core dependencies installed successfully!')"

# Full test (from project directory)
.venv/Scripts/python.exe test_uv_setup.py
```

**Note**: You may see import warnings for realesrgan/basicsr due to version compatibility, but the models should still work correctly.

## Usage

### Quick Start - Optimized for Chinese Documents

Use the pre-configured script with best practices:

```bash
# If using UV environment
.venv/Scripts/python.exe enhance_chinese_docs.py

# Or with UV directly
uv run python enhance_chinese_docs.py

# If using traditional pip/venv
python enhance_chinese_docs.py
```

This uses optimized settings based on community experience:
- `realesr-general-x4v3` model (best for text)
- 2x upscaling (preserves thin strokes)
- Two-stage processing with downsample to 300 DPI
- Background cleaning and text enhancement

### Manual Configuration

Process all images with custom settings:

```bash
# If using UV environment
.venv/Scripts/python.exe src/enhance.py --input-dir data/input --output-dir data/output

# Or with UV directly
uv run python src/enhance.py --input-dir data/input --output-dir data/output

# If using traditional pip/venv
python src/enhance.py --input-dir data/input --output-dir data/output
```

### Best Practice for Text Documents (Based on Research)

```bash
python src/enhance.py \
  --model realesrgan \
  --model-variant realesr-general-x4v3 \
  --scale 2 \
  --two-stage \
  --target-dpi 300 \
  --preprocess \
  --postprocess \
  --enhance-text \
  --clean-background \
  --auto-detect
```

### For Pure Line Art/Calligraphy

```bash
python src/enhance.py \
  --model realesrgan \
  --model-variant realesrgan-animevideo-xsx2 \
  --scale 2 \
  --two-stage \
  --target-dpi 300
```

### Command Line Arguments

#### Input/Output Options
- `--input-dir`: Directory containing input images (default: `data/input`)
- `--output-dir`: Directory for enhanced images (default: `data/output`)

#### Model Options
- `--model`: Choose model: `realesrgan`, `swinir`, or `bsrgan` (default: `realesrgan`)
- `--model-variant`: Specific Real-ESRGAN variant:
  - `realesr-general-x4v3`: Best for documents/text (default)
  - `realesr-general-wdn-x4v3`: With denoising
  - `RealESRGAN_x4plus`: General purpose
  - `realesrgan-animevideo-xsx2`: Best for line art/strokes
- `--scale`: Upscaling factor: 2 or 4 (default: 2)
- `--device`: Device to use: `cuda` or `cpu` (default: `cuda`)

#### Enhancement Options
- `--preprocess`: Apply preprocessing for old documents
- `--postprocess`: Apply postprocessing for better clarity
- `--enhance-text`: Apply text-specific enhancements
- `--clean-background`: Clean aged paper background
- `--background-threshold`: Threshold for background cleaning 0-255 (default: 240)
- `--two-stage`: Use two-stage processing (upscale then downsample) - **Recommended**
- `--target-dpi`: Target DPI for output (e.g., 300 for OCR, 600 for printing)
- `--auto-detect`: Auto-detect content type and select best model

#### Performance Options
- `--batch-size`: Number of images to process in parallel (default: 1)
- `--tile-size`: Tile size for large images (default: 1024)
- `--max-memory-gb`: Maximum GPU memory to use (default: 6.0)

#### Output Options
- `--quality`: JPEG quality 1-100 (default: 95)
- `--resume`: Resume from previous run

## Examples

### Process with Maximum Quality
```bash
python src/enhance.py --model realesrgan --scale 4 --preprocess --postprocess --enhance-text --quality 100
```

### Fast Processing (2x scale)
```bash
python src/enhance.py --model realesrgan --scale 2 --device cuda
```

### Resume Previous Session
```bash
python src/enhance.py --resume
```

### CPU Processing (slower but no GPU required)
```bash
python src/enhance.py --device cpu --tile-size 512
```

## Project Structure

```
Super_Resolution/
├── src/
│   ├── enhance.py          # Main processing script
│   ├── models.py           # Model implementations
│   └── utils.py            # Helper functions
├── data/
│   ├── input/              # Your original images
│   └── output/             # Enhanced images
├── models/                 # Downloaded model weights
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Tips for Best Results

1. **For Very Old/Damaged Documents**: Use all enhancement options:
   ```bash
   python src/enhance.py --preprocess --postprocess --enhance-text --clean-background
   ```

2. **For Large Images**: The script automatically tiles large images to prevent memory issues

3. **Quality vs Speed**: 
   - Use `--scale 2` for faster processing
   - Use `--scale 4` for maximum quality

4. **Memory Issues**: If you encounter OOM errors:
   - Reduce `--tile-size` (e.g., 512)
   - Reduce `--max-memory-gb`
   - Use CPU processing with `--device cpu`

## Output

Enhanced images are saved with the naming pattern:
`{original_name}_x{scale}_enhanced.jpg`

A `progress.json` file in the output directory tracks:
- Completed files
- Failed files with error messages
- Processing timestamps

## Troubleshooting

1. **CUDA out of memory**: Reduce tile size or use CPU mode
2. **Model download fails**: Check internet connection, models are downloaded on first use
3. **Import errors**: Ensure all dependencies are installed, especially `realesrgan` and `basicsr`

## Performance

On a modern GPU (RTX 3070 or better):
- ~5-15 seconds per image at 4x scale
- ~2-5 seconds per image at 2x scale

Processing time depends on image size and selected options.

## Credits

This project uses the following models:
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [SwinIR](https://github.com/JingyunLiang/SwinIR) 
- [BSRGAN](https://github.com/cszn/BSRGAN)