import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import requests
import os
from typing import Optional, Tuple


class RealESRGANModel:
    """Real-ESRGAN model wrapper for super resolution"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.current_model = None
        
    def download_model(self, model_name: str = 'realesr-general-x4v3'):
        """Download pretrained model weights"""
        model_urls = {
            # Best for documents and text
            'realesr-general-x4v3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth',
            'realesr-general-wdn-x4v3': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            
            # General purpose
            'RealESRGAN_x4plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'RealESRGAN_x2plus': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            
            # For line art and strokes (anime models work well for calligraphy)
            'realesrgan-animevideo-xsx2': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-animevideo-xsx2.pth',
            'RealESRGAN_x4plus_anime_6B': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
        }
        
        if model_name not in model_urls:
            raise ValueError(f"Model {model_name} not supported")
            
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{model_name}.pth'
        
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} model...")
            response = requests.get(model_urls[model_name], stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"Progress: {downloaded/total_size*100:.1f}%", end='\r')
            print(f"\n{model_name} downloaded successfully!")
            
        return model_path
    
    def load_model(self, model_name: str = 'realesr-general-x4v3', scale: int = 2):
        """Load the RealESRGAN model"""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.archs.srvgg_arch import SRVGGNetCompact
            from realesrgan import RealESRGANer
            
            # Model configurations
            if model_name == 'realesr-general-x4v3':
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
                netscale = 4
            elif model_name == 'realesr-general-wdn-x4v3':
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
                netscale = 4
            elif model_name == 'realesrgan-animevideo-xsx2':
                model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
                netscale = 2
            elif model_name in ['RealESRGAN_x4plus', 'RealESRGAN_x4plus_anime_6B']:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23 if 'anime' not in model_name else 6, num_grow_ch=32, scale=4)
                netscale = 4
            elif model_name == 'RealESRGAN_x2plus':
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                netscale = 2
            else:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
                netscale = 4
            
            model_path = self.model_path or self.download_model(model_name)
            
            # Configure dni_weight for models that support it
            dni_weight = [0.1, 0.1] if model_name == 'realesr-general-wdn-x4v3' else None
            
            self.model = RealESRGANer(
                scale=netscale,
                model_path=model_path,
                dni_weight=dni_weight,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=True if self.device.type == 'cuda' else False,
                device=self.device
            )
            
            self.current_model = model_name
            print(f"{model_name} model loaded successfully on {self.device}")
            
        except ImportError:
            print("Please install Real-ESRGAN: pip install realesrgan basicsr")
            raise
    
    def enhance(self, image: Image.Image, outscale: float = 2.0, model_name: str = None) -> Image.Image:
        """Enhance a single image"""
        if self.model is None or (model_name and model_name != self.current_model):
            self.load_model(model_name=model_name or 'realesr-general-x4v3', scale=int(outscale))
            
        # Convert PIL to numpy
        img = np.array(image)
        
        # Enhance
        output, _ = self.model.enhance(img, outscale=outscale)
        
        # Convert back to PIL
        return Image.fromarray(output)


class SwinIRModel:
    """SwinIR model wrapper for super resolution"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def load_model(self, scale: int = 4):
        """Load the SwinIR model"""
        try:
            # This is a simplified version - in production you'd load actual SwinIR
            print(f"SwinIR model would be loaded here (scale={scale})")
            # Placeholder for actual SwinIR implementation
            
        except Exception as e:
            print(f"Error loading SwinIR: {e}")
            raise
    
    def enhance(self, image: Image.Image, scale: int = 4) -> Image.Image:
        """Enhance a single image with SwinIR"""
        # Placeholder implementation
        return image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)


class BSRGANModel:
    """BSRGAN model wrapper for blind super resolution"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def load_model(self):
        """Load the BSRGAN model"""
        try:
            # Placeholder for BSRGAN implementation
            print("BSRGAN model would be loaded here")
            
        except Exception as e:
            print(f"Error loading BSRGAN: {e}")
            raise
    
    def enhance(self, image: Image.Image, scale: int = 4) -> Image.Image:
        """Enhance a single image with BSRGAN"""
        # Placeholder implementation
        return image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)


def get_model(model_name: str = 'realesrgan', device: str = 'cuda'):
    """Factory function to get the appropriate model"""
    models = {
        'realesrgan': RealESRGANModel,
        'swinir': SwinIRModel,
        'bsrgan': BSRGANModel
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Model {model_name} not supported. Choose from: {list(models.keys())}")
    
    return models[model_name.lower()](device=device)