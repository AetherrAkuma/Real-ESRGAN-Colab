import argparse
import cv2
import glob
import os
import torch

# import spandrel
from spandrel import ModelLoader, ImageModelDescriptor

from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def load_model_via_spandrel(model_path: str, device: torch.device):
    """
    Load a model file (pth / safetensors etc) via spandrel.
    Returns an ImageModelDescriptor (includes .model, .architecture, .scale etc).
    """
    loader = ModelLoader(device)
    descriptor = loader.load_from_file(model_path)
    if not isinstance(descriptor, ImageModelDescriptor):
        raise RuntimeError(f"Loaded descriptor is not an image model: {descriptor}")
    return descriptor

def main():
    """Inference demo for Real-ESRGAN with spandrel universal loader."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='RealESRGAN_x4plus',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | '
              'RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-general-x4v3'))
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument(
        '-dn',
        '--denoise_strength',
        type=float,
        default=0.5,
        help=('Denoise strength. Only used for the “realesr-general-x4v3” model'))
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='Optional: explicit model path (pth / safetensors)')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision)')
    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using same as input')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='GPU device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if (args.gpu_id is not None and torch.cuda.is_available()) else "cpu")

    # Determine model_name normalization
    args.model_name = args.model_name.split('.')[0]

    # Setup default architecture / scale / URL list as fallback
    netscale = None
    default_urls = []
    if args.model_name == 'RealESRGAN_x4plus':
        netscale = 4
        default_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif args.model_name == 'RealESRNet_x4plus':
        netscale = 4
        default_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':
        netscale = 4
        default_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif args.model_name == 'RealESRGAN_x2plus':
        netscale = 2
        default_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif args.model_name == 'realesr-animevideov3':
        netscale = 4
        default_urls = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif args.model_name == 'realesr-general-x4v3':
        netscale = 4
        default_urls = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    # Determine model path
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            for url in default_urls:
                model_path = load_file_from_url(
                    url=url, model_dir=os.path.dirname(model_path), progress=True, file_name=None
                )

    # Use spandrel to load the model descriptor
    descriptor = load_model_via_spandrel(model_path, device)
    print(f"Loaded model architecture: {descriptor.architecture}, scale: {descriptor.scale}, tags: {descriptor.tags}")

    # Build restorer using descriptor
    model_module = descriptor.model  # torch.nn.Module
    # netscale fallback to descriptor.scale if unset
    if netscale is None:
        netscale = descriptor.scale

    dni_weight = None
    if args.model_name == 'realesr-general-x4v3' and args.denoise_strength != 1:
        # If descriptor supports a secondary WDN model, you may need to adapt
        # Here we just demonstrate combining weights
        # This part might need custom logic depending on descriptor details
        dni_weight = [args.denoise_strength, 1 - args.denoise_strength]

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model_module,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id
    )

    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=args.outscale,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler
        )

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: failed to read {path}, skipping.")
            continue

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if face_enhancer is not None:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try setting --tile to a smaller number.')
            continue

        # Determine extension
        if args.ext == 'auto':
            ext = extension[1:]
        else:
            ext = args.ext
        if img_mode == 'RGBA':
            ext = 'png'

        if args.suffix == '':
            save_path = os.path.join(args.output, f'{imgname}.{ext}')
        else:
            save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{ext}')

        cv2.imwrite(save_path, output)
        print(f"Saved: {save_path}")

if __name__ == '__main__':
    main()
