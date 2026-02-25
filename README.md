# AI Influencer Studio Library

Professional dual-GPU SDXL inference engine with modular architecture and Telegram adapter.

## Overview

AI Influencer Studio is a production-grade Python library for high-performance SDXL image generation. It features:

- **Dual-GPU Pipeline**: Stage 1 generation on CUDA:0, Stage 2 refinement on CUDA:1
- **Dynamic LoRA Management**: Identity preservation and detail enhancement with VRAM-safe loading
- **Advanced Scheduler Support**: DPMSolver++, Euler-a, DDIM with automatic v_prediction detection
- **Async Queue System**: Background worker with real-time progress tracking
- **Memory Optimization**: Intelligent VRAM management with automatic cleanup
- **Modular Architecture**: Clean separation between engine core and adapters

## Architecture

```
studio/
├── core/           # Core inference engine
│   ├── engine.py       # Main StudioEngine orchestrator
│   ├── pipeline.py     # SDXL pipeline management
│   ├── models.py       # Model loading and LoRA handling
│   ├── memory.py       # VRAM management utilities
│   ├── download.py     # Civitai/HuggingFace downloaders
│   ├── prompts.py      # Prompt builder logic
│   └── logging_utils.py # Metrics and logging
├── schema/         # Data structures
│   ├── params.py       # Generation parameters
│   ├── errors.py       # Custom exceptions
│   └── state.py        # Runtime state management
└── adapters/       # External interfaces
    ├── telegram_bot.py # Telegram bot implementation
    ├── handlers.py     # Command and callback handlers
    └── worker.py       # Async queue worker
```

## Installation

### Requirements

- Python 3.10+
- CUDA 12.1+
- 24GB+ total VRAM (dual GPU recommended)
- Ubuntu/Linux environment

### Setup

```bash
# Install PyTorch with CUDA support
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install library dependencies
pip install xformers==0.0.27.post2 diffusers transformers accelerate safetensors pillow imageio opencv-python-headless requests python-telegram-bot

# Clone repository
git clone https://github.com/preethamdev05/ai-influencer-studio-lib.git
cd ai-influencer-studio-lib

# Install as package
pip install -e .
```

## Usage

### Direct Engine Usage

```python
import asyncio
from studio import StudioEngine
from studio.schema import GenerationParams

async def main():
    # Initialize engine
    engine = StudioEngine()
    await engine.load_models()
    
    # Create generation request
    params = GenerationParams(
        base_model="juggernaut",
        prompt="beautiful woman, professional photography",
        negative_prompt="low quality, blurry",
        width=832,
        height=1216,
        steps=25,
        cfg=7.0,
        use_refiner=True,
        loras=["detail", "identity"]
    )
    
    # Generate image
    result = await engine.generate(params)
    result.image.save("output.png")
    print(f"Generation took {result.total_time:.2f}s")

asyncio.run(main())
```

### Telegram Bot Mode

```bash
# Set bot token
export TELEGRAM_BOT_TOKEN="your_bot_token_here"

# Run bot
python run_bot.py
```

## Features

### Model Support

- **Base Models**: Juggernaut, Lustify, IntoRealism
- **Refiner**: RealVisXL V5.0 (fp16 optimized)
- **LoRAs**: Detail enhancer, Identity preservation

### Memory Management

- Automatic VRAM reserve calculation (3.5GB base + model requirements)
- LoRA unloading when VRAM headroom insufficient
- Post-generation cleanup with gc + CUDA cache clearing
- VAE tiling for high-resolution outputs

### Prompt Engineering

- Automatic quality tag injection
- Negative prompt standardization
- Style-specific enhancements
- Multi-stage prompt building (base + refiner)

### Telegram Interface

- Interactive model/style selection
- Resolution presets (Portrait, Landscape, Square)
- Advanced settings (CFG, steps, sampler)
- Real-time progress updates
- Queue status monitoring
- Image regeneration and upscaling

## Configuration

Environment variables:

```bash
TELEGRAM_BOT_TOKEN=your_token       # Required for bot mode
DEBUG_MODE=true                     # Enable verbose logging
MODELS_DIR=/path/to/models          # Custom model directory
GPU_STAGE1=0                        # Primary GPU index
GPU_STAGE2=1                        # Refiner GPU index
```

## Performance

### Benchmark (RTX 4090 + RTX 3090)

- 832x1216 @ 25 steps: ~8-12s
- 1024x1024 @ 30 steps: ~10-14s
- With refiner (15 steps): +4-6s
- LoRA loading overhead: ~2-3s

### Optimization Tips

1. Use DPMSolver++ for speed (default)
2. Enable refiner only for final outputs
3. Cache models on dedicated SSD
4. Monitor VRAM with `engine.get_status()`
5. Use FP16 models for memory efficiency

## Development

### Project Structure

- **Core Layer**: Pure inference logic, no external dependencies
- **Schema Layer**: Type definitions and validation
- **Adapter Layer**: External interface implementations (Telegram, REST API, etc.)

### Adding New Adapters

```python
from studio import StudioEngine
from studio.schema import GenerationParams

class CustomAdapter:
    def __init__(self):
        self.engine = StudioEngine()
    
    async def initialize(self):
        await self.engine.load_models()
    
    async def handle_request(self, user_input):
        params = GenerationParams.from_dict(user_input)
        result = await self.engine.generate(params)
        return result.image
```

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce resolution or batch size
- Disable refiner temporarily
- Clear VRAM: `engine.core.memory.cleanup_memory()`

**Slow Generation**
- Check GPU utilization with `nvidia-smi`
- Verify models loaded on correct devices
- Disable xformers if causing issues

**Model Download Failures**
- Check internet connection
- Verify Civitai API access
- Manually place models in `models/` directory

## License

MIT License - See LICENSE file for details

## Credits

- SDXL by Stability AI
- Diffusers library by Hugging Face
- Community model creators on Civitai

## Support

For issues and feature requests, open a GitHub issue.
