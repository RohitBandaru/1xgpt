# V-JEPA2-AC World Model for 1X Challenge

This module implements a V-JEPA2-AC based world model as a replacement for the GENIE baseline in the 1X robotics challenge.

## Architecture

### Core Components

1. **V-JEPA2-AC Backbone**: Pretrained ViT-Giant encoder (1.7B params) + action-conditioned predictor
2. **Action Embedding**: Converts 1X scalar action tokens to embeddings for V-JEPA2-AC
3. **Factorized Token Heads**: Replaces continuous V-JEPA2-AC output with discrete token prediction
4. **Optional Flow Matching**: Improves representation learning over standard L1 loss

### Key Advantages

- **Scale**: 1.7B params vs 138M GENIE baseline (~12x larger)
- **Pretraining**: Leverages internet-scale video understanding
- **Action Conditioning**: Natural integration with robot actions
- **Factorized Output**: Compatible with 1X challenge evaluation format

## Usage

### Basic Training

```python
from vjepa import create_vjepa_model

# Create model with default config
model = create_vjepa_model(
    config_path="vjepa/configs/vjepa_vitg_1x.json"
)

# Training loop (similar to GENIE)
outputs = model(
    input_ids=batch["input_ids"],      # [B, T*H*W] image tokens
    action_tokens=batch["actions"],     # [B, T] scalar action tokens  
    labels=batch["labels"]              # [B, T*H*W] ground truth
)

loss = outputs.loss
loss.backward()
```

### With Flow Matching

```python
# Enable flow matching for better representations
model = create_vjepa_model(
    config_path="vjepa/configs/vjepa_vitg_1x_flow.json"
)

outputs = model(input_ids, action_tokens, labels)
total_loss = outputs.loss  # Combined token + flow loss
token_loss = outputs.token_loss
flow_loss = outputs.flow_loss
```

## Configuration

### Model Configurations

- `vjepa_vitg_1x.json`: Basic V-JEPA2-AC + factorized tokens
- `vjepa_vitg_1x_flow.json`: With flow matching loss

### Key Parameters

```python
VJEPAConfig(
    model_name="vit_ac_giant",       # V-JEPA2-AC model size
    pretrained=True,                 # Use pretrained weights
    num_factored_vocabs=2,           # Factorized vocabulary 
    factored_vocab_size=512,         # 512^2 = 262144 total vocab
    action_vocab_size=65536,         # uint16 action tokens
    use_flow_matching=False,         # Enable flow matching loss
)
```

## Expected Performance

### Target Metrics (1X Challenge)
- **Baseline GENIE**: 8.79 cross-entropy loss
- **Target**: <8.0 loss (win $10k prize)
- **Expected V-JEPA2-AC**: <7.5 loss (significant improvement)

### Performance Factors
1. **Model Scale**: 12x more parameters than baseline
2. **Pretraining**: Internet-scale video understanding
3. **Action Conditioning**: Better temporal modeling
4. **Flow Matching**: Improved representation quality

## Integration with 1X Pipeline

This module is designed as a drop-in replacement for GENIE:

```python
# Replace GENIE import
# from genie.st_mask_git import STMaskGIT
from vjepa.model import VJEPAWorldModel

# Use same data loading and training loop
# from data import get_maskgit_collator  # Still compatible
# model = VJEPAWorldModel(config) instead of STMaskGIT(config)
```

## Commands

### Training
```bash
# Train V-JEPA model with cross-entropy loss (recommended for 1X challenge)
python train.py --vjepa_config vjepa/configs/vjepa_vitg_1x.json --output_dir data/vjepa_model --max_eval_steps 10

# Train with L1 loss (experimental - similar to original V-JEPA2-AC)
python train.py --vjepa_config vjepa/configs/vjepa_vitg_1x_l1.json --output_dir data/vjepa_l1_model --max_eval_steps 10
```

### Generation
```bash
# Generate frames from trained V-JEPA model
python vjepa/generate.py --checkpoint_dir data/vjepa_model/final_checkpt

# Generate with specific parameters
python vjepa/generate.py --checkpoint_dir data/vjepa_model/final_checkpt --output_dir data/vjepa_generated --example_ind 100
```

### Visualization
```bash
# Visualize generated frames
python visualize.py --token_dir data/vjepa_generated

# Visualize specific example
python visualize.py --token_dir data/vjepa_generated --example_ind 100
```

### Evaluation
```bash
# Evaluate trained V-JEPA model
python vjepa/evaluate.py --checkpoint_dir data/vjepa_model/final_checkpt

# Evaluate with specific parameters
python vjepa/evaluate.py --checkpoint_dir data/vjepa_model/final_checkpt --batch_size 8 --max_examples 1000
```

### Pretrained Models
```bash
# Use pretrained V-JEPA baseline (when available)
python vjepa/generate.py --checkpoint_dir 1x-technologies/VJEPA_1B --output_dir data/vjepa_baseline_generated

# Evaluate pretrained model
python vjepa/evaluate.py --checkpoint_dir 1x-technologies/VJEPA_1B
```

## Performance Comparison

| Model | Loss Type | Parameters | Expected Loss | Status |
|-------|-----------|------------|---------------|--------|
| GENIE (baseline) | Cross-entropy | 138M | 8.79 | ✅ Current |
| V-JEPA + CE | Cross-entropy | 1.7B | <7.5 | 🚧 Implementing |
| V-JEPA + L1 | L1 | 1.7B | <7.0 | 🔬 Experimental |

## TODO

- [ ] Integrate actual V-JEPA2 loading (currently placeholder)
- [ ] Implement generation/sampling methods (vjepa/generate.py)
- [ ] Add evaluation utilities (vjepa/evaluate.py)
- [ ] Update train.py to support --vjepa_config flag
- [ ] Optimize memory usage for large model
- [ ] Add action preprocessing for 1X scalar tokens