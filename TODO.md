# 1X World Model Challenge - TODO

## High Priority

- [ ] **Multi-GPU Support for Evaluation Scripts**
  - [ ] Add distributed evaluation to `genie/evaluate.py`
  - [ ] Add distributed evaluation to `vjepa/evaluate.py` 
  - [ ] Support data parallelism for faster evaluation
  - [ ] Add `--gpu_ids` argument for device selection
  - [ ] Test with 2x4090 setup for faster iteration

## V-JEPA Implementation

- [x] Create V-JEPA module structure
- [x] Implement factorized token prediction heads
- [x] Add action embedding for scalar tokens
- [x] Update train.py to support --vjepa_config
- [x] Add frozen backbone option for transfer learning
- [ ] Integrate actual V-JEPA2-AC weights loading
- [ ] Implement autoregressive generation in vjepa/generate.py
- [ ] Optimize memory usage for 1.7B parameter model

## Performance Optimization

- [x] Add --skip_lpips flag for fast evaluation
- [ ] Implement gradient checkpointing for V-JEPA
- [ ] Add mixed precision training support
- [ ] Optimize data loading pipeline
- [ ] Profile memory usage and bottlenecks

## Experimentation

- [ ] Train V-JEPA with frozen backbone baseline
- [ ] Compare V-JEPA vs GENIE on validation set
- [ ] Experiment with different loss functions (L1 vs cross-entropy)
- [ ] Test action conditioning effectiveness
- [ ] Hyperparameter sweep for learning rates

## Model Integration

- [ ] Download and integrate V-JEPA2-AC pretrained weights
- [ ] Test end-to-end V-JEPA pipeline
- [ ] Validate factorized vocabulary compatibility
- [ ] Ensure 1X challenge evaluation format compliance

## Infrastructure

- [ ] Set up wandb logging for experiments
- [ ] Create evaluation benchmarking scripts
- [ ] Add model checkpointing best practices
- [ ] Document training commands and configs

## Challenge Submission

- [ ] Achieve <8.0 loss target (currently 8.79 baseline)
- [ ] Prepare submission package (source code + build script)
- [ ] Document FLOPS usage and external data
- [ ] Test submission format compliance

## Documentation

- [ ] Update README with latest results
- [ ] Document multi-GPU setup instructions
- [ ] Add troubleshooting guide for common issues
- [ ] Create performance comparison tables

---

**Next Steps:**
1. Implement multi-GPU evaluation for faster iteration
2. Train V-JEPA frozen backbone baseline
3. Compare performance against GENIE baseline