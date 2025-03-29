# Ego4D Multiturn Video QA Inference Pipeline

This project runs multi-turn prompting on the Ego4D dataset using large vision-language models (e.g., Qwen2.5-VL-72B-Instruct), leveraging dynamic batching, multi-GPU inference, and caching for efficient evaluation.

---

## 📦 Project Structure

```
.
├── main.py                # Entrypoint
├── config.py              # Config with dynamic batch size & model settings
├── dataset.py             # Ego4D instance loader
├── sampler.py             # Uniform & dense frame sampler
├── prompt_builder.py      # Prompt templates per style and turn
├── inference_engine.py    # Batched inference with multi-GPU support
├── batch_runner.py        # Multi-turn evaluation loop, caching, metrics
├── cached_stage_outputs.json  # Stores reusable outputs per video & turn
├── model_outputs.json     # Full outputs per instance
├── model_outputs.csv      # Simplified spreadsheet export
├── accuracy_summary.png   # Bar chart of accuracy by query type
```

---

## 🚀 How to Run

```bash
python main.py --config coarse
```

This will:
1. Load dataset annotations and video metadata
2. Sample frames per instance (uniform or dense)
3. Run multi-turn prompts (e.g., scenario, segments, question)
4. Use cache for reusable turns
5. Run batched inference using all available GPUs
6. Evaluate correctness (segment coverage)
7. Save `.json`, `.csv`, and `.png` with results

---

## ⚙️ Configuration (in `config.py`)

- `prompt_style`: Options include `frame_labeling`, `stage_analysis`, etc.
- `sampling_strategy`: `uniform` or `dense`
- `batch_size`: Dynamically estimated based on available GPU memory (~12GB per sample)
- `ignore_cache`: Set `True` to force rerun without cache

---

## 📊 Outputs

- `model_outputs.json`: Full inference records (prompt, response, flags)
- `model_outputs.csv`: Summary per instance
- `accuracy_summary.png`: Bar chart of accuracy per query type (based on first word)

---

## 📈 Evaluation Logic

Model outputs are scanned for segments (e.g., `0s - 128s`). If the clip’s `clip_start_sec` to `clip_end_sec` falls within **any** predicted segment, it is marked as correct.

---

## 🧠 Prompting

Each prompt style defines:
- Number of turns
- Which turns depend on the query
- Which turns are cacheable across queries from the same video

Prompt construction adapts per turn index, and uses frame timestamps, labels, and optionally prior outputs.

---

## 💡 Tips

- Use a smaller model (e.g. Qwen2.5-VL-7B) for debugging
- Override `batch_size` in `Config` for testing
- Extend `PromptBuilder` with new styles for experiments

---

## 📍 Dependencies

- PyTorch
- HuggingFace Transformers
- ImageIO, OpenCV
- Matplotlib

---

## 🧪 TODOs

- Add confusion-style breakdowns
- Retry batch with smaller size on OOM
- Add query clustering by semantic category
- CLI args for quick config switching

---

Feel free to open issues or share results!

