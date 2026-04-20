# Local SFT Dataset

Generated with:

```bash
python3 scripts/build_local_sft_dataset.py
```

Files:

- `opal_local_sft_small.jsonl`: small supervised fine-tuning dataset
- `opal_local_sft_small_manifest.json`: dataset metadata

Format:

- one JSON object per line
- each object contains `messages`
- each sample is a prompt/response pair from the repo's codex-path experiment artifacts

Example schema:

```json
{
  "id": "sobol-codex-gpt5/sobol_pc_gpt5",
  "group": "sobol-codex-gpt5",
  "source_prompt": "src/results/sobol-codex-gpt5/sobol_pc_gpt5_prompt_1.txt",
  "source_response": "src/results/sobol-codex-gpt5/sobol_pc_gpt5_response_1.txt",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
