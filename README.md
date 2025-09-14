# Multimodal Trait & Emotion Recognition through Agentic AI Pipeline

A modular, **multi‑agent conversational AI** pipeline that infers **emotions** and **Big Five personality traits** from interview‑style inputs enriched with behavioral metadata, then generates **empathic, trait‑aware responses**. The system coordinates dedicated **Perception**, **Inference**, **Retrieval‑Memory**, and **Dialogue** agents and supports multiple LLM backbones (e.g., **LLaMA 3.2‑1B**, **Falcon‑RW‑1B**).

> Repository: `Sam-Titan/Multimodal-Trait-and-Emotion-Recognition-through-Agentic-AI-Pipeline`

---

## ✨ Key Features

- **Agentic workflow**: *Observe → Reflect → Act → Self‑Audit*
- **Perception Agent** for emotion recognition from enriched text/metadata
- **Inference Agent** for Big Five (OCEAN) trait estimation
- **Retrieval‑Augmented Memory** for contextual continuity across turns
- **Dialogue Agent** for personality/emotion‑aware response generation
- **Multi‑backbone** support (LLaMA 3.2‑1B, Falcon‑RW‑1B)
- **Lightweight evaluation** via comparison utilities

---

## 🏗️ Architecture (High‑Level)

```
User Input + Behavioral Metadata
        │
        ▼
  [Perception Agent] ──► Emotion labels
        │
        ▼
  [Inference Agent] ──► Big Five (OCEAN) scores
        │
        ▼
  [Retrieval Memory]  ──► Context from past interactions (RAG)
        │
        ▼
  [Dialogue Agent]    ──► Empathic, trait‑aware response
```

---

## 📁 Repository Layout (top‑level)

```
Multimodal-Trait-and-Emotion-Recognition-through-Agentic-AI-Pipeline/
├─ Llama_dialogue_agent.py         # Dialogue agent using LLaMA 3.2‑1B
├─ Falcon_dialogue_agent.py        # Dialogue agent using Falcon‑RW‑1B
├─ compare_models.py               # Simple comparison / benchmarking utility
├─ aligned_data_balanced.json      # Sample aligned data (balanced)
├─ aligned_data_with_traits.json   # Sample aligned data with trait labels
├─ llama_aligned_traits.json       # LLaMA‑specific aligned trait examples
├─ falcon_aligned_traits.json      # Falcon‑specific aligned trait examples
└─ README.md
```

> Languages: **Python**

---

## ⚙️ Setup

### 1) Clone
```bash
git clone https://github.com/Sam-Titan/Multimodal-Trait-and-Emotion-Recognition-through-Agentic-AI-Pipeline.git
cd Multimodal-Trait-and-Emotion-Recognition-through-Agentic-AI-Pipeline
```

### 2) Environment
Create a virtual environment (Python ≥3.10 recommended):
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
If a `requirements.txt` exists:
```bash
pip install -r requirements.txt
```
If not, a minimal stack for local experiments usually includes:
```bash
pip install transformers accelerate datasets sentencepiece
# add: tiktoken, langchain, faiss-cpu, numpy, pandas as needed
```

> Tip: For HF models that require gated access or large downloads, prefer GPU machines and set `HF_HOME` to a disk with sufficient space.

---

## 🚀 Quickstart

### A) Run the LLaMA‑based dialogue agent
```bash
python Llama_dialogue_agent.py
```
### B) Run the Falcon‑based dialogue agent
```bash
python Falcon_dialogue_agent.py
```
### C) Compare backbones
```bash
python compare_models.py
```

> Note: These scripts are intended as starting points. Open the files and adjust model IDs, prompts, and evaluation settings to your environment (GPU/CPU, quantization, context length, etc.).

---

## 🧩 Data

The repository ships a few **aligned JSON** files for quick tests and demos:
- `aligned_data_balanced.json` – balanced examples across labels
- `aligned_data_with_traits.json` – examples enriched with OCEAN traits
- `llama_aligned_traits.json`, `falcon_aligned_traits.json` – backbone‑specific aligned sets

You can extend the schema with additional metadata (e.g., *response latency*, *body language flags*, *speech prosody bins*) to improve the Perception/Inference stages.

---

## 🔧 Configuration Ideas

- **Backbones**: swap HF model IDs or add quantized variants (e.g., 4‑bit)
- **RAG**: wire a vector store (FAISS/Chroma) for retrieval memory
- **Prompting**: add system prompts for *self‑audit* and *bias checks*
- **Scoring**: log empathy/latency/diversity metrics during comparisons
- **Safety**: add refusal/grounding rules for risky or medical claims

---

## 🧪 Evaluation (lightweight)

- **Backbone comparison** using `compare_models.py`
- Track **latency**, **response diversity**, **empathy heuristics**
- Optionally add external human ratings or pairwise preference voting

---

## 📦 Deployment Notes

- Set environment variables (e.g., model IDs, API keys if any)
- Prefer GPU runners; enable `torch.compile()` or `accelerate`
- Containerize with a minimal `Dockerfile` and `CMD` to run your chosen agent

---

## 🗺️ Roadmap

- Add `requirements.txt` and CLI flags
- Integrate a vector DB for Retrieval Memory
- Extend Perception to multimodal I/O (ASR, vision embeddings)
- Add JSON logging + eval dashboards
- Basic unit tests and CI

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit: `git commit -m "feat: add my feature"`
4. Push: `git push origin feat/my-feature`
5. Open a Pull Request

---

## 📜 License

If no license file is present, treat the code as **All Rights Reserved** by the author. Please open an issue to discuss reuse/distribution.

---

## 🙌 Acknowledgements

Built around compact open LLMs (e.g., **LLaMA 3.2‑1B**, **Falcon‑RW‑1B**) with an **agentic design** (Perception → Inference → Retrieval Memory → Dialogue). Inspired by modern RAG and multi‑agent orchestration patterns.
