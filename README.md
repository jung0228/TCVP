# Video Moment Retrieval

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pipeline that collects and analyzes YouTube video comments, combines them with video segment captions, and automatically generates search queries for **moment retrieval**.

---

## Table of Contents

- [Features](#-features)
- [Getting Started](#-getting-started)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data Flow](#-data-flow)
- [Configuration](#%EF%B8%8F-configuration)
- [License](#-license)

---

## Features

- **Popular video discovery**: Build video lists by channel/category using `yt-dlp`
- **Comment crawling**: Per-video comment collection with parallel workers
- **Comment preprocessing**: Timestamp extraction, deduplication, language filtering (e.g. English)
- **Video caption generation**: Segment-level visual/audio descriptions (FFmpeg + AI)
- **Segment integration**: Merge multiple segments into single visual/audio captions (LLM)
- **Modality gating**: Visual/audio classification from comment–caption similarity (Sentence Transformer)
- **Moment query generation**: Retrieval queries from classified data (OpenAI / Letsur)

---

## Getting Started

### Requirements

- **Python** 3.8+
- **yt-dlp**: YouTube metadata and download
- **FFmpeg**: Video segment extraction (`scripts/generate_captions_range.py`)
- **API keys** (optional): OpenAI or Letsur AI Gateway for segment integration and query generation

### Quick install

```bash
# Clone the repo
git clone https://github.com/your-username/VideoMomentRetrieval.git
cd VideoMomentRetrieval

# Install dependencies
pip install -r requirements.txt
pip install yt-dlp   # or: brew install yt-dlp
# FFmpeg: brew install ffmpeg (macOS) / sudo apt install ffmpeg (Linux)
```

### Environment variables (when using LLMs)

```bash
# OpenAI
export OPENAI_API_KEY='your-openai-api-key'

# Or Letsur AI Gateway
export LETSUR_API_KEY='your-letsur-api-key'
export LETSUR_MODEL='gpt-4.1'  # optional
```

---

## Installation

### 1. Python packages

```bash
pip install -r requirements.txt
```

Main dependencies: `youtube-comment-downloader`, `yt-dlp`, `openai`, `sentence-transformers`, `scikit-learn`, `pandas`, `torch`, `transformers`, `langdetect`, etc. See `requirements.txt` for the full list.

### 2. System tools

| Tool | Purpose | Install |
|------|---------|---------|
| **yt-dlp** | Video metadata/download | `pip install yt-dlp` or `brew install yt-dlp` |
| **FFmpeg** | Video segment processing | `brew install ffmpeg` (macOS) / `sudo apt install ffmpeg` (Linux) |

### 3. Data directories

These directories are used by the pipeline (scripts may create them if missing):

- `csv/` — Channel/video mapping and merged comment CSVs
- `Comments/` — Per-video comment CSVs
- `captions_by_video/` — Caption, integrated, classified, and query JSON files

---

## Usage

Run all commands **from the project root**. The pipeline runs in order **0 → 1 → 2 → 3 → (4) → 5 → 6 → 7 → 8**. See [docs/pipeline.md](docs/pipeline.md) for detailed inputs/outputs.

### Step 0: Channel setup (manual)

Prepare `csv/channel_categories.csv` with channel names and categories.

### Step 1: Build popular video list

```bash
# Single channel
python scripts/find_popular_videos.py @channel_handle --top 10 --save --channel-name "Channel Name" --category "category"

# Batch (uses csv/channel_categories.csv)
python scripts/find_popular_videos.py --batch --top 10
```

Example categories: `sport`, `gaming`, `comedy`, `entertainment`, `talk show`, `podcast`, `making`

### Step 2: Crawl comments

```bash
python scripts/crawl_comments.py --workers 5
```

- `--max-comments`: Max comments per video (default: unlimited)
- `--workers`: Number of parallel workers (default: 5)
- `--output-dir`: Comment output directory (default: `Comments`)

### Step 3: Merge comments and language filter

```bash
python scripts/yt_merge_with_dedup_lang.py
```

Keeps target language(s) (e.g. English), deduplicates, and writes `csv/merged_filtered_comments_with_dedup_lang.csv`.

### Step 4: Download videos (optional)

```bash
# All channels
python scripts/download_videos.py --resolution 360

# Specific channels only
python scripts/download_videos.py --resolution 360 --channels Channel1 Channel2
```

- `--resolution`: `360`, `480`, `720`, `1080` (default: `360`)
- `--output-dir`: Output path (default: `/home/elicer/yt_dataset/youtube_videos`)

### Step 5: Generate captions (segments + analysis)

```bash
python scripts/generate_captions_range.py 1 100
```

- Arguments: start comment index, end comment index (omit end to process to the end)
- Options: `--device-map`, `--workers-segments`, `--workers-analysis`, `--max-memory`, etc.

Output: `captions_by_video/captions_<video_id>.json`

### Step 6: Integrate segments

```bash
# OpenAI
python scripts/segment_integrator.py captions_by_video --folder

# Letsur
python scripts/segment_integrator_staxai.py captions_by_video --folder
```

Output: `*_integrated.json`

### Step 7: Modality gating

```bash
python scripts/modality_gating.py captions_by_video_integrated --folder
```

Output: `*_classified.json`

### Step 8: Moment query generation

```bash
# OpenAI
python scripts/query_generator.py --input captions_by_video_classified --folder

# Letsur
python scripts/query_generator_staxai.py --input captions_by_video_classified --folder
```

Output: `*_moment_queries.json`

---

## Project Structure

```
VideoMomentRetrieval/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── csv/                          # Channel/video mapping and merged comment CSVs
│   ├── channel_categories.csv
│   ├── video_id_mapping.csv
│   └── merged_filtered_comments_with_dedup_lang.csv
├── Comments/                     # Per-video crawled comments
├── captions_by_video/            # Caption, integrated, classified, query JSONs
├── config/                       # Config examples
│   ├── README.md
│   └── channel_categories.example.csv
├── docs/
│   └── pipeline.md               # Pipeline details
└── scripts/                      # All pipeline scripts
    ├── find_popular_videos.py        # 1. Popular video discovery
    ├── crawl_comments.py             # 2. Comment crawling
    ├── yt_merge_with_dedup_lang.py   # 3. Comment merge & language filter
    ├── download_videos.py            # 4. Video download (optional)
    ├── generate_captions_range.py    # 5. Caption generation
    ├── OptimizedMomentQueryGenerator.py  # Core module for caption generation
    ├── segment_integrator.py         # 6. Segment integration (OpenAI)
    ├── segment_integrator_staxai.py  # 6. Segment integration (Letsur)
    ├── modality_gating.py            # 7. Modality gating
    ├── query_generator.py            # 8. Query generation (OpenAI)
    └── query_generator_staxai.py     # 8. Query generation (Letsur)
```

---

## Data Flow

```
csv/channel_categories.csv
        ↓
scripts/find_popular_videos.py  →  csv/video_id_mapping.csv
        ↓
scripts/crawl_comments.py       →  Comments/<video_id>_comments.csv
        ↓
scripts/yt_merge_with_dedup_lang.py  →  csv/merged_filtered_comments_with_dedup_lang.csv
        ↓
scripts/download_videos.py (optional)  →  {output_dir}/{category}/{channel}/{video_id}.mp4
        ↓
scripts/generate_captions_range.py  →  captions_by_video/captions_<video_id>.json
        ↓
scripts/segment_integrator*.py  →  *_integrated.json
        ↓
scripts/modality_gating.py  →  *_classified.json
        ↓
scripts/query_generator*.py  →  *_moment_queries.json
```

---

## ⚙️ Configuration

### CSV format

- **csv/video_id_mapping.csv**: `video_id`, `channel_name`, `category`
- **csv/channel_categories.csv**: Channel and category info for batch processing

### Video output path

The default output path for `download_videos.py` is `/home/elicer/yt_dataset/youtube_videos`. Override with `--output-dir`.

---

## License

This project is distributed under the [MIT License](LICENSE).
