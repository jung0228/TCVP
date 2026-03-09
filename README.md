# Video Moment Retrieval
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pipeline that collects and analyzes YouTube video comments, combines them with video segment captions, and automatically generates search queries for **moment retrieval**.
![TCVP Pipeline](figures/TCVP%20pipeline.png)

🔗 Project Page: https://jung0228.github.io/TCVP/

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Flow](#data-flow)
- [License](#license)

---

## Features

- **Popular video discovery**: Build video lists by channel/category using `yt-dlp`
- **Comment crawling**: Per-video comment collection with parallel workers
- **Comment preprocessing**: Timestamp extraction, deduplication, language filtering
- **Video caption generation**: Segment-level visual/audio descriptions (FFmpeg + Qwen2.5-Omni-7B)
- **Segment integration**: Merge multiple segments into single captions (OpenAI)
- **Modality gating**: Visual/audio classification from comment–caption similarity (Qwen3-Embedding-8B)
- **Moment query generation**: Retrieval queries from classified data (OpenAI)

---

## Installation

### Requirements

- **Python** 3.8+
- **yt-dlp**: YouTube metadata and download
- **FFmpeg**: Video segment extraction
- **CUDA GPU**: Recommended for Qwen model inference
- **OpenAI API key**: For segment integration and query generation

```bash
git clone https://github.com/jung0228/TCVP.git
cd TCVP
pip install -r requirements.txt
# FFmpeg: brew install ffmpeg (macOS) / sudo apt install ffmpeg (Linux)
```

### Environment variables

```bash
export OPENAI_API_KEY='your-openai-api-key'
```

---

## Usage

Run all commands **from the project root**. Pipeline order: **1 → 2(optional) → 3 → 4**

### Step 1: Collect data (video discovery + comments + filtering)

```bash
# All channels in csv/channel_categories.csv
python pipeline/collect_data.py --top 10 --workers 5

# Single channel
python pipeline/collect_data.py --channel @TED --channel-name "TED" --category "talk show"
```

Output: `csv/video_id_mapping.csv`, `Comments/<video_id>_comments.csv`, `csv/merged_filtered_comments_with_dedup_lang.csv`

### Step 2: Download videos (optional)

```bash
python pipeline/download_videos.py --resolution 360
```

### Step 3: Generate captions + integrate segments

```bash
python pipeline/generate_captions.py 1 --end 100
```

Output: `captions_by_video/captions_<video_id>_integrated.json`

### Step 4: Classify modality + generate moment queries

```bash
python pipeline/generate_queries.py
```

Output: `captions_by_video/captions_<video_id>_moment_queries.json`

---

## Project Structure

```
TCVP/
├── pipeline/
│   ├── collect_data.py               # [Step 1] Video discovery + comment crawling + filtering
│   ├── download_videos.py            # [Step 2] Video download (optional)
│   ├── generate_captions.py          # [Step 3] Caption generation + segment integration
│   ├── generate_queries.py           # [Step 4] Modality classification + query generation
│   │
│   ├── OptimizedMomentQueryGenerator.py  # Core module for caption generation
│   │
│   └── (individual scripts — can also be run separately)
│       ├── find_popular_videos.py
│       ├── crawl_comments.py
│       ├── yt_merge_with_dedup_lang.py
│       ├── generate_captions_range.py
│       ├── segment_integrator.py
│       ├── segment_integrator_staxai.py
│       ├── modality_gating.py
│       ├── query_generator.py
│       └── query_generator_staxai.py
├── csv/                              # Channel/video mapping and merged comments
├── Comments/                         # Per-video crawled comments
├── captions_by_video/                # Integrated captions and moment query JSONs
├── figures/
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Data Flow

```
csv/channel_categories.csv
        ↓
[Step 1] collect_data.py
        ↓  csv/video_id_mapping.csv
        ↓  Comments/<video_id>_comments.csv
        ↓  csv/merged_filtered_comments_with_dedup_lang.csv
        ↓
[Step 2] download_videos.py (optional)
        ↓
[Step 3] generate_captions.py     →  captions_by_video/captions_<video_id>_integrated.json
        ↓
[Step 4] generate_queries.py      →  captions_by_video/captions_<video_id>_moment_queries.json
```

---

## License

This project is distributed under the [MIT License](LICENSE).
