# Video Moment Retrieval Pipeline

Step-by-step data flow and inputs/outputs.

## Overview

```
Channel/category setup → Popular video discovery → Comment crawl → Merge/filter → (optional) Video download
    → Caption generation → Segment integration → Modality gating → Moment query generation
```

## Step-by-step

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0 | (manual) | - | `csv/channel_categories.csv` |
| 1 | `find_popular_videos.py` | Channel URL / `csv/channel_categories.csv` | `csv/video_id_mapping.csv` |
| 2 | `crawl_comments.py` | `csv/video_id_mapping.csv` | `Comments/<video_id>_comments.csv` |
| 3 | `yt_merge_with_dedup_lang.py` | `Comments/*.csv`, `csv/video_id_mapping.csv` | `csv/merged_filtered_comments_with_dedup_lang.csv` |
| 4 | `download_videos.py` (optional) | `csv/video_id_mapping.csv` | `{output_dir}/{category}/{channel}/{video_id}.mp4` |
| 5 | `generate_captions_range.py` | Comment CSV, video files | `captions_by_video/captions_<video_id>.json` |
| 6 | `segment_integrator.py` | `captions_by_video/*.json` | `*_integrated.json` |
| 7 | `modality_gating.py` | `*_integrated.json` | `*_classified.json` |
| 8 | `query_generator.py` | `*_classified.json` | `*_moment_queries.json` |

## Directory layout

- **csv/**  
  Channel/video mapping and merged comment CSVs.
- **Comments/**  
  Per-video crawled comment CSVs.
- **captions_by_video/**  
  Per-video caption, integrated, classified, and query JSONs.
- **docs/**  
  Pipeline and structure documentation.
