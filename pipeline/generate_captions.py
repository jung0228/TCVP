#!/usr/bin/env python3
"""
캡션 생성 + 세그먼트 통합 파이프라인 (Step 5+6)
- Qwen2.5-Omni-7B로 비디오 세그먼트 분석
- GPT API로 visual/audio 통합 캡션 생성
- 중간 파일 없이 captions_<video_id>_integrated.json 직접 저장
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from OptimizedMomentQueryGenerator import OptimizedMomentQueryGenerator
from generate_captions_range import (
    GeneratorPool, parse_max_memory, safe_video_filename, restore_video_id,
    ensure_dir, convert_json_types, read_comments, process_video_comments,
)
from segment_integrator import SegmentIntegrator

LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("generate_captions")


def integrate_video_results(integrator: SegmentIntegrator, results: List[Dict]) -> List[Dict]:
    """각 댓글의 segments를 통합하여 integrated_visual/audio 추가"""
    integrated = []
    for i, comment in enumerate(results, 1):
        if comment.get('analysis_type') == 'failed' or not comment.get('segments'):
            integrated.append(comment)
            continue
        try:
            logger.info("세그먼트 통합 중 (%d/%d)", i, len(results))
            result = integrator.integrate_comment(comment)
            integrated.append(result)
            time.sleep(0.5)  # API rate limit
        except Exception as e:
            logger.warning("통합 실패 (comment_index=%s): %s", comment.get('comment_index'), e)
            integrated.append(comment)
    return integrated


def save_integrated_video(video_id: str, results: List[Dict], summary: Dict, output_dir: Path) -> Path:
    """통합된 결과를 *_integrated.json으로 저장"""
    output_dir = ensure_dir(output_dir)
    safe_name = safe_video_filename(video_id)
    output_path = output_dir / f"captions_{safe_name}_integrated.json"

    payload = {
        "video_id": video_id,
        "total_comments": summary["total_comments"],
        "successful_comments": summary["successful_comments"],
        "failed_comments": summary["failed_comments"],
        "success_rate": summary["success_rate"],
        "comments": results,
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(convert_json_types(payload), fp, ensure_ascii=False, indent=2)

    logger.info("저장 완료: %s", output_path)
    return output_path


def run(
    start_idx: int,
    end_idx: Optional[int] = None,
    device_map: str = "auto",
    device_cycle: Optional[List[str]] = None,
    max_memory: Optional[Dict] = None,
    max_workers_segments: int = 3,
    max_workers_analysis: int = 2,
    ffmpeg_threads: Optional[int] = None,
    skip_videos: Optional[Iterable[str]] = None,
    sequential_ffmpeg: bool = True,
) -> None:
    comments_df, total_comments = read_comments(start_idx, end_idx)
    if comments_df.empty:
        logger.warning("처리할 댓글이 없습니다.")
        return

    end_idx = int(comments_df.index[-1]) + 1
    logger.info("처리 범위: %s ~ %s (%s개)", start_idx, end_idx, len(comments_df))

    grouped = comments_df.groupby("video_id")
    video_ids = list(grouped.groups.keys())

    if skip_videos:
        skip_set = {v.strip() for v in skip_videos if v.strip()}
        video_ids = [v for v in video_ids if v not in skip_set]

    # 이미 처리된 비디오 제거 (*_integrated.json 기준)
    captions_dir = Path(PROJECT_ROOT) / "captions_by_video"
    if captions_dir.exists():
        existing_ids = {
            restore_video_id(Path(f).stem.replace("captions_", "").replace("_integrated", ""))
            for f in os.listdir(captions_dir)
            if f.startswith("captions_") and f.endswith("_integrated.json")
        }
        before = len(video_ids)
        video_ids = [v for v in video_ids if v not in existing_ids]
        if len(video_ids) != before:
            logger.info("이미 처리된 비디오 %s개 제외, 남은 %s개", before - len(video_ids), len(video_ids))
            comments_df = comments_df[comments_df["video_id"].isin(video_ids)]
            grouped = comments_df.groupby("video_id")

    if not video_ids:
        logger.info("새로 처리할 비디오가 없습니다.")
        return

    # Qwen-Omni 모델 초기화
    logger.info("Qwen2.5-Omni 모델 초기화 중...")
    devices: List[str] = []
    if device_cycle:
        for item in device_cycle:
            if isinstance(item, str) and "," in item:
                devices.extend(t.strip() for t in item.split(",") if t.strip())
            elif item:
                devices.append(str(item).strip())
    if not devices:
        devices = [device_map]

    generator_kwargs = {
        "model_path": "Qwen/Qwen2.5-Omni-7B",
        "segment_duration": 6,
        "num_segments": 3,
        "max_batch_size": 3,
        "max_new_tokens": 150,
        "video_fps": 24,
        "ffmpeg_preset": "ultrafast",
        "ffmpeg_crf": 28,
        "use_model_compile": True,
        "use_flash_attention": True,
        "torch_dtype_str": "float16",
        "max_workers_segments": max_workers_segments,
        "max_workers_analysis": max_workers_analysis,
        "ffmpeg_threads": ffmpeg_threads,
        "max_memory": max_memory,
    }

    def make_factory(target_device: str) -> Callable[[], OptimizedMomentQueryGenerator]:
        def _factory(device_name: str = target_device) -> OptimizedMomentQueryGenerator:
            logger.info("GPU %s 모델 초기화...", device_name)
            return OptimizedMomentQueryGenerator(device_map=device_name, **generator_kwargs)
        return _factory

    generator_pool = GeneratorPool(make_factory(device) for device in devices)
    logger.info("Qwen-Omni 초기화 완료 (%s개 인스턴스)", len(generator_pool))

    # GPT 통합기 초기화
    integrator = SegmentIntegrator()

    overall_start = time.time()
    success_total = failure_total = 0

    for idx, video_id in enumerate(video_ids, 1):
        logger.info("====== 비디오 %s/%s: %s ======", idx, len(video_ids), video_id)
        video_comments = grouped.get_group(video_id)

        # Step 5: 캡션 생성 (Qwen-Omni)
        video_results, video_summary = process_video_comments(
            generator_pool=generator_pool,
            video_id=video_id,
            video_comments=video_comments,
            sequential_ffmpeg=sequential_ffmpeg,
        )

        success_total += int(video_summary["successful_comments"])
        failure_total += int(video_summary["failed_comments"])

        # Step 6: 세그먼트 통합 (GPT, in-memory)
        logger.info("세그먼트 통합 시작: %s", video_id)
        integrated_results = integrate_video_results(integrator, video_results)

        # 최종 저장 (*_integrated.json)
        save_integrated_video(video_id, integrated_results, video_summary, captions_dir)

        # GPU 메모리 정리
        try:
            import torch
            if torch.cuda.is_available():
                current = torch.cuda.current_device()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                torch.cuda.set_device(current)
        except Exception:
            pass

    overall_elapsed = time.time() - overall_start
    logger.info("완료 (성공=%s, 실패=%s, 총시간=%.2fs)", success_total, failure_total, overall_elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="캡션 생성 + 세그먼트 통합 (Step 5+6)")
    parser.add_argument("start", type=int, help="시작 댓글 번호 (1부터)")
    parser.add_argument("--end", type=int, help="끝 댓글 번호")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--device-cycle", type=str, help="예: 'cuda:0,cuda:1'")
    parser.add_argument("--max-memory", type=str, help="예: 'cuda:0=38GiB'")
    parser.add_argument("--workers-segments", type=int, default=3)
    parser.add_argument("--workers-analysis", type=int, default=2)
    parser.add_argument("--ffmpeg-threads", type=int)
    parser.add_argument("--skip-video", action="append")
    parser.add_argument("--parallel-ffmpeg", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))
    max_memory = parse_max_memory(args.max_memory)

    skip_videos: List[str] = []
    if args.skip_video:
        for item in args.skip_video:
            skip_videos.extend(t.strip() for t in item.split(",") if t.strip())

    device_cycle = None
    if args.device_cycle:
        device_cycle = [t.strip() for t in args.device_cycle.split(",") if t.strip()]

    run(
        start_idx=args.start,
        end_idx=args.end,
        device_map=args.device_map,
        device_cycle=device_cycle,
        max_memory=max_memory,
        max_workers_segments=max(args.workers_segments, 1),
        max_workers_analysis=max(args.workers_analysis, 1),
        ffmpeg_threads=args.ffmpeg_threads if args.ffmpeg_threads and args.ffmpeg_threads > 0 else None,
        skip_videos=skip_videos or None,
        sequential_ffmpeg=not args.parallel_ffmpeg,
    )


if __name__ == "__main__":
    main()
