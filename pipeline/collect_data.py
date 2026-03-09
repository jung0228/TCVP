#!/usr/bin/env python3
"""
데이터 수집 파이프라인 (Step 1+2+3)
영상 탐색 → 댓글 크롤링 → 필터링/병합
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import find_popular_videos as fpv
import crawl_comments as cc
from yt_merge_with_dedup_lang import merge_filtered_files_with_dedup_lang


def main():
    parser = argparse.ArgumentParser(description='데이터 수집 파이프라인 (Step 1+2+3)')
    parser.add_argument('--top', type=int, default=10, help='채널당 상위 영상 수 (기본: 10)')
    parser.add_argument('--workers', type=int, default=5, help='댓글 크롤링 워커 수 (기본: 5)')
    parser.add_argument('--languages', nargs='+', default=['en'], help='필터링 언어 (기본: en)')
    parser.add_argument('--top-comments', type=int, default=20, help='영상당 상위 댓글 수 (기본: 20)')
    parser.add_argument('--channel', help='단일 채널 처리 (예: @TED)')
    parser.add_argument('--channel-name', help='채널명 (--channel과 함께 사용)')
    parser.add_argument('--category', help='카테고리 (--channel과 함께 사용)')
    args = parser.parse_args()

    comments_dir = os.path.join(PROJECT_ROOT, 'Comments')

    # Step 1: 인기 영상 탐색
    print('\n' + '='*60)
    print('Step 1/3: 인기 영상 탐색')
    print('='*60)
    if args.channel:
        fpv.process_single_channel(
            args.channel, args.top, 'views', None, True,
            args.channel_name, args.category
        )
    else:
        fpv.process_batch_channels(args.top, 'views')

    # Step 2: 댓글 크롤링
    print('\n' + '='*60)
    print('Step 2/3: 댓글 크롤링')
    print('='*60)
    os.makedirs(comments_dir, exist_ok=True)
    top5_videos = cc.get_top5_per_channel()
    total = len(top5_videos)
    success = skip = fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for idx, video in enumerate(top5_videos, 1):
            futures.append(executor.submit(cc.process_video, video, comments_dir, None, total, idx))
            time.sleep(0.5)
        for future in as_completed(futures):
            r = future.result()
            if r['status'] == 'success':
                success += 1
            elif r['status'] == 'skip':
                skip += 1
            else:
                fail += 1

    print(f'\n댓글 크롤링 완료: 성공 {success}, 건너뜀 {skip}, 실패 {fail}')

    # Step 3: 병합 및 필터링
    print('\n' + '='*60)
    print('Step 3/3: 댓글 병합 및 필터링')
    print('='*60)
    merge_filtered_files_with_dedup_lang(target_languages=args.languages, top_n=args.top_comments)

    print('\n✅ 데이터 수집 완료!')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n⚠️ 중단되었습니다.')
        sys.exit(1)
