#!/usr/bin/env python3
"""
채널의 인기 영상을 찾아주는 스크립트
yt-dlp를 사용하여 채널의 영상 정보를 가져오고 조회수, 좋아요 등으로 정렬합니다.
"""

import sys
import json
import csv
import subprocess
import argparse
import time
import os
from operator import itemgetter

# Project root is parent of scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CSV_DIR = os.path.join(PROJECT_ROOT, 'csv')
MAPPING_FILE = os.path.join(CSV_DIR, 'video_id_mapping.csv')
CHANNEL_CATEGORIES_FILE = os.path.join(CSV_DIR, 'channel_categories.csv')


def get_channel_videos_fast(channel_url, max_videos=None):
    """채널의 기본 영상 정보를 빠르게 가져옵니다 (조회수만)"""
    print(f"📺 1단계: 기본 정보를 빠르게 가져오는 중...")
    start_time = time.time()
    
    # yt-dlp 명령어 (flat-playlist로 빠르게)
    cmd = [
        'yt-dlp',
        '--flat-playlist',
        '--dump-json',
        '--playlist-end', str(max_videos) if max_videos else '999999',
        channel_url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # 각 줄이 하나의 JSON 객체
        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    video_data = json.loads(line)
                    
                    # None 값 처리
                    view_count = video_data.get('view_count')
                    duration = video_data.get('duration')
                    
                    videos.append({
                        'video_id': video_data.get('id', ''),
                        'title': video_data.get('title', 'Unknown'),
                        'views': int(view_count) if view_count is not None else 0,
                        'likes': 0,  # 나중에 채움
                        'comments': 0,  # 나중에 채움
                        'duration': int(duration) if duration is not None else 0,
                        'upload_date': video_data.get('upload_date', 'Unknown')
                    })
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        
        elapsed_time = time.time() - start_time
        print(f"   ⏱️  소요 시간: {elapsed_time:.2f}초 ({len(videos)}개 영상, 평균 {elapsed_time/len(videos)*100:.2f}초/100개)")
        
        return videos
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 오류 발생: {e}")
        return []


def get_video_details(video_ids):
    """특정 영상들의 상세 정보를 가져옵니다 (좋아요, 댓글 수)"""
    print(f"📊 2단계: 상위 {len(video_ids)}개 영상의 상세 정보를 가져오는 중...")
    start_time = time.time()
    
    details = {}
    for idx, video_id in enumerate(video_ids, 1):
        elapsed = time.time() - start_time
        avg_time = elapsed / idx if idx > 0 else 0
        remaining = avg_time * (len(video_ids) - idx)
        print(f"   진행: {idx}/{len(video_ids)} - {video_id} (예상 남은 시간: {remaining:.0f}초)", end='\r')
        
        cmd = [
            'yt-dlp',
            '--skip-download',
            '--dump-json',
            '--no-warnings',
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            video_data = json.loads(result.stdout)
            
            # None 값 처리
            like_count = video_data.get('like_count')
            comment_count = video_data.get('comment_count')
            
            details[video_id] = {
                'likes': int(like_count) if like_count is not None else 0,
                'comments': int(comment_count) if comment_count is not None else 0
            }
        except Exception:
            details[video_id] = {'likes': 0, 'comments': 0}
    
    elapsed_time = time.time() - start_time
    print()  # 줄바꿈
    print(f"   ⏱️  소요 시간: {elapsed_time:.2f}초 ({len(video_ids)}개 영상, 평균 {elapsed_time/len(video_ids):.2f}초/개, {elapsed_time/len(video_ids)*100:.2f}초/100개)")
    
    return details


def get_channel_videos(channel_url, max_videos=None, top_n=50):
    """채널의 영상 정보를 2단계로 가져옵니다"""
    # 1단계: 모든 영상의 기본 정보 (빠름)
    videos = get_channel_videos_fast(channel_url, max_videos)
    
    if not videos:
        return []
    
    print(f"✅ 1단계 완료: {len(videos)}개 영상의 기본 정보를 가져왔습니다.")
    
    # 조회수 기준으로 정렬
    videos_sorted = sorted(videos, key=lambda x: x['views'], reverse=True)
    
    # 상위 N개만 선택 (기본 50개)
    top_videos = videos_sorted[:min(top_n, len(videos_sorted))]
    top_video_ids = [v['video_id'] for v in top_videos]
    
    # 2단계: 상위 N개의 상세 정보 (좋아요, 댓글 수)
    details = get_video_details(top_video_ids)
    
    # 상세 정보 업데이트
    no_comment_count = 0
    for video in videos:
        if video['video_id'] in details:
            video['likes'] = details[video['video_id']]['likes']
            video['comments'] = details[video['video_id']]['comments']
            
            # 댓글 없는 영상 카운트
            if video['comments'] == 0 and video['video_id'] in top_video_ids:
                no_comment_count += 1
    
    print(f"✅ 2단계 완료: 상위 {len(top_video_ids)}개 영상의 상세 정보를 가져왔습니다.")
    if no_comment_count > 0:
        print(f"⚠️  댓글이 없는 영상: {no_comment_count}개")
    print()
    
    return videos


def format_number(num):
    """숫자를 읽기 쉬운 형식으로 변환 (예: 1,234,567)"""
    return f"{num:,}"


def format_duration(seconds):
    """초를 시간 형식으로 변환 (예: 1:23:45)"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def display_videos(videos, sort_by='views', top_n=10):
    """영상 목록을 정렬하여 표시"""
    
    # 정렬
    if sort_by == 'views':
        videos_sorted = sorted(videos, key=itemgetter('views'), reverse=True)
        print(f"\n🔥 조회수 기준 상위 {top_n}개 영상:\n")
    elif sort_by == 'likes':
        videos_sorted = sorted(videos, key=itemgetter('likes'), reverse=True)
        print(f"\n👍 좋아요 기준 상위 {top_n}개 영상:\n")
    elif sort_by == 'ratio':
        # 좋아요/조회수 비율
        for v in videos:
            v['like_ratio'] = v['likes'] / v['views'] if v['views'] > 0 else 0
        videos_sorted = sorted(videos, key=itemgetter('like_ratio'), reverse=True)
        print(f"\n📈 좋아요 비율 기준 상위 {top_n}개 영상:\n")
    else:
        videos_sorted = videos
    
    # 상위 N개만 선택
    top_videos = videos_sorted[:top_n]
    
    # 테이블 형식으로 출력
    print("=" * 140)
    print(f"{'순위':<4} {'Video ID':<15} {'조회수':<12} {'좋아요':<10} {'댓글':<10} {'길이':<8} {'제목':<50}")
    print("=" * 140)
    
    for idx, video in enumerate(top_videos, 1):
        video_id = video['video_id']
        title = video['title'][:47] + '...' if len(video['title']) > 50 else video['title']
        views = format_number(video['views'])
        likes = format_number(video['likes'])
        comments = format_number(video['comments'])
        duration = format_duration(video['duration'])
        
        print(f"{idx:<4} {video_id:<15} {views:<12} {likes:<10} {comments:<10} {duration:<8} {title:<50}")
    
    print("=" * 140)
    
    return top_videos


def save_to_mapping(videos, channel_name, category, min_count=1):
    """csv/video_id_mapping.csv에 추가"""
    import csv
    import os
    
    os.makedirs(CSV_DIR, exist_ok=True)
    mapping_file = MAPPING_FILE
    
    # 기존 video_id와 채널별 카운트 읽기
    existing_ids = set()
    channel_count = {}
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row['video_id'])
                ch_name = row['channel_name']
                channel_count[ch_name] = channel_count.get(ch_name, 0) + 1
    
    # 이미 해당 채널의 영상이 min_count개 이상 있으면 스킵
    current_count = channel_count.get(channel_name, 0)
    if current_count >= min_count:
        print(f"\n⏭️  이미 {channel_name} 채널의 영상이 {current_count}개 있습니다. 건너뜁니다.")
        return
    
    # 새로운 영상만 추가
    new_videos = [v for v in videos if v['video_id'] not in existing_ids]
    
    if not new_videos:
        print(f"\n⚠️  추가할 새로운 영상이 없습니다. (현재 {current_count}개)")
        return
    
    # 파일에 추가
    with open(mapping_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for video in new_videos:
            writer.writerow([video['video_id'], channel_name, category])
    
    print(f"\n✅ {len(new_videos)}개의 새로운 영상이 {mapping_file}에 추가되었습니다. (총 {current_count + len(new_videos)}개)")


def check_channel_count(channel_name, min_count=1):
    """채널의 기존 영상 개수 확인"""
    import os
    
    mapping_file = MAPPING_FILE
    
    if not os.path.exists(mapping_file):
        return 0
    
    channel_count = 0
    with open(mapping_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['channel_name'] == channel_name:
                channel_count += 1
    
    return channel_count


def process_single_channel(channel, top_n, sort_by, max_videos, save, channel_name, category):
    """단일 채널 처리"""
    # 저장 모드일 때 이미 충분한 영상이 있으면 스킵
    if save:
        current_count = check_channel_count(channel_name)
        if current_count >= 1:
            print(f"⏭️  이미 {channel_name} 채널의 영상이 {current_count}개 있습니다. 건너뜁니다.\n")
            return True
    
    # 채널 URL 형식 확인
    channel_url = channel
    if not channel_url.startswith('http'):
        if channel_url.startswith('@'):
            # 공백을 제거하거나 URL 인코딩
            channel_handle = channel_url.replace(' ', '')
            channel_url = f"https://www.youtube.com/{channel_handle}/videos"
        else:
            # 공백을 제거하거나 URL 인코딩
            channel_handle = channel.replace(' ', '')
            channel_url = f"https://www.youtube.com/@{channel_handle}/videos"
    
    print(f"🔍 채널 URL: {channel_url}")
    
    # 영상 정보 가져오기
    videos = get_channel_videos(channel_url, max_videos)
    
    if not videos:
        print("❌ 영상을 가져올 수 없습니다.")
        return False
    
    print(f"✅ 총 {len(videos)}개의 영상 정보를 가져왔습니다.\n")
    
    # 영상 표시
    top_videos = display_videos(videos, sort_by, top_n)
    
    # 매핑 파일에 저장
    if save:
        if not channel_name or not category:
            print("\n⚠️  --channel-name과 --category를 함께 지정해야 합니다.")
            response = input("채널명을 입력하세요: ")
            channel_name = response.strip()
            response = input("카테고리를 입력하세요 (sport/gaming/comedy/entertainment/talk show/podcast/making): ")
            category = response.strip()
        
        save_to_mapping(top_videos, channel_name, category)
    
    return True


def process_batch_channels(top_n, sort_by, type_filter=None, category_filter=None, channel_filter=None):
    """channel_categories.csv의 여러 채널을 배치 처리"""
    import os
    
    # channel_categories.csv 읽기
    if not os.path.exists(CHANNEL_CATEGORIES_FILE):
        print(f"❌ channel_categories.csv 파일을 찾을 수 없습니다. (경로: {CHANNEL_CATEGORIES_FILE})")
        return
    
    channels = []
    with open(CHANNEL_CATEGORIES_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 필터링
            if type_filter and row['type'] != type_filter:
                continue
            if category_filter and row['category'] != category_filter:
                continue
            if channel_filter and row['channel_name'] not in channel_filter:
                continue
            
            channels.append({
                'channel_name': row['channel_name'],
                'category': row['category'],
                'type': row['type']
            })
    
    if not channels:
        print("❌ 조건에 맞는 채널이 없습니다.")
        return
    
    print(f"📋 총 {len(channels)}개 채널의 인기 영상을 찾습니다.")
    print(f"⚙️  설정: 채널당 상위 {top_n}개, {sort_by} 기준\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, channel in enumerate(channels, 1):
        print(f"\n{'#'*80}")
        print(f"진행: [{idx}/{len(channels)}]")
        print(f"채널: {channel['channel_name']} | 카테고리: {channel['category']}")
        print(f"{'#'*80}")
        
        success = process_single_channel(
            f"@{channel['channel_name']}",
            top_n,
            sort_by,
            None,  # max_videos
            True,  # save
            channel['channel_name'],
            channel['category']
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
        
        # API 제한 고려
        if idx < len(channels):
            import time
            print("\n⏳ 잠시 대기 중...")
            time.sleep(3)
    
    # 최종 결과
    print(f"\n{'='*80}")
    print("🎉 배치 작업 완료!")
    print(f"✅ 성공: {success_count}/{len(channels)}개 채널")
    print(f"❌ 실패: {fail_count}/{len(channels)}개 채널")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='채널의 인기 영상을 찾습니다')
    parser.add_argument('channel', nargs='?', help='채널 URL 또는 채널 ID (@channelname 또는 URL)')
    parser.add_argument('--top', type=int, default=10, help='상위 몇 개 영상을 표시할지 (기본: 10)')
    parser.add_argument('--sort', choices=['views', 'likes', 'ratio'], default='views',
                        help='정렬 기준: views(조회수), likes(좋아요), ratio(좋아요 비율)')
    parser.add_argument('--max', type=int, help='가져올 최대 영상 수 (기본: 모든 영상)')
    parser.add_argument('--save', action='store_true', help='결과를 video_id_mapping.csv에 추가')
    parser.add_argument('--channel-name', help='매핑 파일에 저장할 채널명 (--save와 함께 사용)')
    parser.add_argument('--category', help='매핑 파일에 저장할 카테고리 (--save와 함께 사용)')
    
    # 배치 처리 옵션
    parser.add_argument('--batch', action='store_true', help='channel_categories.csv의 모든 채널 처리')
    parser.add_argument('--type', choices=['audio', 'visual', 'mixed'], help='특정 타입만 처리 (배치 모드)')
    parser.add_argument('--category-filter', help='특정 카테고리만 처리 (배치 모드)')
    parser.add_argument('--channels', nargs='+', help='특정 채널만 처리 (배치 모드)')
    
    args = parser.parse_args()
    
    # 배치 모드
    if args.batch:
        process_batch_channels(
            args.top,
            args.sort,
            args.type,
            args.category_filter,
            args.channels
        )
        return
    
    # 단일 채널 모드
    if not args.channel:
        parser.error('채널을 지정하거나 --batch 옵션을 사용하세요')
    
    process_single_channel(
        args.channel,
        args.top,
        args.sort,
        args.max,
        args.save,
        args.channel_name,
        args.category
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        sys.exit(1)
