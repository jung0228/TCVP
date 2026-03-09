#!/usr/bin/env python3
"""
모달리티 분류 + 쿼리 생성 파이프라인 (Step 7+8)
- Qwen3-Embedding-8B로 모달리티 분류
- GPT API로 moment retrieval 쿼리 생성
- 중간 파일 없이 captions_<video_id>_moment_queries.json 직접 저장
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from modality_gating import CommentSimilarityClassifier
from query_generator import LLMBasedMomentRetrievalQueryGenerator


def process_file(
    classifier: CommentSimilarityClassifier,
    generator: LLMBasedMomentRetrievalQueryGenerator,
    input_file: str,
    output_file: str,
) -> None:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    comments = data.get('comments', [])
    print(f"\n📂 {os.path.basename(input_file)}: {len(comments)}개 댓글 처리 시작")

    stats = {"visual_queries": 0, "audio_queries": 0, "unrelated_queries": 0}
    generated_queries = []

    for i, comment in enumerate(comments, 1):
        print(f"  [{i}/{len(comments)}] 처리 중...")

        # Step 7: 모달리티 분류 (Qwen embedding)
        classified = classifier.classify_comment_data(comment)

        # Step 8: 쿼리 생성 (GPT)
        try:
            query_result = generator.generate_moment_retrieval_query(classified)
            generated_queries.append(query_result)
            qt = query_result.get("query_type", "unrelated")
            if qt == "visual_focused":
                stats["visual_queries"] += 1
            elif qt == "audio_focused":
                stats["audio_queries"] += 1
            else:
                stats["unrelated_queries"] += 1
        except Exception as e:
            print(f"  ❌ 쿼리 생성 실패: {e}")
            continue

    output_data = {
        "generated_queries": generated_queries,
        "processing_info": {
            "input_file": input_file,
            "output_file": output_file,
            "total_comments": len(comments),
            "processed_queries": len(generated_queries),
            "generated_at": datetime.now().isoformat(),
        },
        "statistics": stats,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"  ✅ 저장 완료: {output_file}")
    print(f"  📊 Visual {stats['visual_queries']}, Audio {stats['audio_queries']}, Unrelated {stats['unrelated_queries']}")


def main():
    parser = argparse.ArgumentParser(description="모달리티 분류 + 쿼리 생성 (Step 7+8)")
    parser.add_argument(
        "input", nargs="?",
        default=os.path.join(PROJECT_ROOT, "captions_by_video"),
        help="*_integrated.json 파일 또는 폴더 경로 (기본: captions_by_video/)",
    )
    parser.add_argument("-o", "--output", help="출력 파일 또는 폴더 경로")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3-Embedding-8B", help="임베딩 모델")
    parser.add_argument("--api-key", help="OpenAI API 키")
    args = parser.parse_args()

    # 모델 초기화 (한 번만)
    print("🔄 Qwen3-Embedding 모델 초기화 중...")
    classifier = CommentSimilarityClassifier(model_name=args.model)
    print("🔄 OpenAI 클라이언트 초기화 중...")
    generator = LLMBasedMomentRetrievalQueryGenerator(api_key=args.api_key)

    if os.path.isdir(args.input):
        # 폴더 모드: *_integrated.json 파일 모두 처리
        input_files = glob.glob(os.path.join(args.input, "*_integrated.json"))
        output_dir = args.output or args.input

        if not input_files:
            print(f"❌ {args.input}에서 *_integrated.json 파일을 찾을 수 없습니다.")
            return

        print(f"\n📁 {len(input_files)}개 파일 처리 시작")
        skipped = 0

        for input_file in input_files:
            base = os.path.splitext(os.path.basename(input_file))[0]
            out_name = base.replace("_integrated", "_moment_queries") + ".json"
            output_file = os.path.join(output_dir, out_name)

            if os.path.exists(output_file):
                print(f"⏭️  건너뜀 (이미 존재): {out_name}")
                skipped += 1
                continue

            process_file(classifier, generator, input_file, output_file)

        print(f"\n🎉 완료! 처리 {len(input_files) - skipped}개, 건너뜀 {skipped}개")

    else:
        # 단일 파일 모드
        base = os.path.splitext(args.input)[0]
        output_file = args.output or (base.replace("_integrated", "_moment_queries") + ".json")

        if os.path.exists(output_file):
            print(f"⏭️  이미 존재합니다: {output_file}")
            return

        process_file(classifier, generator, args.input, output_file)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 중단되었습니다.")
        sys.exit(1)
