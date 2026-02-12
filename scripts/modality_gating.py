#!/usr/bin/env python3
"""
댓글-캡션 유사도 기반 분류 시스템
통합된 visual/audio 캡션과 원본 댓글의 유사도를 측정하여 댓글 유형을 분류

사용법:
1. 패키지 설치: pip install sentence-transformers scikit-learn numpy
2. 실행: python comment_similarity_classifier.py input_integrated.json
3. 결과: input_classified.json 파일 생성
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime

class CommentSimilarityClassifier:
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B"):
        """
        댓글 분류기 초기화
        
        Args:
            model_name: 사용할 sentence transformer 모델명
        """
        print(f"🔄 임베딩 모델 로딩 중: {model_name}")
        
        # Qwen3 모델 최적화 설정 (단일 GPU 사용)
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(
                model_name,
                model_kwargs={
                    "attn_implementation": "flash_attention_2", 
                    "torch_dtype": torch.float16,  # fp16 설정
                    "device_map": None  # 자동 분산 비활성화
                },
                tokenizer_kwargs={"padding_side": "left"},
                device=device
            )
            print(f"✅ 모델 로딩 완료 (flash_attention_2 + fp16, {device})")
        except Exception as e:
            print(f"⚠️  flash_attention_2 실패, 기본 설정으로 로딩: {e}")
            try:
                # flash_attention 없이 fp16만 시도
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.model = SentenceTransformer(
                    model_name,
                    model_kwargs={"torch_dtype": torch.float16},
                    device=device
                )
                print(f"✅ 모델 로딩 완료 (fp16, {device})")
            except Exception as e2:
                print(f"⚠️  fp16도 실패, 기본 설정으로 로딩: {e2}")
                self.model = SentenceTransformer(model_name)
                print("✅ 모델 로딩 완료 (기본 설정)")
        
        # 분류 임계값 설정 (적절한 필터링)
        self.relevance_threshold = 0.3  # 영상 관련성 임계값 (적절한 수준)
        self.modality_threshold = 0.05  # visual vs audio 차이 임계값
        self.visual_weight = 1.1  # 시각적 유사도 가중치
    
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리"""
        if not text:
            return ""
        
        # 기본 정리
        text = text.strip()
        
        # 타임스탬프 제거 (예: 21:02, 1:30 등)
        import re
        text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text)
        
        # 이모지 및 특수문자 정리 (필요시)
        # text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분리 (개선된 버전)"""
        import re
        
        if not text:
            return []
        
        # 문장 분리: 구분자를 포함하여 분할 (lookahead 사용)
        # 마침표, 느낌표, 물음표 뒤에 공백이나 문장 끝이 오는 경우만 분할
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # 디버깅: 분할 결과 출력
        print(f"🔍 원본 텍스트: {text[:100]}...")
        print(f"🔍 분할된 문장 수: {len(sentences)}")
        for i, sent in enumerate(sentences[:3]):  # 처음 3개만 출력
            print(f"  문장 {i+1}: '{sent[:50]}...'")
        
        # 각 문장 정리
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:  # 빈 문장만 제거
                continue
            
            # 문장이 마침표로 끝나지 않으면 마침표 추가
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            cleaned_sentences.append(sentence)
        
        print(f"🔍 최종 문장 수: {len(cleaned_sentences)}")
        return [s.strip() for s in cleaned_sentences if s.strip()]
    
    def calculate_similarity(self, comment: str, caption: str) -> float:
        """댓글과 캡션 간의 코사인 유사도 계산 (Qwen3 프롬프트 활용)"""
        if not comment or not caption:
            return 0.0
        
        # 텍스트 전처리
        comment = self.preprocess_text(comment)
        caption = self.preprocess_text(caption)
        
        if not comment or not caption:
            return 0.0
        
        try:
            # Qwen3의 프롬프트 기능 활용
            # comment는 query로, caption은 document로 처리
            comment_embedding = self.model.encode([comment], prompt_name="query")
            caption_embedding = self.model.encode([caption])
            
            # 유사도 계산
            similarity = self.model.similarity(comment_embedding, caption_embedding)[0][0]
            return float(similarity)
            
        except Exception as e:
            print(f"⚠️  프롬프트 기능 실패, 기본 방식 사용: {e}")
            # 프롬프트 실패 시 기본 방식으로 fallback
            embeddings = self.model.encode([comment, caption])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
    
    def create_short_chunks(self, text: str, max_chunk_size: int = 1) -> List[str]:
        """텍스트를 짧은 청크로 분할 (문장 개수 기준)"""
        if not text:
            return []
        
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []
        
        chunks = []
        for i in range(0, len(sentences), max_chunk_size):
            chunk = " ".join(sentences[i:i + max_chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def calculate_sentence_level_similarity(self, comment: str, caption: str) -> Dict:
        """댓글과 캡션을 짧은 청크 단위로 비교하여 최대 유사도와 매칭 청크 반환"""
        if not comment or not caption:
            return {"max_similarity": 0.0, "matched_sentence": "", "all_similarities": []}
        
        # 캡션을 짧은 청크로 분리 (1문장씩)
        caption_chunks = self.create_short_chunks(caption, max_chunk_size=1)
        
        if not caption_chunks:
            return {"max_similarity": 0.0, "matched_sentence": "", "all_similarities": []}
        
        # 각 청크와 댓글 간의 유사도 계산
        similarities = []
        for chunk in caption_chunks:
            sim = self.calculate_similarity(comment, chunk)
            similarities.append({
                "sentence": chunk,
                "similarity": sim
            })
        
        # 최대 유사도와 해당 청크 찾기
        max_sim_data = max(similarities, key=lambda x: x["similarity"])
        
        return {
            "max_similarity": max_sim_data["similarity"],
            "matched_sentence": max_sim_data["sentence"],
            "all_similarities": similarities
        }
    
    def classify_comment_relevance(self, comment: str, visual_caption: str, audio_caption: str) -> Dict:
        """
        댓글의 영상 관련성 및 모달리티 분류 (문장 단위 비교)
        
        Args:
            comment: 원본 댓글
            visual_caption: 통합된 visual 캡션
            audio_caption: 통합된 audio 캡션
            
        Returns:
            Dict: 분류 결과 및 유사도 점수
        """
        # 문장 단위로 각 캡션과의 유사도 계산
        visual_result = self.calculate_sentence_level_similarity(comment, visual_caption)
        audio_result = self.calculate_sentence_level_similarity(comment, audio_caption)
        
        # 최대 유사도 추출 (시각적 유사도에 가중치 적용)
        visual_similarity = visual_result["max_similarity"] * self.visual_weight
        audio_similarity = audio_result["max_similarity"]
        
        # 전체 캡션과의 전반적 유사도 (기존 방식)
        combined_caption = f"{visual_caption} {audio_caption}"
        overall_similarity = self.calculate_similarity(comment, combined_caption)
        
        # 영상 관련성 판단 (더 엄격한 조건)
        max_similarity = max(visual_similarity, audio_similarity, overall_similarity)
        
        # 기본 임계값 + 추가 조건들 (조정된 조건)
        is_video_related = (
            max_similarity >= self.relevance_threshold and
            max_similarity > 0.15 and  # 최소 유사도 보장 (조정)
            (visual_similarity >= 0.1 or audio_similarity >= 0.1)  # 최소 하나는 어느 정도 유사해야 함 (조정)
        )
        
        # 모달리티 분류 (이분법적 분류: visual vs audio)
        modality_type = "unknown"
        matched_sentence = ""
        
        if is_video_related:
            # visual과 audio 중 더 높은 유사도를 가진 쪽으로 분류 (더 엄격한 조건)
            similarity_diff = abs(visual_similarity - audio_similarity)
            
            if visual_similarity > audio_similarity and similarity_diff >= self.modality_threshold:
                modality_type = "visual"
                matched_sentence = visual_result["matched_sentence"]
            elif audio_similarity > visual_similarity and similarity_diff >= self.modality_threshold:
                modality_type = "audio"
                matched_sentence = audio_result["matched_sentence"]
            else:
                # 차이가 명확하지 않으면 더 높은 쪽으로, 그래도 애매하면 visual
                if visual_similarity >= audio_similarity:
                    modality_type = "visual"
                    matched_sentence = visual_result["matched_sentence"]
                else:
                    modality_type = "audio"
                    matched_sentence = audio_result["matched_sentence"]
        else:
            modality_type = "unrelated"
        
        return {
            "is_video_related": is_video_related,
            "modality_type": modality_type,
            "matched_sentence": matched_sentence,
            "similarity_scores": {
                "visual_similarity": round(visual_similarity, 4),
                "visual_similarity_raw": round(visual_result["max_similarity"], 4),
                "audio_similarity": round(audio_similarity, 4),
                "overall_similarity": round(overall_similarity, 4),
                "max_similarity": round(max_similarity, 4)
            },
            "sentence_details": {
                "visual_sentences": len(visual_result["all_similarities"]),
                "audio_sentences": len(audio_result["all_similarities"]),
                "best_visual_match": {
                    "sentence": visual_result["matched_sentence"][:100] + "..." if len(visual_result["matched_sentence"]) > 100 else visual_result["matched_sentence"],
                    "similarity": round(visual_similarity, 4)
                },
                "best_audio_match": {
                    "sentence": audio_result["matched_sentence"][:100] + "..." if len(audio_result["matched_sentence"]) > 100 else audio_result["matched_sentence"],
                    "similarity": round(audio_similarity, 4)
                }
            },
            "confidence": round(max_similarity, 4)
        }
    
    def classify_comment_data(self, comment_data: Dict) -> Dict:
        """하나의 댓글 데이터 분류"""
        comment = comment_data.get('comment', '')
        visual_caption = comment_data.get('integrated_visual', '')
        audio_caption = comment_data.get('integrated_audio', '')
        
        if not comment:
            print("⚠️  빈 댓글 발견")
            return comment_data
        
        if not visual_caption and not audio_caption:
            print("⚠️  캡션 데이터 없음")
            return comment_data
        
        # 분류 실행
        classification = self.classify_comment_relevance(comment, visual_caption, audio_caption)
        
        # 결과를 원본 데이터에 추가
        result = comment_data.copy()
        result['classification'] = classification
        
        # 로그 출력
        comment_preview = comment[:50] + "..." if len(comment) > 50 else comment
        print(f"📝 '{comment_preview}' → {classification['modality_type']} (신뢰도: {classification['confidence']:.3f})")
        
        return result
    
    def process_integrated_file(self, input_file: str, output_file: str = None) -> str:
        """통합된 JSON 파일 전체 처리"""
        # 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_classified.json"
        
        # 이미 처리된 파일인지 확인
        if os.path.exists(output_file):
            print(f"⏭️  이미 분류된 파일 - 건너뛰기: {output_file}")
            return output_file
        
        comments = data.get('comments', [])
        print(f"🔄 총 {len(comments)}개 댓글 분류 시작")
        
        # 분류 통계
        stats = {
            "total": len(comments),
            "video_related": 0,
            "unrelated": 0,
            "visual": 0,
            "audio": 0
        }
        
        # 각 댓글 분류
        classified_comments = []
        skipped_comments = 0
        
        for i, comment in enumerate(comments, 1):
            print(f"\n📊 댓글 {i}/{len(comments)} 처리 중...")
            
            # 이미 분류된 댓글인지 확인
            if comment.get('classification'):
                print(f"⏭️  이미 분류됨 - 건너뛰기")
                classified_comments.append(comment)
                skipped_comments += 1
                
                # 기존 분류 결과로 통계 업데이트
                classification = comment.get('classification', {})
                if classification.get('is_video_related', False):
                    stats["video_related"] += 1
                    modality = classification.get('modality_type', 'unknown')
                    if modality in ['visual', 'audio'] and modality in stats:
                        stats[modality] += 1
                else:
                    stats["unrelated"] += 1
                continue
            
            try:
                classified_comment = self.classify_comment_data(comment)
                classified_comments.append(classified_comment)
                
                # 통계 업데이트
                classification = classified_comment.get('classification', {})
                if classification.get('is_video_related', False):
                    stats["video_related"] += 1
                    # 영상 관련 댓글의 모달리티만 카운트
                    modality = classification.get('modality_type', 'unknown')
                    if modality in ['visual', 'audio'] and modality in stats:
                        stats[modality] += 1
                else:
                    stats["unrelated"] += 1
                    
            except Exception as e:
                print(f"❌ 댓글 {i} 분류 실패: {e}")
                classified_comments.append(comment)
        
        # 결과 저장
        result_data = data.copy()
        result_data['comments'] = classified_comments
        result_data['classification_stats'] = stats
        result_data['classification_metadata'] = {
            'processed_at': datetime.now().isoformat(),
            'model_used': self.model.get_sentence_embedding_dimension(),
            'relevance_threshold': self.relevance_threshold,
            'modality_threshold': self.modality_threshold,
            'total_processed': len(classified_comments)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        # 결과 출력
        print(f"\n🎉 분류 완료! 결과: {output_file}")
        print(f"📊 댓글 처리 통계: 새로 분류 {len(comments) - skipped_comments}개, 건너뛴 {skipped_comments}개")
        print(f"\n📈 분류 통계:")
        print(f"  전체: {stats['total']}")
        print(f"  영상 관련: {stats['video_related']}개 ({stats['video_related']/stats['total']*100:.1f}%)")
        print(f"  영상 무관: {stats['unrelated']}개 ({stats['unrelated']/stats['total']*100:.1f}%)")
        print(f"  Visual: {stats['visual']}개 ({stats['visual']/stats['video_related']*100 if stats['video_related'] > 0 else 0:.1f}%)")
        print(f"  Audio: {stats['audio']}개 ({stats['audio']/stats['video_related']*100 if stats['video_related'] > 0 else 0:.1f}%)")
        
        return output_file
    
    def process_folder(self, folder_path: str, output_folder: str = None):
        """폴더 내 모든 통합된 JSON 파일 일괄 처리"""
        import glob
        
        if output_folder is None:
            output_folder = folder_path + "_classified"
        
        # 출력 폴더 생성
        os.makedirs(output_folder, exist_ok=True)
        
        # 통합된 JSON 파일 찾기 (*_integrated.json)
        integrated_files = glob.glob(os.path.join(folder_path, "*_integrated.json"))
        
        if not integrated_files:
            print(f"❌ {folder_path}에서 *_integrated.json 파일을 찾을 수 없습니다.")
            return
        
        print(f"🔄 {len(integrated_files)}개 통합 파일 분류 시작")
        
        # 분류 통계 (전체)
        total_stats = {
            "total": 0,
            "video_related": 0,
            "unrelated": 0,
            "visual": 0,
            "audio": 0
        }
        
        skipped_files = 0
        for i, json_file in enumerate(integrated_files, 1):
            filename = os.path.basename(json_file)
            print(f"\n📁 파일 {i}/{len(integrated_files)}: {filename}")
            
            # 출력 파일명 생성
            base_name = os.path.splitext(filename)[0].replace("_integrated", "")
            output_file = os.path.join(output_folder, f"{base_name}_classified.json")
            
            # 이미 처리된 파일인지 확인
            if os.path.exists(output_file):
                print(f"⏭️  이미 분류된 파일 - 건너뛰기: {output_file}")
                skipped_files += 1
                continue
            
            try:
                # 파일별 처리
                self.process_integrated_file(json_file, output_file)
                
                # 파일별 통계 누적
                with open(output_file, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                
                file_stats = result_data.get('classification_stats', {})
                for key in total_stats:
                    total_stats[key] += file_stats.get(key, 0)
                
                print(f"✅ {filename} 분류 완료")
                
            except Exception as e:
                print(f"❌ {filename} 분류 실패: {e}")
        
        # 전체 통계 출력
        print(f"\n🎉 모든 파일 분류 완료! 결과 폴더: {output_folder}")
        print(f"📊 파일 처리 통계: 새로 분류 {len(integrated_files) - skipped_files}개, 건너뛴 {skipped_files}개")
        print(f"\n📈 전체 분류 통계:")
        print(f"  전체 댓글: {total_stats['total']}개")
        print(f"  영상 관련: {total_stats['video_related']}개 ({total_stats['video_related']/total_stats['total']*100 if total_stats['total'] > 0 else 0:.1f}%)")
        print(f"  영상 무관: {total_stats['unrelated']}개 ({total_stats['unrelated']/total_stats['total']*100 if total_stats['total'] > 0 else 0:.1f}%)")
        print(f"  Visual: {total_stats['visual']}개 ({total_stats['visual']/total_stats['video_related']*100 if total_stats['video_related'] > 0 else 0:.1f}%)")
        print(f"  Audio: {total_stats['audio']}개 ({total_stats['audio']/total_stats['video_related']*100 if total_stats['video_related'] > 0 else 0:.1f}%)")
        
        return output_folder
    
    def demo_single_comment(self, input_file: str):
        """단일 댓글 분류 데모 (상세 버전)"""
        print("\n=== 단일 댓글 분류 데모 ===")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        first_comment = data['comments'][0]
        
        print(f"\n📝 원본 댓글:")
        print(f"  {first_comment.get('comment', '')}")
        
        visual_caption = first_comment.get('integrated_visual', '')
        audio_caption = first_comment.get('integrated_audio', '')
        
        print(f"\n🎥 Visual 캡션 (전체):")
        print(f"  {visual_caption}")
        
        print(f"\n🎵 Audio 캡션 (전체):")
        print(f"  {audio_caption}")
        
        # 문장 분할 과정 보여주기
        print(f"\n✂️ Visual 캡션 문장 분할 과정:")
        visual_sentences = self.split_into_sentences(visual_caption)
        print(f"  분할된 문장 수: {len(visual_sentences)}")
        for i, sentence in enumerate(visual_sentences, 1):
            print(f"  문장 {i}: {sentence}")
        
        print(f"\n✂️ Audio 캡션 문장 분할 과정:")
        audio_sentences = self.split_into_sentences(audio_caption)
        print(f"  분할된 문장 수: {len(audio_sentences)}")
        for i, sentence in enumerate(audio_sentences, 1):
            print(f"  문장 {i}: {sentence}")
        
        # 청크 분할 보여주기 (1문장씩)
        visual_chunks = self.create_short_chunks(visual_caption, max_chunk_size=1)
        audio_chunks = self.create_short_chunks(audio_caption, max_chunk_size=1)
        
        print(f"\n📦 Visual 청크 (1문장씩, {len(visual_chunks)}개):")
        for i, chunk in enumerate(visual_chunks, 1):
            print(f"  청크 {i}: {chunk}")
        
        print(f"\n📦 Audio 청크 (1문장씩, {len(audio_chunks)}개):")
        for i, chunk in enumerate(audio_chunks, 1):
            print(f"  청크 {i}: {chunk}")
        
        # 각 문장별 유사도 계산
        comment = first_comment.get('comment', '')
        visual_result = self.calculate_sentence_level_similarity(comment, visual_caption)
        audio_result = self.calculate_sentence_level_similarity(comment, audio_caption)
        
        print(f"\n📊 Visual 문장별 유사도:")
        for i, sim_data in enumerate(visual_result['all_similarities'], 1):
            print(f"  문장 {i}. {sim_data['similarity']:.4f} - {sim_data['sentence']}")
        
        print(f"\n📊 Audio 문장별 유사도:")
        for i, sim_data in enumerate(audio_result['all_similarities'], 1):
            print(f"  문장 {i}. {sim_data['similarity']:.4f} - {sim_data['sentence']}")
        
        # 분류 결과
        result = self.classify_comment_data(first_comment)
        classification = result['classification']
        
        print(f"\n🎯 최종 분류 결과:")
        print(f"  영상 관련성: {classification['is_video_related']}")
        print(f"  모달리티: {classification['modality_type']}")
        print(f"  매칭된 문장: {classification['matched_sentence']}")
        print(f"  신뢰도: {classification['confidence']:.4f}")
        
        print(f"\n📈 유사도 점수:")
        for key, score in classification['similarity_scores'].items():
            print(f"  {key}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description='댓글-캡션 유사도 기반 분류 도구')
    parser.add_argument('input_path', nargs='?', help='통합된 JSON 파일 또는 폴더 경로')
    parser.add_argument('-o', '--output', help='출력 파일 또는 폴더 경로')
    parser.add_argument('-m', '--model', default="Qwen/Qwen3-Embedding-8B", help='임베딩 모델명')
    parser.add_argument('--demo', action='store_true', help='단일 댓글 데모 실행')
    parser.add_argument('--folder', action='store_true', help='폴더 전체 처리 모드')
    
    args = parser.parse_args()
    
    if not args.input_path:
        print("사용법:")
        print("  python comment_similarity_classifier.py input_integrated.json")
        print("  python comment_similarity_classifier.py captions_by_video_integrated --folder")
        print("  python comment_similarity_classifier.py input_integrated.json --demo")
        return
    
    try:
        classifier = CommentSimilarityClassifier(model_name=args.model)
        
        if args.demo:
            classifier.demo_single_comment(args.input_path)
        elif args.folder or os.path.isdir(args.input_path):
            # 폴더 처리 모드
            classifier.process_folder(args.input_path, args.output)
        else:
            # 단일 파일 처리 모드
            classifier.process_integrated_file(args.input_path, args.output)
            
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
