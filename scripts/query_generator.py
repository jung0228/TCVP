#!/usr/bin/env python3
"""
Moment Retrieval Query Generator (LLM-based)

이 스크립트는 분류된 댓글 데이터를 기반으로 ChatGPT API를 사용하여 
moment retrieval을 위한 지능적이고 정교한 검색 쿼리를 생성합니다. 
모달리티 타입에 따라 특화된 쿼리를 생성하여 긴 영상에서 정확한 순간을 찾을 수 있도록 합니다.
"""

import json
import re
import os
import glob
from typing import Dict, List, Any, Optional
from datetime import datetime

# OpenAI 임포트 (설치되지 않은 경우를 위한 처리)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("경고: OpenAI 라이브러리가 설치되지 않았습니다. 모킹 모드로 실행됩니다.")

class LLMBasedMomentRetrievalQueryGenerator:
    def __init__(self, api_key: str = None):
        """OpenAI API 클라이언트 초기화"""
        if not OPENAI_AVAILABLE:
            raise ValueError("OpenAI 라이브러리가 설치되지 않았습니다. 'pip install openai'로 설치하세요.")
            
        if api_key:
            self.client = OpenAI(
                base_url="https://api.openai.com/v1",
                api_key=api_key
            )
        elif os.getenv('OPENAI_API_KEY'):
            self.client = OpenAI(
                base_url="https://api.openai.com/v1",
                api_key=os.getenv('OPENAI_API_KEY')
            )
        else:
            raise ValueError("OpenAI API 키가 필요합니다. --api-key 옵션이나 OPENAI_API_KEY 환경변수를 설정하세요.")

    def extract_comment_context(self, comment_data: Dict[str, Any]) -> Dict[str, str]:
        """댓글에서 컨텍스트 정보를 추출합니다."""
        comment_text = comment_data.get("comment", "")
        
        # 시간 정보 추출 (예: "2:05", "3:47")
        timestamp_match = re.search(r'(\d+):(\d+)', comment_text)
        timestamp = None
        if timestamp_match:
            minutes, seconds = timestamp_match.groups()
            timestamp = f"{minutes}:{seconds}"
        
        # 특별한 표현이나 인용구 추출
        quotes = re.findall(r'"([^"]*)"', comment_text)
        
        # ASR 텍스트 추출 (integrated_audio에서 음성 내용 추출)
        integrated_audio = comment_data.get("integrated_audio", "")
        asr_text = "N/A"
        
        # 음성 관련 키워드가 있는지 확인
        speech_indicators = ["says", "speaking", "voice", "talking", "commentator", "narrator"]
        if any(indicator in integrated_audio.lower() for indicator in speech_indicators):
            # 실제 음성 내용 추출 시도
            speech_patterns = [
                r'says[^"]*"([^"]*)"',  # "says 'something'"
                r'voice[^"]*"([^"]*)"',  # "voice 'something'"
                r'speaking[^"]*"([^"]*)"',  # "speaking 'something'"
            ]
            
            for pattern in speech_patterns:
                match = re.search(pattern, integrated_audio, re.IGNORECASE)
                if match:
                    asr_text = match.group(1)
                    break
        
        return {
            "timestamp": timestamp,
            "quotes": quotes,
            "asr_text": asr_text,
            "raw_comment": comment_text
        }

    def generate_llm_query(self, comment_data: Dict[str, Any], context: Dict[str, str]) -> str:
        """LLM을 사용하여 지능적인 moment retrieval 쿼리를 생성합니다."""
        modality_type = comment_data.get("classification", {}).get("modality_type", "unrelated")
        integrated_visual = comment_data.get("integrated_visual", "")
        integrated_audio = comment_data.get("integrated_audio", "")
        comment_text = comment_data.get("comment", "")
        confidence = comment_data.get("classification", {}).get("confidence", 0.0)
        
        # LLM 프롬프트 구성
        system_prompt = """You are a multimodal retrieval expert aligning user comments with precise video moments.

GOAL
Generate retrieval-ready search queries that reflect **modality-specific cues**.
Every query must include explicit sensory signals (visual or auditory) that can be objectively detected.
Use ASR speech data when available to identify what was said or how it sounded.

HARD CONSTRAINTS
- Never include timestamps, durations, or ordinal/temporal words.
- The query must be ≥28 words (or ≥220 characters) and richly descriptive.
- The query must include at least **three explicit cues from the target modality**:
  - If modality == "visual":
    - Describe visible entities (people, facial expression, clothing, color, lighting, camera angle, movement, or spatial composition).
    - Mention concrete scene elements (background, props, gestures, camera motion).
    - Do NOT refer to audio, dialogue, or sounds.
  - If modality == "audio":
    - Describe what is *heard*: speech content (from ASR), speaker tone, background music, environmental sounds, crowd or laughter, rhythm, or silence.
    - Include actual or paraphrased words spoken if ASR text is available.
    - Do NOT refer to visual appearance or scene composition.

INTEGRATE SPEECH (for audio modality)
- If ASR text exists, integrate naturally in quotes.
- Example: "a man says 'we finally made it' in a relieved tone over soft rainfall."

OUTPUT FORMAT (JSON)
{
  "modality": "visual" | "audio",
  "query_long": "detailed retrieval query (timestamp-free, modality-pure, ≥28 words)",
  "keywords": ["core search terms"],
  "speech_reference": "actual words or paraphrase from ASR (only for audio)",
  "negatives": ["exclude confusing or similar moments"],
  "rationale": "why these modality-specific cues localize the moment precisely",
  "self_check": {
    "length_ok": true|false,
    "timestamp_free": true|false,
    "modality_pure": true|false,
    "specificity_ok": true|false
  }
}"""

        user_prompt = f"""INPUT CONTEXT
Comment data:
- Comment: "{comment_text}"
- Modality: {modality_type}
- Confidence: {confidence:.3f}

Video analysis:
- Visual: {integrated_visual}
- Audio: {integrated_audio}
- ASR (speech transcript): {context.get('asr_text', 'N/A')}

Quotes (semantic anchors): {context.get('quotes', [])}

TASK
Generate a {modality_type}-specific retrieval query following all constraints above.
Ensure the description relies solely on that modality's observable or audible evidence."""

        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            # JSON 응답 파싱
            response_text = response.choices[0].message.content.strip()
            try:
                import json
                parsed_response = json.loads(response_text)
                return parsed_response
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 텍스트 그대로 반환
                return {
                    "modality": modality_type,
                    "query_long": response_text,
                    "keywords": [],
                    "speech_reference": "",
                    "negatives": [],
                    "rationale": "Failed to parse JSON response",
                    "self_check": {
                        "length_ok": False,
                        "timestamp_free": False,
                        "modality_pure": False,
                        "specificity_ok": False
                    }
                }
            
        except Exception as e:
            print(f"LLM 쿼리 생성 오류: {e}")
            return {
                "modality": modality_type,
                "query_long": f"Error generating query: {str(e)}",
                "keywords": [],
                "speech_reference": "",
                "negatives": [],
                "rationale": "Error occurred during generation",
                "self_check": {
                    "length_ok": False,
                    "timestamp_free": False,
                    "modality_pure": False,
                    "specificity_ok": False
                }
            }



    def generate_moment_retrieval_query(self, comment_data: Dict[str, Any]) -> Dict[str, Any]:
        """댓글 데이터를 기반으로 LLM을 사용하여 moment retrieval 쿼리를 생성합니다."""
        context = self.extract_comment_context(comment_data)
        modality_type = comment_data.get("classification", {}).get("modality_type", "unrelated")
        
        # 기본 정보
        result = {
            "comment_index": comment_data.get("comment_index"),
            "video_id": comment_data.get("video_id"),
            "comment": comment_data.get("comment"),
            "timestamp": context["timestamp"],
            "modality_type": modality_type,
            "confidence": comment_data.get("classification", {}).get("confidence", 0.0),
            "generated_at": datetime.now().isoformat()
        }
        
        # LLM을 사용한 쿼리 생성
        if modality_type == "unrelated":
            result["query"] = "This comment is unrelated to video content, making moment retrieval difficult."
            result["query_type"] = "unrelated"
            result["llm_response"] = {
                "modality": "unrelated",
                "query_long": "Unrelated content",
                "keywords": [],
                "speech_reference": "",
                "negatives": [],
                "rationale": "Comment not related to video content",
                "self_check": {
                    "length_ok": False,
                    "timestamp_free": True,
                    "modality_pure": False,
                    "specificity_ok": False
                }
            }
        else:
            llm_response = self.generate_llm_query(comment_data, context)
            result["query"] = llm_response.get("query_long", "Query generation failed")
            result["query_type"] = f"{modality_type}_focused"
            result["llm_response"] = llm_response
        
        # 추가 메타데이터
        result["metadata"] = {
            "original_confidence": comment_data.get("classification", {}).get("confidence", 0.0),
            "similarity_scores": comment_data.get("classification", {}).get("similarity_scores", {}),
            "matched_sentence": comment_data.get("classification", {}).get("matched_sentence", ""),
            "context_quotes": context["quotes"]
        }
        
        return result

    def process_file(self, input_file: str, output_file: str):
        """JSON 파일을 처리하여 moment retrieval 쿼리를 생성합니다."""
        print(f"입력 파일 읽는 중: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        comments = data.get("comments", [])
        print(f"총 {len(comments)}개의 댓글을 처리합니다.")
        
        # 각 댓글에 대해 쿼리 생성
        generated_queries = []
        for i, comment in enumerate(comments):
            print(f"처리 중: {i+1}/{len(comments)} - 댓글 인덱스 {comment.get('comment_index', 'N/A')}")
            
            try:
                query_result = self.generate_moment_retrieval_query(comment)
                generated_queries.append(query_result)
            except Exception as e:
                print(f"오류 발생 (댓글 {comment.get('comment_index', 'N/A')}): {e}")
                continue
        
        # 결과 저장
        output_data = {
            "generated_queries": generated_queries,
            "processing_info": {
                "input_file": input_file,
                "output_file": output_file,
                "total_comments": len(comments),
                "processed_queries": len(generated_queries),
                "generated_at": datetime.now().isoformat()
            },
            "statistics": {
                "visual_queries": len([q for q in generated_queries if q["query_type"] == "visual_focused"]),
                "audio_queries": len([q for q in generated_queries if q["query_type"] == "audio_focused"]),
                "unrelated_queries": len([q for q in generated_queries if q["query_type"] == "unrelated"])
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"결과 저장 완료: {output_file}")
        print(f"처리된 쿼리 수: {len(generated_queries)}")
        print(f"통계: Visual {output_data['statistics']['visual_queries']}, "
              f"Audio {output_data['statistics']['audio_queries']}, "
              f"Unrelated {output_data['statistics']['unrelated_queries']}")
        
        return output_data

    def process_folder(self, input_folder: str, output_folder: str = None, pattern: str = "*_classified.json"):
        """폴더 내의 모든 분류된 JSON 파일을 처리합니다."""
        if output_folder is None:
            output_folder = input_folder
        
        # 입력 폴더에서 패턴에 맞는 파일들 찾기
        search_pattern = os.path.join(input_folder, pattern)
        input_files = glob.glob(search_pattern)
        
        if not input_files:
            print(f"패턴 '{pattern}'에 맞는 파일을 찾을 수 없습니다: {input_folder}")
            return
        
        print(f"발견된 파일 수: {len(input_files)}")
        
        all_results = []
        
        for input_file in input_files:
            print(f"\n처리 중: {input_file}")
            
            # 출력 파일명 생성
            base_name = os.path.basename(input_file)
            name_without_ext = os.path.splitext(base_name)[0]
            output_file = os.path.join(output_folder, f"{name_without_ext}_moment_queries.json")
            
            try:
                result = self.process_file(input_file, output_file)
                all_results.append(result)
                print(f"완료: {output_file}")
            except Exception as e:
                print(f"파일 처리 오류 ({input_file}): {e}")
                continue
        
        # 전체 통계 출력
        if all_results:
            total_visual = sum(r["statistics"]["visual_queries"] for r in all_results)
            total_audio = sum(r["statistics"]["audio_queries"] for r in all_results)
            total_unrelated = sum(r["statistics"]["unrelated_queries"] for r in all_results)
            total_processed = sum(r["processing_info"]["processed_queries"] for r in all_results)
            
            print(f"\n=== 전체 처리 결과 ===")
            print(f"처리된 파일 수: {len(all_results)}")
            print(f"총 처리된 쿼리 수: {total_processed}")
            print(f"Visual 쿼리: {total_visual}")
            print(f"Audio 쿼리: {total_audio}")
            print(f"Unrelated 쿼리: {total_unrelated}")
        
        return all_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM 기반 Moment Retrieval Query Generator")
    parser.add_argument("--input", "-i", help="입력 파일 또는 폴더 경로")
    parser.add_argument("--output", "-o", help="출력 파일 또는 폴더 경로")
    parser.add_argument("--folder", "-f", action="store_true", help="폴더 처리 모드")
    parser.add_argument("--pattern", "-p", default="*_classified.json", help="파일 패턴 (폴더 모드에서 사용)")
    parser.add_argument("--api-key", help="OpenAI API 키 (선택사항)")
    
    args = parser.parse_args()
    
    # API 키 설정
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not OPENAI_AVAILABLE:
        print("OpenAI 라이브러리가 설치되지 않았습니다. 'pip install openai'로 설치하세요.")
        return
    elif not api_key:
        print("OpenAI API 키가 필요합니다. --api-key 옵션이나 OPENAI_API_KEY 환경변수를 설정하세요.")
        return
    
    generator = LLMBasedMomentRetrievalQueryGenerator(api_key=api_key)
    
    if args.folder:
        # 폴더 처리 모드
        input_folder = args.input or "/home/elicer/yt_dataset/final_pipeline"
        output_folder = args.output or input_folder
        
        print(f"폴더 처리 모드: {input_folder}")
        print(f"출력 폴더: {output_folder}")
        print(f"파일 패턴: {args.pattern}")
        
        generator.process_folder(input_folder, output_folder, args.pattern)
    else:
        # 단일 파일 처리 모드
        input_file = args.input or "/home/elicer/yt_dataset/final_pipeline/test_comments_raw_integrated_short_classified.json"
        output_file = args.output or "/home/elicer/yt_dataset/final_pipeline/moment_retrieval_queries_llm.json"
        
        print(f"단일 파일 처리 모드: {input_file}")
        print(f"출력 파일: {output_file}")
        
        generator.process_file(input_file, output_file)

if __name__ == "__main__":
    main()
