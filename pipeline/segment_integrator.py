#!/usr/bin/env python3
"""
YouTube 비디오 분석 세그먼트 통합 도구
여러 세그먼트를 ChatGPT API로 하나의 통합된 visual/audio 코멘트로 변환

사용법:
1. OpenAI API 키 설정: export OPENAI_API_KEY='your-key-here'
2. 실행: python segment_integrator.py input.json
3. 결과: input_integrated.json 파일 생성

필요한 패키지: pip install openai
"""

import json
from openai import OpenAI
import time
import re
import os
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple

class SegmentIntegrator:
    def __init__(self, api_key: str = None):
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
            raise ValueError("OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY 설정하거나 직접 전달하세요.")
    
    def extract_visual_audio(self, segments: List[Dict]) -> Tuple[List[str], List[str]]:
        """세그먼트에서 visual/audio 정보 분리"""
        visual_parts = []
        audio_parts = []
        
        for segment in segments:
            query = segment.get('query', '')
            time_range = segment.get('time_range', '')
            
            # 문장 단위로 더 세밀하게 분리
            sentences = self.split_into_detailed_sentences(query)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_lower = sentence.lower()
                
                # Visual/Audio 키워드로 분류
                visual_keywords = ['visual', 'visually', 'we can see', 'we can observe', 'shot', 'background', 'appears', 'shows']
                audio_keywords = ['audio', 'speaking', 'saying', 'dialogue', 'music', 'sound']
                
                if any(keyword in sentence_lower for keyword in visual_keywords) and 'audio' not in sentence_lower:
                    visual_parts.append(f"[{time_range}] {sentence}")
                elif any(keyword in sentence_lower for keyword in audio_keywords):
                    audio_parts.append(f"[{time_range}] {sentence}")
                elif not any(keyword in sentence_lower for keyword in audio_keywords):
                    visual_parts.append(f"[{time_range}] {sentence}")
        
        return visual_parts, audio_parts
    
    def split_into_detailed_sentences(self, text: str) -> List[str]:
        """텍스트를 더 세밀하게 문장 단위로 분리 (짧은 여러 개의 문장으로)"""
        if not text:
            return []
        
        # 1단계: 기본 문장 분리 (마침표, 느낌표, 물음표 기준)
        sentences = re.split(r'[.!?]+', text)
        
        # 2단계: 각 문장을 더 세밀하게 분리
        detailed_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 쉼표와 and, but, so, because 등으로 추가 분리
            # 먼저 and, but, so, because 등의 연결어로 분리
            parts = re.split(r'\s+(?:and|but|so|because|while|when|where|if|though|although|however|therefore|moreover|furthermore|additionally|also|then|next|after|before)\s+', sentence, flags=re.IGNORECASE)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                
                # 쉼표로 추가 분리 (하지만 너무 짧으면 합치기)
                comma_parts = re.split(r',\s*', part)
                
                if len(comma_parts) > 1:
                    # 각 쉼표 부분이 너무 짧으면 다음 부분과 합치기
                    combined_parts = []
                    i = 0
                    while i < len(comma_parts):
                        current_part = comma_parts[i].strip()
                        if not current_part:
                            i += 1
                            continue
                        
                        # 현재 부분이 너무 짧으면 다음 부분과 합치기
                        if len(current_part.split()) < 3 and i + 1 < len(comma_parts):
                            next_part = comma_parts[i + 1].strip()
                            if next_part:
                                combined_parts.append(f"{current_part}, {next_part}")
                                i += 2
                            else:
                                combined_parts.append(current_part)
                                i += 1
                        else:
                            combined_parts.append(current_part)
                            i += 1
                    
                    detailed_sentences.extend(combined_parts)
                else:
                    detailed_sentences.append(part)
        
        # 3단계: 너무 짧은 문장들 정리 (3단어 미만은 제거하거나 합치기)
        final_sentences = []
        i = 0
        while i < len(detailed_sentences):
            current = detailed_sentences[i].strip()
            if not current:
                i += 1
                continue
            
            # 너무 짧은 문장이면 다음 문장과 합치기 시도
            if len(current.split()) < 3 and i + 1 < len(detailed_sentences):
                next_sentence = detailed_sentences[i + 1].strip()
                if next_sentence and len(next_sentence.split()) < 5:  # 다음 문장도 짧으면 합치기
                    final_sentences.append(f"{current} {next_sentence}")
                    i += 2
                else:
                    final_sentences.append(current)
                    i += 1
            else:
                final_sentences.append(current)
                i += 1
        
        return [s.strip() for s in final_sentences if s.strip()]
    
    def create_prompt(self, segments: List[Dict]) -> str:
        """ChatGPT API용 프롬프트 생성 - 동적 세그먼트 수 처리"""
        segment_count = len(segments)
        
        # 세그먼트들을 S1, S2, S3, ... 형태로 변환
        segment_texts = []
        for i, segment in enumerate(segments, 1):
            query = segment.get('query', '').strip()
            if query:
                segment_texts.append(f'S{i}: "{query}"')
        
        input_text = "\n".join(segment_texts)
        
        return f"""You are a video segment consolidator.
Your job is to merge multiple consecutive caption segments into one coherent caption
with **many short, detailed sentences** that fully preserve all information.

Output ONLY valid JSON. No explanations.

TASK:
- Input: {segment_count} consecutive captions (S1..S{segment_count}).
- Combine them into one unified description, keeping **every visual and audio detail**.
- Write as **many sentences as needed** to cover all details — even small ones.
- Each sentence must be short and atomic (under 12–15 words).
- Avoid long or complex clauses. One concrete action or observation per sentence.
- Maintain the chronological flow of events across all segments.
- Split the output into:
  - Visual: what is *seen* — people, actions, objects, movement, background, colors, lighting, camera angle, on-screen text.
  - Audio: what is *heard* — speech, tone, background sounds, music, ambient noise, silence, laughter, etc.
- Do NOT summarize or merge different events into one sentence.
- Do NOT include timestamps, durations, or temporal phrases ("then", "afterward").
- Write in present tense, neutral and descriptive tone.

OUTPUT FORMAT:
{
  "visual_caption": "<many short sentences describing only visible content>",
  "audio_caption": "<many short sentences describing only audible content>"
}

INPUT:
{input_text}"""
    
    def call_chatgpt(self, prompt: str) -> Dict:
        """ChatGPT API 호출 및 JSON 파싱"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-5-mini-2025-08-07",
                messages=[
                    {"role": "system", "content": "You are a video caption consolidator. Always output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=1500,
                reasoning_effort="low"
            )
            
            content = response.choices[0].message.content.strip()
            
            # 디버그: 실제 API 응답 확인
            print(f"🔍 API 응답 내용:")
            print(f"'{content}'")
            print(f"🔍 응답 길이: {len(content)}")
            
            # JSON 파싱 시도
            try:
                return json.loads(content)
            except json.JSONDecodeError as je:
                print(f"JSON 파싱 실패: {je}")
                print(f"파싱 실패한 내용: '{content[:200]}...'")
                # JSON이 아닌 경우 기본 구조로 반환
                return {
                    "visual_caption": content,
                    "audio_caption": ""
                }
                
        except Exception as e:
            print(f"API 호출 실패: {e}")
            return {
                "visual_caption": f"통합 실패: {str(e)}",
                "audio_caption": ""
            }
    
    def integrate_comment(self, comment_data: Dict) -> Dict:
        """하나의 코멘트 통합 - 새로운 JSON 방식"""
        segments = comment_data.get('segments', [])
        
        if not segments:
            return comment_data
        
        result = comment_data.copy()
        
        # 새로운 프롬프트로 통합 실행
        prompt = self.create_prompt(segments)
        integration_result = self.call_chatgpt(prompt)
        
        # 결과 저장
        result['integrated_visual'] = integration_result.get('visual_caption', '')
        result['integrated_audio'] = integration_result.get('audio_caption', '')
        
        print(f"✅ 통합 완료 (visual + audio)")
        return result
    
    def process_file(self, input_file: str, output_file: str = None):
        """전체 파일 처리"""
        # 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_integrated.json"
        
        comments = data.get('comments', [])
        print(f"🔄 총 {len(comments)}개 코멘트 처리 시작")
        
        # 각 코멘트 처리
        integrated_comments = []
        skipped_count = 0
        
        for i, comment in enumerate(comments, 1):
            print(f"\n📝 코멘트 {i}/{len(comments)}")
            print(f"내용: {comment.get('comment', '')[:50]}...")
            
            # 이미 처리된 코멘트인지 확인
            if (comment.get('integrated_visual') and comment.get('integrated_audio')):
                print(f"⏭️  이미 처리됨 - 건너뛰기")
                integrated_comments.append(comment)
                skipped_count += 1
                continue
            
            try:
                integrated = self.integrate_comment(comment)
                integrated_comments.append(integrated)
                time.sleep(1)  # API 제한 방지
            except Exception as e:
                print(f"❌ 실패: {e}")
                integrated_comments.append(comment)
        
        # 결과 저장
        result = data.copy()
        result['comments'] = integrated_comments
        result['processing_info'] = {
            'processed_at': datetime.now().isoformat(),
            'original_file': input_file,
            'total_processed': len(integrated_comments),
            'skipped_already_processed': skipped_count
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 완료! 결과: {output_file}")
        print(f"📊 처리 통계: 새로 처리 {len(integrated_comments) - skipped_count}개, 건너뛴 {skipped_count}개")
        return output_file
    
    def process_folder(self, folder_path: str, output_folder: str = None):
        """폴더 내 모든 JSON 파일 일괄 처리"""
        import glob
        
        if output_folder is None:
            output_folder = folder_path + "_integrated"
        
        # 출력 폴더 생성
        os.makedirs(output_folder, exist_ok=True)
        
        # JSON 파일 찾기
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        
        if not json_files:
            print(f"❌ {folder_path}에서 JSON 파일을 찾을 수 없습니다.")
            return
        
        print(f"🔄 {len(json_files)}개 파일 처리 시작")
        
        skipped_files = 0
        for i, json_file in enumerate(json_files, 1):
            filename = os.path.basename(json_file)
            print(f"\n📁 파일 {i}/{len(json_files)}: {filename}")
            
            # 출력 파일명 생성
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_folder, f"{base_name}_integrated.json")
            
            # 이미 처리된 파일인지 확인
            if os.path.exists(output_file):
                print(f"⏭️  이미 처리된 파일 - 건너뛰기: {output_file}")
                skipped_files += 1
                continue
            
            try:
                self.process_file(json_file, output_file)
                print(f"✅ {filename} 처리 완료")
            except Exception as e:
                print(f"❌ {filename} 처리 실패: {e}")
        
        print(f"\n🎉 모든 파일 처리 완료! 결과 폴더: {output_folder}")
        print(f"📊 파일 처리 통계: 새로 처리 {len(json_files) - skipped_files}개, 건너뛴 {skipped_files}개")
        return output_folder

def demo_single():
    """단일 코멘트 데모"""
    print("\n=== 단일 코멘트 데모 ===")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY 환경변수 필요")
        return
    
    try:
        with open('test_moment_query_result_20comments.json', 'r') as f:
            data = json.load(f)
        
        integrator = SegmentIntegrator()
        first_comment = data['comments'][0]
        
        print(f"원본: {first_comment.get('comment', '')}")
        print(f"세그먼트: {len(first_comment.get('segments', []))}개")
        
        result = integrator.integrate_comment(first_comment)
        
        print(f"\n📺 Visual Caption: {result.get('integrated_visual', 'N/A')}")
        print(f"\n🔊 Audio Caption: {result.get('integrated_audio', 'N/A')}")
        
    except Exception as e:
        print(f"데모 실패: {e}")

def main():
    parser = argparse.ArgumentParser(description='세그먼트 통합 도구')
    parser.add_argument('input_path', nargs='?', help='입력 JSON 파일 또는 폴더 경로')
    parser.add_argument('-o', '--output', help='출력 파일 또는 폴더')
    parser.add_argument('-k', '--api-key', help='OpenAI API 키')
    parser.add_argument('--demo', action='store_true', help='데모 실행')
    parser.add_argument('--folder', action='store_true', help='폴더 전체 처리 모드')
    
    args = parser.parse_args()
    
    # 데모 모드
    if args.demo:
        demo_single()
        return
    
    # 입력 경로 확인
    if not args.input_path:
        print("사용법:")
        print("  python segment_integrator.py input.json")
        print("  python segment_integrator.py captions_by_video --folder")
        print("  python segment_integrator.py --demo")
        print("\n환경변수 설정:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        integrator = SegmentIntegrator(api_key=args.api_key)
        
        if args.folder or os.path.isdir(args.input_path):
            # 폴더 처리 모드
            integrator.process_folder(args.input_path, args.output)
        else:
            # 단일 파일 처리 모드
            integrator.process_file(args.input_path, args.output)
            
    except Exception as e:
        print(f"❌ 오류: {e}")

if __name__ == "__main__":
    main()
