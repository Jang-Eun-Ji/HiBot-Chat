# test_hwpx_extraction_fixed.py
# Fixed version of HWPX text extraction

import os
import win32com.client
import tempfile
import zipfile
import xml.etree.ElementTree as ET

NS = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}

def extract_text_from_hwpx_debug(hwpx_path):
    """HWPX 파일에서 텍스트 추출 (디버그 버전)"""
    try:
        print(f"🔍 HWPX 텍스트 추출 시작: {hwpx_path}")
        
        with zipfile.ZipFile(hwpx_path, "r") as z:
            with z.open("Contents/section0.xml") as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
        print(f"✅ XML 파싱 성공, 루트: {root.tag}")
        
        texts = []
        
        # 네임스페이스를 사용하여 텍스트 요소 찾기
        text_elements = root.iter(f"{{{NS['hp']}}}t")
        count = 0
        
        for t in text_elements:
            if t.text and t.text.strip():
                texts.append(t.text.strip())
                count += 1
                if count <= 5:  # 처음 5개 출력
                    print(f"  텍스트 {count}: {t.text.strip()[:50]}")
        
        print(f"📊 총 {count}개의 텍스트 요소 발견")
        
        # 표(table) 처리
        table_count = 0
        for table in root.iter(f"{{{NS['hp']}}}tbl"):
            table_count += 1
            for tr in table.iter(f"{{{NS['hp']}}}tr"):
                row = []
                for tc in tr.iter(f"{{{NS['hp']}}}tc"):
                    cell_texts = [t.text.strip() for t in tc.iter(f"{{{NS['hp']}}}t") if t.text and t.text.strip()]
                    if cell_texts:
                        row.append(" ".join(cell_texts))
                if row:
                    texts.append(" | ".join(row))
        
        print(f"📊 총 {table_count}개의 표 처리")
        
        full_text = "\n".join(texts)
        print(f"✅ 최종 텍스트 길이: {len(full_text)} 문자")
        
        return full_text
        
    except Exception as e:
        print(f"❌ HWPX 텍스트 추출 실패: {e}")
        import traceback
        traceback.print_exc()
        return ""

def convert_hwp_to_hwpx(hwp_path):
    """HWP 파일을 임시 HWPX 파일로 변환"""
    try:
        hwp = win32com.client.Dispatch("HWPFrame.HwpObject")
        
        # 임시 HWPX 파일 경로 생성
        temp_dir = tempfile.gettempdir()
        hwpx_filename = os.path.splitext(os.path.basename(hwp_path))[0] + ".hwpx"
        hwpx_path = os.path.join(temp_dir, hwpx_filename)
        
        print(f"📄 HWP 파일 열기: {hwp_path}")
        
        # HWP 파일 열기
        hwp.Open(hwp_path, "HWP", "forceopen:true")
        
        print(f"💾 HWPX로 변환 중: {hwpx_path}")
        
        # HWPX 형식으로 저장
        hwp.SaveAs(hwpx_path, "HWPX", "version:1.0")
        
        # 문서 닫기
        hwp.XHwpDocuments.Close(isDirty=False)
        hwp.Quit()
        
        return hwpx_path
    except Exception as e:
        print(f"❌ HWP → HWPX 변환 실패 ({hwp_path}): {e}")
        return None

def extract_text_from_hwp(hwp_path):
    """HWP 파일에서 텍스트 추출"""
    # HWP → HWPX 변환
    hwpx_path = convert_hwp_to_hwpx(hwp_path)
    if not hwpx_path:
        return ""
    
    try:
        # HWPX에서 텍스트 추출
        text = extract_text_from_hwpx_debug(hwpx_path)
        
        # 임시 파일 정리
        if os.path.exists(hwpx_path):
            os.remove(hwpx_path)
        
        return text
    except Exception as e:
        print(f"❌ 텍스트 추출 실패: {e}")
        # 임시 파일 정리
        if os.path.exists(hwpx_path):
            os.remove(hwpx_path)
        return ""

def test_hwp_files():
    """HWP 파일 처리 테스트"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "../hibot-chat-docs-hwp")
    
    print("🚀 HWP 파일 처리 테스트 시작 (수정 버전)")
    print("DATA_PATH:", DATA_PATH)
    
    if not os.path.exists(DATA_PATH):
        print("❌ HWP 폴더가 없습니다:", DATA_PATH)
        return
    
    hwp_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".hwp")]
    print(f"📋 총 {len(hwp_files)}개의 HWP 파일 발견")
    
    # 첫 번째 파일만 테스트
    if hwp_files:
        test_file = hwp_files[0]
        print(f"\n🎯 테스트 대상: {test_file}")
        
        hwp_path = os.path.join(DATA_PATH, test_file)
        text = extract_text_from_hwp(hwp_path)
        
        if text:
            print(f"\n✅ 텍스트 추출 성공!")
            print("=" * 50)
            print("추출된 텍스트 미리보기 (처음 1000자):")
            print(text[:1000])
            print("=" * 50)
            
            # 텍스트 파일로 저장
            output_path = os.path.join(BASE_DIR, f"test_output_{test_file.replace('.hwp', '.txt')}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"💾 전체 텍스트를 저장했습니다: {output_path}")
        else:
            print("❌ 텍스트 추출 실패")
    else:
        print("❌ HWP 파일이 없습니다.")

if __name__ == "__main__":
    test_hwp_files()