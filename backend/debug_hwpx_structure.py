# debug_hwpx_structure.py
# Debug script to inspect HWPX file structure

import os
import win32com.client
import tempfile
import zipfile
import xml.etree.ElementTree as ET

def convert_hwp_to_hwpx_debug(hwp_path):
    """HWP 파일을 임시 HWPX 파일로 변환하고 구조 분석"""
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
        
        # HWPX 파일 구조 분석
        debug_hwpx_structure(hwpx_path)
        
        return hwpx_path
    except Exception as e:
        print(f"❌ HWP → HWPX 변환 실패 ({hwp_path}): {e}")
        return None

def debug_hwpx_structure(hwpx_path):
    """HWPX 파일 구조 분석"""
    try:
        print(f"\n🔍 HWPX 파일 구조 분석: {hwpx_path}")
        
        with zipfile.ZipFile(hwpx_path, "r") as z:
            print("📁 ZIP 파일 내용:")
            for name in z.namelist():
                print(f"  - {name}")
            
            # Contents/section0.xml 확인
            if "Contents/section0.xml" in z.namelist():
                print("\n📄 section0.xml 분석:")
                with z.open("Contents/section0.xml") as xml_file:
                    content = xml_file.read().decode('utf-8')
                    print(f"XML 크기: {len(content)} 바이트")
                    print("첫 500자:")
                    print(content[:500])
                    
                    # XML 파싱 시도
                    try:
                        tree = ET.parse(z.open("Contents/section0.xml"))
                        root = tree.getroot()
                        print(f"\nXML 루트 태그: {root.tag}")
                        print(f"XML 네임스페이스: {root.attrib}")
                        
                        # 모든 태그 찾기
                        all_tags = set()
                        for elem in root.iter():
                            all_tags.add(elem.tag)
                        
                        print(f"발견된 태그들: {sorted(list(all_tags))}")
                        
                        # 텍스트 요소 찾기
                        text_elements = []
                        for elem in root.iter():
                            if elem.text and elem.text.strip():
                                text_elements.append((elem.tag, elem.text.strip()[:100]))
                        
                        print(f"\n텍스트가 있는 요소들 ({len(text_elements)}개):")
                        for tag, text in text_elements[:10]:  # 처음 10개만
                            print(f"  {tag}: {text}")
                        
                    except Exception as e:
                        print(f"❌ XML 파싱 실패: {e}")
            else:
                print("❌ Contents/section0.xml 파일이 없습니다.")
                
    except Exception as e:
        print(f"❌ HWPX 구조 분석 실패: {e}")

def test_first_hwp():
    """첫 번째 HWP 파일 테스트"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "../hibot-chat-docs-hwp")
    
    print("🚀 HWP 파일 구조 분석 테스트 시작")
    
    if not os.path.exists(DATA_PATH):
        print("❌ HWP 폴더가 없습니다:", DATA_PATH)
        return
    
    hwp_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(".hwp")]
    
    if hwp_files:
        test_file = hwp_files[0]
        print(f"\n🎯 테스트 대상: {test_file}")
        
        hwp_path = os.path.join(DATA_PATH, test_file)
        hwpx_path = convert_hwp_to_hwpx_debug(hwp_path)
        
        # 임시 파일 정리
        if hwpx_path and os.path.exists(hwpx_path):
            os.remove(hwpx_path)
    else:
        print("❌ HWP 파일이 없습니다.")

if __name__ == "__main__":
    test_first_hwp()