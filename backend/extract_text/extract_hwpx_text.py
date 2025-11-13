import os
import zipfile
import xml.etree.ElementTree as ET
from tqdm import tqdm

NS = {"hp": "http://www.hancom.co.kr/hwpml/2011/paragraph"}

def extract_text_from_hwpx(hwpx_path):
    """정상 ZIP 기반 HWPX 파일에서 텍스트+표 추출"""
    try:
        with zipfile.ZipFile(hwpx_path, "r") as z:
            # 대부분의 본문은 Contents/section0.xml 안에 있습니다
            with z.open("Contents/section0.xml") as xml_file:
                tree = ET.parse(xml_file)
                root = tree.getroot()
    except Exception as e:
        print(f"❌ {hwpx_path} 파싱 실패: {e}")
        return ""

    texts = []

    # 문단 추출
    for p in root.iter(f"{{{NS['hp']}}}p"):
        t_list = [t.text.strip() for t in p.iter(f"{{{NS['hp']}}}t") if t.text]
        if t_list:
            texts.append("".join(t_list))

    # 표 추출
    for table in root.iter(f"{{{NS['hp']}}}tbl"):
        for tr in table.iter(f"{{{NS['hp']}}}tr"):
            row = []
            for tc in tr.iter(f"{{{NS['hp']}}}tc"):
                cell_texts = [t.text.strip() for t in tc.iter(f"{{{NS['hp']}}}t") if t.text]
                if cell_texts:
                    row.append(" ".join(cell_texts))
            if row:
                texts.append(" | ".join(row))

    return "\n".join(texts)

def extract_all_hwpx(input_dir, output_dir):
    """폴더 내 모든 hwpx 파일 처리"""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".hwpx")]

    for f in tqdm(files, desc="📄 파일 처리 중"):
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, f.replace(".hwpx", ".txt"))
        text = extract_text_from_hwpx(in_path)
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(text)
        print(f"✅ 변환 완료: {f}")

if __name__ == "__main__":
    input_dir = r"C:\company_rules_hwpx"
    output_dir = r"C:\company_rules_text"
    extract_all_hwpx(input_dir, output_dir)
