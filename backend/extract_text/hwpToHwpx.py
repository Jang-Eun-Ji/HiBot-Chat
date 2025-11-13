import win32com.client
import os

input_folder = r"C:\company_rules" #변환 전(hwp) 경로
output_folder = r"C:\company_rules_hwpx" #변환 후(hwpx) 저장 경로

os.makedirs(output_folder, exist_ok=True)

# 한글 실행
hwp = win32com.client.Dispatch("HWPFrame.HwpObject")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".hwp"):
        hwp_path = os.path.join(input_folder, filename)
        hwpx_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".hwpx")

        print(f"변환 중: {filename}")

        # 파일 열기
        hwp.Open(hwp_path, "HWP", "forceopen:true")

        # HWPX 형식으로 저장
        # SaveAs의 매개변수는 (파일경로, 파일형식, 옵션)
        # 형식: "HWPX" 대신 "HWPX", "version:1.0" 형태로 줘야 안전합니다.
        hwp.SaveAs(hwpx_path, "HWPX", "version:1.0")

        # 현재 문서 닫기
        hwp.XHwpDocuments.Close(isDirty=False)

# 한글 종료
hwp.Quit()

print("✅ 모든 hwp 파일이 hwpx로 변환되었습니다.")
