import win32com.client
import os

def convert_hwp_to_hwpx(input_folder, output_folder):
    """
    HWP 파일들을 HWPX 파일로 변환하는 함수
    
    Args:
        input_folder: HWP 파일들이 있는 폴더 경로
        output_folder: HWPX 파일들을 저장할 폴더 경로
    """
    os.makedirs(output_folder, exist_ok=True)

    # 한글 실행
    hwp = win32com.client.Dispatch("HWPFrame.HwpObject")
    
    try:
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
    finally:
        # 한글 종료
        hwp.Quit()

    print("✅ 모든 hwp 파일이 hwpx로 변환되었습니다.")

if __name__ == "__main__":
    # 직접 실행할 때만 변환 수행
    input_folder = r"../hibot-chat-docs-hwp"  # 변환 전(hwp) 경로
    output_folder = r"/hibot-chat-docs-hwpx"  # 변환 후(hwpx) 저장 경로
    convert_hwp_to_hwpx(input_folder, output_folder)
