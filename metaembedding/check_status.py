import pandas as pd
import os

print("🔍 데이터 상태 점검 중...")

# 1. 파일 존재 여부 확인
if not os.path.exists("processed_data.pkl"):
    print("❌ 'processed_data.pkl' 파일이 없습니다. Step 1(analysis_viz.py)을 먼저 실행하세요!")
    exit()

# 2. 데이터 열어서 확인
df = pd.read_pickle("processed_data.pkl")
print(f"✅ 전체 데이터 개수: {len(df)}개")

# 3. INPEX 내부 데이터가 들어있는지 확인
inpex_data = df[df['source_type'] == 'Internal (INPEX)']
count = len(inpex_data)

if count > 0:
    print(f"✅ INPEX 내부 데이터가 {count}개 확인되었습니다.")
    print("   -> 이제 'python generate_report.py'를 실행하면 무조건 리포트가 나옵니다.")
else:
    print("⚠️ 전체 데이터는 있는데, 'INPEX 내부 데이터'가 0개입니다.")
    print("   -> Step 1의 전처리 과정에서 필터링되었거나 로드되지 않았습니다.")
    print("   -> inpexdata 폴더 경로와 CSV 파일 상태를 확인해 주세요.")