import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# ------------------------------------------------
# 설정
# ------------------------------------------------
DATA_FILE = "processed_data.pkl"
EMBED_FILE = "embeddings.npy"
REPORT_FILE = "INPEX_Insight_Report.csv"

# ------------------------------------------------
# 데이터 로드
# ------------------------------------------------
print("📂 저장된 데이터 불러오는 중...")

if not os.path.exists(DATA_FILE) or not os.path.exists(EMBED_FILE):
    print("❌ 저장된 데이터가 없습니다. 'analysis_viz.py'를 먼저 실행해주세요.")
    exit()

df_combined = pd.read_pickle(DATA_FILE)
embeddings = np.load(EMBED_FILE)

print(f"✅ 데이터 로드 완료: {len(df_combined)}행")

# ------------------------------------------------
# 유사도 매칭 (Cosine Similarity)
# ------------------------------------------------
print("🔍 유사도 분석 및 리포트 작성 중...")

# 내부 vs 외부 데이터 인덱스 분리
internal_indices = df_combined[df_combined['source_type'] == 'Internal (INPEX)'].index
external_indices = df_combined[df_combined['source_type'] == 'External (Academia)'].index

if len(internal_indices) == 0:
    print("❌ 분석할 INPEX 내부 데이터가 없습니다.")
    exit()

# 벡터 추출
internal_emb = embeddings[internal_indices]
external_emb = embeddings[external_indices]

# 코사인 유사도 계산 (속도가 매우 빠름)
similarity_matrix = cosine_similarity(internal_emb, external_emb)

# ------------------------------------------------
# 리포트 데이터 구성
# ------------------------------------------------
report_data = []

for i, internal_idx in enumerate(internal_indices):
    inpex_filename = df_combined.loc[internal_idx, 'filename']
    inpex_desc_full = df_combined.loc[internal_idx, 'description']
    
    # 미리보기 텍스트 (너무 길면 자르기)
    inpex_preview = inpex_desc_full[:100] + "..." if len(inpex_desc_full) > 100 else inpex_desc_full
    
    # 해당 파일의 유사도 점수들
    scores = similarity_matrix[i]
    
    # 상위 5개(Top 5) 인덱스 찾기
    top_k_indices = np.argsort(scores)[::-1][:5]
    
    for rank, ext_idx_rel in enumerate(top_k_indices):
        real_ext_idx = external_indices[ext_idx_rel]
        
        score = scores[ext_idx_rel]
        match_desc = df_combined.loc[real_ext_idx, 'description']
        match_category = df_combined.loc[real_ext_idx, 'category']
        
        report_data.append({
            'INPEX_File': inpex_filename,
            'INPEX_Desc_Preview': inpex_preview,
            'Rank': rank + 1,
            'Similarity_Score': f"{score:.4f}", # 소수점 4자리
            'Matched_External_Category': match_category,
            'Matched_External_Description': match_desc
        })

# ------------------------------------------------
# 파일 저장
# ------------------------------------------------
report_df = pd.DataFrame(report_data)
report_df.to_csv(REPORT_FILE, index=False, encoding='utf-8-sig')

print(f"✅ [Step 2 완료] 매칭 리포트가 생성되었습니다: {REPORT_FILE}")
print("   (엑셀에서 파일을 열어보세요)")