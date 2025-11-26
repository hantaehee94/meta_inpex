import matplotlib
# 서버(GUI 없는 환경)에서 실행 시 에러 방지를 위해 백엔드 설정 (import pyplot 이전에 해야 함)
matplotlib.use('Agg') 

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 머신러닝 라이브러리
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan

# ------------------------------------------------
# 설정값
# ------------------------------------------------
# __file__은 스크립트 실행 시에만 작동하므로, Jupyter 등에서는 절대 경로 직접 입력 권장
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(".").resolve() # Jupyter/Interactive 환경 대응

DATA_FILE = BASE_DIR / "ieee_dataport_all_categories.csv"
TEXT_COLUMN = "description"
CATEGORY_COLUMN = "category"
MAX_ROWS = None
MAX_CATEGORY_LEGEND = 15

# ------------------------------------------------
# 1. 데이터 로드 및 정제
# ------------------------------------------------
print(f"데이터 파일 경로: {DATA_FILE}")
if not DATA_FILE.exists():
    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {DATA_FILE}")

print("데이터 로드 중...")
df = pd.read_csv(DATA_FILE)

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"'{TEXT_COLUMN}' 컬럼을 찾을 수 없습니다. CSV 헤더를 확인하세요.")

# 전처리: 결측치 및 공백 제거
df = df[df[TEXT_COLUMN].notna() & (df[TEXT_COLUMN].astype(str).str.strip() != "")]

if MAX_ROWS and len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42)

df = df.reset_index(drop=True)
print(f"총 데이터 개수: {len(df)}개")

# ------------------------------------------------
# 2. 임베딩 (Semantic Embedding)
# ------------------------------------------------
print("임베딩 모델 로드 및 변환 중...")
# GPU가 있으면 자동으로 사용합니다.
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df[TEXT_COLUMN].tolist(), show_progress_bar=True)

# ------------------------------------------------
# 3. 차원 축소 (UMAP)
# ------------------------------------------------
print("차원 축소(UMAP) 진행 중...")
# random_state를 고정해도 UMAP 버전에 따라 미세한 차이가 있을 수 있습니다.
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

df['x'] = embedding_2d[:, 0]
df['y'] = embedding_2d[:, 1]

# ------------------------------------------------
# 4. 군집화 (Clustering) - 최적화 적용
# ------------------------------------------------
print("군집화(HDBSCAN) 진행 중...")
# [변경점] 고차원 벡터 대신 UMAP 결과(2D)를 사용하면 시각화와 군집의 일치도가 높아집니다.
# min_cluster_size: 이 크기보다 작은 덩어리는 잡음(-1)으로 처리
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10, prediction_data=True)
df['cluster'] = clusterer.fit_predict(embedding_2d) 

# 노이즈(-1)를 제외한 군집 수 계산
unique_clusters = set(df['cluster'])
num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
print(f"발견된 군집 수: {num_clusters} (노이즈 제외)")

# ------------------------------------------------
# 5. 시각화 (Clustering)
# ------------------------------------------------
print("군집 결과 시각화 생성 중...")
plt.figure(figsize=(14, 10))

# 팔레트 설정: 군집이 많을 경우를 대비해 색상 확장
palette_name = 'tab10' if num_clusters <= 10 else 'turbo'

scatter = sns.scatterplot(
    data=df, 
    x='x', 
    y='y', 
    hue='cluster', 
    palette=palette_name,
    s=5,            # 점 크기 조절 (14000개면 10은 조금 클 수 있음)
    alpha=0.7,      
    legend='full' if num_clusters <= 20 else False # 군집이 20개 넘으면 범례 끄기
)

plt.title(f'Semantic Clustering (Clusters: {num_clusters})', fontsize=16)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')

# 범례가 켜져 있을 때만 위치 조정
if num_clusters <= 20:
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.tight_layout()
plt.savefig('clustering_result.png', dpi=300)
plt.close() # 메모리 해제 (중요)

# ------------------------------------------------
# 6. 시각화 (Category - Optional)
# ------------------------------------------------
if CATEGORY_COLUMN in df.columns:
    print("카테고리 기준 시각화 생성 중...")
    category_series = df[CATEGORY_COLUMN].fillna("Unknown").astype(str)
    
    # 상위 N개 카테고리만 남기고 나머지는 'Other' 처리
    top_categories = category_series.value_counts().nlargest(MAX_CATEGORY_LEGEND).index
    df['category_for_plot'] = np.where(category_series.isin(top_categories), category_series, 'Other')

    # 'Other'는 회색으로 표시하기 위해 팔레트 커스텀 가능 (여기선 자동)
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='category_for_plot',
        palette='tab20',
        s=5,
        alpha=0.6
    )
    plt.title(f'UMAP Visualization by Category (Top {MAX_CATEGORY_LEGEND})', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('clustering_by_category.png', dpi=300)
    plt.close() # 메모리 해제
else:
    print(f"'{CATEGORY_COLUMN}' 컬럼이 없어 카테고리 시각화를 건너뜁니다.")

print("모든 작업 완료! 결과 이미지를 확인하세요.")

"""
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 머신러닝 라이브러리
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
import hdbscan

# ------------------------------------------------
# 설정값
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "ieee_dataport_all_categories.csv"
TEXT_COLUMN = "description"  # row 13 descriptions
CATEGORY_COLUMN = "category"  # 첫 번째 속성
MAX_ROWS = None  # 필요 시 임의로 제한 (예: 12000)
MAX_CATEGORY_LEGEND = 15  # 색상으로 구분할 최대 카테고리 수

# 1. 데이터 로드 및 정제
print("데이터 로드 중...")
df = pd.read_csv(DATA_FILE)

if TEXT_COLUMN not in df.columns:
    raise ValueError(f"'{TEXT_COLUMN}' 컬럼을 찾을 수 없습니다. CSV 헤더를 확인하세요.")

# 설명 텍스트가 있는 행만 사용
df = df[df[TEXT_COLUMN].notna() & (df[TEXT_COLUMN].astype(str).str.strip() != "")]

# 필요 시 샘플링
if MAX_ROWS and len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42)

df = df.reset_index(drop=True)
print(f"총 데이터 개수: {len(df)}개 (실제 CSV 기반)")

# 2. 임베딩 (Semantic Embedding)
# 'all-MiniLM-L6-v2'는 속도와 성능 밸런스가 좋아 1~2만 개 데이터에 적합합니다.
print("임베딩 모델 로드 및 변환 중 (시간이 조금 소요됩니다)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df[TEXT_COLUMN].tolist(), show_progress_bar=True)

# 3. 차원 축소 (UMAP)
# 14000개 데이터는 t-SNE보다 UMAP이 훨씬 빠르고 구조를 잘 보존합니다.
# n_neighbors: 이웃의 수 (보통 15~50 사이, 크면 더 큰 구조를 봄)
# min_dist: 점들 간의 최소 거리 (작으면 뭉치고, 크면 퍼짐)
print("차원 축소(UMAP) 진행 중...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

df['x'] = embedding_2d[:, 0]
df['y'] = embedding_2d[:, 1]

# 4. 군집화 (Clustering)
# 옵션 A: 군집 개수를 모를 때 (HDBSCAN 추천 - 자동으로 개수 찾아줌)
print("군집화(HDBSCAN) 진행 중...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
df['cluster'] = clusterer.fit_predict(embeddings) # 원본 임베딩 사용이 정확도가 높음

# 옵션 B: 군집 개수를 명확히 알 때 (K-Means 사용)
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=5, random_state=42)
# df['cluster'] = kmeans.fit_predict(embeddings)

# 노이즈(-1) 처리: HDBSCAN은 분류되지 않은 데이터를 -1로 표시합니다.
num_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'] else 0)
print(f"발견된 군집 수: {num_clusters}")

# 5. 시각화
plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    data=df, 
    x='x', 
    y='y', 
    hue='cluster', 
    palette='tab10', # 군집이 많으면 'turbo'나 'viridis' 등 사용
    s=10,            # 점 크기 (데이터가 많으므로 작게)
    alpha=0.6,       # 투명도 (겹친 점 확인용)
    legend='full'
)

plt.title(f'Semantic Clustering of {len(df)} IEEE DataPort Rows', fontsize=16)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')

# 범례 위치 조정 (데이터가 많으면 범례가 그림을 가릴 수 있음)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.tight_layout()

# 결과 저장 및 보여주기
plt.savefig('clustering_result.png', dpi=300)
plt.show()

# 6. 카테고리 기준 시각화 (선택적)
if CATEGORY_COLUMN in df.columns:
    print("카테고리 기준 시각화 생성 중...")
    category_series = df[CATEGORY_COLUMN].fillna("Unknown").astype(str)
    top_categories = category_series.value_counts().nlargest(MAX_CATEGORY_LEGEND).index
    df['category_for_plot'] = np.where(category_series.isin(top_categories), category_series, 'Other')

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='category_for_plot',
        palette='tab20',
        s=10,
        alpha=0.6,
        legend='full'
    )
    plt.title(f'UMAP Visualization Colored by Category (Top {MAX_CATEGORY_LEGEND} + Other)', fontsize=16)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('clustering_by_category.png', dpi=300)
    plt.show()

    print("카테고리 기준 시각화 완료! 'clustering_by_category.png' 파일이 생성되었습니다.")
else:
    print(f"'{CATEGORY_COLUMN}' 컬럼이 없어 카테고리 시각화를 건너뜁니다.")

print("완료! 'clustering_result.png' 파일이 생성되었습니다.")
"""