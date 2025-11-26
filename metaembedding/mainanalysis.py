import pandas as pd
import numpy as np
import glob
import os
import re
import csv
import io
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from adjustText import adjust_text # 텍스트 겹침 방지

# ------------------------------------------------
# 1. 환경 설정
# ------------------------------------------------
EXTERNAL_DATA_PATH = "ieee_dataport_all_categories.csv" 
INTERNAL_DATA_DIR = "inpexdata"  # 폴더명 확인

# [Mac 폰트 설정]
plt.rcParams['font.family'] = 'AppleGothic' 
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------
# 2. 데이터 파싱 함수 (Robust Version)
# ------------------------------------------------
def parse_internal_files(directory):
    if not os.path.exists(directory):
        print(f"❌ '{directory}' 폴더가 없습니다.")
        return pd.DataFrame()

    files = glob.glob(str(Path(directory) / "*.csv"))
    parsed_data = []

    print(f"\n📂 내부 데이터 폴더에서 {len(files)}개 파일 발견. 분석 시작...")

    for f_path in files:
        filename = Path(f_path).name
        try:
            with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text = f.read()
        except: continue

        if "Zscaler" in full_text or "<!DOCTYPE HTML>" in full_text: continue

        description = ""
        title_candidate = ""
        
        # [전략 A] JSON 패턴
        pattern_desc = re.compile(r'"{1,2}description"{1,2}\s*:\s*"{1,2}(.*?)"{1,2}\s*[,}]', re.IGNORECASE | re.DOTALL)
        match_desc = pattern_desc.search(full_text)
        if match_desc:
            description = match_desc.group(1).replace('""', '"').replace('\\n', ' ')

        if not description:
            pattern_title = re.compile(r'"{1,2}title"{1,2}\s*:\s*"{1,2}(.*?)"{1,2}\s*[,}]', re.IGNORECASE | re.DOTALL)
            match_title = pattern_title.search(full_text)
            if match_title:
                title_candidate = match_title.group(1).replace('""', '"').replace('\\n', ' ')

        # [전략 B] CSV 스트림 파싱
        if not description:
            try:
                f_io = io.StringIO(full_text)
                reader = csv.reader(f_io)
                for row in reader:
                    if not row: continue
                    first_col = str(row[0]).strip().lower()
                    
                    if any(k in first_col for k in ["abstract", "description", "summary"]):
                        candidates = [c for c in row[1:] if len(str(c).strip()) > 0]
                        if candidates:
                            clean_desc = ", ".join(candidates)
                            if len(clean_desc) > 10:
                                description = clean_desc
                                break 
                    
                    if not title_candidate and any(k in first_col for k in ["title", "full title", "dataset name"]):
                        candidates = [c for c in row[1:] if len(str(c).strip()) > 0]
                        if candidates:
                            title_candidate = ", ".join(candidates)
            except: pass

        final_text = description
        if not final_text or len(final_text) < 5:
            if title_candidate and len(title_candidate) > 2:
                final_text = title_candidate

        if final_text and len(final_text) > 2:
            final_text = final_text.strip(' "').replace('""', '"')
            parsed_data.append({
                'description': final_text,
                'category': 'Internal Asset',
                'source_type': 'Internal (INPEX)',
                'filename': filename
            })

    return pd.DataFrame(parsed_data)

# ------------------------------------------------
# 3. 데이터 로드 및 통합
# ------------------------------------------------
df_internal = parse_internal_files(INTERNAL_DATA_DIR)
if df_internal.empty:
    print("❌ 내부 데이터를 로드하지 못했습니다.")
    exit()

print("외부 데이터 로드 중...")
try:
    df_external = pd.read_csv(EXTERNAL_DATA_PATH)
    col = 'description' if 'description' in df_external.columns else df_external.columns[1]
    df_external = df_external[[col, 'category']].rename(columns={col: 'description'})
    df_external['source_type'] = 'External (Academia)'
    df_external['filename'] = 'IEEE DataPort'
except:
    print("❌ 외부 데이터 파일을 찾을 수 없습니다.")
    exit()

df_combined = pd.concat([df_external, df_internal], ignore_index=True)
df_combined['description'] = df_combined['description'].fillna("").astype(str)
df_combined = df_combined[df_combined['description'].str.strip().str.len() > 2]
df_combined = df_combined.reset_index(drop=True)

# ------------------------------------------------
# 4. 임베딩 및 저장 (Step 2를 위한 준비) 🌟 중요
# ------------------------------------------------
print("🚀 텍스트 임베딩 생성 중...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df_combined['description'].tolist(), show_progress_bar=True)

# 데이터와 임베딩을 파일로 저장 (피클 및 넘파이 포맷)
print("💾 중간 데이터 저장 중 (processed_data.pkl, embeddings.npy)...")
df_combined.to_pickle("processed_data.pkl")
np.save("embeddings.npy", embeddings)

# ------------------------------------------------
# 5. 시각화 (t-SNE)
# ------------------------------------------------
print("cy 위치 좌표 계산 중 (t-SNE)...")
tsne = TSNE(n_components=2, perplexity=50, max_iter=1000, random_state=42, n_jobs=-1)
embedding_2d = tsne.fit_transform(embeddings)

df_combined['x'] = embedding_2d[:, 0]
df_combined['y'] = embedding_2d[:, 1]

print("🎨 그래프 생성 중...")
plt.figure(figsize=(16, 12))

sns.scatterplot(
    data=df_combined[df_combined['source_type'] == 'External (Academia)'],
    x='x', y='y', color='lightgray', s=30, alpha=0.3, linewidth=0, label='External Knowledge'
)

internal_points = df_combined[df_combined['source_type'] == 'Internal (INPEX)']
sns.scatterplot(
    data=internal_points,
    x='x', y='y', color='red', s=150, marker='X', edgecolor='black', label='INPEX Assets'
)

texts = []
for i, row in internal_points.iterrows():
    texts.append(plt.text(
        x=row['x'], y=row['y'], s=row['filename'], 
        fontsize=9, fontweight='bold', color='black'
    ))

print("🧩 텍스트 위치 최적화 중...")
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black', lw=0.5), 
            force_points=0.2, force_text=0.5, expand_points=(1.2, 1.2))

plt.title('INPEX Data Asset Map', fontsize=18)
plt.legend(loc='upper right')
plt.tight_layout()

plt.savefig('INPEX_Map_Final.png', dpi=300)
print("✅ [Step 1 완료] 그래프 저장됨 & 리포트용 데이터 저장됨.")
plt.show()


"""
# ------------------------------------------------
# 2. INPEX 내부 데이터 전처리 함수 (최종 수정판)
# ------------------------------------------------

import pandas as pd
import numpy as np
import json
import glob
import os
import re
import csv
import io
from pathlib import Path

# ------------------------------------------------
# 2. INPEX 내부 데이터 전처리 함수 (최종_완결판)
# ------------------------------------------------
def parse_internal_files(directory):
    if not os.path.exists(directory):
        print(f"❌ '{directory}' 폴더가 없습니다.")
        return pd.DataFrame()

    files = glob.glob(str(Path(directory) / "*.csv"))
    parsed_data = []

    print(f"\n📂 내부 데이터 폴더에서 {len(files)}개 파일 발견. 정밀 분석 시작...")

    for f_path in files:
        filename = Path(f_path).name
        
        # 1. 파일 전체 텍스트로 읽기
        try:
            with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                full_text = f.read()
        except Exception as e:
            print(f"❌ 파일 읽기 에러 ({filename}): {e}")
            continue

        # 2. 보안 차단 파일 건너뛰기
        if "Zscaler" in full_text or "<!DOCTYPE HTML>" in full_text:
            print(f"⚠️ [Skip] 보안 차단된 파일: {filename}")
            continue

        description = ""
        
        # [전략 A] JSON 패턴 파싱 (복잡하게 꼬인 JSON 데이터용)
        # 예: Historic_Clean_Energy... (CSV 안에 JSON이 들어있는 경우)
        if not description:
            # 정규식: "description": "..." 또는 ""description"": ""..."" 패턴 찾기
            pattern = re.compile(r'"{1,2}description"{1,2}\s*:\s*"{1,2}(.*?)"{1,2}\s*[,}]', re.IGNORECASE | re.DOTALL)
            match = pattern.search(full_text)
            if match:
                description = match.group(1).replace('""', '"').replace('\\n', ' ')

        # [전략 B] CSV 스트림 파싱 (IEA, Australian 프로젝트 파일용) 🌟 핵심 수정 🌟
        # Pandas 대신 csv 라이브러리로 한 줄씩 유연하게 읽음
        if not description or len(description) < 10:
            try:
                # 이미 읽은 텍스트(full_text)를 메모리 파일처럼 취급
                f_io = io.StringIO(full_text)
                reader = csv.reader(f_io) # csv 모듈은 칸 수가 달라도 에러 안 남!
                
                for row in reader:
                    if not row: continue # 빈 줄 패스
                    
                    # 첫 번째 칸에서 키워드 찾기 (Abstract, Description 등)
                    first_col = str(row[0]).strip()
                    if any(k in first_col.lower() for k in ["abstract", "description", "summary"]):
                        
                        # 키워드 이후의 모든 칸을 검사
                        # (IEA 파일처럼 쉼표로 문장이 쪼개진 경우 다시 합쳐줌)
                        candidates = [c for c in row[1:] if len(str(c).strip()) > 0]
                        
                        if candidates:
                            # 쪼개진 문장들을 다시 자연스럽게 연결
                            clean_desc = ", ".join(candidates)
                            
                            # 내용이 충분히 길면(20자 이상) 채택
                            if len(clean_desc) > 20:
                                description = clean_desc
                                break
            except Exception as e:
                # print(f"CSV 파싱 에러: {e}") 
                pass

        # 3. 결과 저장
        if description and len(description) > 20:
            # 지저분한 기호 최종 정리
            description = description.strip(' "').replace('""', '"')
            parsed_data.append({
                'description': description,
                'category': 'Internal Asset',
                'source_type': 'Internal (INPEX)',
                'filename': filename
            })
        else:
            print(f"⚠️ 설명 추출 실패 (내용 없음): {filename}")

    return pd.DataFrame(parsed_data)

# ------------------------------------------------
# 3. Main Logic process for embedding and clustering
# ------------------------------------------------
# (1) 데이터 로드
df_internal = parse_internal_files(INTERNAL_DATA_DIR)
print(f"✅ 내부 데이터: {len(df_internal)}건")

try:
    df_external = pd.read_csv(EXTERNAL_DATA_PATH)
    desc_col = 'description' if 'description' in df_external.columns else df_external.columns[1]
    title_col = 'title' if 'title' in df_external.columns else None
    keep_cols = ['category']
    if title_col:
        keep_cols.append(title_col)
    keep_cols.append(desc_col)
    df_external = df_external[keep_cols].rename(columns={desc_col: 'description'})
    if title_col:
        df_external = df_external.rename(columns={title_col: 'title'})
    else:
        df_external['title'] = df_external['description']
    df_external['source_type'] = 'External (Academia)'
    df_external['filename'] = 'IEEE DataPort'
except:
    print("❌ 외부 데이터 파일을 찾을 수 없습니다.")
    exit()

if not df_internal.empty:
    df_internal = df_internal.copy()
    # 내부 데이터는 별도 title이 없으므로 파일명을 title로 사용
    df_internal['title'] = df_internal['filename']

df_combined = pd.concat([df_external, df_internal], ignore_index=True, sort=False)

# description이 비어있거나 NaN이면 title을 대신 사용
# (외부/내부 데이터에서 결측 설명을 자동 보완)
def pick_description(row):
    desc = row.get('description', '')
    if isinstance(desc, str) and desc.strip():
        return desc.strip()
    title = row.get('title', '')
    if isinstance(title, str) and title.strip():
        return title.strip()
    return ""

df_combined['description'] = df_combined.apply(pick_description, axis=1)
df_combined = df_combined[df_combined['description'].astype(str).str.strip() != ""].reset_index(drop=True)
print(f"📊 총 데이터: {len(df_combined)}개")

# (2) 임베딩
print("🚀 텍스트 분석 중...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df_combined['description'].tolist(), show_progress_bar=True)

# (3) 차원 축소 (t-SNE 사용)
print("cy 데이터 지도 그리는 중 (t-SNE)...")
# perplexity: 보통 데이터 수의 1/100 정도 혹은 30~50 사용
# n_jobs=-1: 가능한 모든 CPU 코어 사용 (맥북 성능 활용)
tsne = TSNE(n_components=2, perplexity=50, max_iter=1000, random_state=42, n_jobs=-1)
embedding_2d = tsne.fit_transform(embeddings)

df_combined['x'] = embedding_2d[:, 0]
df_combined['y'] = embedding_2d[:, 1]

# (4) 시각화
print("🎨 그래프 출력...")
plt.figure(figsize=(14, 9))
sns.scatterplot(data=df_combined[df_combined['source_type'] == 'External (Academia)'],
                x='x', y='y', color='lightgray', s=20, alpha=0.4, linewidth=0, label='External Knowledge')
sns.scatterplot(data=df_combined[df_combined['source_type'] == 'Internal (INPEX)'],
                x='x', y='y', color='red', s=100, marker='X', edgecolor='white', label='INPEX Assets')

plt.title('INPEX Data Asset Map (t-SNE)', fontsize=18)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
plt.savefig('INPEX_Map_tSNE.png', dpi=300)
"""