'''
# 전체 속성 보기
import pandas as pd
path = "/Users/hantaehee/Downloads/Meta-Data/Ver.2/ieee_dataport_all_categories.csv"

# usecols 매개변수를 제거하여 모든 열을 읽어옵니다.
df_full_cols = pd.read_csv(
    path, 
    nrows=100) 

print(df_full_cols.head()) # 처음 5줄을 출력하여 확인
print(df_full_cols.columns) # 모든 컬럼 이름을 출력하여 확인
'''

'''
# 속성 하나씩 뽑아서 보기

import pandas as pd
path = "/Users/hantaehee/Downloads/Meta-Data/Ver.2/ieee_dataport_all_categories.csv"

df_sample = pd.read_csv(
    path, 
    usecols=['resource_type'],
    nrows=100)
print(df_sample)
print(df_sample.columns)  # 컬럼 구조도 같이 확인 가능
'''

'''
Index(['category', 'doi', 'title', 'publication_year', 'url', 'publisher',
       'resource_type', 'license', 'subjects', 'creators', 'updated',
       'published', 'description']
'''


'''
# 랜덤 행 보기
import pandas as pd
import numpy as np # 랜덤 시드 설정을 위해 필요

path = "/Users/hantaehee/Downloads/Meta-Data/Ver.2/ieee_dataport_all_categories.csv"

# 1. 모든 열을 읽어옵니다. (nrows=100 제거)
# 파일 크기가 매우 크지 않다면 이 방법이 가장 정확합니다.
df_full = pd.read_csv(path) 

# 2. DataFrame.sample() 메서드를 사용하여 랜덤으로 100개 행을 선택합니다.
# random_state를 설정하면 매번 동일한 랜덤 샘플을 얻을 수 있습니다.
df_random_sample = df_full.sample(n=100, random_state=42)

print(df_random_sample.head())
print(len(df_random_sample))
print(df_random_sample.columns)
'''


import pandas as pd

df = pd.read_csv("/Users/hantaehee/Downloads/Meta-Data/Ver.2/ieee_dataport_all_categories.csv")
sample20 = df.sample(20, random_state=42)

pd.set_option('display.max_colwidth', None)
print(sample20.shape)
print(sample20['publisher'].head(20))
# print(df['category'].isna().sum())
# print(df['category'].isna().mean() * 100)
# print(df['category'].eq('Other').sum())
# print(df['category'].eq('Other').mean() * 100)