import pandas as pd
path = "/Users/hantaehee/Downloads/Meta-Data/Ver.2/ieee_dataport_all_categories.csv"

df_sample = pd.read_csv(
    path, 
    usecols=['category','title',],
    nrows=100)
print(df_sample)
print(df_sample.columns)  # 컬럼 구조도 같이 확인 가능