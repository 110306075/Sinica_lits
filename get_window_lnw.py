import pandas as pd

# df already loaded from hu_stats.csv
# 1) Parenchyma / liver window  (mean ± 2·std)
df = pd.read_csv('./hu_stats.csv')
liver_mu  = df["LiverMean"].mean()
liver_std = df["LiverStd"].mean()

low_liver  = liver_mu - 2 * liver_std
high_liver = liver_mu + 2 * liver_std
WW_liver   = high_liver - low_liver
WL_liver   = (high_liver + low_liver) / 2

print(f"Liver window   →  WL : {WL_liver:.1f}  |  WW : {WW_liver:.1f}")


tum_lo   = df["TumorMean"].quantile(0.05)     
tum_hi   = df["TumorMean"].quantile(0.95)    
liv_q25  = df["LiverMean"].quantile(0.25)     

margin   = 5                                  
low_tum  = tum_lo - margin
high_tum = min(tum_hi + margin, liv_q25)      

WW_tum = high_tum - low_tum
WL_tum = (high_tum + low_tum) / 2

print(f"Tumour window  →  WL : {WL_tum:.1f}  |  WW : {WW_tum:.1f}")
