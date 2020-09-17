import pandas as pd


ptufile  = './data/AMOS/20191216/PTU_R06_15.csv'
visfile  = './data/AMOS/20191216/VIS_R06_15.csv'
windfile = './data/AMOS/20191216/WIND_R06_15.csv'

ptu = pd.read_csv(
    ptufile,
    index_col='LOCALDATE (BEIJING)',
    usecols=['LOCALDATE (BEIJING)', 'PAINS (HPA)', 'QNH AERODROME (HPA)', 'QFE R06 (HPA)', 'TEMP (°C)', 'RH (%)', 'DEWPOINT (°C)'],
    parse_dates=True)

vis = pd.read_csv(
    visfile,
    index_col='LOCALDATE (BEIJING)',
    usecols=['LOCALDATE (BEIJING)', 'RVR_1A', 'MOR_1A', 'LIGHTS'],
    parse_dates=True)

wind = pd.read_csv(
    windfile,
    index_col='LOCALDATE (BEIJING)',
    usecols=['LOCALDATE (BEIJING)', 'WS2A (MPS)', 'WD2A', 'CW2A (MPS)'],
    parse_dates=True)

df = pd.merge(vis, ptu, how='left', left_index=True, right_index=True)
df = pd.merge(df, wind, how='left', left_index=True, right_index=True)
df = df.interpolate()
df.to_csv('./data/AMOS/processed/20191216.csv', float_format='%.2f') 