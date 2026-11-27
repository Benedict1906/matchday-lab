import numpy as np
import pandas as pd
from engine import LABELS

def enrich(r):
 r=r.copy();d=pd.to_datetime(r.date);r['week']=(d-pd.to_timedelta(d.dt.weekday,unit='D')).dt.strftime('%Y-%m-%d');p=np.sort(r[LABELS].to_numpy(float),axis=1);r['confidence']=p[:,-1];r['gap']=p[:,-1]-p[:,-2];return r

def club_points(r):
 parts=[]
 for club,w in [('home','Home win'),('away','Away win')]:
  parts.append(pd.DataFrame({'Club':r[club],'Expected points':3*r[w]+r.Draw,'Actual points':3*r.actual.eq(w).astype(int)+r.actual.eq('Draw').astype(int)}))
 t=pd.concat(parts).groupby('Club').agg(**{'Played':('Club','size'),'Expected points':('Expected points','sum'),'Actual points':('Actual points','sum')});t['Above / below']=t['Actual points']-t['Expected points'];return t.sort_values(['Above / below','Club'],ascending=[False,True])

def comparable(r,x):
 lo=min(9,int(float(x.confidence)*10))/10;hi=lo+.1;q=r[(r.date<x.date)&(r.confidence>=lo)&((r.confidence<hi) if hi<1 else (r.confidence<=1))];return lo,hi,len(q),int(q['pick'].eq(q.actual).sum())

def goal_rates(m,x):
 v=pd.to_numeric(x[m['features']],errors='coerce').to_numpy(float);z=(np.where(np.isfinite(v),v,m['median'])-m['mean'])/m['scale'];return np.exp(np.array(m['coef'])@z+m['intercept'])
