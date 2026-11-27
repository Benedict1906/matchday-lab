from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import skellam
STATS=['points','goals_for','goals_against','xg','xga','shots','shots_on_target']
LABELS=['Away win','Draw','Home win']
class History:
 def __init__(self): self.rows=defaultdict(list); self.rating=defaultdict(lambda:1500.)
 def team(self,t,opp,venue,day,season):
  past=self.rows[t]; f={}; v=np.array([r[4] for r in past[-10:]])
  for n in (5,10):
   for j,k in enumerate(STATS):
    a=v[-n:,j] if len(v) else []; a=np.asarray(a); f[f'{k}_{n}']=np.nanmean(a) if np.isfinite(a).any() else np.nan
  sc=[r[4] for r in past if r[1]==season]; f['season_ppg']=np.mean([x[0] for x in sc]) if sc else np.nan; f['season_gd']=np.mean([x[1]-x[2] for x in sc]) if sc else np.nan; f['rest']=min((day-past[-1][0]).days,30) if past else np.nan; f['elo']=self.rating[t]
  for name,rows in [('venue',[r for r in past if r[3]==venue][-5:]),('head_to_head',[r for r in past if r[2]==opp][-5:])]: f[name]=np.mean([r[4][0] for r in rows]) if rows else np.nan
  return f,''.join({0:'L',1:'D',3:'W'}[r[4][0]] for r in past[-5:])
 def fixture(self,h,a,d,s):
  f={}; forms=[]
  for prefix,t,o,v in [('home',h,a,1),('away',a,h,0)]:
   z,form=self.team(t,o,v,d,s);f.update({prefix+'_'+k:v for k,v in z.items()});forms.append(form)
  return f,forms
 def add(self,r):
  self.rows[r.team].append((r.match_date,r.season,r.opponent,r.home_indicator,[getattr(r,k) for k in STATS]));
  if r.home_indicator:
   h,a=r.team,r.opponent;e=1/(1+10**((self.rating[a]-self.rating[h]-60)/400));delta=20*({3:1,1:.5,0:0}[r.points]-e);self.rating[h]+=delta;self.rating[a]-=delta

def predict(m,frame):
 x=frame[m['features']].to_numpy(dtype=float);x=(np.where(np.isfinite(x),x,m['median'])-m['mean'])/m['scale'];z=x@np.array(m['coef']).T+m['intercept'];
 if m['kind']=='logistic':return softmax(z,axis=1)
 h,a=np.exp(z).T;return np.column_stack([skellam.cdf(-1,h,a),skellam.pmf(0,h,a),skellam.sf(0,h,a)])
