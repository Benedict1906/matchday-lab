import sys,json
from engine import *
from sklearn.linear_model import LogisticRegression,PoissonRegressor
from sklearn.metrics import log_loss,accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler

def run(path):
 g=pd.read_excel(path,sheet_name='Team_Games',usecols=['match_id','season','match_date','team','opponent','home_indicator']+STATS).sort_values('match_date');g.match_date=pd.to_datetime(g.match_date);g.to_csv('history.csv.gz',index=False); hist=History(); rows=[]
 for d,b in g.groupby('match_date'):
  for r in b[b.home_indicator==1].itertuples():
   f,forms=hist.fixture(r.team,r.opponent,d,r.season);rows.append(dict(**f,home_form=forms[0],away_form=forms[1],date=d,home=r.team,away=r.opponent,season=r.season,match_id=r.match_id,actual={0:'Away win',1:'Draw',3:'Home win'}[r.points],hg=r.goals_for,ag=r.goals_against))
  for r in b.itertuples():hist.add(r)
 df=pd.DataFrame(rows); years=df.season.str[:4].astype(int); test=years==2024;val=years.between(2022,2023);tr=years<2022;fields=list(f);base=[k for k in fields if not k.endswith(('elo','venue','head_to_head'))]; candidates=[('Rolling form','logistic',base),('Form + matchup','logistic',fields),('Goal model','poisson',fields)]
 def fit(mask,kind,cols):
  x=df.loc[mask,cols];med=x.median();sc=StandardScaler();z=sc.fit_transform(x.fillna(med));
  if kind=='logistic':
   l=LogisticRegression(C=.1,max_iter=600).fit(z,df.actual[mask]);assert list(l.classes_)==LABELS;coef=l.coef_;inter=l.intercept_
  else:
   ls=[PoissonRegressor(alpha=.3,max_iter=400).fit(z,df.loc[mask,k]) for k in ['hg','ag']];coef=np.array([l.coef_ for l in ls]);inter=np.array([l.intercept_ for l in ls])
  return dict(kind=kind,features=cols,median=med.tolist(),mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),coef=coef.tolist(),intercept=inter.tolist())
 scores=[]
 for name,kind,cols in candidates:
  m=fit(tr,kind,cols);p=predict(m,df[val]);y=df.actual[val];scores.append(dict(candidate=name,accuracy=accuracy_score(y,np.array(LABELS)[p.argmax(1)]),log_loss=log_loss(y,p,labels=LABELS)))
 i=np.argmin([r['log_loss'] for r in scores]);name,kind,cols=candidates[i];m=fit(tr|val,kind,cols);out=df[test].copy();p=predict(m,out);out[LABELS]=p;out['pick']=np.array(LABELS)[p.argmax(1)];market=pd.read_excel(path,sheet_name='Model_Matches',usecols=['match_id','market_prob_away','market_prob_draw','market_prob_home']).set_index('match_id');market.columns=LABELS;out['market']=out.match_id.map(market.dropna().idxmax(axis=1));out.to_csv('replay.csv',index=False);json.dump(dict(model=m,selection=name,validation=scores),open('model.json','w'));print(name,len(out),sum(out.pick==out.actual),scores)
if __name__=='__main__':run(sys.argv[1])
