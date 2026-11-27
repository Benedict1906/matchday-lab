import json,datetime as dt
from pathlib import Path
import streamlit as st
from engine import *
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Matchday Lab | EPL',page_icon='⚽',layout='wide')
@st.cache_data
def load():return pd.read_csv(ROOT/'replay.csv',keep_default_na=False),pd.read_csv(ROOT/'history.csv.gz',parse_dates=['match_date']),json.loads((ROOT/'model.json').read_text())
r,g,meta=load();m=meta['model'];last=g.match_date.max();teams=sorted(set(r.home)|set(r.away));colors=['#4f9cf9','#aab9bc','#179d82']
st.markdown('''<div style="background:linear-gradient(105deg,#082c31,#12504c);color:white;border-radius:16px;padding:25px 32px;border-left:8px solid #96e8ae"><small style="letter-spacing:3px;color:#b0f0ca">FOOTBALL ANALYTICS &nbsp; / &nbsp; EPL</small><h1 style="color:white;margin:0;padding:6px 0">Matchday Lab</h1><span>Read the form. Make your pick. Reveal the result.</span></div>''',unsafe_allow_html=True)
st.caption(f"Senior project • Emeka Olisa   |   Data through {last:%d %b %Y}   |   No betting odds used in training")
def bars(vals,h,a):
 for nm,v,c,box in zip([h,'Draw',a],vals[::-1],colors[::-1],st.columns(3)):
  with box:
   st.metric(nm,f'{float(v):.1%}');st.markdown(f'<div style="background:#dee8e5;height:8px;border-radius:8px"><div style="background:{c};width:{100*v:.2f}%;height:8px;border-radius:8px"></div></div>',unsafe_allow_html=True)
def form(s):return ' '.join({'W':'🟢 W','D':'⚪ D','L':'🔴 L'}[x] for x in s) or 'No prior games'
from insights import enrich,club_points,comparable,goal_rates
r=enrich(r);weeks=sorted(r.week.unique());default_week=weeks[min(14,len(weeks)-1)]
def weekname(w):
 start=pd.Timestamp(w);return f'{start:%d %b} – {start+pd.Timedelta(days=6):%d %b %Y}'
def team_pick(x):return {'Home win':x.home,'Draw':'Draw','Away win':x.away}[x['pick']]
def trust(x):
 lo,hi,n,k=comparable(r,x);st.write(f'Among earlier 2024/25 fixtures, **{n}** had a top pick in the same probability band ({lo:.0%} to below {hi:.0%}).')
 if n:st.write(f'The top pick was correct **{k} of {n} times ({k/n:.1%})**. This pools home, draw and away picks; it is not a team-specific win rate.')
 if n<30:st.warning('Fewer than 30 comparisons: too little evidence for a stable rate.')
 st.caption('Only dates before this fixture count. Past hit rates describe this sample, not a guarantee for this match.')
t1,t2,t4,t3=st.tabs(['Matchweek Finder','Model Scorecard','Expected vs. Actual','Fixture Simulator'])
with t1:
 st.subheader('Find your match');st.caption('2024/25 replay • Calendar weeks run Monday–Sunday, not official league rounds. No final scores in the finder.')
 club_filter=st.selectbox('Browse a club',['All clubs']+teams);pool=r if club_filter=='All clubs' else r[(r.home==club_filter)|(r.away==club_filter)];available=sorted(pool.week.unique());a,b=st.columns(2);wk=a.selectbox('Week starting',available,index=available.index(default_week) if default_week in available else 0,format_func=weekname);order=b.selectbox('Sort fixtures',['Strongest favorite','Closest call','Kickoff date']);q=pool[pool.week==wk].sort_values({'Strongest favorite':'confidence','Closest call':'gap','Kickoff date':'date'}[order],ascending=order!='Strongest favorite');st.caption('Strongest favorite = largest top probability. Closest call = smallest gap between the top two outcomes.')
 table=pd.DataFrame({'Date':q.date,'Fixture':q.home+' vs '+q.away,'Model pick':q.apply(team_pick,axis=1),'Top chance':q.confidence.map('{:.1%}'.format),'Gap (pts)':(100*q.gap).round(1)});st.dataframe(table,hide_index=True,use_container_width=True)
 i=st.selectbox('Explore a fixture',q.index,format_func=lambda j:f'{r.at[j,"home"]} vs {r.at[j,"away"]}');x=r.loc[i];st.divider();st.caption(f'MATCH PREVIEW   /   {x.date}');st.header(f'{x.home} vs {x.away}');bars(x[LABELS].to_numpy(float),x.home,x.away);st.write(f'**Model pick: {team_pick(x)} • {x.confidence:.1%}.** The other two outcomes together account for {1-x.confidence:.1%}.')
 for box,t,s in zip(st.columns(2),[x.home,x.away],[x.home_form,x.away_form]):box.write(f'**{t}** · last five EPL results');box.caption(form(s)+'  ·  oldest first')
 st.markdown('#### Ask Matchday Lab');question=st.radio('Choose a question',['Why this pick?','Can I trust this forecast?','What was this week’s biggest upset?'],horizontal=True)
 if question=='Why this pick?':
  if m['kind']!='logistic':
   gh,ga=goal_rates(m,x);st.write(f'The model estimates **{gh:.2f} goals for {x.home}** and **{ga:.2f} for {x.away}**. It combines possible scorelines into home-win, draw, and away-win chances; **{team_pick(x)}** has the largest share.')
  measures={'points_5':'Points / game · last 5','xg_5':'Expected goals / game · last 5','shots_on_target_5':'Shots on target / game · last 5','venue':'Points / game · last 5 home or away','elo':'Opponent-adjusted Elo','rest':'Days since EPL match · capped at 30'}
  st.caption('Matchup facts, not a ranking of model drivers. Missing history is shown as —; the model uses training medians.')
  for key,phrase in [('xg_5','created more expected goals per game'),('shots_on_target_5','put more shots on target per game')]:
   h,a=pd.to_numeric([x['home_'+key],x['away_'+key]],errors='coerce');
   if np.isfinite([h,a]).all() and round(h,2)!=round(a,2):st.write(f'• **{x.home if h>a else x.away}** {phrase} over its last five EPL matches ({max(h,a):.2f} vs {min(h,a):.2f}).')
  st.table(pd.DataFrame({'Pre-match input':measures.values(),**{t:[f'{float(x[p+k]):.2f}' if x[p+k]!='' else '—' for k in measures] for p,t in [('home_',x.home),('away_',x.away)]}}).set_index('Pre-match input'))
 elif question=='Can I trust this forecast?':trust(x)
 else:
  st.caption('Biggest upset = the result the model gave the lowest probability among all fixtures that week. This can spoil another match.')
  if st.button('Reveal this week’s biggest upset',key='upset'+wk):
   full=r[r.week==wk];probs=full.apply(lambda v:v[v.actual],axis=1);u=full.loc[probs.idxmin()];st.info(f'{u.home} {u.hg} – {u.ag} {u.away} · {u.actual}. The model gave that outcome {probs.min():.1%}.')
 guess=st.radio('Your pick',[x.home,'Draw',x.away],index=None,horizontal=True,key=f'guess{i}');
 if st.toggle('Reveal match result',key=f'reveal{i}'):
  winner={'Home win':x.home,'Draw':'Draw','Away win':x.away}[x.actual];st.info(f'Full time: {x.home} {x.hg} – {x.ag} {x.away}  |  Model: '+('correct' if x['pick']==x.actual else 'incorrect')+(f'  |  Your pick: {"correct" if guess==winner else "incorrect"}' if guess else ''))
with t4:
 st.subheader('Which clubs beat their forecast?');st.write('Expected points are a forecast average, not predicted league standings. A 50% win chance and a 25% draw chance give 3 × 0.50 + 0.25 = 1.75 expected points.');end=st.selectbox('Include results through the end of',weeks,index=len(weeks)-1,format_func=weekname);tot=club_points(r[r.week<=end]);club=st.selectbox('Spotlight a club',sorted(tot.index),index=sorted(tot.index).index('Arsenal') if 'Arsenal' in tot.index else 0);v=tot.loc[club];a,b,c=st.columns(3);a.metric('Expected points',f'{v["Expected points"]:.1f}');b.metric('Actual points',int(v['Actual points']));c.metric('Above / below forecast',f'{v["Above / below"]:+.1f}');st.caption('Ranked by actual minus expected points. A positive gap alone does not establish why the club exceeded expectations.');st.dataframe(tot.round(1),use_container_width=True)
with t2:
 st.subheader('Season scorecard');club2=st.selectbox('Evaluate a club',['League-wide']+teams);z=r if club2=='League-wide' else r[(r.home==club2)|(r.away==club2)];correct=z['pick']==z.actual;ss=st.columns(3);ss[0].metric('Forecast accuracy',f'{correct.mean():.1%}');ss[1].metric('Correct picks',f'{correct.sum()} / {len(z)}');ss[2].metric('Draws identified',f'{((z["pick"]=="Draw")&(z.actual=="Draw")).sum()} / {(z.actual=="Draw").sum()}');base=[('Matchday Lab',correct.mean()),('Always home',(z.actual=='Home win').mean()),('Closing favorite · odds available',(z.loc[z.market!='','market']==z.loc[z.market!='','actual']).mean())];
 if club2=='League-wide':base.append(('Original app · same season',170/380))
 st.dataframe(pd.DataFrame([(k,f'{v:.1%}') for k,v in base],columns=['Approach','Accuracy']),hide_index=True);st.markdown('#### Where did the model miss?');cm=pd.crosstab(z['pick'],z.actual).reindex(index=LABELS,columns=LABELS,fill_value=0);cm.index.name='Prediction ↓ / actual →';st.table(cm);st.caption('Read across a row: the diagonal is correct. The draw row counts games the model actually called draws.')
 with st.expander('How the model was chosen'):
  st.write('Trained through 2021/22. Selected the lowest log loss on 2022/23–2023/24, then refit through 2023/24. The table is validation, not test. Lower log loss rewards better probabilities.');st.dataframe(pd.DataFrame(meta['validation']).round(4),hide_index=True);st.write(f"Selected: {meta['selection']} · {len(m['features'])} inputs. 2024/25 is a historical comparison, not a new blind test.")
 st.download_button('Download these predictions',z[['date','home','away',*LABELS,'pick','actual']].to_csv(index=False),'matchday_predictions.csv')
with t3:
 st.subheader('Build a matchup');st.write('Choose two clubs from the 2024/25 field. Only games before your chosen date count toward form. This is a scenario, not a live schedule.');a,b,c=st.columns(3);h=a.selectbox('Home club',teams,index=teams.index('Arsenal'));away=b.selectbox('Away club',[t for t in teams if t!=h]);day=pd.Timestamp(c.date_input('Kickoff date',value=(last+pd.Timedelta(days=1)).date(),min_value=dt.date(2024,8,1)));past=g[g.match_date<day];cut=past.match_date.max();st.caption(f'Latest completed match used: {cut:%d %b %Y} • No injuries, transfers or confirmed lineups included.')
 if day>last:st.warning('Beyond workbook coverage: this is a what-if using frozen team history, not a current-season forecast.')
 hist=History();
 for row in past.itertuples():hist.add(row)
 yr=day.year-int(day.month<7);feat,_=hist.fixture(h,away,day,f'{yr}-{str(yr+1)[-2:]}');bars(predict(m,pd.DataFrame([feat]))[0],h,away)
