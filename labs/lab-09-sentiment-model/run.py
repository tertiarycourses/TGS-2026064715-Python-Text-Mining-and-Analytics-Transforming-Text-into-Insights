#!/usr/bin/env python3
"""Self-contained synthetic hotel-review text analytics lab. No network access required."""
from pathlib import Path
import csv, re, json, collections, math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
DATA=Path(__file__).with_name('hotel_reviews.csv')
OUT=Path(__file__).with_name('output'); OUT.mkdir(exist_ok=True)
MODE='sentiment'
df=pd.read_csv(DATA)
texts=df['text'].fillna('').astype(str)
print('rows',len(df),'unique IDs',df.review_id.nunique(),'missing text',df.text.isna().sum())
if MODE=='profile':
    print('CRISP-DM: business understanding -> data understanding -> preparation -> modelling -> evaluation -> deployment')
    print('Question: Which hotel/service aspects predict negative review share?')
    print(df.groupby('hotel')['sentiment'].value_counts(normalize=True).round(2))
if MODE=='ingest':
    print('schema',df.dtypes.to_dict()); print('duplicates',df.review_id.duplicated().sum())
    print('class balance',df.sentiment.value_counts().to_dict())
    assert df.review_id.is_unique and df.text.notna().all()
if MODE=='tokens':
    stops={'the','and','was','my','a','in','to','on','at','is'}
    for t in texts.head(8):
        raw=re.findall(r"[A-Za-z']+",t.lower())
        print({'raw':raw,'filtered':[x for x in raw if x not in stops]})
if MODE=='normalize':
    try:
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        stem=PorterStemmer(); lem=WordNetLemmatizer()
        words=['booking','booked','rooms','delayed','service','services']
        print([(w,stem.stem(w),lem.lemmatize(w)) for w in words])
    except (ImportError, LookupError):
        print('Install optional nltk with: python -m pip install nltk')
        print('Fallback vocabulary:',sorted(set(re.findall(r'[a-z]+',' '.join(texts).lower())))[:30])
if MODE=='tf':
    for ngram in [(1,1),(1,2)]:
        v=CountVectorizer(stop_words='english',ngram_range=ngram,min_df=2)
        X=v.fit_transform(texts)
        top=sorted(zip(X.sum(axis=0).A1,v.get_feature_names_out()),reverse=True)[:12]
        print('ngram',ngram,'shape',X.shape,'top',top)
if MODE=='tfidf':
    v=TfidfVectorizer(stop_words='english',min_df=2)
    X=v.fit_transform(texts)
    for i in [0,1,2]:
        pairs=sorted(zip(X[i].toarray().ravel(),v.get_feature_names_out()),reverse=True)[:8]
        print(df.review_id[i],pairs)
    print('IDF examples',[(w,round(v.idf_[v.vocabulary_[w]],3)) for w in ['room','booking'] if w in v.vocabulary_])
if MODE=='entities':
    hotels=['Harbour View Hotel','Orchard Central Hotel','Marina Bay Suites']
    aspects=['booking','room','breakfast','support','invoice','app']
    links=collections.Counter()
    for row in df.itertuples():
        for aspect in aspects:
            if aspect in row.text.lower(): links[(row.hotel,aspect)]+=1
    print('Entity-aspect links',links.most_common(15))
    (OUT/'links.json').write_text(json.dumps([{'hotel':h,'aspect':a,'count':n} for (h,a),n in links.items()],indent=2))
if MODE=='cluster':
    v=TfidfVectorizer(stop_words='english',min_df=2)
    X=v.fit_transform(texts)
    for k in [2,3,4,5]:
        m=KMeans(n_clusters=k,random_state=42,n_init=10).fit(X)
        print('k',k,'inertia',round(m.inertia_,2))
    pipe=make_pipeline(TfidfVectorizer(stop_words='english'),LogisticRegression(max_iter=1000))
    Xtr,Xte,ytr,yte=train_test_split(texts,df.sentiment,test_size=.25,stratify=df.sentiment,random_state=42)
    pipe.fit(Xtr,ytr); print(classification_report(yte,pipe.predict(Xte),zero_division=0))
if MODE=='sentiment':
    Xtr,Xte,ytr,yte=train_test_split(texts,df.sentiment,test_size=.25,stratify=df.sentiment,random_state=42)
    pipe=make_pipeline(TfidfVectorizer(stop_words='english',ngram_range=(1,2)),LogisticRegression(max_iter=1000))
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    print(classification_report(yte,pred,zero_division=0)); print('confusion',confusion_matrix(yte,pred,labels=pipe.classes_))
    pd.DataFrame({'text':Xte,'truth':yte,'prediction':pred}).to_csv(OUT/'holdout_predictions.csv',index=False)
if MODE=='summary':
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tab=pd.crosstab(df.hotel,df.sentiment,normalize='index').reindex(columns=['negative','neutral','positive']).fillna(0)
    print(tab.round(2)); tab.to_csv(OUT/'sentiment_by_hotel.csv')
    ax=tab.plot.bar(stacked=True,color=['#EF6B6B','#A9B8C2','#16A6A0'],figsize=(9,5))
    ax.set_ylabel('Share of reviews'); ax.set_ylim(0,1); plt.tight_layout(); plt.savefig(OUT/'sentiment_by_hotel.png',dpi=150)
    aspects=['booking','room','breakfast','support','invoice','app']
    issues={a:int(((df.sentiment=='negative') & texts.str.contains(a,case=False)).sum()) for a in aspects}
    print('negative aspects',issues)
