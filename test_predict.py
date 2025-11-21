import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load data
base = __import__('os').path.dirname(__import__('os').path.abspath(__file__))
df = pd.read_csv(__import__('os').path.join(base, 'data.csv'))
features = ['N','P','K','temperature','humidity','ph','rainfall']
X = df[features]
y = df['label']

model = DecisionTreeClassifier(random_state=42)
model.fit(X,y)

# Define some test user inputs
tests = [
    {'N':90,'P':42,'K':43,'temperature':20.8,'humidity':82.0,'ph':6.5,'rainfall':202.9},
    {'N':20,'P':80,'K':20,'temperature':18.8,'humidity':51.7,'ph':6.3,'rainfall':75.3},
    {'N':110,'P':48,'K':39,'temperature':24.9,'humidity':80.36,'ph':6.01,'rainfall':142.06},
    {'N':80,'P':20,'K':45,'temperature':26.6,'humidity':60.18,'ph':5.7,'rainfall':127.7},
]

for i,t in enumerate(tests,1):
    df_t = pd.DataFrame([t])
    pred = model.predict(df_t)[0]
    proba = None
    try:
        proba = model.predict_proba(df_t)[0]
    except Exception:
        pass
    print(f"Test {i}: input={t} -> pred={pred} proba={proba}")
