import pandas as pd
import plotly.express as px

df = pd.read_csv('heart.csv')
df.describe()
df.head(10)
print (df)

# Select your Variables for plotting
print(df[['thal', 'age']])
print(df[['thal', 'cp']])
print(df[['thal', 'chol']])
df = pd.read_csv('heart.csv')
fig = px.line(df, x = 'thal', y = 'cp', title='sampletest')
fig1 = px.line(df, x = 'thal', y = 'age', title='sampletest')
fig2= px.line(df, x = 'thal', y = 'chol', title='sampletest')

# PrintResult
#fig.show()
#fig1.show()
#fig2.show()

fig3 = px.scatter(df,x='target', y='thal',color='red',title='sampletest')
fig3.show()
