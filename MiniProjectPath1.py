import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


'''
 The following is the starting code for path1 for data reading to make your first step easier.
 'dataset_1' is the clean data for path1.
'''


with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_1 = df
#print(dataset_1[15620:25350].to_string()) #This line will print out the first 35 rows of your data



# Part 1:
dataset_1.columns
#remove (userIDS)students that watched less than five videos
l=(dataset_1['userID'].value_counts()>=5).index #counts the index 
p=pandas.DataFrame(dataset_1['userID'].value_counts()>=5) #makes separate dataframe for various data points which have vid counts >= 5
p.reset_index(inplace=True) 
l=list(p[p['userID']==True]['index'].values) #this prints out user IDsthat watched 5 videos or more

#creating new dataframe with only valid UserIDS.
m=pandas.DataFrame(columns=list(dataset_1.columns)) #making changes to coloums. providing list of coulms through a for loop
for i in list(dataset_1['userID'].value_counts().index): # couloums will remain the same, the data will be updated. This removes user IDs with count less than 5
    if i in l: #this looks at the original data and checks from that 
        m=pandas.concat([m, dataset_1[dataset_1['userID']==i]], axis=0) #concatates userIds only in l. We will not add the counts less than 5


x = dataset_1[['VidID','fracSpent','fracComp','fracPlayed','fracPaused','numPauses','avgPBR','stdPBR','numRWs','numFFs','s']]
model = KMeans(n_clusters= 6,max_iter=300,random_state=50)
model.fit(x) #fit the k means object to the data
centroid = model.cluster_centers_
labels = model.labels_
score = silhouette_score(x,labels)
print("The final score that we have gotten from compiling a k means with 6 clusters is: ")
print(score)
print("The closer the score is to 1, the better it is ")

wcss = [] #Plotting the elbow method
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(wcss)
plt.plot(wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Part 2:
#counting s(i.e. s=1) for every userID
cnt=m.groupby('userID')['s'].sum()
cnt
m['s'].sum()
m[m['userID']=='3a06c1d4cadf969114ac27d0e8743c32984ca9ff'] #displaying records from m with provided userID

# group rows by user ID and calculate sum of scores for each group
sum_scores = m.groupby("userID")["s"].sum()
avg_fracSpent = m.groupby("userID")["fracSpent"].mean()
avg_fracPause=m.groupby("userID")["fracPaused"].mean()

# create new column with sum of scores for each user ID
m["sum_score"] = m["userID"].map(sum_scores)
m["avg_fracSpent"] = m["userID"].map(avg_fracSpent)
m["avg_fracPause"] = m["userID"].map(avg_fracPause)

print("This is the updated datframe:")
print(m)

#Finding score for each userID for checking relation with fracSpent and fracPaused using Linear Regression model
s=[]
for i in l:
    s.append(m[m['userID']==i]['s'].sum()/len(m[m['userID']==i]))
s

#created new dataframe for specified columns in Task 2 to feed in Linear Regression
df_avgPer=m[['userID','avg_fracSpent','avg_fracPause','sum_score']]
df_avgPer=df_avgPer.drop_duplicates('userID')
df_avgPer['final_score']=s
print("This is the new dataframe that was used in the linear regression model")
print(df_avgPer)

# Model 1: Linear regression for relation between fracSpent and final_score
model1=LinearRegression()  #creating model for relation between fracSpent and finalScore
model1.fit(df_avgPer[['avg_fracSpent']],df_avgPer['final_score'])
#finding score
model1.score(df_avgPer[['avg_fracSpent']],df_avgPer['final_score'])
yp=model1.predict(df_avgPer[['avg_fracSpent']])
#finding r2 (r squared) score 
score = r2_score(df_avgPer['final_score'],yp)
print("The r square score is:")
print(score)
#finding mse
meansquared= mean_squared_error(df_avgPer['final_score'],yp)
print("The mean suqared score is:")
print(meansquared)

# # Model 2: Linear regression for relation between fracPaused and final_score
model2=LinearRegression()  #creating model for relation between fracPaused and finalScore
model2.fit(df_avgPer[['avg_fracPause']],df_avgPer['final_score'])
#finding score
model2.score(df_avgPer[['avg_fracPause']],df_avgPer['final_score'])
yp=model2.predict(df_avgPer[['avg_fracPause']])
#finding r2 (r squared) score 
r2_score(df_avgPer['final_score'],yp)
#finding mse
mean_squared_error(df_avgPer['final_score'],yp)


#  Part 3: Visualization using scatter plot

#visualization of relation between fracPaused and fracSpent
sns.scatterplot(m['fracSpent'],m['fracPaused'])
#visualization for relation between fracSpent and final_score
sns.scatterplot(df_avgPer['avg_fracSpent'],df_avgPer['final_score'])
#visualization of relation between fracPaused and final_score
sns.scatterplot(df_avgPer['avg_fracPause'],df_avgPer['final_score'])

# ### As we can see in above graphs, some outliers are available in the dataset which might be affecting the score of the models. 




