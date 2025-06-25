import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
df = pd.read_csv("C:\\Users\\ranat\\OneDrive\\Desktop\\Summer Training\\Datasets\\Iris.csv")

# Basic Insights
print("Dataframe:", df)
print("Info :",df.info())
print("Describe :",df.describe())
print("Top 10 rows",df.head(10))
print("No. of Rows and Columns ", df.shape)
print("Total Value Count of each variable :", df.count())
print("No. of Unique Values :",df.nunique())
print(df.isnull().sum())
duplicates = df[df.duplicated()]
print(duplicates)
df = df.drop_duplicates(keep='first', ignore_index=True)
print("Number of duplicates after removal:", df.duplicated().sum())

#0. countplot-count of each species
sns.countplot(data=df, x='Species', hue='Species', palette="rocket", legend=False)
plt.title("Count of Each Species")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

# 1. Pairplot
# Shows relationships between all feature pairs by species for pattern detection.
# Outcome: Clear species separation.

plt.figure(figsize=(6, 4))
sns.pairplot(df, hue='Species', diag_kind='hist', palette=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.suptitle('Pairplot of Iris Features', y=1.02)
plt.show()

# 2. 4-plot subplot with tight_layout
# Combines multiple views to compare distributions and trends efficiently. 
# Outcome: Distinct species characteristics.

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

#This confirms that petal length is a strong differentiator between setosa and the other two species.
sns.boxplot(x='Species', y='PetalLengthCm', data=df, ax=axes[0, 0], color='#FF9F55')
axes[0, 0].set_title('Box Plot of Petal Length')

#sepal length alone is less effective for clear species separation but shows a gradual increase in average length
sns.violinplot(x='Species', y='SepalLengthCm', data=df, ax=axes[0, 1], color='#6A0572')
axes[0, 1].set_title('Violin Plot of Sepal Length')

#combining sepal and petal length provides better species discrimination, with setosa being the most distinct
sns.scatterplot(x='SepalLengthCm', y='PetalLengthCm', hue='Species', data=df, ax=axes[1, 0], palette=['#FFD60A', '#00C4B4', '#8A2BE2'])
axes[1, 0].set_title('Scatter Plot: Sepal vs Petal Length')

#The overlap between versicolor and virginica suggests sepal width is less distinctive, but setosa’s narrower distribution sets it apart
sns.kdeplot(data=df[df['Species'] == 'Iris-setosa']['SepalWidthCm'], label='Setosa', ax=axes[1, 1], color='#FF2E63')
sns.kdeplot(data=df[df['Species'] == 'Iris-versicolor']['SepalWidthCm'], label='Versicolor', ax=axes[1, 1], color='#00FF00')
sns.kdeplot(data=df[df['Species'] == 'Iris-virginica']['SepalWidthCm'], label='Virginica', ax=axes[1, 1], color='#FF4500')
axes[1, 1].set_title('KDE Plot of Sepal Width')
axes[1, 1].legend()
plt.tight_layout()
plt.show()

# 3. Heatmap
# Displays feature correlations for relationship strength. 
# Outcome: High petal correlation, low sepal width correlation.
plt.figure(figsize=(8, 6))
corr_mat = df.drop(['Id', 'Species'], axis=1).corr()
sns.heatmap(corr_mat, annot=True, cmap='YlOrRd', vmin=-1, vmax=1, cbar_kws={'shrink': .5})
plt.title('Correlation Heatmap of Iris Features')
plt.show()

# 4. Bar plot
# Compares mean feature values by species for differentiation.
# Outcome: Setosa with smaller petals, virginica larger.

mean_vals = df.groupby('Species').mean()
mean_vals.drop('Id', axis=1).plot(kind='bar', figsize=(10, 6), color=['#FF1493', '#00CED1', '#FFA500', '#ADFF2F'])
plt.title('Mean Feature Values by Species')
plt.xlabel('Species')
plt.ylabel('Mean Value (cm)')
plt.xticks(rotation=45)
plt.legend(title='Features')
plt.show()

# Outlier detection and removal
# Removes outliers using IQR for cleaner data.
# Outcome: Reduced extreme values.

sns.boxplot(data=df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
plt.title("Boxplot to Detect Outliers in Iris Features")
plt.show()
Q1 = df.drop(['Id', 'Species'], axis=1).quantile(0.25)
Q3 = df.drop(['Id', 'Species'], axis=1).quantile(0.75)
IQR = Q3 - Q1
df_clean = df[~((df.drop(['Id', 'Species'], axis=1) < (Q1 - 1.5 * IQR)) | (df.drop(['Id', 'Species'], axis=1) > (Q3 + 1.5 * IQR))).any(axis=1)]

# Machine Learning Section
# Encodes labels, splits data, and uses KNN to classify, optimizing k.
# High accuracy with best k.

# Encode species
le = LabelEncoder()
df_clean['Species'] = le.fit_transform(df_clean['Species'])
for i, label in enumerate(le.classes_):
    print(f"{i} → {label}")  # 0 → Iris-setosa, 1 → Iris-versicolor, 2 → Iris-virginica

# Prepare data
X = df_clean.drop(['Id', 'Species'], axis=1)  # Features: SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm
y = df_clean['Species']  # Target: encoded species
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% test

# KNN with k=3
model = KNeighborsClassifier(n_neighbors=3)  # Uses 3 nearest neighbors for classification
model.fit(X_train, y_train)  # Trains the model
y_pred = model.predict(X_test)  # Predicts on test data
print("Accuracy with k=3:", accuracy_score(y_test, y_pred))  
print(classification_report(y_test, y_pred))  # showing how well each species is classified

# KNN with k=5
model1 = KNeighborsClassifier(n_neighbors=5)  # Uses 5 nearest neighbors
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print("Accuracy with k=5:", accuracy_score(y_test, y_pred1)) 
print(classification_report(y_test, y_pred1))  

# KNN with k=9
model2 = KNeighborsClassifier(n_neighbors=9)  # Uses 9 nearest neighbors
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("Accuracy with k=9:", accuracy_score(y_test, y_pred2))  # May show lower or stable accuracy (e.g., ~95%), as larger k smooths decisions
print(classification_report(y_test, y_pred2))  

# Optimize k
acc_scores = []  #store accuracy for each k
for k in range(1, 20):  # Tests k 
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    acc_scores.append(acc)  #accuracy for each k

# Plot accuracy vs k
plt.figure(figsize=(8, 6))
plt.plot(range(1, 20), acc_scores, marker='o', color='#1E90FF')  
plt.title('KNN Accuracy for Different k Values')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
 #Graph shows accuracy trend, with a peak helping choose the best k value
plt.show() 