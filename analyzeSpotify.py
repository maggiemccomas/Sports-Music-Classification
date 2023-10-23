
# Import libraries

import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l1
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import MinMaxScaler
from keras.losses import CategoricalCrossentropy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Load data in from csv files

def load_sport_data(sport_name):
    sportDF = pd.DataFrame()

    csv_files = [file for file in os.listdir('Data') if sport_name in file and file.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join('Data', file)
        df = pd.read_csv(file_path, index_col = 0)
        df['sport'] = sport_name

        sportDF = pd.concat([sportDF, df], ignore_index = True)
    
    return sportDF

nhlDF = load_sport_data('nhl')
nflDF = load_sport_data('nfl')
fifaDF = load_sport_data('fifa')
nbaDF = load_sport_data('nba')

# bar graph

def plot_avg_audio_features(df, title, bar_color, outline, file):
    avg_audio_vibes = df[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']]
    plt.figure(figsize=(10, 5))
    avg_audio_vibes.mean().plot.bar(color = bar_color, edgecolor= outline)
    plt.title(title)
    plt.xticks(rotation=0)
    plt.savefig(file)

plot_avg_audio_features(nhlDF, 'NHL - Mean Values of Audio Features', '#95A3AE', '#8394A1', 'NHLMeanFeatures.png')
plot_avg_audio_features(nflDF, 'NFL - Mean Values of Audio Features', '#D50A0A', '#a40808', 'NFLMeanFeatures.png')
plot_avg_audio_features(fifaDF, 'FIFA - Mean Values of Audio Features', '#005391', '#002745', 'FIFAMeanFeatures.png')
plot_avg_audio_features(nbaDF, 'NBA - Mean Values of Audio Features', '#C9082A','#7f051b', 'NBAMeanFeatures.png')

# histograms

dataframes = [nhlDF, nflDF, fifaDF, nbaDF]

for df in dataframes:
    df['length'] = (df['length'] / 60000).round(0)

features_hist = ['length', 'popularity', 'acousticness', 'danceability', 'energy',
                 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

def plot_feature_distribution(df, bar_color, outline_color, title):
    for feature in features_hist:
        sns.displot(df, x=feature, kde=True, color=bar_color, edgecolor=outline_color, height = 6)
        plt.title(title)
        plt.tight_layout()
        filename = title[:3] + f'{feature}'
        plt.savefig(filename)

plot_feature_distribution(nhlDF, '#D1D7DB', '#95A3AE', 'NHL - Audio Feature Histogram')
plot_feature_distribution(nflDF, '#013369', '#011a36', 'NFL - Audio Feature Histogram')
plot_feature_distribution(fifaDF, '#005391', '#002745', 'FIFA - Audio Feature Histogram')
plot_feature_distribution(nbaDF, '#17408B', '#0c2249', 'NBA - Audio Feature Histogram')

# heatmap

features_to_plot = ['popularity', 'acousticness', 'danceability', 'energy', 'liveness', 'loudness',
                   'speechiness', 'tempo', 'valence']

def plot_heatmap(df, title, file):
    heat_map = df[features_to_plot]
    mask = np.triu(np.ones_like(heat_map.corr(), dtype=bool))
    f, ax = plt.subplots(figsize=(11, 8))
    cmap = sns.color_palette('vlag', as_cmap = True)
    sns.heatmap(heat_map.corr(), mask=mask, cmap=cmap, vmin=0, vmax=0.5,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
    plt.title(title)
    plt.savefig(file)

plot_heatmap(nhlDF, 'NHL - Audio Feature Heatmap', 'NHLHeatmap.png')
plot_heatmap(nflDF, 'NFL - Audio Feature Heatmap', 'NFLHeatmap.png')
plot_heatmap(fifaDF, 'FIFA - Audio Feature Heatmap', 'FIFAHeatmap.png')
plot_heatmap(nbaDF, 'NBA - Audio Feature Heatmap', 'NBAHeatmap.png')

# Combine the different dataframes

sports_musicDF = pd.concat([nhlDF, nflDF, fifaDF, nbaDF], ignore_index = True)


# Encode the sports: nhl - 0, nfl - 1, fifa - 2, nba -3

sports = sports_musicDF['sport'].unique()
sport_to_label = {sport: label for label, sport in enumerate(sports)}
sports_musicDF['sport_encoded'] = sports_musicDF['sport'].map(sport_to_label)

# Feature corolation with dependent variable

correlations = {}

for feature in features_to_plot:
    correlation = sports_musicDF[feature].corr(sports_musicDF['sport_encoded'])
    correlations[feature] = correlation

plt.figure(figsize=(10, 5))
plt.bar(correlations.keys(), correlations.values(), color='#0A3161')
plt.title('Audio Features Correlation with Sport')
plt.xlabel('Features')
plt.ylabel('Pearson Correlation')
plt.xticks(rotation=45)
plt.savefig('SportCorrelation.png')

# Scale data

scaler = MinMaxScaler()
columns_to_scale = ['popularity', 'loudness', 'tempo']
sports_musicDF[columns_to_scale] = scaler.fit_transform(sports_musicDF[columns_to_scale])

# Seperate into testin and training

X = sports_musicDF.drop(columns = ['name', 'album', 'artist', 'release_date', 'length', 'instrumentalness', 'time_signature', 'sport', 'sport_encoded']).values
y = sports_musicDF[['sport_encoded']].values.ravel()

random_state = random.randint(1, 1000) 

X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.2, random_state = random_state)
X_valid, X_test, y_valid, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state = random_state)

# K-Nearest-Neighbor

k_range = range(1, 50)
scores_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=10, shuffle=True)
    score_acc_list = []

    for train_index, test_index in kf.split(X_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        knn.fit(X_train_fold, y_train_fold)
        y_pred = knn.predict(X_test_fold)
        score_acc_list.append(accuracy_score(y_test_fold, y_pred))
    scores_list.append(np.mean(score_acc_list))


# Plot the results

plt.figure(figsize=(10, 6))
plt.plot(k_range, scores_list, marker='o', linestyle='-')
plt.title('KNN Testing Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Testing Accuracy')
plt.grid(True)
plt.xticks(k_range)
plt.savefig('KValue.png')

# KNN

best_k_index = scores_list.index(max(scores_list))
knn = KNeighborsClassifier(n_neighbors = best_k_index)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_valid)

# create a confusion matrix

cm = confusion_matrix(y_valid, y_pred_knn)

plt.figure(figsize=(10, 8))
ax = plt.subplot()
sns.heatmap(cm,annot=True,ax=ax)

labels = sports_musicDF['sport_encoded'].tolist()
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('KNN Confusion Matrix')
ax.xaxis.set_ticklabels(sports)
ax.yaxis.set_ticklabels(sports)
plt.savefig('KNNMatrix.png')

print("KNN Accuracy Score: ", accuracy_score(y_valid, y_pred_knn))

# Random Forest

rf = RandomForestClassifier(n_estimators = 100, random_state=2)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_valid)

# create a confusion matrix

cm = confusion_matrix(y_valid, y_pred_rf)

plt.figure(figsize=(10, 8))
ax = plt.subplot()
sns.heatmap(cm,annot=True,ax=ax)

labels = sports_musicDF['sport_encoded'].tolist()
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Random Forest Confusion Matrix')
ax.xaxis.set_ticklabels(sports)
ax.yaxis.set_ticklabels(sports)
plt.savefig('RandomMatrix.png')

print("Random Forest Accuracy Score: ", accuracy_score(y_valid, y_pred_rf))

# Voting Classifier

voting_classifier = VotingClassifier(estimators=[('knn', knn), ('rf', rf)], voting='hard')
voting_classifier.fit(X_train, y_train)
y_pred = voting_classifier.predict(X_test)

# create a confusion matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
ax = plt.subplot()
sns.heatmap(cm,annot=True,ax=ax)

labels = sports_musicDF['sport_encoded'].tolist()
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Voting Classifier Confusion Matrix')
ax.xaxis.set_ticklabels(sports)
ax.yaxis.set_ticklabels(sports)
plt.savefig('VotingMatrix.png')

print("Voting CLassifier Accuracy Score: ", accuracy_score(y_test, y_pred))

# Neural Network

def classification_model():
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(12, activation='relu', kernel_regularizer=l1(0.01)))
    model.add(Dense(24, activation='tanh'))
    model.add(Dense(4, activation='softmax'))
    opt = Adam(learning_rate=0.0001)
    loss = CategoricalCrossentropy()
    
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    return model

y_train_encoded = to_categorical(y_train, num_classes=4)
y_test_encoded = to_categorical(y_test, num_classes=4)

classifier = KerasClassifier(build_fn=classification_model, epochs=50, batch_size=16, verbose=0)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
history = classifier.fit(X_train, y_train_encoded, validation_split=0.05, callbacks=[es])
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Neural Network Accuracy Score: ", accuracy)

# Confusion Matrix

confusion = confusion_matrix(y_test_encoded.argmax(axis=1), y_pred.argmax(axis=1))

print("Confusion Matrix:")
print(confusion)

# correlation analysis

result = permutation_importance(classifier, X_test, y_test_encoded, n_repeats=30, random_state=0)
importance = result.importances_mean

plt.figure(figsize=(10, 10))
plt.bar(features_to_plot, importance, color = "#D50A0A", edgecolor = '#a40808')
plt.xlabel('Feature')
plt.ylabel('Permutation Importance')
plt.title('Feature Importance via Permutation')
plt.xticks(rotation=0)
plt.savefig('Permutation.png')