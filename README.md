# Sports-Music-Classification
This project is focused on classifying music tracks based on their audio features into four different categories: NHL, NFL, FIFA, and NBA. The goal is to explore the relationship between musical attributes and sports playlists. This porject involves data collection using the Spotify API, data processing, feature analysis, and machine learning classification.

### Data Collection
This project collects data from Spotify soundtrack playlists related to various sport video games (EA Sports NHL, Madden, FC/FIFA and NBA2K). These playlists contain music tracks, and infomration about these tracks was extracted using the Spotify API.

### Data Analysis
This project expolres the audio features of the music tracks, including:
- Acousticness
- Danceability
- Energy
- Instrumentalness
- Liveness
- Loudness
- Speechiness
- Valence

These features are analyzed and visualized through bar graphs, histograms, and heatmaps for each sports genere. 

### Machine Learning
This project employs machine learning techniques to classify music tracks into sports categories. 
- K-Nearest-Neighbor(KNN)
  - KNN is used for classification, and the optimal value of K is determined using a cross-validation approach.
- Random Forest
  - Random Forest is another classification model that is utilized for this task.
- Voting Classifier
  - A Voting CLassifier combines the predictions from KNN and Random Forest to imporve classification accuracy.
- Neural Network
  - A neural network model is developed to classify tracks. This model consists of multiple layers and uses various activations functions.

### Model Evaluation
The accuracy of each model is evaluated, and confusion matrices are generated to understand the classification performance.

### Feature Importance
Feature Importance is analzyed to identify which audio features contribute the most to the classification task.
