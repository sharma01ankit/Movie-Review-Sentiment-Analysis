# Movie Review Sentiment Analysis

### Overview
This project involves sentiment analysis of movie reviews, predicting whether a given review is positive or negative. The dataset contains thousands of reviews labeled as "positive" or "negative." The focus is on building a robust NLP pipeline and experimenting with various machine learning models for text classification.

---

### Features
- Text preprocessing, including tokenization, stopword removal, stemming, and lemmatization.
- Feature extraction using Bag of Words and TF-IDF techniques.
- Model training and evaluation using Logistic Regression, Random Forest, and Gradient Boosting.
- Performance metrics: accuracy, precision, recall, F1-score, and ROC Curve.

---

### Approach
1. **Data Preprocessing**:
   - Removed duplicates from the dataset.
   - Cleaned the text by removing special characters, stopwords, and HTML tags.
   - Normalized text using stemming and lemmatization.

2. **Feature Engineering**:
   - Bag of Words (CountVectorizer).
   - TF-IDF (Term Frequency-Inverse Document Frequency).

3. **Model Training**:
   - Logistic Regression (baseline model).
   - Random Forest for feature importance and non-linear decision boundaries.
   - Gradient Boosting for performance improvement.

4. **Evaluation**:
   - Confusion Matrix.
   - ROC Curve.
   - Classification Report.

---

### Results
- Logistic Regression achieved **63% accuracy** on the test set.
- Gradient Boosting showed competitive performance with fine-tuned hyperparameters.
- TF-IDF features provided better results than Bag of Words.

---

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/Movie_Review_Sentiment.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### Usage
1. Place the dataset in the `data/` folder.
2. Run the script:
   ```bash
   python src/movie_review_analysis.py
   ```
3. Review the outputs, including performance metrics and visualizations.

---

### Dependencies
- `pandas`
- `numpy`
- `nltk`
- `scikit-learn`
- `matplotlib`

---

### Visual Outputs
Include visual outputs like:
1. ROC Curve.
2. Confusion Matrix.
3. Sample word cloud (if available).

---

### License
This project is licensed under the MIT License.
