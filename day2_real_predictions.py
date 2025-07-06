import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

print("🌴 Love Island AI - Day 2: Real Predictions (FIXED)")

# Your real comments (replace with your actual collected data)
real_comments = [
    {"text": "Ace and Chelley overplayed their hand and now they're scrambling", "couple": "Chelley and Ace", "sentiment": "negative"},
    {"text": "Cackling at Ace and Chelley saying whole america loves them", "couple": "Chelley and Ace", "sentiment": "negative"},
    {"text": "Does anyone else feel like Ace and Chelley give off sibling vibes", "couple": "Chelley and Ace", "sentiment": "negative"},
    {"text": "Chelley tryna gas Chris up now that she's seen Ace's true colors", "couple": "Chelley and Ace", "sentiment": "negative"},
    {"text": "Huda's actions in the past weren't okay, but I genuinely think she and Chris have a good connection", "couple": "Chris and Huda", "sentiment": "positive"},
    {"text": "Amaya and Clarke have such natural chemistry together", "couple": "Amaya and Clarke", "sentiment": "positive"},
    {"text": "Pepe and Iris seem forced, like they're just trying to stay in the villa", "couple": "Pepe and Iris", "sentiment": "negative"},
    {"text": "Amaya and Zak had zero compatibility from day one", "couple": "Amaya and Zak", "sentiment": "negative"},
    {"text": "Chelley and Chris would never work, too much drama", "couple": "Chelley and Chris", "sentiment": "negative"},
    {"text": "Amaya and Bryan are actually really sweet together", "couple": "Amaya and Bryan", "sentiment": "positive"},
    {"text": "Zak and Olandria have great energy, they balance each other", "couple": "Zak and Olandria", "sentiment": "positive"},
    {"text": "Ace is just using Chelley for the game", "couple": "Chelley and Ace", "sentiment": "negative"},
    {"text": "Chris and Huda's connection feels genuine despite the drama", "couple": "Chris and Huda", "sentiment": "positive"},
    {"text": "Pepe seems like he's settling for Iris", "couple": "Pepe and Iris", "sentiment": "negative"},
    {"text": "Amaya looks happiest with Bryan, they complement each other", "couple": "Amaya and Bryan", "sentiment": "positive"}
]

# Convert to DataFrame
df = pd.DataFrame(real_comments)
print(f"\n📊 Collected {len(df)} comments about Love Island couples")
print(df.head())

# Show sentiment distribution
print(f"\n📈 Sentiment Distribution:")
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)

# Check if we have enough data for each sentiment
min_samples = min(sentiment_counts.values)
print(f"\n⚠️  Minimum samples per class: {min_samples}")

if min_samples < 2:
    print("🚨 Warning: Not enough samples for reliable training. Need at least 2 samples per sentiment.")
    print("Consider collecting more balanced data or using a different approach.")

# Create features from text using TF-IDF
print(f"\n🔍 Converting text to AI-readable features...")
vectorizer = TfidfVectorizer(
    max_features=50,  # Reduced for small dataset
    stop_words='english',
    ngram_range=(1, 2),  # Include both single words and pairs
    min_df=1,  # Include words that appear at least once
    max_df=0.8  # Exclude words that appear in more than 80% of documents
)

X = vectorizer.fit_transform(df['text'])
y = df['sentiment']

# Show the most important features
feature_names = vectorizer.get_feature_names_out()
print(f"\n🔤 Found {len(feature_names)} text features")

# Use stratified split to ensure both classes are represented
if len(sentiment_counts) > 1 and min_samples >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,  # Smaller test size for small dataset
        random_state=42,
        stratify=y  # Ensure both classes in train and test
    )
    
    # Train the model
    print(f"\n🧠 Training your Love Island prediction model...")
    model = MultinomialNB(alpha=0.1)  # Smoothing for small dataset
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy: {accuracy:.2%}")
    
    # Show detailed results
    print(f"\n📋 Detailed Results:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
else:
    # For very small datasets, train on all data
    print(f"\n🧠 Training on all data (dataset too small for train/test split)...")
    model = MultinomialNB(alpha=0.1)
    model.fit(X, y)
    
    # Test on training data (not ideal but necessary for small datasets)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\n🎯 Training Accuracy: {accuracy:.2%}")

# Test with new comments
print(f"\n🔮 Testing with new comments:")
test_comments = [
    "They have amazing chemistry",
    "He's being so fake with her", 
    "I love their connection",
    "This is so forced",
    "They're perfect together"
]

for comment in test_comments:
    comment_vector = vectorizer.transform([comment])
    prediction = model.predict(comment_vector)[0]
    probability = model.predict_proba(comment_vector)[0]
    confidence = max(probability)
    print(f"Comment: '{comment}'")
    print(f"Prediction: {prediction} (confidence: {confidence:.2%})")
    print()

# Show most important words for each sentiment
print(f"\n🔍 Most Important Words for Predictions:")
if hasattr(model, 'feature_log_prob_'):
    feature_names = vectorizer.get_feature_names_out()
    
    for i, sentiment in enumerate(model.classes_):
        # Get top features for this sentiment
        top_indices = model.feature_log_prob_[i].argsort()[-10:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"{sentiment.upper()}: {', '.join(top_words)}")

# Analyze each couple's breakup risk
print(f"\n💔 Couple Breakup Risk Analysis:")
couple_risks = []

for couple in df['couple'].unique():
    couple_data = df[df['couple'] == couple]
    negative_count = len(couple_data[couple_data['sentiment'] == 'negative'])
    total_count = len(couple_data)
    risk_percentage = (negative_count / total_count) * 100
    
    # Get average confidence from our model
    couple_texts = couple_data['text'].tolist()
    if couple_texts:
        couple_vectors = vectorizer.transform(couple_texts)
        predictions = model.predict_proba(couple_vectors)
        
        # Calculate average negative probability
        negative_class_index = list(model.classes_).index('negative') if 'negative' in model.classes_ else 0
        avg_negative_prob = np.mean([p[negative_class_index] for p in predictions])
        
        couple_risks.append({
            'couple': couple,
            'risk_percentage': risk_percentage,
            'ai_confidence': avg_negative_prob,
            'comment_count': total_count
        })

# Sort by risk percentage
couple_risks.sort(key=lambda x: x['risk_percentage'], reverse=True)

print(f"\n🏆 COUPLE RANKINGS (Most to Least Risk):")
for i, risk in enumerate(couple_risks, 1):
    risk_emoji = "🔥" if risk['risk_percentage'] >= 75 else "⚠️" if risk['risk_percentage'] >= 50 else "💚"
    print(f"{i}. {risk_emoji} {risk['couple']}: {risk['risk_percentage']:.0f}% breakup risk")
    print(f"   AI confidence: {risk['ai_confidence']:.2%} | Comments analyzed: {risk['comment_count']}")

# Data quality insights
print(f"\n📊 Data Quality Report:")
print(f"Total comments: {len(df)}")
print(f"Unique couples: {len(df['couple'].unique())}")
print(f"Sentiment balance: {dict(sentiment_counts)}")
print(f"Average comment length: {df['text'].str.len().mean():.0f} characters")

# Predictions for tonight's episode
print(f"\n🔮 PREDICTIONS FOR TONIGHT'S EPISODE:")
high_risk_couples = [r for r in couple_risks if r['risk_percentage'] >= 75]
if high_risk_couples:
    print(f"🚨 Watch out for drama with: {', '.join([r['couple'] for r in high_risk_couples])}")

safe_couples = [r for r in couple_risks if r['risk_percentage'] <= 25]
if safe_couples:
    print(f"💚 Likely to stay strong: {', '.join([r['couple'] for r in safe_couples])}")

print(f"\n✅ Day 2 Complete! Your AI model is now making real predictions!")
print(f"💡 Next steps: Watch tonight's episode and see if your predictions come true!")