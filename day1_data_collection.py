import requests
import json
import pandas as pd
from datetime import datetime

# This is your first Love Island AI code!
print("🌴 Love Island AI - Day 1 Data Collection")

# We'll start with Reddit data (no API needed for now)
# First, let's create some sample data to test with

# Sample Love Island comments (you'll replace this with real data)
sample_comments = [
    {"text": "They have no chemistry, it's so forced", "couple": "Couple A", "sentiment": "negative"},
    {"text": "I love how genuine their connection is", "couple": "Couple B", "sentiment": "positive"},
    {"text": "He's giving me the ick, she deserves better", "couple": "Couple C", "sentiment": "negative"},
    {"text": "They're perfect for each other, so cute", "couple": "Couple D", "sentiment": "positive"},
    {"text": "This relationship is toxic af", "couple": "Couple E", "sentiment": "negative"},
    {"text": "Their connection is so deep and real", "couple": "Couple F", "sentiment": "positive"}
]

# Convert to DataFrame (like Excel spreadsheet)
df = pd.DataFrame(sample_comments)

# Display the data
print("\n📊 Sample Love Island Comments:")
print(df)

# Count positive vs negative
sentiment_counts = df['sentiment'].value_counts()
print(f"\n📈 Sentiment Analysis:")
print(f"Positive comments: {sentiment_counts.get('positive', 0)}")
print(f"Negative comments: {sentiment_counts.get('negative', 0)}")

# Calculate breakup risk for each couple
print(f"\n💔 Breakup Risk Analysis:")
for couple in df['couple'].unique():
    couple_data = df[df['couple'] == couple]
    negative_count = len(couple_data[couple_data['sentiment'] == 'negative'])
    total_count = len(couple_data)
    risk_percentage = (negative_count / total_count) * 100
    print(f"{couple}: {risk_percentage:.0f}% breakup risk")

print(f"\n✅ Day 1 Complete! You just ran your first Love Island AI analysis!")