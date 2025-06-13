# Mount Drive first (only once per session)
from google.colab import drive
drive.mount('/content/drive')

# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from collections import Counter
from wordcloud import WordCloud
import re

# Load Data
file_path = '/content/drive/MyDrive/Colab Notebooks/cultural_fest_feedback.xlsx'
df = pd.read_excel(file_path)

# Data Cleaning
df.drop_duplicates(inplace=True)
df['Comments'] = df['Comments'].fillna("")
df['Year'] = pd.Categorical(df['Year'], categories=['First', 'Second', 'Third', 'Fourth'], ordered=True)

# Basic Stats
print("Average Overall Rating:", df['OverallRating'].mean())
print("Rating by Year:\n", df.groupby('Year')['OverallRating'].mean())
print("Rating by Department:\n", df.groupby('Department')['OverallRating'].mean())

# Average Rating by Department
plt.figure(figsize=(8,4))
sns.barplot(x='Department', y='OverallRating', data=df, estimator='mean', ci=None)
plt.title("Average Overall Rating by Department")
plt.xticks(rotation=45)
plt.ylim(0,5)
plt.show()

# Category-wise Rating Distributions
for cat in ['FoodRating', 'OrganizationRating', 'PerformanceRating', 'VenueRating']:
    plt.figure(figsize=(5,4))
    sns.countplot(x=cat, data=df, order=[1,2,3,4,5])
    plt.title(f"{cat} Distribution")
    plt.show()

# Sentiment Analysis
df['Sentiment'] = df['Comments'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['SentimentLabel'] = df['Sentiment'].apply(lambda s: "Positive" if s > 0.05 else "Negative" if s < -0.05 else "Neutral")

# Pie chart for sentiment
sent_counts = df['SentimentLabel'].value_counts()
plt.figure(figsize=(5,5))
plt.pie(sent_counts, labels=sent_counts.index, autopct='%1.0f%%', startangle=140)
plt.title("Sentiment Distribution")
plt.show()

# Sentiment by Department
plt.figure(figsize=(8,5))
sns.countplot(x='Department', hue='SentimentLabel', data=df)
plt.title("Sentiment by Department")
plt.xticks(rotation=45)
plt.show()

# WordClouds
positive_comments = " ".join(df[df['SentimentLabel'] == "Positive"]['Comments'])
negative_comments = " ".join(df[df['SentimentLabel'] == "Negative"]['Comments'])

plt.figure(figsize=(10,5))
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_comments)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Positive Comments")
plt.show()

plt.figure(figsize=(10,5))
wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_comments)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Negative Comments")
plt.show()

# Complaint Keywords after 'disliked'
words = []
for comment in df[df['SentimentLabel'] == 'Negative']['Comments']:
    match = re.search(r'disliked\s(.*)', comment, re.IGNORECASE)
    if match:
        text = match.group(1)
        for word in re.findall(r'\w+', text.lower()):
            if word not in ('and', 'the', 'a', 'an', 'was', 'is', 'it'):
                words.append(word)

complaints = Counter(words).most_common(10)
complaints_df = pd.DataFrame(complaints, columns=['Keyword', 'Frequency'])

# Barplot for complaint keywords
plt.figure(figsize=(8,4))
sns.barplot(x='Frequency', y='Keyword', data=complaints_df)
plt.title("Top Complaint Keywords")
plt.show()

# Save to Excel
output_path = "/content/drive/MyDrive/Colab Notebooks/processed_feedback.xlsx"
df.to_excel(output_path, index=False)
print("Saved to:", output_path)
