```python
import pandas as pd
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# Load the dataset
try:
    df = pd.read_csv('youtube_trendings.csv')
except FileNotFoundError:
    print("Error: 'youtube_trendings.csv' not found. Please ensure the file is in the current directory.")
    exit()

# --- Step 1 & 2: Create title_word_count ---
# Convert 'Video' column to string type to handle potential non-string entries
df['Video'] = df['Video'].astype(str)
df['title_word_count'] = df['Video'].apply(lambda x: len(x.split()))

# --- Step 3: Clean 'Video views' column ---
# Remove commas from 'Video views'
df['Video views'] = df['Video views'].astype(str).str.replace(',', '', regex=False)
# Convert to numeric, coercing errors to NaN
df['Video views'] = pd.to_numeric(df['Video views'], errors='coerce')
# Fill resulting NaNs with 0
df['Video views'] = df['Video views'].fillna(0)

# --- Step 4: Calculate and print Pearson correlation coefficient ---
# Ensure both columns are numeric and have no NaNs for correlation calculation
# (title_word_count is already numeric, Video views is cleaned)
correlation = df['title_word_count'].corr(df['Video views'])
print(f"Pearson correlation between 'title_word_count' and 'Video views': {correlation:.4f}")
print("-" * 50)

# --- Step 5: Filter the dataset for videos published in 2023 or 2024 ---
# Convert 'published' column to datetime objects
df['published_datetime'] = pd.to_datetime(df['published'], errors='coerce')

# Filter for years 2023 or 2024
df_filtered_years = df[df['published_datetime'].dt.year.isin([2023, 2024])]

# --- Step 6 & 7: Process titles and find 5 most common words for filtered videos ---
all_words = []
stop_words = set(stopwords.words('english'))

# Iterate through titles of filtered videos
for title in df_filtered_years['Video']:
    # Convert to lowercase
    title = title.lower()
    # Remove punctuation (keep only letters and spaces)
    title = re.sub(r'[^\w\s]', '', title)
    # Split into words and remove stop words
    words = [word for word in title.split() if word not in stop_words and word.isalpha()] # .isalpha() to remove numbers/empty strings

    all_words.extend(words)

# Find the 5 most common words
if all_words:
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(5)
    print("5 most common words in titles from 2023/2024 videos (after processing):")
    for word, count in most_common_words:
        print(f"- '{word}': {count} occurrences")
else:
    print("No video titles found for 2023/2024 or no words remained after processing.")

```