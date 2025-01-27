import pandas as pd
import random
from datetime import datetime, timedelta

# Sample data generation
num_samples = 10

# Generating random song numbers and orders
song_numbers = [i for i in range(1, 21)]
data = []
for _ in range(num_samples):
    num_songs = random.randint(3, 6)
    input_order = random.sample(song_numbers, num_songs)
    output_order = random.sample(input_order, num_songs)
    rating = round(random.uniform(3.0, 5.0), 2)
    data.append((input_order, output_order, rating))

# Generating additional columns
dates = [datetime(2023, 7, 1) + timedelta(days=random.randint(0, 30)) for _ in range(num_samples)]
performance_metrics = [round(random.uniform(0.0, 100.0), 2) for _ in range(num_samples)]
themes = ["Christmas", "Easter", "Wedding", "Gospel", "Classical", "Contemporary"]
theme_genre = [random.choice(themes) for _ in range(num_samples)]

# Create a DataFrame with the generated data
df = pd.DataFrame(data, columns=["SongNum", "Order", "Rating"])
df["Order"] = df["Order"].apply(lambda x: str(x)[1:-1].replace(",", ""))
df["Date/Time"] = dates
df["Performance Metrics"] = performance_metrics
df["Theme/Genre"] = theme_genre

# Display the generated fake data
print(df)
