import matplotlib.pyplot as plt
from db import init
from dotenv import load_dotenv
from bson.objectid import ObjectId
import numpy as np
import pandas as pd

load_dotenv()

# client = init()
# db = client["lychte"]

df = pd.read_csv('./oracle-test/norm-2-correct-flags/data.csv')

labels = df["start"].to_numpy()
timings = df["timing"].to_numpy()
anomallies = df["isAnomally"].to_numpy()
currentPredictions = df["currentPrediction"].to_numpy()

plt.plot(range(len(labels)), timings)
# plt.plot(range(len(labels)), currentPredictions, "y")
plt.scatter(range(len(labels)), list(timings[i] if anomallies[i] == 1 else None for i in range(len(timings))), facecolors="none", edgecolors="r", zorder=999)
plt.title("Real-Time Anomally Detection Results")
plt.ylabel("Response Time (ms)")

plt.show()
