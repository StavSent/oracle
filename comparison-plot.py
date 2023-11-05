import matplotlib.pyplot as plt
from db import init
from dotenv import load_dotenv
from bson.objectid import ObjectId
import numpy as np
import pandas as pd

load_dotenv()

# client = init()
# db = client["lychte"]

# df5_memory = pd.read_csv('./oracle-test/norm-5-no-update/usages.csv')
# df4_memory = pd.read_csv('./oracle-test/norm-4-live-update/usages.csv')

# original_size4 = len(df4_memory["memory_percentage"].to_numpy())
# original_size5 = len(df5_memory["memory_percentage"].to_numpy())

# memory_percentages5 = df5_memory["memory_percentage"].to_numpy()
# memory_percentages5 = memory_percentages5[317:len(memory_percentages5)]
# memory_percentages4 = df4_memory["memory_percentage"].to_numpy()
# memory_percentages4 = memory_percentages4[317:len(memory_percentages4)]

# print(len(memory_percentages4))
# print(len(memory_percentages5))

# plt.plot(range(317,original_size5), memory_percentages5)
# plt.plot(range(317,original_size4), memory_percentages4)

# plt.legend(["a. No Dynamic Resource Allocation", "b. With Dynamic Resource Allocation"])
# plt.title("Usage Plot")
# plt.ylabel("Memory Percentage Usage")
# plt.ylim(top=100)  # adjust the top leaving bottom unchanged
# plt.ylim(bottom=0)  # adjust the bottom leaving top unchanged

# plt.show()

df5_data = pd.read_csv('./oracle-test/norm-5-no-update/data.csv')
df4_data = pd.read_csv('./oracle-test/norm-4-live-update/data.csv')

labels5 = df5_data["start"].to_numpy()
labels5 = labels5[10:len(labels5)]
timings5 = df5_data["timing"].to_numpy()
timings5 = timings5[10:len(timings5)]
anomallies5 = df5_data["isAnomally"].to_numpy()
anomallies5 = anomallies5[10:len(anomallies5)]
currentPredictions5 = df5_data["currentPrediction"].to_numpy()
currentPredictions5 = currentPredictions5[10:len(currentPredictions5)]

labels4 = df4_data["start"].to_numpy()
timings4 = df4_data["timing"].to_numpy()
anomallies4 = df4_data["isAnomally"].to_numpy()
currentPredictions4 = df4_data["currentPrediction"].to_numpy()

plt.plot(range(len(labels5)), timings5)
plt.scatter(range(len(labels5)), list(timings5[i] if anomallies5[i] == 1 else None for i in range(len(timings5))), facecolors="none", edgecolors="r", zorder=999, label='_nolegend_')

plt.plot(range(len(labels4)), timings4)
plt.scatter(range(len(labels4)), list(timings4[i] if anomallies4[i] == 1 else None for i in range(len(timings4))), facecolors="none", edgecolors="g", zorder=999, label='_nolegend_')
plt.legend(["a. No Dynamic Resource Allocation", "b. With Dynamic Resource Allocation"], loc='upper left')
plt.title("Real-Time Anomally Detection Results")
plt.ylabel("Response Time (ms)")

plt.show()

