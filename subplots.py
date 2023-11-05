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

fig = plt.figure()

plt0 = fig.add_subplot(111)
plt0.set_ylabel("Response Time (ms)", labelpad=15)
plt0.spines['top'].set_color('none')
plt0.spines['bottom'].set_color('none')
plt0.spines['left'].set_color('none')
plt0.spines['right'].set_color('none')
plt0.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


plt1 = fig.add_subplot(221)
plt1.plot(range(420,610), timings[420:610])
plt1.scatter(range(420,610), list(timings[i] if anomallies[i] == 1 else None for i in range(420,610)), facecolors="none", edgecolors="r", zorder=999)
plt1.title.set_text("a.")

plt2 = fig.add_subplot(222)
plt2.plot(range(1650,1750), timings[1650:1750])
plt2.scatter(range(1650,1750), list(timings[i] if anomallies[i] == 1 else None for i in range(1650,1750)), facecolors="none", edgecolors="r", zorder=999)
plt2.title.set_text("b.")

plt3 = fig.add_subplot(223)
plt3.plot(range(3170,3600), timings[3170:3600])
plt3.scatter(range(3170,3600), list(timings[i] if anomallies[i] == 1 else None for i in range(3170,3600)), facecolors="none", edgecolors="r", zorder=999)
plt3.title.set_text("c.")

plt4 = fig.add_subplot(224)
plt4.plot(range(4050,4100), timings[4050:4100])
plt4.scatter(range(4050,4100), list(timings[i] if anomallies[i] == 1 else None for i in range(4050,4100)), facecolors="none", edgecolors="r", zorder=999)
plt4.title.set_text("d.")

# plt.plot(range(len(labels)), timings)
# # plt.plot(range(len(labels)), currentPredictions, "y")
# plt.scatter(range(len(labels)), list(timings[i] if anomallies[i] == 1 else None for i in range(len(timings))), facecolors="none", edgecolors="r", zorder=999)
# plt.title("Real-Time Anomally Detection Results")
# plt.ylabel("Response Time (ms)")

plt.show()
