import matplotlib.pyplot as plt
from db import init
from dotenv import load_dotenv
from bson.objectid import ObjectId
import numpy as np
import pandas as pd

load_dotenv()

# client = init()
# db = client["lychte"]

df = pd.read_csv('./oracle-test/norm-5-no-update/usages.csv')

cpu_percentages = df["cpu_percentage"].to_numpy()
memory_percentages = df["memory_percentage"].to_numpy()
memory_percentages = memory_percentages[7:len(memory_percentages)]

# plt.plot(range(len(cpu_percentages)), cpu_percentages)
# plt.plot(range(len(labels)), currentPredictions, "y")
plt.plot(range(len(memory_percentages)), memory_percentages)
plt.title("Usage Plot")
plt.ylabel("Memory Percentage Usage")
plt.ylim(top=100)  # adjust the top leaving bottom unchanged
plt.ylim(bottom=0)  # adjust the bottom leaving top unchanged

plt.show()
