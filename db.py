from pymongo import MongoClient
from bson.objectid import ObjectId
import os
import sys
import traceback

# pymongo is synchronous - no need for await await
def init():
	try:
		return MongoClient(os.getenv("CLUSTER_URL"))
	except Exception as e:
		traceback.print_exc(file=sys.stdout)
		print("\nError: {}".format(e))
		sys.exit(1)

def getLatestResponses(db, api_id, b = 3):
	responses = db.responses.aggregate([
		{ "$match": { "api": ObjectId(api_id), "hasFailed": False, "hasError": False } },
		{ "$sort": { "createdAt": -1 } },
		{ "$limit": (2 * b) + 1 },	# b = 3 but we need at least 8 last responses to check on which step is currently the algorithm  
		{ "$project": {
			"timing": "$timings.duration",
			"futurePrediction": 1,
			"currentPrediction": 1,
			"AARE": 1,
			"thd": 1,
		} }
	])

	return list(responses)

def getApiOptions(db, api_id):
	api = db.apis.find_one({ "_id": ObjectId(api_id) }, { "anomallyDetection": 1 })
	return api["anomallyDetection"]