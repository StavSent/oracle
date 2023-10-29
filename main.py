import os
import sys
import schedule
import traceback
import time
import tensorflow as tf
from bson.objectid import ObjectId
from dotenv import load_dotenv
from db import init, getLatestResponses, getApiOptions
from RePAD import RePAD, RePAD_predict, RePAD_threshold, AARE

load_dotenv()
b = 3

def func(db, api_ids):
    for api_id in api_ids:
        options = getApiOptions(db=db, api_id=api_id)
        if options["isActive"] is not True: continue

        responses = getLatestResponses(db, api_id, b)
        responses = responses[::-1]

        last_response = responses[len(responses)-1]
        timings = [res.get("timing") for res in responses if res.get("timing") is not None]
        aares = [res.get("AARE") for res in responses if res.get("AARE") is not None]
        future_predictions = [res.get("futurePrediction") for res in responses if res.get("futurePrediction") is not None]
        current_predictions = [res.get("currentPrediction") for res in responses if res.get("currentPrediction") is not None]
        
        model = tf.keras.models("./models/{}".format(api_id)) if os.path.exists("./models/{}".format(api_id)) else None

        if ((last_response.get("AARE") is not None) or (last_response.get("futurePrediction") is not None)):
            continue

        if (len(timings) >= b):
            if len(aares) <= (b-1):
                y_pred = RePAD(data=timings[len(timings)-b:len(timings)], api_id=api_id)[0]
                print("y_pred: {}".format(y_pred))
                print("last response id: {}".format(last_response.get("_id")))

                aare = None
                if len(future_predictions) >= b:
                    aare = AARE(
                        u=timings[len(timings)-b:len(timings)],
                        u_hat=future_predictions[len(future_predictions)-b:len(future_predictions)]
                    )

                print("aare: {}".format(aare))
                update = { "$set": {
                    "futurePrediction": y_pred
                } }

                if aare is not None:
                    update["$set"]["AARE"] = aare

                db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, update)
            elif options["repadFlag"] is True:
                u_hat = []
                u_pred = 0
                if len(current_predictions) == 0:
                    u_hat = future_predictions[len(future_predictions)-b:len(future_predictions)]
                    u_pred = u_hat[-1]
                elif len(current_predictions) == 1:
                    u_pred = RePAD_predict(x_pred=timings[len(timings)-2], api_id=api_id)
                    u_hat = [future_predictions[-1], current_predictions[0], u_pred]
                else:
                    u_pred = RePAD_predict(x_pred=timings[len(timings)-2], api_id=api_id)
                    u_hat = current_predictions[len(current_predictions)-(b-1):len(current_predictions)]
                    u_hat.append(u_pred)
                
                aare = AARE(u=timings[len(timings)-b:len(timings)], u_hat=u_hat)
                db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "AARE": aare, "currentPrediction": u_pred } })

                thd = RePAD_threshold(db=db, api_id=api_id)

                if (aare <= thd):
                    # db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "isAnomally": False } })
                    continue
                else:
                    y_pred, model = RePAD(data=timings[len(timings)-b-1:len(timings)-1], api_id=api_id, save=False)
                    u_hat[-1] = y_pred

                    aare = AARE(u=timings[len(timings)-b:len(timings)], u_hat=u_hat)
                    db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "AARE": aare, "currentPrediction": y_pred } })

                    thd = RePAD_threshold(db=db, api_id=api_id)

                    if (aare <= thd):
                        db.api.update_one({ "_id": ObjectId(api_id) }, { "$set": { "anomallyDetection.repadFlag": True } })
                        model.save("./models/{}.keras".format(api_id))
                    else:
                        db.api.update_one({ "_id": ObjectId(api_id) }, { "$set": { "anomallyDetection.repadFlag": False } })
                        db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "isAnomally": True } })
                    
            else:
                y_pred, model = RePAD(data=timings[len(timings)-b-1:len(timings)-1], api_id=api_id, save=False)

                u_hat = []
                if len(current_predictions) == 0:
                    u_hat = future_predictions[len(future_predictions)-b:len(future_predictions)]
                elif len(current_predictions) == 1:
                    u_hat = [future_predictions[-1], current_predictions[0], u_pred]
                else:
                    u_hat = current_predictions[len(current_predictions)-(b-1):len(current_predictions)]
                    u_hat.append(u_pred)

                aare = AARE(u=timings[len(timings)-b:len(timings)], u_hat=u_hat)
                db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "AARE": aare, "currentPrediction": y_pred } })
                thd = RePAD_threshold(db=db, api_id=api_id)

                if (aare <= thd):
                    db.api.update_one({ "_id": ObjectId(api_id) }, { "$set": { "anomallyDetection.repadFlag": True } })
                    model.save("./models/{}.keras".format(api_id))
                else:
                    db.api.update_one({ "_id": ObjectId(api_id) }, { "$set": { "anomallyDetection.repadFlag": False } })
                    db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "isAnomally": True } })
    
                

if __name__ == "__main__":      
    client = init()
    db = client["lychte"]
    api_ids = os.getenv("API_IDS").split(",")

    try:
        schedule.every(30).seconds.do(lambda: func(db=db, api_ids=api_ids))

        while True:
            schedule.run_pending()
            time.sleep(10)
    except KeyboardInterrupt:
        print("Shutdown requested...exiting")
    except Exception:
        traceback.print_exc(file=sys.stdout)


    client.close()
    sys.exit(0)