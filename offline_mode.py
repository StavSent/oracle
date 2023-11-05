import os
import sys
import traceback
import time
import datetime
from timeit import default_timer as timer
import statistics
import schedule
import tensorflow as tf
from bson.objectid import ObjectId
from dotenv import load_dotenv
from db import init, getApiOptions, getResposesById
from RePAD import RePAD_lstm, RePAD_predict, RePAD_threshold, AARE

load_dotenv()
b = 3

def offline_mode(db, api_ids):
    for api_id in api_ids:
        count = 0
        retrain_count = 0
        elapsed_time = []
        epochs = []

        all_responses = db.responses.find({ "api": ObjectId(api_id), "hasFailed": False, "hasError": False }, { "_id": 1 }).sort("createdAt", 1)
        all_response_ids = [x["_id"] for x in all_responses]
        response_id_batches = [all_response_ids[max(i-((b*2)+2), 0):i] for i in range(3, len(all_response_ids)+1)]
        for response_id_batch in response_id_batches:
            start = timer()
            count += 1
            print("progress: {}%", (count / len(response_id_batches) * 100))
            responses = getResposesById(db=db, ids=response_id_batch)
            responses = responses[::-1]

            options = getApiOptions(db=db, api_id=api_id)
            if options["isActive"] is not True: continue
            norm = options["norm"]
            # max_timing = options["max"] or 0
            # min_timing = options["min"] or 0

            last_response = responses[len(responses)-1]

            timings = [res.get("timing") for res in responses if res.get("timing") is not None]
            aares = [res.get("AARE") for res in responses if res.get("AARE") is not None]
            future_predictions = [res.get("futurePrediction") for res in responses if res.get("futurePrediction") is not None]
            current_predictions = [res.get("currentPrediction") for res in responses if res.get("currentPrediction") is not None]
            
            model = tf.keras.models("./models/{}".format(api_id)) if os.path.exists("./models/{}".format(api_id)) else None

            if ((last_response.get("AARE") is not None) or (last_response.get("futurePrediction") is not None)):
                print("{} successfully skipped {}".format(datetime.datetime.now(), last_response.get("_id")))
                continue

            if (len(timings) >= b):
                # if (max_timing == 0 and min_timing == 0):
                #     min_timing, max_timing = getMinMax(db=db, api_id=api_id)
                #     db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": {
                #         "anomallyDetection.repadFlag": True,
                #         "anomallyDetection.min": min_timing,
                #         "anomallyDetection.max": max_timing,
                #     } })


                if len(aares) <= (b-1):
                    y_pred, norm, n_epochs = RePAD_lstm(data=timings[len(timings)-b:len(timings)], api_id=api_id)[0:3]
                    epochs.append(n_epochs)
                    retrain_count += 1
                    db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": { "anomallyDetection.norm": norm } })
                    # new_min_timing, new_max_timing = getMinMax(db=db, api_id=api_id)
                    # db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": {
                    #     "anomallyDetection.min": new_min_timing,
                    #     "anomallyDetection.max": new_max_timing,
                    # } })    

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
                    print("{} -> flag: true -> {}".format(datetime.datetime.now(), last_response.get("_id")))
                    u_hat = []
                    u_pred = 0
                    print("currentPreodictions length: {}".format(len(current_predictions)))
                    if len(current_predictions) == 0:
                        u_hat = future_predictions[len(future_predictions)-b:len(future_predictions)]
                        u_pred = u_hat[-1]
                    elif len(current_predictions) == 1:
                        u_pred = RePAD_predict(x_pred=[timings[-3], timings[-2]], norm=norm, api_id=api_id)
                        u_hat = [future_predictions[-1], current_predictions[0], u_pred]
                    else:
                        u_pred = RePAD_predict(x_pred=[timings[-3], timings[-2]], norm=norm, api_id=api_id)
                        print("u_pred: {}".format(u_pred))
                        u_hat = current_predictions[len(current_predictions)-(b-1):len(current_predictions)]
                        u_hat.append(u_pred)
                    
                    aare = AARE(u=timings[len(timings)-b:len(timings)], u_hat=u_hat)
                    db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "AARE": aare, "currentPrediction": u_pred } })

                    thd = RePAD_threshold(db=db, api_id=api_id)

                    if (aare <= thd):
                        db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "thd": thd } })
                        continue
                    else:
                        y_pred, norm, n_epochs, model = RePAD_lstm(data=timings[len(timings)-b-1:len(timings)-1], api_id=api_id, save=False)
                        epochs.append(n_epochs)
                        retrain_count += 1
                        u_hat[-1] = y_pred

                        aare = AARE(u=timings[len(timings)-b:len(timings)], u_hat=u_hat)
                        db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "AARE": aare, "currentPrediction": y_pred } })

                        thd = RePAD_threshold(db=db, api_id=api_id)

                        if (aare <= thd):
                            # new_min_timing, new_max_timing = getMinMax(db=db, api_id=api_id)
                            db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": {
                                "anomallyDetection.repadFlag": True,
                                "anomallyDetection.norm": norm
                                # "anomallyDetection.min": new_min_timing,
                                # "anomallyDetection.max": new_max_timing,
                            } })

                            db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "thd": thd } })
                            model.save("./models/{}.keras".format(api_id))
                        else:
                            print("changing flag to false")
                            db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": { "anomallyDetection.repadFlag": False } })
                            db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "isAnomally": True, "thd": thd } })
                        
                else:
                    print("{} -> flag: false -> {}".format(datetime.datetime.now(), last_response.get("_id")))
                    y_pred, norm, n_epochs, model = RePAD_lstm(data=timings[len(timings)-b-1:len(timings)-1], api_id=api_id, save=False)
                    epochs.append(n_epochs)
                    retrain_count += 1

                    u_hat = []
                    if len(current_predictions) == 0:
                        u_hat = future_predictions[len(future_predictions)-b:len(future_predictions)]
                    elif len(current_predictions) == 1:
                        u_hat = [future_predictions[-1], current_predictions[0], y_pred]
                    else:
                        u_hat = current_predictions[len(current_predictions)-(b-1):len(current_predictions)]
                        u_hat.append(y_pred)

                    aare = AARE(u=timings[len(timings)-b:len(timings)], u_hat=u_hat)
                    db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "AARE": aare, "currentPrediction": y_pred } })
                    thd = RePAD_threshold(db=db, api_id=api_id)

                    if (aare <= thd):
                        # new_min_timing, new_max_timing = getMinMax(db=db, api_id=api_id)
                        print("chaning flag to true")
                        db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": {
                            "anomallyDetection.repadFlag": True,
                            "anomallyDetection.norm": norm,
                            # "anomallyDetection.min": new_min_timing,
                            # "anomallyDetection.max": new_max_timing,
                        } })
                        db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "thd": thd } })
                        model.save("./models/{}.keras".format(api_id))
                    else:
                        print("chaning flag to false")
                        db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": { "anomallyDetection.repadFlag": False } })
                        db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "isAnomally": True, "thd": thd } })

            end = timer()
            elapsed_time.append(end - start)

        print("retrain_count: {}".format(retrain_count))
        print("mean_elapsed_time {}".format(statistics.mean(elapsed_time)))
        print("std_elapsed_time {}".format(statistics.stdev(elapsed_time)))
        print("mean_number_of_epochs {}".format(statistics.mean(epochs)))

if __name__ == "__main__":      
    client = init()
    db = client["lychte"]
    api_ids = os.getenv("API_IDS").split(",")

    try:
        offline_mode(db=db, api_ids=api_ids)
    except KeyboardInterrupt:
        print("Shutdown requested...exiting")
    except Exception:
        traceback.print_exc(file=sys.stdout)


    client.close()
    sys.exit(0)