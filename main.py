import os
import sys
import schedule
import traceback
import time
import tensorflow as tf
import datetime
from bson.objectid import ObjectId
from dotenv import load_dotenv
from db import init, getLatestResponses, getApiOptions
from RePAD import RePAD_lstm, RePAD_predict, RePAD_threshold, AARE
from kafka import KafkaConsumer, KafkaProducer
from json import loads, dumps

# KAFKA_BROKER="localhost:9092"
# KAFKA_CONSUMER_TOPIC="usages"
# KAFKA_PRODUCER_TOPIC="updates"

load_dotenv()

kafka_broker = os.getenv("KAFKA_BROKER")
kafka_consumer_topic = os.getenv("KAFKA_CONSUMER_TOPIC")
kafka_producer_topic = os.getenv("KAFKA_PRODUCER_TOPIC")

b = 3

consumer = KafkaConsumer(
    kafka_consumer_topic,
    bootstrap_servers=[kafka_broker],
    auto_offset_reset='earliest',
    group_id='oracle',
    consumer_timeout_ms=10000,
    value_deserializer=lambda x: loads(x.decode("utf-8")))
producer = KafkaProducer(value_serializer=lambda v: dumps(v).encode('utf-8'))

def func(db, api_ids):
    for api_id in api_ids:
        options = getApiOptions(db=db, api_id=api_id)
        if options["isActive"] is not True: continue
        norm = options["norm"]

        responses = getLatestResponses(db, api_id, b)
        responses = responses[::-1]

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
            if len(aares) <= (b-1):
                y_pred, norm = RePAD_lstm(data=timings[len(timings)-b:len(timings)], api_id=api_id)[0:2]
                db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": { "anomallyDetection.norm": norm } })   

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
                    y_pred, norm, model = RePAD_lstm(data=timings[len(timings)-b-1:len(timings)-1], api_id=api_id, save=False)
                    u_hat[-1] = y_pred

                    aare = AARE(u=timings[len(timings)-b:len(timings)], u_hat=u_hat)
                    db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "AARE": aare, "currentPrediction": y_pred } })

                    thd = RePAD_threshold(db=db, api_id=api_id)

                    if (aare <= thd):
                        db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": {
                            "anomallyDetection.repadFlag": True,
                            "anomallyDetection.norm": norm
                        } })

                        db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "thd": thd } })
                        model.save("./models/{}.keras".format(api_id))
                    else:
                        print("changing flag to false")
                        db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": { "anomallyDetection.repadFlag": False } })
                        db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "isAnomally": True, "thd": thd } })

                        # latest_message = None
                        # while latest_message is None:
                        #     msg = next(consumer)
                        #     latest_message = msg
                        #     # key = str(msg.key, 'utf-8')
                        #     # if key == api_id: latest_message = msg

                        # cpu_percentage = latest_message.value.get("cpu_percentage")
                        # cpu_update = 1 if cpu_percentage > 80 else (-1 if cpu_percentage < 10 else 0)

                        # memory_percentage = latest_message.value.get("memory_percentage")
                        # memory_update = 1 if memory_percentage > 80 else (-1 if memory_percentage < 10 else 0)

                        # producer.send(kafka_producer_topic, key=latest_message.key, value={
                        #     "cpu_update": cpu_update,
                        #     "memory_update": memory_update
                        # })

                        # producer.flush()

            else:
                print("{} -> flag: false -> {}".format(datetime.datetime.now(), last_response.get("_id")))
                y_pred, norm, model = RePAD_lstm(data=timings[len(timings)-b-1:len(timings)-1], api_id=api_id, save=False)

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
                    print("chaning flag to true")
                    db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": {
                        "anomallyDetection.repadFlag": True,
                        "anomallyDetection.norm": norm,
                    } })
                    db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "thd": thd } })
                    model.save("./models/{}.keras".format(api_id))
                else:
                    print("chaning flag to false")
                    db.apis.update_one({ "_id": ObjectId(api_id) }, { "$set": { "anomallyDetection.repadFlag": False } })
                    db.responses.update_one({ "_id": ObjectId(last_response.get("_id")) }, { "$set": { "isAnomally": True, "thd": thd } })
    
                    # latest_message = None
                    # while latest_message is None:
                    #     msg = next(consumer)
                    #     latest_message = msg
                    #     # key = str(msg.key, 'utf-8')
                    #     # if key == api_id: latest_message = msg

                    # cpu_percentage = latest_message.value.get("cpu_percentage")
                    # cpu_update = 1 if cpu_percentage > 80 else (-1 if cpu_percentage < 20 else 0)

                    # memory_percentage = latest_message.value.get("memory_percentage")
                    # memory_update = 1 if memory_percentage > 80 else (-1 if memory_percentage < 20 else 0)

                    # producer.send(kafka_producer_topic, key=latest_message.key, value={
                    #     "cpu_update": cpu_update,
                    #     "memory_update": memory_update
                    # })

                    # producer.flush()
                    

if __name__ == "__main__":      
    client = init()
    db = client["lychte"]
    api_ids = os.getenv("API_IDS").split(",")

    try:
        schedule.every(5).seconds.do(lambda: func(db=db, api_ids=api_ids))

        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutdown requested...exiting")
    except Exception:
        traceback.print_exc(file=sys.stdout)


    client.close()
    sys.exit(0)