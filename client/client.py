import paho.mqtt.client as mqtt
import numpy as np

import json
import time
import sys

from trainer import Trainer

# total args
n = len(sys.argv)

# check args
if (n != 5):
    print("correct use: python client.py <broker_address> <name> <id> <trainer_mode>.")
    exit()

BROKER_ADDR     = sys.argv[1]
CLIENT_ID       = sys.argv[2]
CLIENT_NUMBER   = int(sys.argv[3])
MODE            = sys.argv[4]
# class for coloring messages on terminal


class color:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD_START = '\033[1m'
    BOLD_END = '\033[0m'
    RESET = "\x1B[0m"

# subscribe to queues on connection


def on_connect(client, userdata, flags, rc):
    subscribe_queues = ['minifed/selectionQueue',
                            'minifed/posAggQueue', 'minifed/stopQueue','minifed/args']
    for s in subscribe_queues:
        client.subscribe(s)

# callback for selectionQueue: if trainer gets chosen, then starts training, else just wait

def on_args(client, userdata, message):
    msg = json.loads(message.payload.decode("utf-8"))
    if msg['id'] == CLIENT_ID:
        trainer.set_args(msg['args'])
        client.publish('minifed/ready', json.dumps({"id":CLIENT_ID}))

def on_message_selection(client, userdata, message):
    msg = json.loads(message.payload.decode("utf-8"))
    if msg['id'] == CLIENT_ID:
        if bool(msg['selected']) == True:
            print(color.BOLD_START + 'new round starting' + color.BOLD_END)
            print(
                f'trainer was selected for training this round and will start training!')
            trainer.train_model()
            results = trainer.all_metrics()
            response = json.dumps({'id': CLIENT_ID, 'weights': [w.tolist(
            ) for w in trainer.get_weights()], 'num_samples': trainer.get_num_samples(), 'metrics': results})
            client.publish('minifed/preAggQueue', response)
            print(f'finished training and sent weights!')
        else:
            print(color.BOLD_START + 'new round starting' + color.BOLD_END)
            print(f'trainer was not selected for training this round')

# callback for posAggQueue: gets aggregated weights and publish validation results on the metricsQueue


def on_message_agg(client, userdata, message):
    print(f'received aggregated weights!')
    # response = json.dumps({'id': CLIENT_ID, 'accuracy': trainer.eval_model(), "metrics": trainer.all_metrics()})
    msg = json.loads(message.payload.decode("utf-8"))
    agg_weights = [np.asarray(w, dtype=np.float32) for w in msg["weights"]]
    results = trainer.all_metrics()
    response = json.dumps({'id': CLIENT_ID, 'accuracy': results["accuracy"], "metrics": results})
    trainer.update_weights(agg_weights)    
    print(f'sending eval metrics!\n')
    client.publish('minifed/metricsQueue', response)

# callback for stopQueue: if conditions are met, stop training and exit process


def on_message_stop(client, userdata, message):
    print(color.RED + f'received message to stop!')
    trainer.set_stop_true()
    exit()


# connect on queue and send register
trainer = Trainer(CLIENT_NUMBER, MODE)
client = mqtt.Client(str(CLIENT_ID))
client.connect(BROKER_ADDR, keepalive=0)
client.on_connect = on_connect
client.message_callback_add('minifed/selectionQueue', on_message_selection)
client.message_callback_add('minifed/posAggQueue', on_message_agg)
client.message_callback_add('minifed/stopQueue', on_message_stop)
client.message_callback_add('minifed/args', on_args)

# start waiting for jobs
client.loop_start()

response = json.dumps({'id': CLIENT_ID, 'accuracy': trainer.eval_model(), "metrics": trainer.all_metrics()})
client.publish('minifed/registerQueue',  response)
print(color.BOLD_START +
      f'trainer {CLIENT_ID} connected!\n' + color.BOLD_END)



while not trainer.get_stop_flag():
    time.sleep(1)

client.loop_stop()
