import json

from kafka import KafkaProducer, KafkaConsumer
from predict import do_predict, get_key_words

# 创建消费者
consumer = KafkaConsumer('j2p', bootstrap_servers=['127.0.0.1:9092'])
producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'], max_request_size=204857600)


def format_result(a: float) -> str:
    if float(a) < 0.5:
        return "正常样本"
    else:
        return "攻击样本"


print('application started.')

# 消费消息
for message_object in consumer:
    key = message_object.key.decode('utf-8')
    value = message_object.value.decode('utf-8')
    message = json.loads(value)
    tests = list(json.loads(message['message']))
    model = message['model']
    print('--------------------start-%s--------------------' % model)
    y_pred = list(do_predict(model, tests))
    keywords = list(get_key_words())
    response = [{"result": format_result(pred), "keyword": str(keyword)} for (pred, keyword) in zip(y_pred, keywords)]
    if key == "file":
        producer.send("p2j-file", json.dumps(response).encode("utf-8"))
        producer.flush()
    else:
        producer.send("p2j-text", json.dumps(response).encode("utf-8"))
        producer.flush()
    print('-------------------- end-%s --------------------' % model)
