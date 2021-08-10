import sys
import os
import json
import datetime
import time
import uuid
import threading
import numpy as np
import cv2

import tensorflow as tf
from openvino.inference_engine import IECore

from azure.iot.device import IoTHubModuleClient
from azure.storage.blob import BlobServiceClient

from ThreadingVideoStream import ThreadingVideoStream
from LCD import LCD

DEVICE_ID = os.environ['IOTEDGE_DEVICEID']
MODEL = "Tensorflow"
PROBABILITY_THRESHOLD = .8
SAVING_DATA_TO_FILE = True
TWIN_CALLBACKS = 0

# モジュール ツインの更新
def twin_update_listener(client):
    global MODEL
    global PROBABILITY_THRESHOLD
    global SAVING_DATA_TO_FILE
    global TWIN_CALLBACKS

    while True:
        data = client.receive_twin_desired_properties_patch()
        
        if "Model" in data:
            MODEL = data["Model"]
        
        if "ProbabilityThreshold" in data:
            PROBABILITY_THRESHOLD = data["ProbabilityThreshold"]

        if "SavingDataToFile" in data:
            SAVING_DATA_TO_FILE = data["SavingDataToFile"]
        
        TWIN_CALLBACKS +=1
        print("Total calls confirmed: %d\n" % TWIN_CALLBACKS)

# 指定したサイズで中心をトリミング
def crop_center(image, cropx, cropy):
    h, w = image.shape[:2]
    startx = w//2-(cropx//2)
    starty = h//2-(cropy//2)

    return image[starty:starty+cropy, startx:startx+cropx]


def main():
    try:
        print("\nPython %s\n" % sys.version)
        print("Press Ctrl-C to exit.")

        global MODEL
        global PROBABILITY_THRESHOLD
        global SAVING_DATA_TO_FILE

        module_client = IoTHubModuleClient.create_from_edge_environment()
        module_client.connect()

        # モジュール ツイン更新の検出のためのスレッドを開始
        twin_update_listener_thread = threading.Thread(target=twin_update_listener, args=(module_client,))
        twin_update_listener_thread.daemon = True
        twin_update_listener_thread.start()

        LOCAL_STORAGE_ACCOUNT_NAME=os.environ['LOCAL_STORAGE_ACCOUNT_NAME']
        LOCAL_STORAGE_ACCOUNT_KEY=os.environ['LOCAL_STORAGE_ACCOUNT_KEY']
        IMAGE_CONTAINER_NAME=os.environ['IMAGE_CONTAINER_NAME']

        # BLOB ストレージ モジュールへの接続
        localblob_connectionString = "DefaultEndpointsProtocol=http;BlobEndpoint=http://" + DEVICE_ID + ":11002/" + LOCAL_STORAGE_ACCOUNT_NAME + ";AccountName=" + LOCAL_STORAGE_ACCOUNT_NAME + ";AccountKey=" + LOCAL_STORAGE_ACCOUNT_KEY + ";"
        blob_service_client = BlobServiceClient.from_connection_string(localblob_connectionString)

        # コンテナがない場合は作成
        try:
            container_client = blob_service_client.get_container_client(IMAGE_CONTAINER_NAME)
        except Exception as e:
            container_client = blob_service_client.create_container(IMAGE_CONTAINER_NAME)
            
        blob_client = None

        # labels.txt から分類ラベルのリストを生成
        labels = []
        labels_filename = "labels.txt"

        with open(labels_filename, 'rt') as lf:
            for l in lf:
                labels.append(l.strip())

        # tensorflow
        graph_def = tf.compat.v1.GraphDef()

        pb_filename = "model.pb"

        # tensorflow グラフのインポート
        with tf.io.gfile.GFile(pb_filename, 'rb') as pf:
            graph_def.ParseFromString(pf.read())
            tf.import_graph_def(graph_def, name='')
        
        sess = tf.compat.v1.Session()

        # OpenVino
        xml_path = "model.xml"
        bin_path = "model.bin"

        ie_core_handler = IECore()
        network = ie_core_handler.read_network(model=xml_path, weights=bin_path)
        executable_network = ie_core_handler.load_network(network, device_name='MYRIAD', num_requests=4)

        stream = ThreadingVideoStream(0)
        stream.start()

        lcd = LCD()
        lcd.clear()

        message_no = 0

        while True:
            try:
                frame = stream.read()

                cropy, cropx = frame.shape[:2]
                min_dim = min(cropx, cropy)
                image = crop_center(frame, min_dim, min_dim)
                
                if MODEL == "Tensorflow":
                    input_node = "Placeholder:0"
                    output_layer = "loss:0"

                    # モデルへの入力サイズを取得
                    input_tensor_shape = sess.graph.get_tensor_by_name(input_node).shape.as_list()
                    network_input_size = input_tensor_shape[1]

                    # 入力サイズで中央をトリミング
                    image = crop_center(image, network_input_size, network_input_size)

                    start = time.time()

                    try:
                        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
                        predictions = sess.run(prob_tensor, {input_node: [image]})

                        # 推測にかかった時間を計測
                        infer_time = time.time() - start
                    except KeyError:
                        print("Could't find classification output layer")
                        exit(1)
                    
                else:
                    input_blob = next(iter(network.input_info))
                    output_blob = next(iter(network.outputs))

                    # モデルへの入力サイズを取得
                    h, w = network.input_info[input_blob].input_data.shape[2:]
                    # 入力サイズにイメージをリサイズ
                    image = cv2.resize(image, (w, h), interpolation = cv2.INTER_LINEAR)
                    image = image.transpose((2, 0, 1))

                    input_name = list(network.input_info.keys())[0]

                    start = time.time()

                    try:
                        predictions = executable_network.infer(inputs={input_name: image})
                        predictions = predictions[output_blob]

                        # 推測にかかった時間を計測
                        infer_time = time.time() - start
                    except KeyError:
                        print("Could't find classification output layer")
                        exit(1)
                
                # 最も高い確率の結果を取得
                highest_probability_index = np.argmax(predictions)
                score = predictions[0][highest_probability_index]

                # LCD ディスプレイに結果を表示
                line1 = "{} ({:.0%})".format(labels[highest_probability_index], float(score))
                line2 = "time: {}ms".format(int(infer_time*1000.0))

                lcd.write(line1, line2)

                # 結果が閾値を超える場合のみ、メッセージの送信、画像の保存を実行
                if score > PROBABILITY_THRESHOLD:
                    # IoT Hub に送信するメッセージの作成
                    now = datetime.datetime.now() + datetime.timedelta(hours=9)

                    messageDict = dict(
                        messageID=message_no,
                        deviceID=DEVICE_ID,
                        datetime=now.isoformat(),
                        model=MODEL,
                        threshold=PROBABILITY_THRESHOLD
                    )

                    messageDict['label'] = labels[highest_probability_index]
                    messageDict['probability'] = "{:.2%}".format(float(score))
                    messageDict['processing_time'] = "{}ms".format(int(infer_time*1000.0))

                    message = json.dumps(messageDict)

                    # メッセージの送信
                    module_client.send_message_to_output(message, "output1")
                    print("Send to IotHub: %s" % message)

                    if SAVING_DATA_TO_FILE:
                        # 画像を保存
                        filename = "/app/images/{}.jpg".format(uuid.uuid4().hex)
                        cv2.putText(frame, "label: {}, probability: {:.2%}".format(labels[highest_probability_index], float(score)), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), thickness=2)
                        cv2.imwrite(filename, frame)
                        print("Saving the image: %s" % filename)

                        # ローカル BLOB へ保存した画像をアップロード
                        blob_client = blob_service_client.get_blob_client(container=IMAGE_CONTAINER_NAME, blob=filename)
                        with open(filename, "rb") as data:
                            blob_client.upload_blob(data)
                    
                    message_no += 1

            except Exception as e:
                print("Unexpected error in main: %s" % str(e))

            time.sleep(45)
            lcd.clear()
            time.sleep(15)
    
    except KeyboardInterrupt:
        print("predict.py stopped")

        stream.stop()
        lcd.stop()


if __name__ == "__main__":
    main()