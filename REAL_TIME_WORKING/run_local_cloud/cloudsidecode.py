import socket
import pickle
import argparse
import torch 
import time
from client import connect_to_server, send_data, receive_response
from cloudsideinference import get_preds
def main():
    parser = argparse.ArgumentParser(description='Inference Local Cloud')
    parser.add_argument('-d', '--model_path', default="/home/h2x/Desktop/UniLCD_RealWorld/REAL_TIME_WORKING/run_local_cloud/model_run_0011.pth", type=str, help='Path to your model')
    args = parser.parse_args()
    while True:
        client_socket = connect_to_server()
        predictions = receive_response(client_socket)
        output = get_preds(args.model_path, predictions)
        send_data(client_socket, output)