import socket
import pickle
import argparse
import torch 
import time
import numpy as np
from client import connect_to_server, send_data, receive_response
from cloudsideinference import get_preds
def main():
    parser = argparse.ArgumentParser(description='Inference Local Cloud')
    parser.add_argument('-d', '--model_path', default="/home/h2x/Desktop/UniLCD_RealWorld/REAL_TIME_WORKING/run_local_cloud/model_run_0011.pth", type=str, help='Path to your model')
    args = parser.parse_args()
    # client_socket = connect_to_server()
    # predictions = receive_response(client_socket)
    # output = get_preds(args.model_path, predictions)
    # send_data(client_socket, output)
    # client_socket = connect_to_server()

    # Assert that the client socket is not None
    # assert client_socket is not None, "Failed to connect to the server, client_socket is None."

    # Receive predictions from the server
    # predictions = receive_response(client_socket)
    predictions = np.load("/opt/app-root/src/UniLCD_RealWorld/well.npy")

    # Assert that predictions are not None and are of expected type
    assert predictions is not None, "Predictions should not be None."
    assert isinstance(predictions, (list, np.ndarray)), "Predictions should be a list or numpy array."

    # Get model predictions based on received data
    output = get_preds(args.model_path, predictions)
    print(output)
    # Assert that output is not None and is of expected type
    assert output is not None, "Output from get_preds should not be None."
    assert isinstance(output, np.ndarray), "Output should be a numpy array."

    # Send data back to the server
    # send_data(client_socket, output)
    print("Yay it works")
'''    while True:
        # client_socket = connect_to_server()
        # predictions = receive_response(client_socket)
        # output = get_preds(args.model_path, predictions)
        # send_data(client_socket, output)
        client_socket = connect_to_server()

        # Assert that the client socket is not None
        assert client_socket is not None, "Failed to connect to the server, client_socket is None."

        # Receive predictions from the server
        predictions = receive_response(client_socket)

        # Assert that predictions are not None and are of expected type
        assert predictions is not None, "Predictions should not be None."
        assert isinstance(predictions, (list, np.ndarray)), "Predictions should be a list or numpy array."

        # Get model predictions based on received data
        output = get_preds(args.model_path, predictions)

        # Assert that output is not None and is of expected type
        assert output is not None, "Output from get_preds should not be None."
        assert isinstance(output, np.ndarray), "Output should be a numpy array."

        # Send data back to the server
        send_data(client_socket, output)

        # Optionally: Assert that data was sent successfully (depends on your send_data implementation)
        # You might need to implement a check inside send_data or confirm that the action was successful.'''


if __name__=="__main__":
    main()