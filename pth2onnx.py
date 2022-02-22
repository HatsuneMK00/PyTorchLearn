# -*- coding: utf-8 -*-
# created by makise, 2022/2/22

"""
convert pth format network to onnx format network
"""

import torch
from gtsrb.cnn_model_small import SmallCNNModel

# load pth model from file
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SmallCNNModel()
model.load_state_dict(torch.load('model/cnn_model_gtsrb_small.pth', map_location=device))
model = model.to(device)

# convert to onnx
torch.onnx.export(model, torch.randn(10, 3, 32, 32), 'model/cnn_model_gtsrb_small.onnx', verbose=True)

# test onnx model
import onnx
import onnxruntime


def test_onnx_model(model_path):
    model_onnx = onnx.load(model_path)
    ort_session = onnxruntime.InferenceSession(model_path)
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_data = torch.randn(input_shape)
    torch_out = ort_session.run(None, {input_name: input_data.numpy()})
    print(torch_out)

test_onnx_model('model/cnn_model_gtsrb_small.onnx')
