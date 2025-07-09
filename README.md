## Improving Inference Throughput of CNN models through Quantization + Parallelel Inference 

Rapid fast inference of AI models like CNN and LLMs have attained focus from industries. Improving the model throughput without compromising the accuracy is much required in EdgeAI applications.

Here we study how CNN models can be quantized and further reduce thier inference through parallel pipeline inference

### Requirements
1. A machine with minimum 16 GB RAM
2. GeForce RTX with atleast 4GB dedicated space (prefered RTX 3050 or above)
3. Python 3.8 - 3.10

### Set up
1. Clone the repo
    ```
    git clone https://github.com/Tejes123/Pipeline-Parallelism-for-CNN.git

    cd Pipeline-Parallelism-for-CNN
    ```

2. Create an python environment and activate
    ```
    pythom -m venv venv

    venv\Scripts\actiate
    ```

3. Install the needed Packages
    ```
    pip install -r requirements.txt

    ```

4. Move to any one Model, eg Resnet.
    ```
    cd Resnet - 3 nodes parallelism
    ```

5. Set up the model files, turn it to ONNX files and futher th TensorRT engines. (you can do it with runnning one file). 

    ##### NOTE: This can take some time for building the tensorrt engines

    For example for Resnet - 3 nodes parallelism,
    ```
    python setup-models.py
    ```


6. The pipeline parallelism uses sockets to communicate between node. Make sure the nodes are connected to the same network. Also Make the ports available, by allowing inbound and outbound firewall rules for the ports. 

    <b>Your Setup is complete!</b>


### How to run?
1. In the respective model folder, you will see three files:
```
first.py (node0)---- For running inference in node 0. This starts the inference and sends the activations to next node.
second.py (node1)--- Recieves activations from node0, processes and sends the activation to node 2
third.py (node2) --- Recieves the activations from node 1 and gives the final output for the input batch
```
    Each File is to be run in 3 seperate machines.

2. In each of the three files, you will see 4 functions:
```
without_fusion_without_tensorrt()  #Baseline
without_fusion_with_tensorrt()
with_fuson_without_tensorrt()
with_fusion_with_tensorrt()
```
3. In each machine, run the same type of function. Start running the function in the order:

```
third.py ---> second.py ----> first.py
```

We run the first.py after the two functions, as first.py starts the inference and there should be sockets open and listening to recieve the activations from first.py


Recordings Doc Link: https://docs.google.com/document/d/1TcfALvlPAt8nJTxgYsxV9HGFnXy5JUbHPFCmC3LvPaE/edit?usp=sharing





