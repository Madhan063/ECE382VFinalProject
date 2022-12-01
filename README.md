# ECE382VFinalProject

## Post-Training Clamping
In this technique, we are going to clamp the model weights, biases or activations and observe how the model’s performance is affected. We perform clamping in the following way: Clamping only the weights and biases of the model - As a part of this method we clamp all the trainable parameters of the model. But we still continue to use inputs and activations in FP32 and observe the model's  characteristics.

For the choice of models we have taken CNN models and language models to observe the effects of clamping. All the numbers are considered as signed numbers during clamping.

### 1. Alexnet
![alt text](https://github.com/Madhan063/ECE382VFinalProject/blob/main/results/alexnet.png)

### 2. Resnet 50
![alt text](https://github.com/Madhan063/ECE382VFinalProject/blob/main/results/resnet50.png)

### 3. GPT2
![alt text](https://github.com/Madhan063/ECE382VFinalProject/blob/main/results/gpt2.png)

### 4. BERT Large Cased
![alt text](https://github.com/Madhan063/ECE382VFinalProject/blob/main/results/bert_large_cased.png)
