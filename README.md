# Low Cost AI Accelerator Based on TPU   
  
### Development Environment :    
- **RTL Simulator** : *ModelSim-Intel FPGA Standard Edition, Version 20.1.1, windows* / *Vivado*   
- **Synthesis Tool** : *Synopsys Design Compiler*
- **Model Training** : *Pytorch*    
   
### Repository Structure :  
```
Low Cost AI Accelerator Based on TPU /
├── Model/                    # Model Training  
│   ├── MLP.py         
│   └── LeNet.py              
│   └── ResNet.py              
│   └── AlexNet.py              
├── RTL/                      # RTL Code  
│   ├── TPU.v                 # Top Module  
│   └── tb_TPU.v              # Testbench  
│   └── TSC.v                 # TPU System Controller  
│   └── Weight_Memory.v              
│   └── Activation_Memory.v
│   └── Compensation_Memory.v
│   └── WPU.v                 # Weight Pre-processing Unit  
│   └── Input_Buffer.v
│   └── RPE.v                 # Reduce-precision Processing Element  
│   └── CPE.v                 # Compensation Processing Element  
│   └── Accumulator.v
│   └── Activation_Function.v
│   └── UB.v                  # Unified Buffer  
├── PE/                       # PE Comparsion resource
├── Src/                      # Simulation resource (.dat, .out)
├── Result_Simulator/         # Check the Answer
├── Area_Analysis/            # Analyze the Area
│   └── area.log              # The system area report
│   └── area.py               # Area calculator (RPE, CPE)  
└── README.md                 
```  


## Tensor Processing Unit (TPU) [1] :  
<img width="1116" height="839" alt="image" src="https://github.com/user-attachments/assets/47d3af4e-3567-4cf8-bcb4-d5f5aa79293b" />   

## Data Flow (OS/WS/IS) [2] [3] :  
The systolic array architecture supports three mainstream dataflow methods: Output Stationary (OS), Weight Stationary (WS), and Input Stationary (IS).    
  
<img width="1846" height="722" alt="image" src="https://github.com/user-attachments/assets/c6f07320-1c2e-4bca-9610-6f9131aaee00" />  
  
We use the Weight Stationary (WS) data flow to implement our TPU architecture.  
The weight data will be preloaded into each Processing Element (PE), and the activation values will be fed into the systolic array using a +45-degree diagonal flow.    
   
<img width="1664" height="877" alt="image" src="https://github.com/user-attachments/assets/c114ffd9-b225-458d-9e16-d64c49b8c25d" />   
   
## Most Significant Runs (MSR) [4] :  

Deep neural network models are typically trained using 32-bit floating-point operations. After training, the resulting weight values are also in 32-bit floating-point format. However, to reduce computational resources and **inference time**, deep neural networks often perform inference computations using fixed-point arithmetic. Since most weights are close to zero, when these weights are converted to fixed-point representation, as shown in the figure below, we often observe consecutive 1s or 0s in the most significant bits. This phenomenon is referred to as ***Most Significant Runs (MSR)***.  

<img width="1793" height="406" alt="image" src="https://github.com/user-attachments/assets/6a8130fa-d0b0-4e50-abb6-fae3c1e7e34c" />   
  
For example, consider the value 0.10534. When converted to a fixed-point format, we get: 0.10534 × 128 = 13.48, which rounds to 13, represented in binary as 00001101. In this case, the leading four zeros can be compressed into a single zero without losing precision.  
Similarly, for the value -0.0784, we have: -0.0784 × 128 = -10.0352, which rounds to -10, represented in binary (two's complement) as 11110110. Here, the leading four ones can also be compressed into a single one without any loss of precision.  
     
We then analyze the proportion of MSR occurrences across different deep neural network models. By quantizing the model weights into 8-bit integers (INT8) using fixed-point representation, we observe that nearly 99% of the weights contain an MSR-4. Since most of the weights are negative values, the four most significant bits (MSR-4) can be compressed into a single bit. This technique not only reduces computational cost and power consumption, but also significantly lowers memory usage.    

| MSR-N / Model | MLP |  LeNet | ResNet | AlexNet | 
|:-----:|:---:|:------:|:------:|:-------:|
| MSR-3 | 99.9% |  99.9% | 99.9% | 99.9% |
| MSR-4 | 99.98% |  98.90% | 99.61% | 99.98% |
| MSR-5 | 98.0% |  88.3% | 99.5% | 99.7% |
| MSR-6 | 78.2% |  53.4% | 99.1% | 97.8% |
| MSR-7 | 40.4% |  27.3% | 85.5% | 84.3% |

As shown above, if we compress weight data containing MSR-4 from 8 bits to 5 bits for computation, then the data without MSR-4 must also be truncated. However, such truncation inevitably introduces some loss in accuracy.  
If we want to avoid this loss of precision, we need to compensate for the truncated bits accordingly.  
    
  
## MSR-4 Analysis : 
By analyzing the distribution of MSR-4 in the trained weights, we found that, on average, only 2.9 out of every 256 weights do not contain MSR-4 patterns. Therefore, for a 256×256 systolic array, we only need 3 rows per column to perform compensation.  
   
| Model         | MLP        | LeNet      | ResNet     | AlexNet    |
|:---------------:|:------------:|:------------:|:------------:|:------------:|
| **Layers (CONV/FC)** | 3(0/3)     | 5(2/3)     | 17(16/1)   | 8(5/3)     |
| **Dataset**       | MNIST      | MNIST      | MNIST      | MNIST      |
| **Input Dimensions** | 28x28    | 28x28      | 28x28      | 28x28      |
| **Output Class**  | 10         | 10         | 10         | 10         |
| **Test Accuracy** | 98.08%     | 98.05%     | 99.61%     | 99.56%     |
| **MSR-4 %**       | 99.98%     | 98.90%     | 99.61%     | 99.98%     |
| **Non-MSR-4 / 256** | 0.1      | **2.9**  | 0.1        | 0.0        |
   
   
In addition, some techniques used during model training to prevent overfitting also contribute to an increased MSR-4 ratio, as they tend to compress the weight distribution. Examples include: reducing the learning rate, L1 regularization, and L2 regularization (weight decay).    
Below is the architecture of the model used in our training :   
  
| Model               | MLP          | LeNet        | ResNet         | AlexNet        |
|:-----:|:---:|:------:|:------:|:-------:|
| **Optimizer**           | Adam             | Adam             | Adam               | Adam               |
| **Learning Rate**       | 0.0001           | 0.000055         | 0.001              | 0.001              |
| **lr Scheduler**<br>(**step_size / gamma**) | -                | -                | 7 / 0.1            | 7 / 0.1            |
| **Loss Function**       | Cross Entropy Loss | Cross Entropy Loss | Cross Entropy Loss | Cross Entropy Loss |
| **Regularization**      | -                | -                | **L2 (λ=1e-4)**  | **L2 (λ=1e-4)**  |
| **Epochs / Batch Size** | 10 / 64          | 10 / 64          | 15 / 64            | 15 / 64            |
  
  
## Proposed TPU Architecture :   
  
<img width="4554" height="2192" alt="RPTPU drawio" src="https://github.com/user-attachments/assets/9252b519-17c7-4bc3-8771-d5a9be06d1d9" />  

  
The above is the proposed TPU architecture. The input weight data is first processed by the Weight Processing Unit (WPU) to determine whether it contains an MSR-4 pattern. If MSR-4 is detected, the leading four bits can be compressed into a single bit, and the least significant bit (LSB) is discarded, as it will be compensated during computation by fixing the LSB to 1 within the Reduced-precision Processing Element (RPE). A Shift Bit = 0 is added in front of the data to indicate that it is an MSR-4 weight. For weights without MSR-4, the leading four bits are preserved, and a Shift Bit = 0 is similarly added to indicate that it is a non-MSR-4 weight. Among the remaining four bits, the lowest three bits are stored in a Compensation Memory, since the LSB will again be fixed to 1 in the Compensation Processing Element (CPE) during computation as a form of expected value compensation. The storage and computation mechanism is illustrated in the figure below:  
  
|<img width="1092" height="578" alt="Design_MSR drawio (2)" src="https://github.com/user-attachments/assets/48697522-db78-4900-a01a-946885f3f35f" /> | <img width="844" height="512" alt="Cal drawio (1)" src="https://github.com/user-attachments/assets/1f6684f6-ed29-429c-8a5d-066877dd4b65" /> | 
|-----------------------------------------------|-------------------------------|
  
An important consideration is that, for signed operations, the compensation weight must include an extra sign bit, which reflects the sign of the original weight value.  
  
# 
Next, the TPU operates using a Weight-Stationary (WS) dataflow. The weights and compensation weights are preloaded from their respective memory blocks into the Reduced-precision Processing Elements (RPEs) and Compensation Processing Elements (CPEs). Once the preloading is complete, the Activation Memory outputs activation values to the Input Buffer, which then feeds them into the systolic array at a +45-degree diagonal angle. Since the shadow array compensation structure on the left side performs computations significantly faster than the right side (completing in just 3 cycles), the results from the left side are first written into a shared accumulator, stored at the target compensation address. By sharing this accumulator with the right side of the array, we can save a significant amount of hardware area, as there's no need to design a separate accumulator specifically for the compensation path. Once the right side finishes its computation, its result is added to the compensation result stored in the accumulator to produce the final correct output, as illustrated in the figure below.    
    
<img width="2584" height="854" alt="Acc drawio" src="https://github.com/user-attachments/assets/eceb0009-4f9f-4e60-abad-f00a223fcf31" />  
    
## RPE / CPE Architecture :   
| <img width="2630" height="1446" alt="PE drawio" src="https://github.com/user-attachments/assets/c13b90f2-0f2b-47ef-bc33-c8cd04cefd16" /> | <img width="2288" height="1412" alt="CPE drawio" src="https://github.com/user-attachments/assets/cdd3e260-c86d-4e87-9eaa-39a110822ab3" /> |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
  

## Weight / Compensation / Activation Memory Architecture :
In this project, for implementation convenience, we made some slight adjustments to the memory architecture by configuring it to output 8 data values per access. In practice, each memory block can be regarded as consisting of 8 individual SRAMs, enabling it to output 8 data entries simultaneously.   
  
<img width="3150" height="698" alt="Memory drawio" src="https://github.com/user-attachments/assets/bc47a240-fb7a-4a97-a1e6-861cafecec3e" />  

## Memory Read Control : 
After the Mem_Write signal is asserted and completed, the system prepares to read from the Weight Memory and Compensation Memory in order to preload the weight data into the PE of the Systolic Array. Therefore, once Mem_Write finishes, I assert Mem_Rd_en on the falling edge to trigger the memory read. Then, on the next falling edge, the Pre_LoadWeight and Pre_LoadCWeight signals are asserted, allowing the previously fetched data to be successfully loaded into the systolic array. The same mechanism applies to the Activation Memory. After the weight preloading is complete, activations are introduced. To improve efficiency, we can assert Mem_Rd_en on the falling edge just before the last weight preload, so that in the next cycle, when Cal is asserted on the falling edge and the PEs begin computation on the rising edge, the corresponding activation data can be immediately provided to the input buffer—thus speeding up the overall operation.  
    
<img width="1479" height="265" alt="image" src="https://github.com/user-attachments/assets/38a219e8-0829-4202-b606-5d9f348363e4" />     
<img width="1483" height="381" alt="image" src="https://github.com/user-attachments/assets/c862e6f0-32f7-44e1-a536-39cbc3576a18" />      
  
## RTL Simulation :   
We use 8x8 Systolic Array and 8x3 Compensation Array to simulate the proposed architecture.  
Therefore, We use 64 x 5bit Weight Memory, 64 x 7bit Activation Memory and 24 x 3bit Compensation Memory to help ours simulation.  

### Answer Check   
We use Pytohn to check the answer, the result in [output.out](./Src/Output.out)  
<img width="888" height="372" alt="image" src="https://github.com/user-attachments/assets/bcb49d35-67db-46d4-82fc-da65306aa883" />

### Activation Function (ReLU Function) :  
In this project implementation, we use the ReLU (Rectified Linear Unit) function as the activation function. The output values generated by the Systolic Array are passed through the ReLU function and then sent to the Unified Buffer.   
You also can go to the [output.out](./Src/Output.out) to see the result after ReLU.  
| <img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/14283b26-61a8-4405-92f1-b9ddc122e373" /> | <img width="962" height="820" alt="image" src="https://github.com/user-attachments/assets/98abd8c4-35ee-4616-8c94-6d09c107353b" />| 
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|



## Accuracy Analysis :  
| **PE Type / Model**                                | **MLP**     | **LeNet**   | **ResNet**  | **AlexNet** |
|:----------------------------------------------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|**Original Precision (Float 32)**                                          | 98.08%      | 98.01%      | 99.61%      | 99.56%      |
| **Quantization Precision (INT8)**                                             | 97.28%      | 97.97%      | 99.09%      | 99.45%      |
| **Truncate 3 bits in MSR4 & Non-MSR4 Weight Data**                                      |  **92.71%** |  **89.20%** |  **11.36%** |  **19.27%** |
|  **Add Expect Value (LSB = 1)**                                        | 97.29%      | 97.44%      | 98.96%      | 99.40%      |
| **Add Expect Value (LSB = 1) & Non-MSR4 Compensation**                                      |  **97.34%** |  **98.00%** |  98.96% |  99.40% |
  
The above results were obtained using Post-Training Quantization (PTQ) with PyTorch. We can observe that when the model is quantized to INT8, the accuracy typically drops by around 0.1% to 1%. However, once we truncate the Non-MSR-4 weights, the accuracy degrades rapidly. Although Non-MSR-4 weights only account for about 1% of the total weights, they can cause significant errors in larger models such as ResNet and AlexNet, where the parameters are more sensitive. To address this issue, we introduce an expected value compensation mechanism and apply the proposed compensation architecture. As a result, the accuracy loss can be effectively recovered, and in some cases—such as MLP and LeNet—the final accuracy even surpasses that of standard INT8 quantization.  

## Hardware Overhead Analysis :  
### PE Comparison  
For each type of PE, we implemented the multiplication units using only basic Half Adders (HA) and Full Adders (FA) to ensure a fair comparison across all designs.  
Below is the hardware area comparison obtained from synthesizing the signed-version PE :    
|Type of PE|PE|RPE|CPE|
|:--:|:--:|:--:|:--:|
|Area | 0% | -18.8% | -30.6% |   
|Power|  0.6743mW | 0.5399mW | 0.4327mW |  
     
For a 256x256 Systolic Array, we overall reduce about -16.64% and each RPE contains approximately 460 fewer gates compared to its original PE.     
You also can go to [Report](./PE) to see our PE report and RTL.   
  
### Input Buffer Comparison  
|Type|Original Input Buffer|Input Buffer|
|:--:|:--:|:--:|
|Area | 0% | +125% |

The Input Buffer incurs an overhead of approximately 3571 additional gates compared to the original design.

# 

Let us now analyze in detail how much overall hardware cost is reduced. For a 256×256 systolic array, the proposed architecture requires a 256×3 compensation array. We can then calculate the total hardware overhead for the systolic array as follows:   
-460x256x256 (Reduced Systolic Array) + 1688x256x3 (Extra Compensation Array) = -28615116 Gate     
    
Next, we consider the Input Buffer overhead, which (when scaled to a 256×256 configuration) adds 
   
+3571 x 32 = +114272 Gate    
  
Total :  -28615116 + 114272 = -28500844 Gate    
Therefore, even with the compensation array included, the overall area of the proposed systolic array remains significantly smaller than the original design.  
  
As for the increase in Compensation Memory and the reduction in Weight Memory and Activation Memory, this part will be temporarily excluded from the discussion. The Weight Memory can be reduced by approximately 3/8, and the Activation Memory by approximately 1/8.  
  
### Each Module  

|Module|Area Percnetage (%)| Gate Count |
|:--:|:--:|:--:|
|**Weight Memory**| 1.8% | 3400 |
|**Activation Memory**| 2.4% |  4658  |
|**Compensation Memory**| 0.9% | 1683 |
|**Input Buffer**| 5.5% | 10700 |
|**TPU System Controller**| 0.5% | 1030 |
|**Unified Buffer**| 6.4% | 12340 |
|**WPU**| 0.2% | 464 |
|**Systolic Array**| 52.25% | 101483 |
|**Accumulator**| 12.11% | 11360 |
|**Activation Function**| 0.11% | 213 | 
|**Compensation Array**| 17.8% | 34545 |



## Reference : 
**[0] Weight-Aware and Reduced-Precision Architecture Designs for Low-Cost AI Accelerators**  
**[1] In-Datacenter Performance Analysis of a Tensor Processing Unit**     
**[2] SCALE-Sim: Systolic CNN Accelerator Simulator**    
**[3] Effective_Runtime_Fault_Detection_for_DNN_Accelerators**  
**[4] Refresh Power Reduction of DRAMs in DNN Systems Using Hybrid Voting and ECC Method**   
**[5] DRUM: A Dynamic Range Unbiased Multiplier for approximate applications**   
**[6] APTPU: Approximate Computing Based Tensor Processing Unit**   


