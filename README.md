# Low Cost AI Accelerator   
Prior research has explored this approach in hardware accelerator designs such as the Approximate Tensor Processing Unit (APTPU) [1],   
which leverages the Dynamic Range Unbiased Multiplier (DRUM) architecture [2].   
DRUM applies a dynamic truncation method that identifies significant bits using leading-one detectors (LODs), 
retains only the most meaningful bits for multiplication, and compensates for truncated bits using an expected-value approach.   
APTPU builds on this by introducing pre-approximate units (PAUs) to handle word-length reduction outside the systolic array, 
enabling approximate computations through dedicated approximate processing elements (APEs).   
This architectural separation improves performance and reduces power consumption.   
However, since APTPU performs dynamic truncation, it must determine the shift amount for each operand at runtime, increasing circuit complexity.   
If a fixed shift amount is used across all operations, the hardware area can be further reduced and performance enhanced.   
Additionally, the PAUs in APTPU are located within the systolic array, meaning that input and weight data stored in memory remains in high precision. 
Pre-truncating these values before storage would reduce memory access overhead.  
Based on the above analysis, we propose a new approach that reduces precision by exploiting the bit-level characteristics of weight values,   
while enabling a fixed shift amount for each operation. Specifically,   
we utilize Most-Significant Runs (MSRs) defined as sequences of consecutive 0s or 1s starting from the most significant bit which are commonly observed in the fixed-point binary   
representation of trained weights in deep learning models, where most values tend to be close to zero, as illustrated in Fig.  
  
<img width="413" height="346" alt="image" src="https://github.com/user-attachments/assets/0e688109-fb36-4551-bc2f-2232173e1ab3" />    
  
If a weight value holds four consecutive identical bits (either 0s or 1s) starting from the most significant bit,   
these bits can be replaced with a single sign bit without degrading the accuracy of model. By exploiting this MSR property,   
the word length of the weight can be effectively reduced. As a result, each PE within the systolic array of the tensor processing unit can use a smaller multiplier for computation.   
This reduction not only decreases the overall area and power consumption but also enables faster computation due to the shorter input word length, thereby improving the overall performance.  





## MSR-4 Analysis : 
All four trained models exhibit MSR-4 distributions covering at least 99% of the weights. In the worst case, only about 2.9 out of every 256 weights are Non-MSR-4.  
| Model       | MLP     | LeNet   | ResNet  | AlexNet |
|:----:|:------:|:-----:|:---------:|:-------:|
| Layers (CONV/FC) | 3(0/3) | 5(2/3) | 17(16/1) | 8(5/3) |
| Dataset     | MNIST   | MNIST   | MNIST   | MNIST   |
| Input Dimensions | 28x28  | 28x28  | 28x28  | 28x28  |
| Output Class | 10      | 10      | 10      | 10      |
| Test Accuracy | 98.08% | 98.05% | 99.61% | 99.56% |
| MSR-4 %     | 99.98%  | 98.90%  | 99.61%  | 99.98%  |
| Non-MSR-4 per 256 weights | 0.1     | 2.9     | 0.1     | 0.0     |

As a result, in a 256×256 Systolic Array, each column would require 3 PEs capable of performing compensation.  


## Model Training Structure :  
| Model       | MLP     | LeNet   | ResNet  | AlexNet |
|:-----------:|:-------:|:-------:|:-------:|:-------:|
| Optimizer   | Adam    | Adam    | Adam    | Adam    |
| Learning Rate | 0.0001 | 0.000055 | 0.001   | 0.001   |
| Ir Scheduler (step_size/gamma) | -       | -       | 7 / 0.1  | 7 / 0.1  |
| Loss Function | Cross Entropy Loss | Cross Entropy Loss | Cross Entropy Loss | Cross Entropy Loss |
| Regularization | -       | -       | L2 (λ=1e-4) | L2 (λ=1e-4) |
| Epochs / Batch Size | 10 / 64 | 10 / 64 | 15 / 64 | 15 / 64 |

During model training, certain methods used to prevent overfitting can also help increase the MSR-4 percentage.   
Examples include lowering the learning rate, applying L1 regularization, and L2 regularization (weight decay).  


## Quantization Accuracy Analysis (Post-train Quantization) :  
| Model       | MLP     | LeNet   | ResNet  | AlexNet |
|:-----------:|:-------:|:-------:|:-------:|:-------:|
| **Original (32x32)** | 98.08%  | 98.05%  | 99.61%  | 99.56%  |
| **Float32 quantize to Q1.7 (8x5)** | 92.71%  | 89.63%  | 11.36%  | 19.27%  |
| **Float32 quantize to Q1.7 and add Expect value compensation (8x5)** | 97.29%  | 97.44%  | 98.96%  | 99.40%  |
| **Float32 quantize to Q1.7 and add Expect value & Non-MSR4 compensation (8x5)** | 97.34%  | 97.63%  | 98.96%  | 99.40%  |

For complex models like ResNet and AlexNet,   
quantization without expectation compensation leads to an accuracy drop below 20%, while with compensation, the accuracy only decreases by less than 1%.  
  


## Proposed TPU Architecture :  
![RPTPU_DATAPATH drawio (6)](https://github.com/user-attachments/assets/fb4c0342-37bb-40c0-9241-e5ba87262708)

## Reduce Processing Element (RPE) :   
![RPE drawio](https://github.com/user-attachments/assets/c790f418-5e94-47a2-b850-18127da7769d)

## Compensation Processing Element (CPE) :  
![CPE drawio](https://github.com/user-attachments/assets/e12d8fac-3e1d-444b-9175-dcd8a724af95)
