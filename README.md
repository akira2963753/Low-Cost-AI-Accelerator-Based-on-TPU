# Low Cost AI Accelerator Based on Systolic Array     
Prior research has explored this approach in hardware accelerator designs such as the Approximate Tensor Processing Unit (APTPU) [1], which leverages the Dynamic Range Unbiased Multiplier (DRUM) architecture [2]. DRUM applies a dynamic truncation method that identifies significant bits using leading-one detectors (LODs), retains only the most meaningful bits for multiplication, and compensates for truncated bits using an expected-value approach. APTPU builds on this by introducing pre-approximate units (PAUs) to handle word-length reduction outside the systolic array, enabling approximate computations through dedicated approximate processing elements (APEs). This architectural separation improves performance and reduces power consumption. However, since APTPU performs dynamic truncation, it must determine the shift amount for each operand at runtime, increasing circuit complexity. If a fixed shift amount is used across all operations, the hardware area can be further reduced and performance enhanced. Additionally, the PAUs in APTPU are located within the systolic array, meaning that input and weight data stored in memory remains in high precision.Pre-truncating these values before storage would reduce memory access overhead.Based on the above analysis, we propose a new approach that reduces precision by exploiting the bit-level characteristics of weight values, while enabling a fixed shift amount for each operation.Specifically, we utilize Most-Significant Runs (MSRs) defined as sequences of consecutive 0s or 1s starting from the most significant bit which are commonly observed in the fixed-point binary representation of trained weights in deep learning models, where most values tend to be close to zero, as illustrated in Figure.  
<img width="1867" height="440" alt="image" src="https://github.com/user-attachments/assets/8b25f99f-a2e1-4d54-872e-b3422aaa75d6" />

  
If a weight value holds four consecutive identical bits (either 0s or 1s) starting from the most significant bit, these bits can be replaced with a single sign bit without degrading the accuracy of model. By exploiting this MSR property, the word length of the weight can be effectively reduced. As a result, each PE within the systolic array of the tensor processing unit can use a smaller multiplier for computation. This reduction not only decreases the overall area and power consumption but also enables faster computation due to the shorter input word length, thereby improving the overall performance.  





## MSR-4 Analysis : 
All four trained models exhibit MSR-4 distributions covering at least 99% of the weights. In the worst case, only about 2.9 out of every 256 weights are Non-MSR-4.  
<img width="1122" height="417" alt="image" src="https://github.com/user-attachments/assets/c4fe2d6a-f449-40fc-8f40-f9ab724513c2" />  


As a result, in a 256×256 Systolic Array, each column would require 3 PEs capable of performing compensation.  


## Model Training Structure :  
<img width="1108" height="427" alt="image" src="https://github.com/user-attachments/assets/ba55a537-6cca-4147-8348-5d5ebfdbcf39" />


During model training, certain methods used to prevent overfitting can also help increase the MSR-4 percentage.   
Examples include lowering the learning rate, applying L1 regularization, and L2 regularization (weight decay).  


## Quantization Accuracy Analysis (Post-train Quantization) :  
<img width="1198" height="473" alt="image" src="https://github.com/user-attachments/assets/b867669b-beb9-4b30-9546-873f6b0632cf" />


For complex models like ResNet and AlexNet,   
quantization without expectation compensation leads to an accuracy drop below 20%, while with compensation, the accuracy only decreases by less than 1%.  
  


## Proposed TPU Architecture :  
![RPTPU_DATAPATH drawio (6)](https://github.com/user-attachments/assets/fb4c0342-37bb-40c0-9241-e5ba87262708)

## Reduce Processing Element (RPE) :   
![RPE drawio](https://github.com/user-attachments/assets/c790f418-5e94-47a2-b850-18127da7769d)

## Compensation Processing Element (CPE) :  
![CPE drawio](https://github.com/user-attachments/assets/e12d8fac-3e1d-444b-9175-dcd8a724af95)
