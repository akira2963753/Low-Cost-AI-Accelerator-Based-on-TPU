## Low Cost AI Accelerator   

### MSR-4 Analysis : 
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

### Model Training Structure :  
| Model       | MLP     | LeNet   | ResNet  | AlexNet |
|-------------|---------|---------|---------|---------|
| Optimizer   | Adam    | Adam    | Adam    | Adam    |
| Learning Rate | 0.0001 | 0.000055 | 0.001   | 0.001   |
| Ir Scheduler (step_size/gamma) | -       | -       | 7 / 0.1  | 7 / 0.1  |
| Loss Function | Cross Entropy Loss | Cross Entropy Loss | Cross Entropy Loss | Cross Entropy Loss |
| Regularization | -       | -       | L2 (λ=1e-4) | L2 (λ=1e-4) |
| Epochs / Batch Size | 10 / 64 | 10 / 64 | 15 / 64 | 15 / 64 |




### Proposed TPU Architecture :  
![RPTPU_DATAPATH drawio (6)](https://github.com/user-attachments/assets/fb4c0342-37bb-40c0-9241-e5ba87262708)

### Reduce Processing Element (RPE) :   
![RPE drawio](https://github.com/user-attachments/assets/c790f418-5e94-47a2-b850-18127da7769d)

### Compensation Processing Element (CPE) :  
![CPE drawio](https://github.com/user-attachments/assets/e12d8fac-3e1d-444b-9175-dcd8a724af95)
