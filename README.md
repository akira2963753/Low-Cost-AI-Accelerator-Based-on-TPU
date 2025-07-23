# Low Cost AI Accelerator Based on Systolic Array 

## What is the *Systolic Array*  :





## What is the *Most Significant Runs (MSR)* :  
通常深度神經網絡模型使用32位元浮點數 (Floating Point) 運算進行訓練。訓練完成後可以獲得32位元的權重值。然而，為了減少計算資源和時間，深度神經網路通常使用定點數運算進行"推論計算"。而由於大部分的權重皆接近於0，因此我們把權重轉換成定點數時，如下圖所示，可以發現在高位元部分常常會有連續的1或是0，我們稱之為*Most Significant Runs (MSR)*。  
<img width="1867" height="440" alt="image" src="https://github.com/user-attachments/assets/8b25f99f-a2e1-4d54-872e-b3422aaa75d6" />
我們接著去分析在不同深度神經網路模型中，MSR數目各自的占比，我們將模型的權重以定點數格式量化成INT8，可以發現幾乎99%都含有MSR-4，由於權重皆是小於0的數字，我們可以將MSR-4這四個位元縮減成一個位元來表示，這不僅可以縮短我們的計算成本、功耗，也能夠降低我們使用的記憶體空間。
<img width="1134" height="700" alt="image" src="https://github.com/user-attachments/assets/2fe2cfa9-9f78-4b57-9199-6fbf8e2026f6" />  
   
Q1.7格式 :   
假設權重是0.0566，我們會將其轉換成INT8定點數(Q1.7格式)，一個Bit代表整數(Sign Bit)，其他7個bit為小數部分  
因此0.0566 * 128 = 7.2448(Round) = 7 = 00000111(2)   
  

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
