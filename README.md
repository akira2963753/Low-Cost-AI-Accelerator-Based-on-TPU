# Low Cost AI Accelerator Based on TPU    

## Tensor Processing Unit (TPU) :
...


## Weight-Stationary Data Flow :   
...

  
## Most Significant Runs (MSR) :  
通常深度神經網絡模型使用32位元浮點數 (Floating Point) 運算進行訓練。訓練完成後可以獲得32位元的權重值。然而，為了減少計算資源和時間，深度神經網路通常使用定點數運算進行"推論計算"。而由於大部分的權重皆接近於0，因此我們把權重轉換成定點數時，如下圖所示，可以發現在高位元部分常常會有連續的1或是0，我們稱之為*Most Significant Runs (MSR)*。  
<img width="900" height="225" alt="image" src="https://github.com/user-attachments/assets/8b25f99f-a2e1-4d54-872e-b3422aaa75d6" />   

---
   
我們接著去分析在不同深度神經網路模型中，MSR數目各自的占比，我們將模型的權重以定點數格式量化成INT8，可以發現幾乎99%都含有MSR-4，由於權重皆是小於0的數字，我們可以將MSR-4這四個位元縮減成一個位元來表示，這不僅可以縮短我們的計算成本、功耗，也能夠降低我們使用的記憶體空間。


| MSR-N / Model | MLP |  LeNet | ResNet | AlexNet | 
|:-----:|:---:|:------:|:------:|:-------:|
| MSR-3 | 99.9% |  99.9% | 99.9% | 99.9% |
| MSR-4 | 99.98% |  98.90% | 99.61% | 99.98% |
| MSR-5 | 98.0% |  88.3% | 99.5% | 99.7% |
| MSR-6 | 78.2% |  53.4% | 99.1% | 97.8% |
| MSR-7 | 40.4% |  27.3% | 85.5% | 84.3% |

由上述可知，如果我們將有MSR-4的權重資料從8位元量化為5位元做計算，則沒有MSR-4的資料也必須要做截斷，這些截斷必定會帶來一些相對應的精確度損失...    
如果我們不想要這些精準度損失，就必須要把被截斷的部分補償回來。  
  
  

## MSR-4 Analysis : 
我們藉由去觀察訓練完的權重MSR-4的分布情形，發現每256個權重中，最差只會有2.9個是沒有MSR-4的權重資料。因此，對於256x256的Systolic Array來說，每個col我只需要3個row來做補償即可。  
| Model         | MLP        | LeNet      | ResNet     | AlexNet    |
|:---------------:|:------------:|:------------:|:------------:|:------------:|
| **Layers (CONV/FC)** | 3(0/3)     | 5(2/3)     | 17(16/1)   | 8(5/3)     |
| **Dataset**       | MNIST      | MNIST      | MNIST      | MNIST      |
| **Input Dimensions** | 28x28    | 28x28      | 28x28      | 28x28      |
| **Output Class**  | 10         | 10         | 10         | 10         |
| **Test Accuracy** | 98.08%     | 98.05%     | 99.61%     | 99.56%     |
| **MSR-4 %**       | 99.98%     | 98.90%     | 99.61%     | 99.98%     |
| **Non-MSR-4 / 256** | 0.1      | **2.9** 🟧 | 0.1        | 0.0        |


此外，在訓練模型時，一些避免overfitting的方法，因為其會將權重分布縮小的特性，也有助於我們提高MSR-4%。  
例如 : 降低學習率、L1 Regularization and L2 Regularization (Weight Decay)    
以下是我們這次訓練的模型結構 :   
| Model               | MLP          | LeNet        | ResNet         | AlexNet        |
|:-----:|:---:|:------:|:------:|:-------:|
| **Optimizer**           | Adam             | Adam             | Adam               | Adam               |
| **Learning Rate**       | 0.0001           | 0.000055         | 0.001              | 0.001              |
| **lr Scheduler**<br>(**step_size / gamma**) | -                | -                | 7 / 0.1            | 7 / 0.1            |
| **Loss Function**       | Cross Entropy Loss | Cross Entropy Loss | Cross Entropy Loss | Cross Entropy Loss |
| **Regularization**      | -                | -                | **L2 (λ=1e-4)**  | **L2 (λ=1e-4)**  |
| **Epochs / Batch Size** | 10 / 64          | 10 / 64          | 15 / 64            | 15 / 64            |


  
## Proposed TPU Architecture :   
![RPTPU drawio](https://github.com/user-attachments/assets/5da91644-8498-4474-ad14-df98444436ce)   
  
以上是我們提出的TPU架構，我們會將輸入的權重資料透過WPU，判斷是否有MSR-4，如果有的話，就可以把前面的4個位元縮減成1個位元，並且將最後一個位元捨去，因為會在RPE內部計算時將LSB固定為1作為期望值補償，但需要再資料前面標示一個Shift Bit = 0，表示其為MSR-4資料。  
而對於沒有MSR-4的資料，則是將前面四個位元保留，後面四個位元中的三個位元存入Compensation Memory，因為一樣CPE內部計算時會將LSB固定為1作為期望值補償，Shift Bit = 0，表示其為Non-MSR-4資料。   
  
接著整個TPU會以Weight-sationary data flow的方式，開始將權重和補償權重從各自的Memory中Pre-load到RPE以及CPE裡面，Pre-load結束後，Activation Memory會輸出Activation到Input Buffer以正45角的方法輸入到Systolic Array裡面。  
  
由於左半邊的Shadow Array補償架構的計算速度一定會比右邊快上不少(只要3Cycle就可以計算完成)，因此，左半邊計算完的結果會先存入Accumulator，與右半邊共用，當右半邊的結果算完後，則會和補償結果相加得到正確的值，如下圖所示。  
![Acc drawio](https://github.com/user-attachments/assets/3f9eb4eb-a362-4aea-9439-404e5581edda)  


## RPE / CPE Structure :   
|![PE drawio (1)](https://github.com/user-attachments/assets/6d8220a3-97a1-43b4-bcf9-a325b713fe92)|![CPE drawio (1)](https://github.com/user-attachments/assets/2507c175-738a-4372-8175-9c798b9057ba)|
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|


## Weight / Compensation / Activation Memory Structure :
在這個專案裡，為了實作之便利性，我們對Memory的結構稍微做了一些調整，設定其一次會輸出8個地址的資料，實際上可以將這些單一塊的Memory看做是8個SRAM，一次輸出8筆資料。
![Memory drawio (1)](https://github.com/user-attachments/assets/7948d795-7941-4c81-9795-a97a43021b8e)

### Memory Read Control 
系統會在Mem_Write訊號Done之後，準備讀出Weight Memory and Compensation Memory的Weight Data pre-load到Systolic Array的PE裡面。因此，在Mem_Write結束的同時，我將Mem_Rd_en在負緣拉起，使Mem讀出資料，下一個負緣Cycle再讓Pre_LoadWeight、Pre_LoadCWeight拉起，讓剛剛那筆資料順利送入到Systolic Array裡面。  
    
<img width="1479" height="265" alt="image" src="https://github.com/user-attachments/assets/38a219e8-0829-4202-b606-5d9f348363e4" />   
    
而Activation Memory也是，系統會在權重Pre-load完後加入Activation，我們可以進一步在最後一個權重Pre-load進來前，在負緣將Mem_Rd_en拉起，這樣在下一個Cycle，負緣拉起Cal，PE正緣讀到開始計算，就可以馬上輸出Activation給Buffer，加快速度。  
     
<img width="1483" height="381" alt="image" src="https://github.com/user-attachments/assets/c862e6f0-32f7-44e1-a536-39cbc3576a18" />  
  



