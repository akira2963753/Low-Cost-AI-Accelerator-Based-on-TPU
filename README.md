# Low Cost AI Accelerator Based on TPU   
This GitHub repository is provided for reference purposes only     
If you encounter any issues, feel free to contact harry2963753@gmail.com    
  
### Development Environment :  
- RTL Simulator : *ModelSim-Intel FPGA Standard Edition, Version 20.1.1, windows*  
- Synthesis Tool : *Synopsys Design Compiler*
- Model Training : *Pytorch*  


  
## Tensor Processing Unit (TPU) [1] :  
<img width="1116" height="839" alt="image" src="https://github.com/user-attachments/assets/47d3af4e-3567-4cf8-bcb4-d5f5aa79293b" />   

## Data Flow (OS/WS/IS) [2] [3] :
在Systolic Array的資料流中，有三種主流資料流的方式，分別是Ouput Staionary(OS), Weight Staionary(WS) and Input Stationay(IS) Data Flow.  
<img width="1846" height="722" alt="image" src="https://github.com/user-attachments/assets/c6f07320-1c2e-4bca-9610-6f9131aaee00" />
  
我們採用的是Weight Staionary (WS) Data Flow來實現我們的TPU架構。  
<img width="1664" height="877" alt="image" src="https://github.com/user-attachments/assets/c114ffd9-b225-458d-9e16-d64c49b8c25d" />   
  
## Most Significant Runs (MSR) [4] :  
通常深度神經網絡模型使用32位元浮點數 (Floating Point) 運算進行訓練。訓練完成後可以獲得32位元的權重值。然而，為了減少計算資源和時間，深度神經網路通常使用定點數運算進行"推論計算"。而由於大部分的權重皆接近於0，因此我們把權重轉換成定點數時，如下圖所示，可以發現在高位元部分常常會有連續的1或是0，我們稱之為*Most Significant Runs (MSR)*。 
<img width="1793" height="406" alt="image" src="https://github.com/user-attachments/assets/6a8130fa-d0b0-4e50-abb6-fae3c1e7e34c" />   
  
舉例來說  
以0.10534來說，我們將其轉換成定點數格式，可以得到0.10534x128=13.48(round)=13=00001101  
因此我們可以將前面的四位元0縮減成一位元的0，也不會損失精準度。   
以-0.0784來說，我們可以得到-0.0784x128=-10.0352(round)=-10=11110110  
一樣可以將前面四位元的1縮減成一位元的1。  
     
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
| **Non-MSR-4 / 256** | 0.1      | **2.9**  | 0.1        | 0.0        |
   
   
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
  
<img width="4554" height="2192" alt="RPTPU drawio" src="https://github.com/user-attachments/assets/9252b519-17c7-4bc3-8771-d5a9be06d1d9" />  

  
以上是我們提出的TPU架構，我們會將輸入的權重資料透過WPU，判斷是否有MSR-4，如果有的話，就可以把前面的4個位元縮減成1個位元，並且將最後一個位元捨去，因為會在RPE內部計算時將LSB固定為1作為期望值補償，但需要再資料前面標示一個Shift Bit = 0，表示其為MSR-4資料。  
而對於沒有MSR-4的資料，則是將前面四個位元保留，Shift Bit = 0，表示其為Non-MSR-4資料，而後面四個位元中的三個位元存入Compensation Memory，因為一樣CPE內部計算時會將LSB固定為1作為期望值補償。  
儲存和計算的方式如下圖所示 :    
|<img width="1092" height="578" alt="Design_MSR drawio (2)" src="https://github.com/user-attachments/assets/48697522-db78-4900-a01a-946885f3f35f" /> | <img width="844" height="512" alt="Cal drawio (1)" src="https://github.com/user-attachments/assets/1f6684f6-ed29-429c-8a5d-066877dd4b65" /> | 
|-----------------------------------------------|-------------------------------|

值得注意的是 : 如果要進行有號數的運算的話，我們必須將Compensation Weight多一個Sign Bit    
(根據他原來的Weight Data的正負號)  

# 
接著整個TPU會以WS Data Flow的方式，開始將權重和補償權重從各自的Memory中Pre-load到RPE以及CPE裡面，Pre-load結束後，Activation Memory會輸出Activation到Input Buffer以正45角的方法輸入到Systolic Array裡面。  
  
由於左半邊的Shadow Array補償架構的計算速度一定會比右邊快上不少(只要3Cycle就可以計算完成)，因此，左半邊計算完的結果會先存入Accumulator，寫入至他要補償的地址。藉由與右半邊共用，這樣可以省下許多硬體面積，不用在為Compensation的部分設計一塊Accumulator，當右半邊的結果算完後，則會和補償結果相加得到正確的值，如下圖所示。    
  
<img width="2584" height="854" alt="Acc drawio" src="https://github.com/user-attachments/assets/eceb0009-4f9f-4e60-abad-f00a223fcf31" />  
  
## RPE / CPE Architecture :   
| <img width="2630" height="1446" alt="PE drawio" src="https://github.com/user-attachments/assets/c13b90f2-0f2b-47ef-bc33-c8cd04cefd16" /> | <img width="2288" height="1412" alt="CPE drawio" src="https://github.com/user-attachments/assets/cdd3e260-c86d-4e87-9eaa-39a110822ab3" /> |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
  

## Weight / Compensation / Activation Memory Architecture :
在這個專案裡，為了實作之便利性，我們對Memory的結構稍微做了一些調整，設定其一次會輸出8個地址的資料，實際上可以將這些單一塊的Memory看做是8個SRAM，一次輸出8筆資料。
<img width="3150" height="698" alt="Memory drawio" src="https://github.com/user-attachments/assets/bc47a240-fb7a-4a97-a1e6-861cafecec3e" />  

## Memory Read Control : 
系統會在Mem_Write訊號Done之後，準備讀出Weight Memory and Compensation Memory的Weight Data pre-load到Systolic Array的PE裡面。因此，在Mem_Write結束的同時，我將Mem_Rd_en在負緣拉起，使Mem讀出資料，下一個負緣Cycle再讓Pre_LoadWeight、Pre_LoadCWeight拉起，讓剛剛那筆資料順利送入到Systolic Array裡面。  
而Activation Memory也是，系統會在權重Pre-load完後加入Activation，我們可以進一步在最後一個權重Pre-load進來前，在負緣將Mem_Rd_en拉起，這樣在下一個Cycle，負緣拉起Cal，PE正緣讀到開始計算，就可以馬上輸出Activation給Buffer，加快速度。  
  
<img width="1479" height="265" alt="image" src="https://github.com/user-attachments/assets/38a219e8-0829-4202-b606-5d9f348363e4" />     
<img width="1483" height="381" alt="image" src="https://github.com/user-attachments/assets/c862e6f0-32f7-44e1-a536-39cbc3576a18" />    

## RTL Simulation :  
我們實現了上述TPU架構，以8x8 Systolic Array with 8 x 3 Compensation Arra來進行電路模擬。  
詳細RTL可以參考[RTL Source](./RTL_Signed(Main))   
  
### Answer Check  
我們利用Python計算正確的結果並將RTL模擬的結果輸出至[Output.out](./RTL_Signed(Main)/Output.out)上面，可以發現結果完全一致。      
<img width="888" height="372" alt="image" src="https://github.com/user-attachments/assets/bcb49d35-67db-46d4-82fc-da65306aa883" />

### Activation Function (ReLU Function) :  
在本次專案實作中，我們的Activation Function採用的是ReLU Function，將剛剛Systolic Array得到的值，經過ReLU後送入到Unified Buffer。    
| <img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/14283b26-61a8-4405-92f1-b9ddc122e373" /> | <img width="900" height="800" alt="image" src="https://github.com/user-attachments/assets/fd5fc851-401a-4309-9184-2b514bec8611" /> | 
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|



## Accuracy Analysis :  
| **PE Type / Model**                                | **MLP**     | **LeNet**   | **ResNet**  | **AlexNet** |
|:----------------------------------------------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|**Original Precision (Float 32)**                                          | 98.08%      | 98.01%      | 99.61%      | 99.56%      |
| **Quantization Precision (INT8)**                                             | 97.28%      | 97.97%      | 99.09%      | 99.45%      |
| **Truncate 3 bits in MSR4 & Non-MSR4 Weight Data**                                      |  **92.71%** |  **89.20%** |  **11.36%** |  **19.27%** |
|  **Add Expect Value (LSB = 1)**                                        | 97.29%      | 97.44%      | 98.96%      | 99.40%      |
| **Add Expect Value (LSB = 1) & Non-MSR4 Compensation**                                      |  **97.34%** |  **98.00%** |  98.96% |  99.40% |

以上是我們利用Pytorch做Post Training Quantization得到的數據，我們可知，當我們將模型量化成INT8時，精準度會下降大概0.1% ~ 1%左右，而當我們將Non-MSR-4截斷掉後，會發現模型的精準度下降的很快，這是因為雖然Non-MSR-4的比例只佔全部的1%左右，但對於參數比較大的ResNet、AlexNet來講卻會造成很嚴重的誤差，因此我們可以加上期望值補償，並套用以上所提出之補償架構，將損失補償回去，最後甚至在MLP、LeNet上達到比原本量化更好的精準度。  

## Hardware Overhead Analysis :  
### PE Comparison  
我們對每種PE皆使用最基礎以HA和FA來搭建而成的乘法單元，以確保公平之比較。    
以下是我們有號數的PE做Synthesis所得的硬體面積大小比較  :  
|Type of PE|PE|RPE|CPE|
|:--:|:--:|:--:|:--:|
|Area | 0% | -13.2% | -33.4% |
每個RPE相比其原來的PE來說約少了357 gate count。  

### Input Buffer Comparison  
|Type|Original Input Buffer|Input Buffer|
|:--:|:--:|:--:|
|Area | 1x | 2.19x |



### Each Module  

|Module|Area Percnetage (%)|
|:--:|:--:|
|**Weight Memory**| 2.83% |
|**Activation Memory**| 4% |
|**Compensation Memory**| 1.48% |
|**Input Buffer**| 5.73% | 
|**WPU**| 0.38% | 
|**Systolic Array**| 64.5% | 
|**Accumulator**| 21.8% | 
|**Activation Function**| 0.21% | 
|**Compensation Array**|  | 


## Reference :  
**[1] In-Datacenter Performance Analysis of a Tensor Processing Unit**     
**[2] SCALE-Sim: Systolic CNN Accelerator Simulator**    
**[3] Effective_Runtime_Fault_Detection_for_DNN_Accelerators**  
**[4] Refresh Power Reduction of DRAMs in DNN Systems Using Hybrid Voting and ECC Method**   
**[5] Weight-Aware and Reduced-Precision Architecture Designs for Low-Cost AI Accelerators**    
**[6] DRUM: A Dynamic Range Unbiased Multiplier for approximate applications**   
**[7] APTPU: Approximate Computing Based Tensor Processing Unit**   


