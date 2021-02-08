# NeuralProtect

## I.Project details

**Authors:** Abay Akturin, Chi Zhang, John-eun Kang

**The Goal:** Design a backdoor detector for BadNets trained on the YouTube Face dataset.

**Useful links:**

1.   [CSAW-HackML-2020 ](https://www.csaw.io/hackml)
2.   [CSAW-HackML-2020 GitHub](https://github.com/csaw-hackml/CSAW-HackML-2020)

**Papers:**
1.   [STRIP: a defence against trojan attacks on deep neural networks](https://arxiv-org.proxy.library.nyu.edu/pdf/1902.06531.pdf)

2.   [Fine-Pruning: Defending Against Backdooring Attacks
on Deep Neural Networks
](https://arxiv.org/pdf/1805.12185.pdf)

3. [Neural Cleanse: Identifying and Mitigating
Backdoor Attacks in Neural Networks
](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf)

## II.File organization
```bash
├── data 
    └── clean_validation_data.h5 // this is clean data used to evaluate the BadNet and design the backdoor defense
    └── clean_test_data.h5
    └── sunglasses_poisoned_data.h5
    └── anonymous_1_poisoned_data.h5
    └── multi-trigger_multi-target
        └── eyebrows_poisoned_data.h5
        └── lipstick_poisoned_data.h5
        └── sunglasses_poisoned_data.h5
        
├── GoodNets
    └── eval_anonymous_1.py //anonymous_1_bd_net.h5
    └── eval_anonymous_2.py //anonymous_2_bd_net.h5
    └── eval_multi.py //multi_trigger_multi_target_bd_net.h5
    └── eval_sunglasses.py // Goodnet for sunglasses_bd_net.h5
        
├── models
    └── sunglasses_bd_net.h5
    └── sunglasses_bd_weights.h5
    └── multi_trigger_multi_target_bd_net.h5
    └── multi_trigger_multi_target_bd_weights.h5
    └── anonymous_1_bd_net.h5
    └── anonymous_1_bd_weights.h5
    └── anonymous_2_bd_net.h5
    └── anonymous_2_bd_weights.h5
    
├── test_images   

├── MLCS_Final_Project_Report.pdf
     
```


## III.Dependencies 

The latest versions available on 12/22/20:
1. keras <br>
2. sys <br>
3. h5py<br>
4. numpy<br>
5. cv2<br>
6. scipy<br>
7. scipy.stats


## IV.Running the tests
The Python files in the GoodNets folder are used to evaluate each GoodNet. 

To run the test, run ``` python3 <good_net_to_evaluate.py> <test_image.png> ```.
The output class is in the range of [0, 1283]. The GoodNet will output 1283 if the test image is poisoned, else the output class is in rage[0, 1282].

Example:
```bash
python3 GoodNets/eval_sunglasses.py test_images/sunglass_poisoned_2.png
```
