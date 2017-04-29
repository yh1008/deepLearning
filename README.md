# homework-v-yh1008  
UNIs:Emily Hua yh2901, Ming Zhou mz2591  

We execuated the script on personal environment, which `environment.yml` is included.  
   
- [x] Task1  Iris+DNN  
highest accuracy is 0.97 achieved with 100 epoches (12 as first hidden layer size, and 6 as the second hidden layer size). For details, please see `task1/Task1-Iris-DNN.ipynb` 

- [x] Task2 MINIST+DNN    
vanila baseline achieves 0.9657, and adding dropout achieves 0.9795 on a DNN with the following parameters(64 epochs, 64 as the first hidden layer size, 32 as the second hidden layer size, 128 batch size, and 0.1 dropout rate)   
For output details please see `task2/Task2-MNIST-DNN.ipynb` and `task2/Task2-gpu.py`, the learning curve plot can be found in `task2/learning_curve_comparison.png`  

- [x] Task3 SVHN+CNN+bachnormalization  
the baseline model achives 0.893 accuracy; after bachnorm, achieves 0.952 accuracy on the test set. For details, please see `task3/Task3-SVHN-CNN.py`

- [x] Task4 VGG+Pets+transferredLearning  
The model achieves 0.755 test accuracy with a retrained MLP (32 epochs, 128 batch size, 32 hidden layer size). For details please see `task4/task4.py` and simply `task4/task4-notebook.ipynb`, which contains all output display.  
To execute `task4/task4.py` make sure you change `path_to_pets = "../pets/"` to let `path_to_pets` point to where `pets` folder sits on your computer.   
