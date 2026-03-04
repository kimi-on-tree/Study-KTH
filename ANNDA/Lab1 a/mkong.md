## 3.1.2

### Task 1

Apply and compare perceptron learning with the delta learning rule in online (sequential) mode on the generated dataset. Adjust the learning rate and study the convergence of the two algorithms.

**task 1 dataset:**

![task1_dataset](D:\study\ANN\Assignments\Lab1\task1_dataset.png)

**Lr = 0.0005:**

![task1_compare_lr=0.0005](D:\study\ANN\Assignments\Lab1\task1_compare_lr=0.0005.png)

**Lr = 0.001:**

![task1_compare_lr=0.001](D:\study\ANN\Assignments\Lab1\task1_compare_lr=0.001.png)

**Lr=0.01:**

![task1_compare_lr=0.01](D:\study\ANN\Assignments\Lab1\task1_compare_lr=0.01.png)

**Lr = 0.1:**

![task1_compare_lr=0.1](D:\study\ANN\Assignments\Lab1\task1_compare_lr=0.1.png)

General conclusion: Changing lr doesn't affect the result of classical perceptron learning but have impact on delta rule

### Task 2

Compare sequential with a batch learning approach for the delta rule. How quickly (in terms of epochs) do the algorithms converge? Please adjust the learning rate and plot the learning curves for each variant. Bear in mind that for sequential learning you should not use the matrix form of the learning rule discussed in section 2.2 and instead perform updates iteratively for each sample. How sensitive is learning to random initialisation?

**task2 dataset:**

![task2_dataset](D:\study\ANN\Assignments\Lab1\task2_dataset.png)

#### Results(left) & Learning curves(right)

**Lr = 0.005:** 

<div style="display:flex; gap:12px;">
  <img src="task2_compare_lr=0.005.png" style="width:48%;">
  <img src="task2_mse_lr=0.005.png" style="width:48%;">
</div>

**Lr = 0.01:**

<div style="display:flex; gap:12px;">
  <img src="task2_compare_lr=0.01.png" style="width:48%;">
  <img src="task2_mse_lr=0.01.png" style="width:48%;">
</div>

**Lr = 0.05:**

<div style="display:flex; gap:12px;">
  <img src="task2_compare_lr=0.05.png" style="width:48%;">
  <img src="task2_mse_lr=0.05.png" style="width:48%;">
</div>

**Lr = 0.1:**

<div style="display:flex; gap:12px;">
  <img src="task2_compare_lr=0.1.png" style="width:48%;">
  <img src="task2_mse_lr=0.1.png" style="width:48%;">
</div>

**General conclusion 1:** Because the task is simple, there is almost no difference between the online and batch methods. However, it can still be observed that the online method converges faster in the early stages.(as shown in picture when lr =0.1)

**General conclusion 2:** The larger the learning rate (lr), the faster the convergence tends to be. But when lr becomes too large(eg. 0.5), the updates can overshoot, causing excessively large jumps in the weights for the delta rule (batch) method, and leads to significant deviations and prevents the algorithm from converging.

#### Different initial conditions

![task2_mse_initialseed=43](D:\study\ANN\Assignments\Lab1\task2_mse_initialseed=43.png)

![task2_mse_initialseed=1](D:\study\ANN\Assignments\Lab1\task2_mse_initialseed=1.png)

![task2_mse_initialseed=35](D:\study\ANN\Assignments\Lab1\task2_mse_initialseed=35.png)

General conclusion: Different initial conditions can lead to different convergence speeds, but the effect is not significant.

## Task 3

Remove the bias, train your network with the delta rule in batch mode and test its behaviour. In what cases would the perceptron without bias converge and classify correctly all data samples? Please verify your hypothesis by adjusting data parameters, mA and mB.

<div style="display:flex; gap:12px;">
  <img src="task3_dataset_symmetrical.png" style="width:48%;">
  <img src="task3_nobias_Symmetrical.png" style="width:48%;">
</div>

<div style="display:flex; gap:12px;">
  <img src="task3_dataset_same_quadrant .png" style="width:48%;">
  <img src="task3_nobias_Same Quadrant.png" style="width:48%;">
</div>

<div style="display:flex; gap:12px;">
  <img src="task3_dataset_normal.png" style="width:48%;">
  <img src="task3_nobias_Normal.png" style="width:48%;">
</div>

General conclusion: When parts of the two classes fall within the same quadrant, or when the two classes collectively spread across three quadrants, correct classification is not possible without a bias term. In contrast, when the two classes are symmetric with respect to the axes or the origin, or when all data points lie in two different quadrants, correct classification can be achieved without adding a bias term.