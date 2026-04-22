# 📌 Self-Pruning Neural Network – Report

## 🔹  Why L1 Penalty on Sigmoid Gates Encourages Sparsity?

The L1 penalty adds a cost proportional to the value of each gate, which pushes many gate values toward zero during training.

Important weights contribute significantly to the model’s predictions, so reducing their gate values would increase the classification loss. Because of this, they resist being pushed to zero and remain active.

On the other hand, weights that have little impact on the output can be reduced without hurting performance, so the L1 penalty dominates and make their gate values close to zero.

As a result, the network naturally keeps only the important connections and prunes away the unnecessary ones, leading to a sparse model.

---

## 🔹 Total Loss Function

The model is trained using the following objective:

$$\mathrm{Total\ Loss}=\mathrm{Classification\ Loss}+\lambda\cdot\mathrm{Sparsity\ Loss}$$

where:

* Classification Loss helps the model improve prediction accuracy
* Sparsity Loss encourages the network to remove unnecessary connections
* λ controls the tradeoff between accuracy and sparsity
---
##  Results: Effect of λ on Accuracy and Sparsity

| Lambda | Test Accuracy (%) | Sparsity (%) |
| ------ | ----------------- | ------------ |
| 0.1    | 80.07             | 51.03        |
| 1.0    | 79.53             | 59.93        |
| 5.0    | 79.24             | 72.21        |
| 10.0   | 78.87             | 79.71        |
| 20.0   | 78.66             | 85.61        |

---
## 🔹 Observations

* Increasing λ increases sparsity in the network.
* This demonstrates the trade-off between model compression and predictive performance.

---

## 🔹Gate Value Distribution (for best model , λ=20)

![Gate Distribution](gate_distribution.png)


The histogram of gate values for the best model shows:

* A large spike at 0 → indicating many pruned connections.
* A smaller cluster near 1 → representing important active connections.

This bimodal distribution confirms that the network successfully learned to prune itself.

---

## 🔹 Conclusion

The self-pruning mechanism effectively reduces the number of active parameters while maintaining high accuracy. This demonstrates that neural networks are highly overparameterized and can be significantly compressed without major performance loss.

The experiment successfully shows how L1 regularization on learnable gates can enable dynamic, data-driven pruning during training.
