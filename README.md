# Optimization Techniques in Deep Learning 🚀

Welcome to my little project where I’ve implemented and explored some of the most powerful **optimization strategies** used in training neural networks. If you're into machine learning or deep learning, this is something you’ll probably run into all the time.

---

## 🌄 What This Project Covers

I’ve built a few core functions that mimic how modern neural networks optimize weights during training. This includes:

### 1. **Adam Optimizer**
Adam is like that smart friend who knows how to balance momentum and adaptivity at the same time. I implemented:

- `initialize_adam()` – to set up the moving averages (first moment `v` and second moment `s`) for gradients.
- `update_parameters_with_adam()` – performs parameter updates using bias-corrected values of `v` and `s`.

The update rule is essentially:
```math
v = \beta_1 \cdot v + (1 - \beta_1) \cdot \text{grad}
```

```math
s = \beta_2 \cdot s + (1 - \beta_2) \cdot \text{grad}^2
```

```math
v_{\text{corrected}} = \frac{v}{1 - \beta_1^t}
```
```math
s_{\text{corrected}} = \frac{s}{1 - \beta_2^t}
```
```math
\text{parameter} = \text{parameter} - \alpha \cdot \frac{v_{\text{corrected}}}{\sqrt{s_{\text{corrected}}} + \varepsilon}
```


---

### 2. **Learning Rate Decay**
I added two types of learning rate decay:

- `update_lr()` – simple exponential decay with epoch count.
  
```math
\alpha = \frac{\alpha_0}{1 + \text{decay\_rate} \times \text{epoch}}
```


- `schedule_lr_decay()` – same idea, but it only kicks in every `time_interval` epochs.

```math
\alpha = \frac{\alpha_0}{1 + \text{decay\_rate} \times \left\lfloor \frac{\text{epoch}}{\text{time\_interval}} \right\rfloor}
```

---

## 🧠 Visual Intuition

Here’s a beautiful image to illustrate what’s happening behind the scenes. Think of the red stick figures as the model trying to "climb" towards the optimal solution, and the winding path as the optimizer helping it reach the peak effectively.

### 📸 optimization-journey.png  
![Optimization Journey](./Optimization%20Image.png)



---

## 🧪 Test Cases

I validated the functions with assertions to ensure they:
- Preserve shapes and types
- Apply updates correctly
- Handle bias correction properly
- Adjust learning rate only at intended intervals

---

## 🙌 Final Thoughts

This was a super insightful project. It gave me a solid understanding of how optimizers like Adam really work behind the scenes and how important it is to manage learning rates during training.


If you found this interesting or have any questions, feel free to connect! 😊







