# Adversarial Evasion Attacks on Neural Networks Using the Adversarial Robustness Toolbox (ART)


## 📄 About

This project demonstrates the implementation of **Evasion Attacks** using the **Adversarial Robustness Toolbox (ART)**. Evasion attacks are a form of adversarial attack where a model is fooled into making incorrect predictions by slightly altering the input data. The project utilizes a neural network built with TensorFlow and Keras to showcase these attacks, focusing on techniques like Fast Gradient Method, Basic Iterative Method, and Projected Gradient Descent.

---

## 🌍 Impact

Evasion attacks highlight the vulnerabilities of machine learning models, especially in critical applications like cybersecurity, autonomous vehicles, and healthcare. Understanding and simulating these attacks is crucial for developing more robust and secure AI systems.

---

## 🔬 Methodology

1. **Data Loading and Preprocessing:** The dataset is loaded and preprocessed to fit the requirements of the neural network model.
2. **Model Creation:** A convolutional neural network (CNN) is built using TensorFlow and Keras, designed to classify images.
3. **Attack Implementation:** Evasion attacks are performed on the model using ART’s built-in methods.
4. **Evaluation:** The model's performance is evaluated before and after the attack to assess its robustness.

---

## 🛠️ Technologies Used

- **TensorFlow & Keras:** For building and training the neural network model.
- **Adversarial Robustness Toolbox (ART):** For implementing evasion attacks.

---

## 🧩 Code Snippets

1. **Model Setup:**
    ```python
    model = tf.keras.models.Sequential([
        layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),        
        layers.Dense(1024 , activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
        )
    ```

2. **Evasion Attack:**
    ```python
    classifier = KerasClassifier(model=model, clip_values=(0, 1))
    attack = FastGradientMethod(estimator=classifier, eps=0.2)
    x_train_adv = attack.generate(x=x_train)
    ```

---

## 🔍 Findings

### 1. Evasion Attack Impact on Model Performance

![a00eb373-f00b-4539-aec6-4d8f76467893](https://github.com/user-attachments/assets/831f43ef-c034-48b9-9f86-9b71e397fb04)

The images above demonstrate how increasing the attack strength (ε) impacts the accuracy of the neural network model. The attack strength level, denoted by ε, represents the degree of noise added to the input data to fool the model. Here's a breakdown of the findings:

- **Low attack strength (ε = 0.01 - 0.05):** The model maintains high accuracy, consistently predicting the digit "7" correctly with minor noise.
- **Moderate attack strength (ε = 0.075 - 0.15):** As noise increases, accuracy drops gradually, indicating the model's increasing difficulty in maintaining correct predictions.
- **High attack strength (ε = 0.2 - 0.3):** The model's accuracy significantly decreases, struggling to correctly classify the digit as the noise distorts the image further.
- **Very High attack strength (ε = 0.5):** The model fails completely, predicting an incorrect digit, showing the model's vulnerability to substantial adversarial noise.

### 2. Comparison of Original and Robust Classifier

![1be24f13-9548-4ac0-a29c-7e99ee41d5e3](https://github.com/user-attachments/assets/2242e64c-f682-4be0-ab8b-03417c455b1a)

The graph depicts a comparison between the original classifier and a robust classifier under varying attack strengths (ε). The robust classifier, likely trained with adversarial training or a similar defense strategy, shows improved resistance to adversarial attacks:

- **Original Classifier:** Performance degrades rapidly as attack strength increases, showing a steep drop in accuracy.
- **Robust Classifier:** Exhibits more resilience, with accuracy degrading less steeply, indicating that adversarial training or similar methods can indeed strengthen the model against evasion attacks.

These findings highlight the importance of adversarial training and defense mechanisms to build more secure and reliable models, particularly in environments where adversarial threats are a concern.

---

## 🔚 Conclusion

The project successfully demonstrates the vulnerabilities of machine learning models to evasion attacks using ART. By understanding these weaknesses, researchers and engineers can develop more resilient models, contributing to the advancement of AI security.
