import matplotlib.pyplot as plt

epochs = list(range(1, 21))

train_losses = [
1.4270,1.3680,1.3000,1.2350,1.1700,
1.1300,1.0500,0.9650,0.9100,0.8550,
0.8100,0.7800,0.7550,0.7450,0.7300,
0.7200,0.7050,0.7000,0.6950,0.6850
]

val_accuracies = [
0.2850,0.3600,0.3850,0.4500,0.4550,
0.4850,0.4950,0.5550,0.5700,0.6250,
0.6300,0.6650,0.6750,0.6750,0.6800,
0.6750,0.6800,0.7250,0.6950,0.7100
]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, train_losses, marker='o')
plt.title("Training Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")

plt.subplot(1,2,2)
plt.plot(epochs, val_accuracies, marker='o')
plt.title("Validation Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")

plt.tight_layout()
plt.show()
