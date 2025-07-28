import pandas as pd

import matplotlib.pyplot as plt

# Data
data = {
    "epoch": list(range(30)),
    "train_loss": [0.1924, 0.0553, 0.0441, 0.0370, 0.0333, 0.0292, 0.0276, 0.0252, 0.0237, 0.0225, 0.0210, 0.0198, 0.0189, 0.0182, 0.0171, 0.0165, 0.0158, 0.0158, 0.0152, 0.0142, 0.0138, 0.0132, 0.0135, 0.0126, 0.0125, 0.0116, 0.0123, 0.0113, 0.0110, 0.0107],
    "val_loss": [0.0645, 0.0488, 0.0392, 0.0370, 0.0348, 0.0303, 0.0290, 0.0278, 0.0302, 0.0253, 0.0259, 0.0234, 0.0234, 0.0228, 0.0223, 0.0231, 0.0213, 0.0221, 0.0222, 0.0204, 0.0214, 0.0206, 0.0202, 0.0203, 0.0205, 0.0216, 0.0208, 0.0202, 0.0205, 0.0199]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o')

# Labels and Title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve for Conv GRU Model')
plt.legend()
plt.grid(True)

# Show Plot
plt.show()
# Save Plot
plt.savefig("loss_curve_conv_gru.png", dpi=300, bbox_inches='tight')