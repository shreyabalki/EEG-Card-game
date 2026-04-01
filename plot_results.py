import matplotlib.pyplot as plt
import numpy as np
import os

labels = ['Type1', 'Type2', 'Type3']
cnn_macro_f1 = [0.461, 0.360, 0.354]
transformer_macro_f1 = [0.460, 0.383, 0.468]

x = np.arange(len(labels))
width = 0.35

plt.figure()
plt.bar(x - width/2, cnn_macro_f1, width, label='CNN')
plt.bar(x + width/2, transformer_macro_f1, width, label='Transformer')

plt.xlabel('Event Type')
plt.ylabel('Macro-F1 Score')
plt.title('CNN vs Transformer Performance')
plt.xticks(x, labels)
plt.legend()

plt.tight_layout()

# 🔴 FORCE SAVE WITH FULL PATH
save_path = os.path.join(os.getcwd(), "figure_results.png")
plt.savefig(save_path, dpi=300)

print("Saved to:", save_path)

plt.close()