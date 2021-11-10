import json
import matplotlib.pyplot as plt

from args import args

with open(args.training_history_filepath, 'r') as fp:
    training_history = json.load(fp)

best_acc = max(training_history['val_acc'])
best_model = training_history['val_acc'].index(best_acc)
best_loss = training_history['val_loss'][best_model]

fig, axs = plt.subplots(2)

axs[0].set_title('Loss')
axs[0].plot(training_history['training_loss'], label='train')
axs[0].plot(training_history['val_loss'], label='val')
axs[0].plot([best_model], [best_loss], 'ro', label='best')
axs[0].legend()

axs[1].set_title('Accuracy')
axs[1].plot(training_history['training_acc'], label='train')
axs[1].plot(training_history['val_acc'], label='val')
axs[1].plot([best_model], [best_acc], 'ro', label='best')
axs[1].legend()

plt.show()