import matplotlib.pyplot as plt

# Model names and their corresponding accuracies
data_name = ['Train', 'Validation', 'Test']
tot_data = [10500, 518, 518]

# Bar width and positions
bar_width = 0.3  # Adjusted bar width for better spacing
x = range(len(data_name))

# Creating the bar plot
plt.figure(figsize=(10, 6))
bars1 = plt.bar(x, tot_data, width=bar_width, label='Total Dataset', color='r')

# Adding labels and title
plt.xlabel('Category', fontsize=12)
plt.ylabel('Amount', fontsize=12)
plt.xticks([p + bar_width / 2 for p in x], data_name)  # Centering the x-ticks
plt.ylim(0, 11000)  # Adjusted y-limit for clarity

# Adding legend
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Displaying the plot
plt.tight_layout()
plt.show()
