import requests
import matplotlib.pyplot as plt
import numpy as np
LORCANA_API_ENDPOINT = "https://api.lorcana-api.com/cards"
if __name__ == "__main__":
    page_num = 1
    results = []
    while True:
        response = requests.get(f'{LORCANA_API_ENDPOINT}/all?page={page_num}&pagesize=1000')
        if not response.json():
            break
        results += response.json()
        page_num += 1
    color_mapping = {
        'Steel': 'gray',
        'Sapphire': 'darkblue',
        'Amethyst': 'orange',
        "Emerald": "green",
        "Amber": "purple",
        "Ruby": "red",
        # Add more color mappings as needed
    }

    # Extracting relevant data for the plot
    costs_by_color = {}  # Dictionary to store costs for each color

    for payload in results:
        color = payload['Color']
        cost = payload['Cost']

        if color not in costs_by_color:
            costs_by_color[color] = []

        costs_by_color[color].append(cost)

    # Plotting overlaid bar plots for each color
    colors = list(costs_by_color.keys())
    num_colors = len(colors)
    bar_width = 0.2
    bar_positions = np.arange(1, 11)  # Assuming costs from 1 to 10

    fig, ax = plt.subplots()

    for i, color in enumerate(colors):
        cost_counts = np.histogram(costs_by_color[color], bins=np.arange(1, 12))[0]
        ax.bar(bar_positions + i * (bar_width + 0.1), cost_counts, bar_width, label=color,
               color=color_mapping.get(color, 'blue'))

    # Adding labels and legend
    ax.set_xlabel('Cost')
    ax.set_ylabel('Count of Cards')
    ax.set_title('Count of Cards by Cost and Color')
    ax.set_xticks(bar_positions + bar_width * (num_colors - 1) / 2)
    ax.set_xticklabels(range(1, 11))
    ax.legend(title='Color')

    ax.tick_params(axis='x', which='both', pad=15)

    # Display the plot
    plt.show()