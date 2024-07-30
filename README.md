import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# Normalize data by making 0 always good and 1 always bad
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    
    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    df_cat['total'] = 1
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count()
    df_cat = df_cat.rename(columns={'total': 'count'})
    
    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', y='count', hue='value', col='cardio', data=df_cat, kind='bar').fig
    
    # Get the figure for the output
    return fig

# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', square=True, linewidths=.5, center=0, cbar_kws={"shrink": .5}, ax=ax)

    # Get the figure for the output
    return fig
main.py
python
Copy code
import unittest
from medical_data_visualizer import draw_cat_plot, draw_heat_map

class TestMedicalDataVisualizer(unittest.TestCase):

    def test_draw_cat_plot(self):
        cat_plot_fig = draw_cat_plot()
        self.assertIsNotNone(cat_plot_fig, "Cat plot figure is None")

    def test_draw_heat_map(self):
        heat_map_fig = draw_heat_map()
        self.assertIsNotNone(heat_map_fig, "Heat map figure is None")

if __name__ == '__main__':
    unittest.main()
Instructions for running tests
To run the tests, execute main.py. The unittest framework will run the defined tests, ensuring your functions for generating the categorical plot and the heat map work as expected.

This setup will help you develop and verify the functionality of your code effectively. If you have any more specific requirements or questions, feel free to ask!

 # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', square=True, linewidths=.5, center=0, cbar_kws={"shrink": .5}, ax=ax)
ChatGPT
Got it! Here's the medical_data_visualizer.py file with the detailed implementation, including the heatmap drawing code and the proper handling of instructions.

medical_data_visualizer.py
python
Copy code
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data
df = pd.read_csv('medical_examination.csv')

# 2. Add 'overweight' column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3. Normalize data by making 0 always good and 1 always bad
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Draw Categorical Plot
def draw_cat_plot():
    # 5. Create DataFrame for cat plot using pd.melt
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6. Group and reformat the data to split it by 'cardio'. Show the counts of each feature.
    df_cat['total'] = 1
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count()
    df_cat = df_cat.rename(columns={'total': 'count'})

    # 7. Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(x='variable', y='count', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 8. Get the figure for the output
    return fig

# 9. Draw Heat Map
def draw_heat_map():
    # 10. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 11. Calculate the correlation matrix
    corr = df_heat.corr()

    # 12. Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 13. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 14. Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', square=True, linewidths=.5, center=0, cbar_kws={"shrink": .5}, ax=ax)

    # 15. Get the figure for the output
    return fig
