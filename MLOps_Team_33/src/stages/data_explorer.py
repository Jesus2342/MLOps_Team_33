import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

class DataExplorer:
    def __init__(self, config_path: str):
        with open(config_path) as config_file:
            self.config = yaml.safe_load(config_file)
        self.target_column = self.config['target']['column']

    def explore_data(self, data: pd.DataFrame):
        """
        Print the first few rows, descriptive statistics, and information about the dataset.
        :param data: Pandas DataFrame to explore.
        """
        print("\nFirst rows of the dataset:\n")
        print(data.head().T)
        print("\nDescriptive statistics:\n")
        print(data.describe())
        print("\nDataset info:\n")
        print(data.info())

    def target_classes_distribution(self, data: pd.DataFrame):
        """
        Plot the class distribution of the target column.
        :param data: Pandas DataFrame containing the target column.
        """
        plt.figure(figsize=(8, 6))
        target_counts = data[self.target_column].value_counts(normalize=True) * 100
        ax = sns.barplot(x=target_counts.index, y=target_counts.values, palette='viridis')
        plt.title(f'{self.target_column} Class Distribution (%)')
        plt.xlabel(self.target_column)
        plt.ylabel('Percentage')

        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                        textcoords='offset points')

        plt.show()

    @staticmethod
    def plot_heat_map(my_df: pd.DataFrame):
        """
        Plot a heatmap of the correlation matrix of the DataFrame.
        :param my_df: Pandas DataFrame to plot.
        """
        cat_encoder = LabelEncoder()
        lista_colmuns = [col for col in my_df.columns]

        for col in lista_colmuns:
            my_df[col] = cat_encoder.fit_transform(my_df[col])

        corr_matrix = my_df.corr()
        plt.figure(figsize=(10, 10))
        plt.title('Relationship between Performance')
        sns.heatmap(corr_matrix, annot=True)
        plt.show()

    @staticmethod
    def pivot_tables(data: pd.DataFrame):
        """
        Plot pivot tables showing the relationship between features and the target column.
        :param data: Pandas DataFrame containing the data.
        """
        x_variables = data.drop(columns=['Performance']).columns
        num_vars = len(x_variables)
        num_cols = 3  # Number of columns for subplots
        num_rows = (num_vars + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
        axes = axes.flatten()

        for i, x_var in enumerate(x_variables):
            pivot = pd.pivot_table(data, index=x_var, columns='Performance', aggfunc='size', fill_value=0)
            sns.heatmap(pivot, annot=True, cmap='Blues', fmt='d', ax=axes[i])
            axes[i].set_title(f'Performance by {x_var}')
            axes[i].tick_params(axis='x', rotation=45)

        # Hide any extra subplots
        for i in range(num_vars, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    def run_all(self, data: pd.DataFrame):
        """
        Run all data exploration methods on the provided DataFrame.
        :param data: Pandas DataFrame to explore.
        """
        self.explore_data(data)
        self.target_classes_distribution(data)
        self.plot_heat_map(data)
        self.pivot_tables(data)

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    explorer = DataExplorer(config_path=args.config)

    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
        data_path = config['data']['dataset_arff']

    data, meta = arff.loadarff(data_path)
    data = pd.DataFrame(data)
    explorer.run_all(data)