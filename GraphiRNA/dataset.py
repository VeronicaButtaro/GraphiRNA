import pandas as pd

from sklearn.model_selection import StratifiedKFold


class MiRNADataProcessor:
    def __init__(self, raw_data_path="../data/raw", processed_data_path="../data/processed"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.df_concat = None

    def load_data(self):

        self.df_na = pd.read_csv(f"{self.raw_data_path}/df.na.csv", sep=',').rename(
            columns={'miRNAme': 'ID'}).set_index('ID')
        self.df_na = self.df_na[['sequence.miRBase']].rename(columns={'sequence.miRBase': 'Sequence'})

        self.df_84 = pd.read_csv(f"{self.raw_data_path}/GSE120584(1).csv", sep=',')
        self.df_84_meta = pd.read_csv(f"{self.raw_data_path}/GSE120584.metadata (1).csv", sep=',').set_index('sample')

        self.df_93 = pd.read_csv(f"{self.raw_data_path}/GSE150693 (1).csv", sep=',')
        self.df_93_meta = pd.read_csv(f"{self.raw_data_path}/GSE150693.metadata (1).csv", sep=',').set_index('sample')
        self.df_93_meta['disease'] = self.df_93_meta['disease'].replace({'MCI-C': 'MCI'})

        self.df_89 = pd.read_csv(f"{self.raw_data_path}/GSE215789 (1).csv", sep=',')
        self.df_89_meta = pd.read_csv(f"{self.raw_data_path}/GSE215789_metadata (1).csv", sep=',').set_index('sample')
        self.df_89_meta['disease'] = self.df_89_meta['disease'].replace({'Control': 'NC'})

        self.df_23 = pd.read_csv(f"{self.raw_data_path}/GSE242923 (1).csv")
        self.df_23_meta = pd.read_csv(f"{self.raw_data_path}/GSE242923.metadata (1).csv").set_index('Sample')
        self.df_23_meta['Disease'] = self.df_23_meta['Disease'].replace({'Control': 'NC', 'AD': 'AD'})

    def preprocess(self):

        self.df_84_gen = self.df_84.drop(columns=[self.df_84.columns[0], self.df_84.columns[2]]).set_index('ID').T
        self.df_93_gen = self.df_93.drop(columns=[self.df_93.columns[0], self.df_93.columns[2]]).set_index('ID').T
        self.df_89_gen = self.df_89.drop(columns=[self.df_89.columns[0], self.df_89.columns[2]]).set_index('ID').T
        self.df_23_gen = self.df_23.drop(columns=[self.df_23.columns[0], self.df_23.columns[2]]).set_index('ID').T


        self.df_84_tot = self.df_84_gen.merge(self.df_84_meta, left_index=True, right_index=True)
        self.df_93_tot = self.df_93_gen.merge(self.df_93_meta, left_index=True, right_index=True)
        self.df_89_tot = self.df_89_gen.merge(self.df_89_meta, left_index=True, right_index=True)
        self.df_23_tot = self.df_23_gen.merge(self.df_23_meta, left_index=True, right_index=True)
        self.df_23_tot['disease'] = self.df_23_tot.pop('Disease')
        self.df_23_tot['country'] =self.df_23_tot.pop('Country')




        self.df_concat = pd.concat([self.df_84_tot, self.df_93_tot, self.df_23_tot])
        self.df_concat = self.df_concat[self.df_concat['disease'] != "MCI-NC"]
        self.df_concat['disease'] = self.df_concat['disease'].replace({'Alzheimer Disease': 'AD'})
        self.df_concat = self.df_concat[~((self.df_concat['age'] >= 85) & (self.df_concat['disease'] == 'AD'))]




    def save_processed_data(self):
        """Salva i dataset preprocessati."""
        self.df_84_tot.to_csv(f"{self.processed_data_path}/df_84_tot.csv")
        self.df_93_tot.to_csv(f"{self.processed_data_path}/df_93_tot.csv")
        self.df_89_tot.to_csv(f"{self.processed_data_path}/df_89_tot.csv")
        self.df_23_tot.to_csv(f"{self.processed_data_path}/df_23_tot.csv")
        self.df_concat.to_csv(f"{self.processed_data_path}/df_concat_final.csv")




    def create_folds(self, n_splits=5):

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(skf.split(self.df_concat, self.df_concat['disease'])):
            train_df = self.df_concat.iloc[train_idx]
            test_df = self.df_concat.iloc[test_idx]

            train_df.to_csv(f"{self.processed_data_path}/train_fold_{fold}.csv")
            test_df.to_csv(f"{self.processed_data_path}/test_fold_{fold}.csv")



if __name__ == "__main__":


    processor = MiRNADataProcessor(
        raw_data_path="../data/raw",
        processed_data_path="../data/processed"
    )

    processor.load_data()
    processor.preprocess()
    processor.save_processed_data()
    processor.create_folds(n_splits=5)
