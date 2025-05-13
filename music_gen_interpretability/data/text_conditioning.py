import pandas as pd
from music_gen_interpretability.data.generic_data_module import GenericDataModule


class TextConditioning(GenericDataModule):
    def setup(self, stage: str = None) -> None:
        super().setup(stage)
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.sample(n=self.num_samples, random_state=self.random_state, replace=False)
        self.df.reset_index(drop=True, inplace=True)

    def prepare_data(self, stage: str = None) -> None:
        super().prepare_data(stage)
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.sample(n=self.num_samples, random_state=self.random_state, replace=False)
        self.df.reset_index(drop=True, inplace=True)
