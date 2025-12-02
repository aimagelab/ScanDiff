from typing import Any

#from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader

# from src.data.collators import DefaultCollator
from src.utils.pylogger import RankedLogger  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


class UnifiedDataModule:
    def __init__(
        self,
        train_datasets,
        val_datasets,
        test_datasets,
        train_collators,
        val_collators,
        test_collators,
        use_abs_coords: bool = True,
        time_in_ms: bool = False,
        max_len: int = 7,
        batch_size_train: int = 64,
        batch_size_validation: int = 1,
        batch_size_test: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        task_embeddings_file: str = None,
    ) -> None:
        super().__init__()

        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        self.train_collators = train_collators
        self.val_collators = val_collators
        self.test_collators = test_collators
        self.use_abs_coords = use_abs_coords
        self.time_in_ms = time_in_ms
        self.max_len = max_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.batch_size_train = batch_size_train
        self.batch_size_validation = batch_size_validation
        self.batch_size_test = batch_size_test
        self.data_train = None
        self.data_test = None
        self.task_embeddings_file = task_embeddings_file

    def setup(self):
        # Load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            self.data_train = ConcatDataset(self.train_datasets)
            self.data_val = ConcatDataset(self.val_datasets)
            self.data_test = ConcatDataset(self.test_datasets)

    def train_dataloader(self) -> DataLoader[Any]:
        assert self.data_train is not None, "Data not loaded. Call `setup` first."
        
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.train_collators[0],
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        assert self.data_test is not None, "Data not loaded. Call `setup` first."
        
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_validation, # equal to train batch size
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.test_collators[0],
            drop_last=self.drop_last,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        assert self.data_test is not None, "Data not loaded. Call `setup` first."
        
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_test,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.test_collators[0],
            drop_last=self.drop_last,
        )

if __name__ == "__main__":
    _ = UnifiedDataModule()