"""Verification of zero data leakage across train, val, and test splits."""
from pathlib import Path
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"


class TestDataSplitsAndLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_path = SPLITS_DIR / "train.csv"
        cls.val_path = SPLITS_DIR / "val.csv"
        cls.test_path = SPLITS_DIR / "test.csv"
        cls.df_train = pd.read_csv(cls.train_path)
        cls.df_val = pd.read_csv(cls.val_path)
        cls.df_test = pd.read_csv(cls.test_path)

    def test_split_files_exist(self):
        self.assertTrue(self.train_path.exists(), f"Missing split file: {self.train_path}")
        self.assertTrue(self.val_path.exists(), f"Missing split file: {self.val_path}")
        self.assertTrue(self.test_path.exists(), f"Missing split file: {self.test_path}")

    def test_zero_base_id_leakage(self):
        train_bases = set(self.df_train["base_id"])
        val_bases = set(self.df_val["base_id"])
        test_bases = set(self.df_test["base_id"])

        train_val_overlap = train_bases.intersection(val_bases)
        train_test_overlap = train_bases.intersection(test_bases)
        val_test_overlap = val_bases.intersection(test_bases)

        self.assertEqual(len(train_val_overlap), 0, f"Leakage detected between train and val: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"Leakage detected between train and test: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"Leakage detected between val and test: {val_test_overlap}")

    def test_sample_and_episode_counts(self):
        self.assertEqual(len(self.df_train), 630, f"Expected 630 train samples, got {len(self.df_train)}")
        self.assertEqual(len(self.df_val), 135, f"Expected 135 val samples, got {len(self.df_val)}")
        self.assertEqual(len(self.df_test), 135, f"Expected 135 test samples, got {len(self.df_test)}")

        self.assertEqual(self.df_train["base_id"].nunique(), 210)
        self.assertEqual(self.df_val["base_id"].nunique(), 45)
        self.assertEqual(self.df_test["base_id"].nunique(), 45)

    def test_role_balance_across_splits(self):
        for name, df in [("train", self.df_train), ("val", self.df_val), ("test", self.df_test)]:
            role_counts = df["role"].value_counts().to_dict()
            counts = list(role_counts.values())
            self.assertTrue(all(c == counts[0] for c in counts), f"Unbalanced roles in {name}: {role_counts}")


if __name__ == "__main__":
    unittest.main()
