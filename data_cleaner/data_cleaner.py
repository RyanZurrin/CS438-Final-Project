import pandas as pd
import numpy as np
import pydicom
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from pydicom.multival import MultiValue
from tqdm.notebook import tqdm  # Import tqdm for Jupyter Notebook


class DataCleaner:
    def __init__(self, data):
        self.df = data.copy()

    def process_ages(self, column_name):
        self.df[column_name] = self.df[column_name].apply(self.convert_age)

    def label_encode(self, columns):
        label_encoders = {}
        for column in columns:
            label_enc = LabelEncoder()
            self.df[column] = label_enc.fit_transform(self.df[column])
            label_encoders[column] = label_enc

    def map_labels(self, column, mapping):
        self.df[column] = self.df[column].map(mapping)
        # Drop rows where the 'label' column is NA after mapping
        self.df = self.df.dropna(subset=[column])
        # Convert the 'label' column to integers
        self.df[column] = self.df[column].astype(int)

    def drop_unnecessary_columns(self, columns):
        self.df.drop(columns, axis=1, inplace=True)

    def handle_missing_values(self):
        self.df = self.df.dropna()

    def convert_age(self, age_str):
        if pd.isnull(age_str) or age_str == "":
            return None
        age_num = int("".join(filter(str.isdigit, age_str)))
        if "Y" in age_str.upper():
            return age_num
        elif "M" in age_str.upper():
            return age_num / 12
        elif "D" in age_str.upper():
            return age_num / 365
        else:
            return age_num

    # def process_ages(self, column_name):
    #     self.df[column_name] = self.df[column_name].apply(self.convert_age)
    #
    # def label_encode(self, columns):
    #     for column in columns:
    #         label_enc = LabelEncoder()
    #         self.df[column] = label_enc.fit_transform(self.df[column])
    #
    # def map_labels(self, column, mapping):
    #     self.df[column] = self.df[column].map(mapping).astype(int)

    @staticmethod
    def expand_column(df, column_name, max_length):
        def expand_row(row):
            row_data = (
                list(row[column_name])
                if isinstance(row[column_name], (list, tuple, MultiValue))
                else [row[column_name]]
            )
            return pd.Series(row_data + [0] * (max_length - len(row_data)))

        expanded_data = df.apply(expand_row, axis=1)
        expanded_data.columns = [f"{column_name}{i+1}" for i in range(max_length)]
        return expanded_data

    @staticmethod
    def extend_multivalue(multival, size, fillvalue=0):
        if isinstance(multival, MultiValue):
            multival = list(multival)
        if isinstance(multival, list):
            return multival + [fillvalue] * (size - len(multival))
        else:
            return [multival] + [fillvalue] * (size - 1)

    def expand_and_extend_columns(self, expand_cols, extend_cols):
        for col in tqdm(expand_cols, desc="Expanding Columns"):
            max_length = (
                self.df[col]
                .apply(lambda x: len(x) if isinstance(x, (list, tuple)) else 1)
                .max()
            )
            expanded_data = self.expand_column(self.df, col, max_length)
            self.df = self.df.drop(col, axis=1).join(expanded_data)
        for col in tqdm(extend_cols, desc="Extending Columns"):
            max_size = max(
                self.df[col].apply(lambda x: len(x) if hasattr(x, "__iter__") else 1)
            )
            expanded_data = self.df[col].apply(self.extend_multivalue, size=max_size)
            for i in range(max_size):
                self.df[f"{col}{i+1}"] = expanded_data.apply(lambda x: x[i])

        self.df.drop(extend_cols, axis=1, inplace=True)

    @staticmethod
    def calculate_statistics(df, column_name):
        df[f"{column_name}_mean"] = df[column_name].apply(lambda x: np.mean(x))
        df[f"{column_name}_median"] = df[column_name].apply(lambda x: np.median(x))
        df[f"{column_name}_std"] = df[column_name].apply(lambda x: np.std(x))
        df[f"{column_name}_min"] = df[column_name].apply(lambda x: np.min(x))
        df[f"{column_name}_max"] = df[column_name].apply(lambda x: np.max(x))
        df[f"{column_name}_percentile_25"] = df[column_name].apply(
            lambda x: np.percentile(x, 25)
        )
        df[f"{column_name}_percentile_50"] = df[column_name].apply(
            lambda x: np.percentile(x, 50)
        )
        df[f"{column_name}_percentile_75"] = df[column_name].apply(
            lambda x: np.percentile(x, 75)
        )

    def process_and_flatten_columns(self):
        self.calculate_statistics(self.df, "histogram")
        # check if there is a column named 'image'
        if "image" in self.df.columns:
            self.df["flattened_image"] = self.df["image"].apply(
                lambda x: x.flatten() if isinstance(x, np.ndarray) else np.array([])
            )
            self.calculate_statistics(self.df, "flattened_image")
            self.df.drop(
                ["histogram", "image", "flattened_image"], axis=1, inplace=True
            )
        else:
            self.df.drop(["histogram"], axis=1, inplace=True)

    def add_correctness_column(self):
        def determine_correctness(row):
            score_threshold = 0.5
            # Correct predictions: (score >= 0.5 and label == 1) or
            # (score < 0.5 and label == 0)
            if (row["score"] >= score_threshold and row["label"] == 1) or (
                row["score"] < score_threshold and row["label"] == 0
            ):
                return 1  # Correct prediction
            else:
                return 0  # Incorrect prediction

        # Apply the function to each row to create the new correctness column
        self.df["target_label"] = self.df.apply(determine_correctness, axis=1)

    # Helper function to create binary masks from bounding boxes
    @staticmethod
    def create_binary_mask(image_shape, bounding_box):
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bounding_box
        mask[y1:y2, x1:x2] = 1
        return mask

    def enforce_numeric_data_types(self):
        non_numeric_columns = self.df.select_dtypes(exclude=np.number).columns

        for column in non_numeric_columns:
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce")

    def balance_classes(self, label):
        # Separate the dataset into minority and majority classe by checking the target label
        # column and seeing which class has fewer samples
        ones = self.df[self.df[label] == 1]
        zeros = self.df[self.df[label] == 0]

        # see which class has fewer samples
        if len(ones) < len(zeros):
            df_minority = ones
            df_majority = zeros
        else:
            df_minority = zeros
            df_majority = ones

        # Downsample the majority class to match the minority class size
        df_majority_downsampled = df_majority.sample(len(df_minority), random_state=42)

        # Combine the downsampled majority class with the original minority class
        self.df = pd.concat([df_minority, df_majority_downsampled])

        # Shuffle the dataset
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

    def scale_data(self, label="target_label", columns=None, scaler=StandardScaler()):
        features_to_scale = self.df.columns.drop(label) if columns is None else columns
        self.df[features_to_scale] = scaler.fit_transform(
            self.df[features_to_scale].values
        )

    def add_multiplicative_features(self, columns_a, columns_b):
        for col_a, col_b in zip(columns_a, columns_b):
            self.df[f"{col_a}_x_{col_b}"] = self.df[col_a] * self.df[col_b]

    def add_polynomial_features(self, columns, degree=2):
        for col in columns:
            self.df[f"{col}^{degree}"] = self.df[col] ** degree

    def clean_data(
        self,
        return_img_mask_data=False,
        balance_classes=False,
        label="target_label",
        scale=True,
        multiplicative_features=None,
        polynomial_features=None,
    ):
        combined_data = None
        print("Cleaning data...")
        print("drop unnecessary columns")
        self.drop_unnecessary_columns(
            [
                "path",
                "StudyInstanceUID",
                "SOPInstanceUID",
                "BreastImplantPresent",
                "PixelSpacing",
                "HalfValueLayer",
                "original_shape",
                "shape",
            ]
        )
        print("handle missing values")
        self.handle_missing_values()
        print("process ages")
        self.process_ages("PatientAge")
        print("label encode")
        self.label_encode(
            [
                "ImageLaterality",
                "Manufacturer",
                "ManufacturerModelName",
                "ViewPosition",
                "DetectorType",
            ]
        )
        print("map labels")
        self.map_labels(
            "label", {"IndexCancer": 1, "PreIndexCancer": 1, "NonCancer": 0}
        )
        print("handle missing values")
        self.handle_missing_values()  # Additional missing value handling after transformations
        print("add correctness column")
        self.add_correctness_column()

        if balance_classes:
            print("balance classes")
            self.balance_classes(label)

        if return_img_mask_data:
            print("create binary masks")
            preprocessed_images = []
            binary_masks = []
            # Wrap the loop with tqdm for a progress bar
            for idx, row in tqdm(
                self.df.iterrows(), total=len(self.df), desc="Processing Data"
            ):
                image = row["image"]
                bounding_box = row["resized_coords"]
                binary_mask = self.create_binary_mask(image.shape, bounding_box)
                preprocessed_images.append(image)
                binary_masks.append(binary_mask)

            preprocessed_images = np.array(preprocessed_images)
            if len(preprocessed_images.shape) == 3:
                preprocessed_images = np.expand_dims(preprocessed_images, axis=-1)
            binary_masks = np.array(binary_masks)
            binary_masks = np.expand_dims(binary_masks, axis=-1)
            combined_data = np.concatenate((preprocessed_images, binary_masks), axis=-1)

        print("expand and extend columns")
        self.expand_and_extend_columns(
            ["coords", "resized_coords"],
            ["FieldOfViewOrigin", "WindowCenter", "WindowWidth"],
        )
        print("process and flatten columns")
        self.process_and_flatten_columns()
        print("handle missing values")
        self.handle_missing_values()  # Additional missing value handling after transformations

        print("enforce numeric data types")
        self.enforce_numeric_data_types()

        if multiplicative_features is not None:
            print("add multiplicative features")
            self.add_multiplicative_features(
                multiplicative_features[0], multiplicative_features[1]
            )

        if polynomial_features is not None:
            print("add polynomial features")
            self.add_polynomial_features(polynomial_features)

        if scale:
            print("scale data")
            self.scale_data(label=label)

        if return_img_mask_data:
            return combined_data, self.df
        else:
            return self.df
