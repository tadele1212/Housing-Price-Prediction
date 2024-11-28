# Housing Price Prediction

This project uses machine learning to predict housing prices based on various features such as area, number of bedrooms, bathrooms, and other amenities. The model is built using **Linear Regression** to predict the price of a house based on the dataset provided.

## Project Description

The goal of this project is to build a machine learning model to predict house prices based on various features:

- **Price**: The target variable (house price).
- **Area**: The area of the house in square feet.
- **Bedrooms**: The number of bedrooms in the house.
- **Bathrooms**: The number of bathrooms in the house.
- **Stories**: The number of stories (floors) in the house.
- **Mainroad**: Whether the house is located on the main road.
- **Guestroom**: Whether the house has a guestroom.
- **Basement**: Whether the house has a basement.
- **Hotwaterheating**: Whether the house has hot water heating.
- **Airconditioning**: Whether the house has air conditioning.
- **Parking**: The number of parking spaces available.
- **Prefarea**: Whether the house is located in a preferred area.
- **Furnishingstatus**: The furnishing status of the house (furnished, semi-furnished, unfurnished).

The model is trained on this dataset using **Linear Regression**, and it predicts the house price based on these features.

## Installation Instructions

To run the project locally, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```

### 2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt`, you can manually install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset

The dataset used in this project is a CSV file containing house-related features. It can be downloaded from Google Drive or any other source, and is placed in the `data/` directory.

The dataset has the following columns:

- **price**: The price of the house (target variable).
- **area**: The area of the house in square feet.
- **bedrooms**: Number of bedrooms in the house.
- **bathrooms**: Number of bathrooms.
- **stories**: Number of stories (floors).
- **mainroad**: Whether the house is on the main road (yes/no).
- **guestroom**: Whether the house has a guestroom (yes/no).
- **basement**: Whether the house has a basement (yes/no).
- **hotwaterheating**: Whether the house has hot water heating (yes/no).
- **airconditioning**: Whether the house has air conditioning (yes/no).
- **parking**: Number of parking spaces.
- **prefarea**: Whether the house is in a preferred area (yes/no).
- **furnishingstatus**: Furnishing status of the house (furnished/semi-furnished/unfurnished).

## Technologies Used

- **Python**: Programming language used to build the model.
- **pandas**: Data manipulation and analysis library.
- **numpy**: Library for numerical operations.
- **matplotlib** and **seaborn**: Data visualization libraries.
- **scikit-learn**: Machine learning library used to build the model.
- **gdown**: Used for downloading datasets from Google Drive.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the model:

```bash
python housing_price_prediction.py
```

This will preprocess the dataset, train the model, and output the evaluation metrics such as Mean Squared Error and R² score.

## Code Explanation

### Data Preprocessing

- **Label Encoding**: Categorical columns (like `yes` and `no`) are converted to numerical values using **LabelEncoder** from scikit-learn.
- **Feature Selection**: The target variable (`price`) is separated from the feature variables.

### Model Training

The model uses **Linear Regression** from `sklearn.linear_model` to train the data and predict the housing prices.

### Model Evaluation

The model is evaluated using:

- **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual values.
- **R² (R-squared)**: Indicates how well the independent variables explain the variance of the dependent variable (house price).
```

### Key Notes:

- Make sure that when you paste the content into GitHub, it should automatically render properly if the Markdown syntax is correct.
- Ensure that the file has a `.md` extension (e.g., `README.md`).
