import pickle
import json
import numpy as np


class RealEstatePredictor:
  def __init__(self, city):
    self.city = city
    self.data_columns_bengaluru = None
    self.data_columns_mumbai=None
    self.locations = None
    self.model = None
    self.load_data()

  def load_data(self):
    if self.city == "Bengaluru":
      filename = "./artifacts/Bengaluru_hp.json"
      model_filename = "./artifacts/bangaluru_hp_prediction.pickle"
    elif self.city == "Mumbai":
      filename = "./artifacts/Mumbai_hp.json"
      model_filename = "./artifacts/Mumbai_hp_prediction.pickle"
    else:
      raise ValueError(f"Unsupported city: {self.city}")

    print(f"Loading {self.city} artifacts...start")
    with open(filename, "r") as f:
      data = json.load(f)
      
      if self.city=="Mumbai":
        self.locations = self.data_columns[12:]
        self.data_columns_mumbai = data['data_columns_mumbai']
      else:
        self.locations = self.data_columns[3:]
        self.data_columns_bengaluru = data['data_columnsbengaluru']

    with open(model_filename, 'rb') as f:
      self.model = pickle.load(f)
    print(f"Loading {self.city} artifacts...done")

def get_estimated_price(self, location, sqft, bhk, bath):
    try:
      loc_index = self.data_columns.index(location.lower())
    except ValueError:
      return -1  # Indicate unsupported location

    x = np.zeros(len(self.data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
      x[loc_index] = 1

    return round(self.model.predict([x])[0], 2)

def get_estimated_price_mumbai(self, location, area, no_of_bedrooms, new_or_resale, gym, lift, clubhouse, swimming_pool):
  try:
    loc_index_mumbai = self.data_columns_mumbai.index(location.lower())
  except ValueError:
    return -1  # Indicate unsupported location

  x = np.zeros(len(self.data_columns_mumbai))

  # Assuming data_columns_mumbai is in the format:
  # ["area", "no. of bedrooms", "new/resale", "gymnasium", "lift available", "clubhouse", "swimming pool", ...]

  x[0] = area
  x[1] = no_of_bedrooms

  # Handle categorical features (assuming one-hot encoding for "new/resale")
  if new_or_resale.lower() == "new":
    x[2] = 1  # One-hot encoding for "New"
  else:
    x[3] = 1  # One-hot encoding for "Resale" (assuming only two categories)

  x[4] = gym  # Boolean value for gym availability
  x[5] = lift  # Boolean value for lift availability
  x[6] = clubhouse  # Boolean value for clubhouse availability
  x[7] = swimming_pool  # Boolean value for swimming pool availability

  # Replace with your trained model for Mumbai
  estimated_price = round(self.model_mumbai.predict([x])[0], 2)
  return estimated_price

def get_location_names(self):
    return self.locations

def get_data_columns(self):
    return self.data_columns


# Usage example
bengaluru_predictor = RealEstatePredictor("Bengaluru")
mumbai_predictor = RealEstatePredictor("Mumbai")

print(bengaluru_predictor.get_location_names())
print(bengaluru_predictor.get_estimated_price('1st Phase JP Nagar', 1000, 3, 3))
# ... (similar calls for other locations)

print(mumbai_predictor.get_location_names())
print(mumbai_predictor.get_estimated_price('Borivali', 1200, 2, 2))  # Example for Mumbai
