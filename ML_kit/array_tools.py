def normalize_array(X):
  return (X - X.min()) / (X.max() - X.min())