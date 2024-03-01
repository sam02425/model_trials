# After evaluating the model
metrics = model.val('')

# Inspect the metrics object
print(dir(metrics))  # List all attributes and methods

# Look for an attribute or method that seems relevant to accessing evaluation metrics
# For example, if you see something like 'precision' or 'get_metrics', it could be what you need

# Example 1: Using a method to get precision
performance_metric = metrics.get_precision()

# Example 2: Accessing an attribute directly
performance_metric = metrics.precision
