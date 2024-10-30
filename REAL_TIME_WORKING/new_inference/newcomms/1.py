import pickle
import torch

# Create a sample tensor
tensor = torch.randn(1, 32, 150, 150)  # Example tensor

# Serialization Verification
test_serialized = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
test_tensor = pickle.loads(test_serialized)

# Check if the original tensor and deserialized tensor are equal
assert torch.equal(tensor, test_tensor), "Tensor serialization/deserialization failed."