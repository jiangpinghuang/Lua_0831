-- Main function for demo in Torch.

require('nn')
require('torch')

-- Functions call functions.
output = nn.Linear(3, 5)(torch.randn(2, 3))
print(output)

