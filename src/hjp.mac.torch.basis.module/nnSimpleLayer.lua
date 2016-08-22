require('nn')

-- Linear module.
module = nn.Linear(10, 5)
mlp = nn.Sequential()
mlp:add(module)

print(module.weight)
print(module.bias)
print(module.gradWeight)
print(module.gradBias)

input = torch.Tensor(10)
output = mlp:forward(input)

print(input)
print(output)

-- Bilinear.
mlp = nn.Sequential()
mlp:add(nn.Bilinear(10, 5, 3))

input = {torch.randn(128, 10), torch.randn(128, 5)}
output = mlp:forward(input)

print(input)
print(output)
