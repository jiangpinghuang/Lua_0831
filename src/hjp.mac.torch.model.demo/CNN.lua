local CNN = {}

function CNN.cnn(inputSize, outputSize, kW, dw)
  lcnn = nn.Sequential()
  lcnn:add(nn.TemporalConvolution(inputSize, outputSize, kW, dW))
  lcnn:add(nn.ReLU())
  lcnn:add(nn.Max(1))

  rcnn = nn.Sequential()
  rcnn:add(nn.TemporalConvolution(inputSize, outputSize, kW, dW))
  rcnn:add(nn.ReLU())
  rcnn:add(nn.Max(1))

  tcnn = nn.ParallelTable()
  tcnn:add(lcnn)
  tcnn:add(rcnn)

  cnn = nn.Sequential()
  cnn:add(tcnn)
  cnn:add(nn.CosineDistance())
  
  return cnn
end