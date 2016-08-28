require 'nn'
require 'string'
require 'hdf5'
require 'nngraph'
require 'cunn'
require 'cutorch'

require 'Model'
require 'Data'

--LinearNoBias from elements library
local LinearNoBias, Linear = torch.class('nn.LinearNoBias', 'nn.Linear')

function LinearNoBias:__init(inputSize, outputSize)
  nn.Module.__init(self)

  self.weight = torch.Tensor(outputSize, inputSize)
  self.gradWeight = torch.Tensor(outputSize, inputSize)

  self:reset()
end

function LinearNoBias:reset(stdv)
  if stdv then
    stdv = stdv * math.sqrt(3)
  else
    stdv = 1./math.sqrt(self.weight:size(2))
  end
  if nn.oldSeed then
    for i=1,self.weight:size(1) do
      self.weight:select(1, i):apply(function()
          return torch.uniform(-stdv, stdv)
        end)
    end
  else
    self.weight:uniform(-stdv, stdv)
  end

  return self
end

function LinearNoBias:updateOutput(input)
  if input:dim() == 1 then
    self.output:resize(self.weight:size(1))
    self.output:mv(self.weight, input)
  elseif input:dim() == 2 then
    local nframe = input:size(1)
    local nElement = self.output:nElement()
    self.output:resize(nframe, self.weight:size(1))
    if self.output:nElement() ~= nElement then
      self.output:zero()
    end
    if not self.addBuffer or self.addBuffer:nElement() ~= nframe then
      self.addBuffer = input.new(nframe):fill(1)
    end
    self.output:addmm(0, self.output, 1, input, self.weight:t())
  else
    error('input must be vector or matrix')
  end

  return self.output
end

function LinearNoBias:accGradParameters(input, gradOutput, scale)
  scale = scale or 1
  if input:dim() == 1 then
    self.gradWeight:addr(scale, gradOutput, input)
  elseif input:dim() == 2 then
    self.gradWeight:addmm(scale, gradOutput:t(), input)
  end
end

cmd = torch.CmdLine()

-- file location
cmd:option('-gpu_file', 'gpu_model.t7','gpu model file')
cmd:option('-cpu_file', 'cpu_model.t7', 'cpu output file')
cmd:option('-gpuid', 2, 'which gpuid to use')
opt = cmd:parse(arg)

function main()
  print('loading gpu model ' .. opt.gpu_file)
  checkpoint = torch.load(opt.gpu_file)
  model, model_opt = checkpoint[1], checkpoint[2]
  if model_opt.cudnn == 1 then
    require 'cudnn'
  end
  cutorch.setDevice(opt.gpuid)
  for i = 1, #model do
    model[i]:double()
  end
  print('saving cpu model to ' .. opt.cpu_file)
  torch.save(opt.cpu_file, {model, model_opt})
end
--main()

