require('torch')
require('nn')
require('nngraph')
require('optim')
require('xlua')
require('sys')
require('lfs')

PIT = {}

include('Model.lua')
include('Util.lua')
include('Vocab.lua')

config = {
  dim       = 100,
  learnRate = 0.01,
  batchSize = 5,
  layerSize = 1,
  regular   = 1e-4,
  modelName = 1,
  charCNN   = 1,
  wordCNN   = 1,
  sentLSTM  = 1,
  hidSize   = 150,
  epochSize = 20,
  crit      = nn.DistKLDivCriterion(),
  vDir      = '/home/hjp/Workshop/Model/data/coling/pit/voc/vocabs.txt',
  eVocDir   = '/home/hjp/Workshop/Model/data/coling/pit/vec/glove.vocab',
  eDimDir   = '/home/hjp/Workshop/Model/data/coling/pit/vec/glove.th',
  trainDir  = '/home/hjp/Workshop/Model/data/coling/pit/train/',
  devDir    = '/home/hjp/Workshop/Model/data/coling/pit/dev/',
  testDir   = '/home/hjp/Workshop/Model/data/coling/pit/test/'  
}

local function train()
  local voc = PIT.Vocab(config.vDir)
  local eVoc, eVec = PIT.readEmb(config.eVocDir, config.eDimDir)
  local vec = torch.Tensor(voc.size, eVec:size(2))
  for i = 1, voc.size do
    local t = voc:token(i)
    if eVoc:contains(t) then
      vec[i] = eVec[eVoc:index(t)]
    else
      vec[i]:uniform(-0.05, 0.05)
    end
  end
  eVoc, eVec = nil, nil
  collectgarbage()
  
  local trainSet  = PIT.readData(config.trainDir, voc)
  local devSet    = PIT.readData(config.devDir, voc)
  local testSet   = PIT.readData(config.testDir, voc) 
  local bestDevScore = 0.0
  local bestTestScore= 0.0
  local name = PIT.wordCNN()
  for j = 1, config.epochSize do 
    timer = torch.Timer()
    local devScore = PIT.demoCNN(trainSet, devSet, vec, name)
    print(string.format("Epoch%3d, pearson: %6.8f, and costs %6.8f s.",j, devScore, timer:time().real))
    if devScore >= bestDevScore then
      bestDevScore = devScore
      local testScore = PIT.predTest(name, testSet, vec)
      if testScore >= bestTestScore then
        bestTestScore = testScore
      end
      local msg = "The current best dev score is: " .. bestDevScore .. ", and best Test Score is: " .. bestTestScore .. "."
      PIT.header(msg)
    end    
  end
end

local function main()
  train()
end

main()