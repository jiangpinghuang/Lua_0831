--[[ 
-- Copyright (c) 2016, by Author.
-- All rights reserved.

-- Neural language model for normalization, based on long short-term memory.
-- LICENSE file in the root directory of this source folder.
]]--

local stringx   = require('pl.stringx')
local file      = require('pl.file')

local ptb_path  = "./data/"

local vocab_idx = 0
local vocab_map = {}

local function replicate(x_inp, batch_size)
  local s = x_inp:size(1)
  local x = torch.zeros(torch.floor(s / batch_size), batch_size)
  
  for i = 1, batch_size do
    local start   = torch.round((i - 1) * s / batch_size) + 1
    local finish  = start + x:size(1) - 1
    x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
  end
  
  return x
end

local function load_data(fname)
  local data = file.read(fname)
  data = stringx.replace(data, '\n', '<eos>')
  data = stringx.split(data)
  
  print(string.format("Load %s, size of data = %d", fname, #data))
  
  local x = torch.zeros(#data)
  
  for i = 1, #data do
    if vocab_map[data[i]] == nil then
      vocab_idx = vocab_idx + 1
      vocab_map[data[i]] = vocab_idx
    end
    x[i] = vocab_map[data[i]]
  end
  
  return x       
end

local function train_data_set(batch_size)
  local x = load_data(ptb_path .. "ptb.train.txt")
  x = replicate(x, batch_size)
  return x
end

local function test_data_set(batch_size)
  local x = load_data(ptb_path .. "ptb.test.txt")
  x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
  return x
end

local function valid_data_set(batch_size)
  local x = load_data(ptb_path .. "ptb.valid.txt")
  x = replicate(x, batch_size)
  return x
end

return {train_data_set = train_data_set,
        valid_data_set = valid_data_set,
        test_data_set  = test_data_set}
