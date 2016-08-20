--[[ 
-- Copyright (c) 2016, by Author.
-- All rights reserved.

-- Neural language model for normalization, based on long short-term memory.
-- LICENSE file in the root directory of this source folder.
]]--

function g_disable_dropout(node) 
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_disable_dropout)
    end
    return
  end
  if string.match(node.__typename, "Dropout") then
    node.train = false
  end
end

function g_enable_dropout(node)
  if type(node) == "table" and node.__typename == nil then
    for i = 1, #node do
      node[i]:apply(g_enable_dropout)
    end
    return 
  end
  if string.match(node.__typename, "Dropout") then
    node.train = true
  end
end

function g_cloneManyTimes(net, T)
  local clones = {}
  local params, gradParams = net:parameters()
  local mem = torch.MemoryFile("w"):binary() 
   
  mem:writeObject(net)
  
  for t = 1, T do
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:clone()
    local cloneParams, cloneGradParams = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGradParams[i]:set(gradParams[i])
    end
    clones[t] = clone
    collectgarbage()
  end
  
  mem:close()  
  
  return clones
end

function g_init_gpu(args)
  local gpuidx = args
  gpuidx = gpuidex[1] or 1
  cutorch.setDevice(gpuidx)
  g_make_deterministic(1)
end

function g_make_deterministic(seed)
  torch.manualSeed(seed)
  cutorch.manualSeed(seed)
  torch.zeros(1, 1):cuda():uniform()
end

function g_replace_table(to, from)
  assert(#to == #from)
  for i = 1, #to do
    to[i]:copy(from[i])
  end
end

function g_f3(f)
  return string.format("%.3f", f)
end

function g_d(f)
  return string.format("%d", torch.round(f))
end
