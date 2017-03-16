require 'image'
require 'torch'
require 'nn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
  batchSize = 1,         -- number of samples to produce
  net = '',              -- path to the discriminator network
  name = 'sun',          -- name of the experiment and prefix of file saved
  gpu = 1,               -- gpu mode. 0 = CPU, 1 = 1st GPU etc.
  nc = 3,                -- # of channels in input
  display  = 0,          -- Display image: 0 = false, 1 = true
  loadSize = 0,          -- resize the loaded image to loadsize maintaining aspect ratio. 0 means don't resize. -1 means scale randomly between [0.5,2] -- see donkey_folder.lua
  fineSize = 128,        -- size of random crops
  nThreads = 1,          -- # of data loading threads to use
  manualSeed = 394,      -- 0 means random seed
  overlapPred = 4,       -- overlapping edges of center with context
  dir = 'val_inputs/',
  output_dir = 'val_features/',
  ext = 'jpg',

  -- Extra Options:
  noiseGen = 0,          -- 0 means false else true; only works if network was trained with noise too.
  noisetype = 'normal',  -- type of noise distribution (uniform / normal)
  nz = 100,              -- length of noise vector if used
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end
if opt.noiseGen == 0 then opt.noiseGen = false end

-- set seed
if opt.manualSeed == 0 then
    opt.manualSeed = torch.random(1, 10000)
end
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- load Context-Encoder
assert(opt.net ~= '', 'provide a discriminator model')
net = util.load(opt.net, opt.gpu)
net:evaluate()

-- initialize variables
input_image_ctx = torch.Tensor(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
local noise
if opt.noiseGen then
    noise = torch.Tensor(opt.batchSize, opt.nz, 1, 1)
    if opt.noisetype == 'uniform' then
        noise:uniform(-1, 1)
    elseif opt.noisetype == 'normal' then
        noise:normal(0, 1)
    end
end

-- port to GPU
if opt.gpu > 0 then
    require 'cunn'
    if pcall(require, 'cudnn') then
        print('Using CUDNN !')
        require 'cudnn'
        net = util.cudnn(net)
    end
    net:cuda()
    input_image_ctx = input_image_ctx:cuda()
    if opt.noiseGen then
        noise = noise:cuda()
    end
else
   net:float()
end
print(net)

-- Create empty table to store file names:
files = {}

-- Go over all files in directory. We use an iterator, paths.files().
for file in paths.files(opt.dir) do
-- for file in paths.files('sun_inputs/') do
   -- We only load files that match the extension
   if file:find(opt.ext .. '$') then
      -- and insert the ones we care about in our table
      table.insert(files, paths.concat(opt.dir,file))
   end
end

-- Check files
if #files == 0 then
   error('given directory doesnt contain any files of type: ' .. opt.ext)
end

-- print('Found files:')
-- print(files)

for i,file in ipairs(files) do
   -- load each image
   image_ctx = image.load(file, nc, 'float')
   image_ctx:resize(opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
   image_ctx:mul(2.0):csub(1.0)

   print('Loaded Image: ', image_ctx:size(1)..' x '..image_ctx:size(2) ..' x '..image_ctx:size(3) .. ' x '..image_ctx:size(4) .. ' ' .. torch.min(image_ctx) .. ' ' .. torch.max(image_ctx))
   real_center = image_ctx[{{},{},{1 + opt.fineSize/4, opt.fineSize/2 + opt.fineSize/4},{1 + opt.fineSize/4, opt.fineSize/2 + opt.fineSize/4}}]:clone() -- copy by value
   -- fill center region with mean value
   image_ctx[{{},{1},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*117.0/255.0 - 1.0
   image_ctx[{{},{2},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*104.0/255.0 - 1.0
   image_ctx[{{},{3},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*123.0/255.0 - 1.0
   input_image_ctx:copy(image_ctx)

   -- run Context-Encoder to inpaint center
   local pred_center
   if opt.noiseGen then
       pred_center = net.modules[1]:forward({input_image_ctx,noise})
   else
       pred_center = net.modules[1]:forward(input_image_ctx)
   end
   pred_center:squeeze()
   print('Prediction: size: ', pred_center:size())
   print('Prediction: Min, Max, Mean, Stdv: ', pred_center:min(), pred_center:max(), pred_center:mean(), pred_center:std())

  --  -- paste predicted center in the context
  --  image_ctx[{{},{},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}]:copy(pred_center[{{},{},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred},{1 + opt.overlapPred, opt.fineSize/2 - opt.overlapPred}}])
  --  -- re-transform scale back to normal
  --  input_image_ctx:add(1):mul(0.5)
  --  image_ctx:add(1):mul(0.5)
  --  pred_center:add(1):mul(0.5)
  --  real_center:add(1):mul(0.5)
  --
   file_name = paths.basename(file)
   output_path = paths.concat(opt.output_dir,file_name)
   torch.save(output_path .. '.7z', pred_center)
   print('Saved predictions to: ', output_path.. '.7z')
end



--
-- -- save outputs in a pretty manner
-- real_center=nil; pred_center=nil;
-- pretty_output = torch.Tensor(2*opt.batchSize, opt.nc, opt.fineSize, opt.fineSize)
-- input_image_ctx[{{},{1},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 1
-- input_image_ctx[{{},{2},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 1
-- input_image_ctx[{{},{3},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 1
-- for i=1,opt.batchSize do
--     -- pretty_output[2*i-1]:copy(input_image_ctx[i])
--     -- pretty_output[2*i]:copy(image_ctx[i])
--     image.save(string.format('sun_outputs/%04d', i-1) .. '.png', image.toDisplayTensor(image_ctx[i]))
--     print('Saved predictions to: ./', string.format('sun_outputs/%04d', i-1) .. '.png')
-- end
-- image.save(opt.name .. '.png', image.toDisplayTensor(pretty_output))
-- print('Saved predictions to: ./', opt.name .. '.png')
