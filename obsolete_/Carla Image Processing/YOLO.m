clearvars -except yolonet yoloml yolojb
close all

% Download yolo network if it does not exist in the current folder (which it won't the 
% first time you run the code). This takes a while. Code snippet courtesy of mathworks.

if exist('yolonet.mat','file') == 0
    url = 'https://www.mathworks.com/supportfiles/gpucoder/cnn_models/Yolo/yolonet.mat';
    websave('yolonet.mat',url);
end

%load yolo network from current folder (this can take a while) if it does not exist in
%workspace (which it won't the first time you run code)
if exist('yolonet') ~= 1
    load yolonet.mat
end

% the first time we run the script, we need to modify yolonet and save it
% with a new name (yoloml)
if exist('yoloml.mat','file') == 0
    display('modifying network')
    
    % extract a layer graph from the network. We need to modify this graph.
    lgraph = layerGraph(yolonet.Layers);
    
    % the yolo network from MATLAB is built like a classifier.
    % We need to convert it to a regression network. This means modifying the
    % last two layers
    lgraph = removeLayers(lgraph,'ClassificationLayer');
    lgraph = removeLayers(lgraph,'softmax');
    
    % According to the original YOLO paper, the last transfer function
    % is not a leaky, but a normal ReLu (I think).
    % In MATLAB, this is equivalent to a leaky ReLu with Scale = 0.
    alayer = leakyReluLayer('Name','linear_25','Scale',0);
    rlayer = regressionLayer('Name','routput');
    lgraph = addLayers(lgraph,rlayer);
    lgraph = replaceLayer(lgraph,'leakyrelu_25',alayer);
    lgraph = connectLayers(lgraph,'FullyConnectedLayer1','routput');
    yoloml = assembleNetwork(lgraph);
    
    %save the network with a new name
    display('saving modified network')
    save yoloml yoloml    
 
% if we have created and saved yoloml but not loaded it to workspace, load
% it now.
elseif exist('yoloml') ~= 1
    display('loading modified network')
    load('yoloml.mat')
end