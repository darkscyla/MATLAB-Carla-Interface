classdef HelperBoundingBoxDetector < matlab.System
    % HelperBoundingBoxDetector A helper class to segment the point cloud
    % into bounding box detections.
    % The step call to the object does the following things:
    %
    % 1. Removes point cloud outside the limits.
    % 2. From the survived point cloud, segments out ground
    % 3. From the obstacle point cloud, forms clusters and puts bounding
    %    box on each cluster.
    
    % Cropping properties
    properties
        XLimits = [-70 70];
        YLimits = [-6 6];
        ZLimits = [-2 10];
    end
   
    % Ground Segmentation Properties
    properties
        GroundMaxDistance = 0.3;
        GroundReferenceVector = [0 0 1];
        GroundMaxAngularDistance = 5;
    end
    
    % Bounding box Segmentation properties
    properties
        SegmentationMinDistance = 1.6;
        MinDetectionsPerCluster = 2;
        MaxZDistanceCluster = 3;
        MinZDistanceCluster = -3;
    end
    
    % Ego vehicle radius to remove ego vehicle point cloud.
    properties
        EgoVehicleRadius = 3;
    end
    
    properties
        MeasurementNoise = blkdiag(eye(3),eye(3));
    end
    
    methods 
        function obj = HelperBoundingBoxDetector(varargin)
            setProperties(obj,nargin,varargin{:})
        end
    end
    
    methods (Access = protected)
        function [bboxDets,obstacleIndices,groundIndices,croppedIndices] = stepImpl(obj,currentPointCloud,time)
            % Crop point cloud
            [pcSurvived,survivedIndices,croppedIndices] = cropPointCloud(currentPointCloud,obj.XLimits,obj.YLimits,obj.ZLimits,obj.EgoVehicleRadius);
            % Remove ground plane
            [pcObstacles,obstacleIndices,groundIndices] = removeGroundPlane(pcSurvived,obj.GroundMaxDistance,obj.GroundReferenceVector,obj.GroundMaxAngularDistance,survivedIndices);
            % Form clusters and get bounding boxes
            detBBoxes = getBoundingBoxes(pcObstacles,obj.SegmentationMinDistance,obj.MinDetectionsPerCluster,obj.MaxZDistanceCluster,obj.MinZDistanceCluster);
            % Assemble detections
            bboxDets = assembleDetections(detBBoxes,obj.MeasurementNoise,time);
        end
    end
end    
    
function detections = assembleDetections(bboxes,measNoise,time)
% This method assembles the detections in objectDetection format.
numBoxes = size(bboxes,2);
detections = cell(numBoxes,1);
for i = 1:numBoxes
    detections{i} = objectDetection(time,cast(bboxes(:,i),'double'),...
        'MeasurementNoise',double(measNoise),'ObjectAttributes',struct);
end
end

function bboxes = getBoundingBoxes(ptCloud,minDistance,minDetsPerCluster,maxZDistance,minZDistance)
    % This method fits bounding boxes on each cluster with some basic
    % rules.
    % Cluster must have atleast minDetsPerCluster points.
    % Its mean z must be between maxZDistance and minZDistance.
    % length, width and height are calculated using min and max from each
    % dimension.
    [labels,numClusters] = pcsegdist(ptCloud,minDistance);
    pointData = ptCloud.Location;
    bboxes = nan(6,numClusters,'like',pointData);
    isValidCluster = false(1,numClusters);
    for i = 1:numClusters
        thisPointData = pointData(labels == i,:);
        meanPoint = mean(thisPointData,1);
        if size(thisPointData,1) > minDetsPerCluster && ...
                meanPoint(3) < maxZDistance && meanPoint(3) > minZDistance
            xMin = min(thisPointData(:,1));
            xMax = max(thisPointData(:,1));
            yMin = min(thisPointData(:,2));
            yMax = max(thisPointData(:,2));
            zMin = min(thisPointData(:,3));
            zMax = max(thisPointData(:,3));
            l = (xMax - xMin);
            w = (yMax - yMin);
            h = (zMax - zMin);
            x = (xMin + xMax)/2;
            y = (yMin + yMax)/2;
            z = (zMin + zMax)/2;
            bboxes(:,i) = [x y z l w h]';
            isValidCluster(i) = l < 20; % max length of 20 meters
        end
    end
    bboxes = bboxes(:,isValidCluster);
end

function [ptCloudOut,obstacleIndices,groundIndices] = removeGroundPlane(ptCloudIn,maxGroundDist,referenceVector,maxAngularDist,currentIndices)
    % This method removes the ground plane from point cloud using
    % pcfitplane.
    [~,groundIndices,outliers] = pcfitplane(ptCloudIn,maxGroundDist,referenceVector,maxAngularDist);
    ptCloudOut = select(ptCloudIn,outliers);
    obstacleIndices = currentIndices(outliers);
    groundIndices = currentIndices(groundIndices);
end

function [ptCloudOut,indices,croppedIndices] = cropPointCloud(ptCloudIn,xLim,yLim,zLim,egoVehicleRadius)
    % This method selects the point cloud within limits and removes the
    % ego vehicle point cloud using findNeighborsInRadius
    locations = ptCloudIn.Location;
    insideX = locations(:,1) < xLim(2) & locations(:,1) > xLim(1);
    insideY = locations(:,2) < yLim(2) & locations(:,2) > yLim(1);
    insideZ = locations(:,3) < zLim(2) & locations(:,3) > zLim(1);
    inside = insideX & insideY & insideZ;
    
    % Remove ego vehicle
    nearIndices = findNeighborsInRadius(ptCloudIn,[0 0 0],egoVehicleRadius);
    nonEgoIndices = true(ptCloudIn.Count,1);
    nonEgoIndices(nearIndices) = false;
    validIndices = inside & nonEgoIndices;
    indices = find(validIndices);
    croppedIndices = find(~validIndices);
    ptCloudOut = select(ptCloudIn,indices);
end


