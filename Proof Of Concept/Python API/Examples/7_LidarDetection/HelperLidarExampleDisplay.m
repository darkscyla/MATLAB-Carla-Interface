classdef HelperLidarExampleDisplay < matlab.System
    % This is a helper class for visualization of tracking using lidar data
    % example. It may be removed or modified in a future release.
    % 
    
    % Copyright 2019 The MathWorks, Inc.
    
    % Public properties for modifying any required property/handle.
    properties
        PointCloudProcessingDisplay
        TrackingDisplay
        CloseUpDisplay
    end
    
    % Figure
    properties (SetAccess = private)
        Figure
    end
    
    % Index of position, velocity, yaw and dimension for track plotting.
    properties
        PositionIndex = [1 3 6];
        VelocityIndex = [2 4 7];
        DimensionIndex = [9 11 13];
        YawIndex = 8;
    end
    

    properties
        RecordGIF = false;
    end
    
    % Changing the color of objects. Each index refers to the index
    % specified in the ColorMap.
    properties (Hidden)    
        ColorMap = [1.0000    1.0000    0.0667
                    0.0745    0.6235    1.0000
                    1.0000    0.4118    0.1608
                    0.3922    0.8314    0.0745
                    0.7176    0.2745    1.0000
                    0.0588    1.0000    1.0000
                    1.0000    0.0745    0.6510];        
        TrackColorIndex = 4
        DetectionColorIndex = 3
        EgoColorIndex = 3
        GroundColorIndex = 5
        RawCloudColorIndex = 2
        ObstacleColorIndex = 3
    end
    
    % Name of the movie for recording the display. If empty, no movie will
    % be recorded.
    properties
        MovieName
    end
    
    % Plot handles for each plot.
    properties (Access = private)
        pPCAxes
        pGroundPlotter
        pRawPCPlotter
        pRawPCPlotter3
        pObstaclePlotter
        pObstaclePlotter2
        pObstaclePlotter3
        pDetectionPlotter
        pDetectionPlotter2
        pDetectionPlotter3
        pTrackPlotter
        pTrackPlotter2
        pTrackPlotter3
        pTrackingPanel
        pCoveragePlotter
        pCoveragePlotter2
        pEgoPlotter
        pTrackingAnalysisTheater
        pImagePlotter
        pFigure;
        pVideoWriter;
        pVideoFrames = {};
        pPreProcessing
        pEgo
        pDetail
        pIsPublishing
    end
    
    methods
        % Constructor
        function obj = HelperLidarExampleDisplay(image,varargin)
            % Create each display
            createPreProcessingDisplay(obj);
            createTrackingAnalysisDisplay(obj,image);
            createCloseUpDisplay(obj);
            setProperties(obj,nargin - 1, varargin{:});
            
            % Setup movie writer.
            if ~isempty(obj.MovieName)
                obj.pVideoWriter = VideoWriter(obj.MovieName);
                obj.pVideoWriter.FrameRate = 10;
            end
        end
        
        % Write movie from recorded frames.
        function writeMovie(obj)
            open(obj.pVideoWriter);
            for i = 1:numel(obj.pVideoFrames)
                thisFrame = obj.pVideoFrames{i};
                writeVideo(obj.pVideoWriter,thisFrame);
            end
            close(obj.pVideoWriter);
        end
        
        function writeAnimatedGIF(obj,frame1,frame2,fName,type)
            switch lower(type(1))
                case 'p' % Pre-processing
                    allFrames = obj.pPreProcessing(frame1:frame2);
                case 'e' % Ego
                    allFrames = obj.pEgo(frame1:frame2);
                case 'd' % Details
                    allFrames = obj.pDetail(frame1:frame2);
            end
            imSize = size(allFrames{1}.cdata);
            im = zeros(imSize(1),imSize(2),1,numel(allFrames),'uint8');
            map = [];
            for i = 1:numel(allFrames)
                if isempty(map)
                    [im(:,:,1,i),map] = rgb2ind(allFrames{i}.cdata,256,'nodither');
                else
                    im(:,:,1,i) = rgb2ind(allFrames{i}.cdata,map,'nodither');
                end
            end
            imwrite(im,map,[fName,'.gif'],'DelayTime',0,'LoopCount',inf);
        end
        
        function set.TrackColorIndex(obj,val)
            setTrackColorIndex(obj,val);
            obj.TrackColorIndex = val;
        end
        
        function setTrackColorIndex(obj,val)
            colorOrder = obj.ColorMap;
            obj.pTrackPlotter.MarkerFaceColor = colorOrder(val,:);
            obj.pTrackPlotter.MarkerEdgeColor = colorOrder(val,:);
            
            obj.pTrackPlotter2.MarkerFaceColor = colorOrder(val,:);
            obj.pTrackPlotter2.MarkerEdgeColor = colorOrder(val,:);
            
            obj.pTrackPlotter3.MarkerFaceColor = colorOrder(val,:);
            obj.pTrackPlotter3.MarkerEdgeColor = colorOrder(val,:);  
        end
        
        function set.DetectionColorIndex(obj,val)
            setDetectionColorIndex(obj,val);
            obj.DetectionColorIndex = val;
        end
        
        function setDetectionColorIndex(obj,val)
            colorOrder = obj.ColorMap;
            obj.pDetectionPlotter.MarkerFaceColor = colorOrder(val,:);
            obj.pDetectionPlotter.MarkerEdgeColor = colorOrder(val,:);
            
            obj.pDetectionPlotter2.MarkerFaceColor = colorOrder(val,:);
            obj.pDetectionPlotter2.MarkerEdgeColor = colorOrder(val,:);
            
            obj.pDetectionPlotter3.MarkerFaceColor = colorOrder(val,:);
            obj.pDetectionPlotter3.MarkerEdgeColor = colorOrder(val,:);
        end

        function set.EgoColorIndex(obj,val)
            setEgoColorIndex(obj,val);
            obj.EgoColorIndex = val;
        end
        function setEgoColorIndex(obj,val)
            colorOrder = obj.ColorMap;
            obj.pEgoPlotter.MarkerFaceColor = colorOrder(val,:);
            obj.pEgoPlotter.MarkerEdgeColor = colorOrder(val,:);
        end
        
        function set.ObstacleColorIndex(obj,val)
            setObstacleColorIndex(obj,val);
            obj.ObstacleColorIndex = val;
        end
        function setObstacleColorIndex(obj,val)
            colorOrder = obj.ColorMap;
            obj.pObstaclePlotter.Color = colorOrder(val,:);
            obj.pObstaclePlotter2.Color = colorOrder(val,:);
            obj.pObstaclePlotter3.Color = colorOrder(val,:);
        end
        
        function set.RawCloudColorIndex(obj,val)
            setRawCloudColorIndex(obj,val);
            obj.RawCloudColorIndex = val;
        end
        function setRawCloudColorIndex(obj,val)
            colorOrder = obj.ColorMap;
            obj.pRawPCPlotter.Color = colorOrder(val,:);
            obj.pRawPCPlotter3.Color = colorOrder(val,:);
        end
        
        function set.GroundColorIndex(obj,val)
            setGroundColorIndex(obj,val);
            obj.GroundColorIndex = val;
        end
        function setGroundColorIndfex(obj,val)
            colorOrder = obj.ColorMap;
            obj.pGroundPlotter.Color = colorOrder(val,:);
        end
        
        % Snapnow 
        function snapnow(obj)
           if obj.pIsPublishing
               g = findall(0,'Tag','PublishFig');
               close(g);
               f = figure('InvertHardCopy','off','Units','normalized','Position',[0.1 0.1 0.8 0.8],'Tag','PublishFig');
               copyobj(copy(obj.Figure.Children),f);
           end
        end
    end
    
    methods (Access = protected)
        function stepImpl(obj,detections,confTracks,lidarScan,obstacleIndices,...
                groundIndices,croppedIndices,currentImage,varargin)
            % Plot tracks.
            trkPlotter = obj.pTrackPlotter;
            trkPlotter2 = obj.pTrackPlotter2;
            trkPlotter3 = obj.pTrackPlotter3;
            [pos,~,posCov,dims,orients,labels,labelsMotion] = ...
                parseTracks(confTracks,obj.PositionIndex,obj.VelocityIndex,obj.DimensionIndex,obj.YawIndex,varargin{:});
            trkPlotter.plotTrack(pos,dims,orients,labels);
            trkPlotter2.plotTrack(pos,dims,orients,labels);
            trkPlotter3.plotTrack(pos,posCov,dims,orients,labelsMotion);
            locations = lidarScan.Location;
            
            % Update the point cloud displays.
            lidarLocations = locations(croppedIndices,:);
            groundLocations = locations(groundIndices,:);
            obstacleLocations = locations(obstacleIndices,:);
            
            % Sample it down
            lidarLocations = lidarLocations(1:5:end,:);
            groundLocations = groundLocations(1:5:end,:);
            obstacleLocations = obstacleLocations(1:5:end,:);
            
            if isvalid(obj.pObstaclePlotter)
                set(obj.pObstaclePlotter,'XData',obstacleLocations(:,1),'YData',obstacleLocations(:,2),'ZData',obstacleLocations(:,3));
                set(obj.pObstaclePlotter2,'XData',obstacleLocations(:,1),'YData',obstacleLocations(:,2),'ZData',obstacleLocations(:,3));
                set(obj.pObstaclePlotter3,'XData',obstacleLocations(:,1),'YData',obstacleLocations(:,2),'ZData',obstacleLocations(:,3));
                set(obj.pGroundPlotter,'XData',groundLocations(:,1),'YData',groundLocations(:,2),'ZData',groundLocations(:,3));
                set(obj.pRawPCPlotter,'XData',lidarLocations(:,1),'YData',lidarLocations(:,2),'ZData',lidarLocations(:,3));
                set(obj.pRawPCPlotter3,'XData',lidarLocations(:,1),'YData',lidarLocations(:,2),'ZData',lidarLocations(:,3));
            end
            
            % Plot detections
            detPlotter = obj.pDetectionPlotter;
            detPlotter2 = obj.pDetectionPlotter2;
            detPlotter3 = obj.pDetectionPlotter3;
            [pos,dims,orients] = parseDetections(detections);
            detPlotter.plotTrack(pos,dims,orients);
            detPlotter2.plotTrack(pos,dims,orients);
            detPlotter3.plotTrack(pos,dims,orients);
            
            % Update the image display
            obj.pImagePlotter.CData = currentImage;
            
            % Store the frame
            if ~isempty(obj.pVideoWriter)
                obj.pVideoFrames{end+1} = getframe(obj.Figure);
            end
            if obj.RecordGIF
                obj.pPreProcessing{end+1} = getframe(obj.pObstaclePlotter.Parent);
                obj.pEgo{end+1} = getframe(obj.pObstaclePlotter2.Parent);
                obj.pDetail{end+1} = getframe(obj.pObstaclePlotter3.Parent);
            end
        end
        
        function createPreProcessingDisplay(obj)
            f = figure('Units','normalized','OuterPosition',[0.1 0.1 0.8 0.8],'InvertHardcopy','off','Visible','off');
            % Display figure if not publishing.
            obj.pIsPublishing = ~isempty(snapnow('get'));
            if ~obj.pIsPublishing
                f.Visible = 'on';
            end
            obj.Figure = f;
            p1 = uipanel('Parent',f,'Position',[0 0 0.6 1],'Title','Lidar Preprocessing and Tracking');
            p1.BackgroundColor = [0.1570 0.1570 0.1570];
            p1.ForegroundColor = [1 1 1];
            ax = axes('Parent',p1);
            obj.pPCAxes = ax;
            set(ax,'XLim',[-50 50],'YLim',[-20 20],'ZLim',[-2 5]);
            ax.NextPlot = 'add';
            ax.Color = [0 0 0];
            grid(ax,'on');
            ax.GridColor = 0.68*[1 1 1];
            ax.GridAlpha = 0.4;
            axis(ax,'equal');
            ax.XLimMode = 'manual';
            ax.YLimMode = 'manual';
            ax.ZLimMode = 'manual';
            ax.XColor = [1 1 1]*0.68;
            ax.YColor = [1 1 1]*0.68;
            ax.ZColor = [1 1 1]*0.68;
            l = legend(ax);
            l.Color = [0 0 0];
            l.TextColor = [1 1 1];
            view(ax,3);
            colorOrder = obj.ColorMap;
            obj.pRawPCPlotter = plot3(ax,nan,nan,nan,'.','MarkerSize',1,'Color',colorOrder(obj.RawCloudColorIndex,:));
            obj.pGroundPlotter = plot3(ax,nan,nan,nan,'.','MarkerSize',1,'Color',colorOrder(obj.GroundColorIndex,:));
            obj.pObstaclePlotter = plot3(ax,nan,nan,nan,'.','MarkerSize',1,'Color',colorOrder(obj.ObstacleColorIndex,:));
            l = legend('Raw point cloud','Segmented ground','Obstacles');
            % Field of view plotter
            hPatch = patch(ax,0,0,[0 0 1],'EdgeColor',[0 0 1],...
                'FaceAlpha',0.4,'DisplayName','Vision field of view');
            obj.pCoveragePlotter = hPatch;
            l.Orientation = 'horizontal';
            l.NumColumns = 2;
            l.FontSize = 10;
            tp1 = theaterPlot('Parent',ax);
            detPlotter = trackPlotter(tp1,'MarkerFaceColor',colorOrder(obj.DetectionColorIndex,:),'MarkerEdgeColor',colorOrder(obj.DetectionColorIndex,:),'MarkerSize',3,'Marker','o');
            trkPlotter = trackPlotter(tp1,'MarkerFaceColor',colorOrder(obj.TrackColorIndex,:),'MarkerEdgeColor',colorOrder(obj.TrackColorIndex,:),'LabelOffset',[0 0 2],'FontSize',12,'Marker','s','MarkerSize',3);
            trkPlotter.plotTrack([nan nan nan],struct('Length',1,'Width',1,'Height',1,'OriginOffset',[0 0 0]),eye(3));
            detPlotter.plotTrack([nan nan nan],struct('Length',1,'Width',1,'Height',1,'OriginOffset',[0 0 0]),eye(3));
            
            % Create two patches to show cuboids in the legend instead of
            % markers.
            l.AutoUpdate = 'on';
            patch(ax,nan,nan,colorOrder(obj.DetectionColorIndex,:),'EdgeColor',colorOrder(obj.DetectionColorIndex,:),'FaceAlpha',0,'DisplayName','Bounding box detections');
            patch(ax,nan,nan,colorOrder(obj.TrackColorIndex,:),'EdgeColor',colorOrder(obj.TrackColorIndex,:),'FaceAlpha',0,'DisplayName','Bounding box tracks');
            l.AutoUpdate = 'off';

            obj.pDetectionPlotter = detPlotter;
            obj.pTrackPlotter = trkPlotter;
            pcProcessingDisplay.ObstaclePlotter = obj.pObstaclePlotter;
            pcProcessingDisplay.RawPointCloudPlotter = obj.pRawPCPlotter;
            pcProcessingDisplay.GroundPlotter = obj.pGroundPlotter;
            pcProcessingDisplay.TheaterDisplay = tp1;
            obj.PointCloudProcessingDisplay = pcProcessingDisplay;
            view(ax,-115,25);
        end
        
        function createTrackingAnalysisDisplay(obj,image)
            p3 = uipanel('Parent',obj.Figure,'Position',[0.6 0.3 0.4 0.7],'Title','Ego Vehicle Display');
            ax2 = axes('Parent',p3);
            ax2.Position = [0.65 0.6 0.3 0.3];
            imdisplay = imshow(image,'Parent',ax2);
            ax2.Title.String = 'Reference Image';
            ax2.Title.Color = [1 1 1];
            obj.pImagePlotter = imdisplay;
            p3.BackgroundColor = [0.1570 0.1570 0.1570];
            p3.ForegroundColor = [1 1 1];
            obj.pTrackingPanel = p3;
            ax = axes('Parent',p3);
            ax.Color = [0 0 0];
            grid(ax,'on');
            ax.GridColor = 0.68*[1 1 1];
            ax.GridAlpha = 0.4;
            axis(ax,'equal');
            set(ax,'XLim',[-10 60],'YLim',[-20 20],'ZLim',[-3 5]);
            hold(ax,'on');
            ax.XLimMode = 'manual';
            ax.YLimMode = 'manual';
            ax.ZLimMode = 'manual';
            ax.XColor = [1 1 1]*0.68;
            ax.YColor = [1 1 1]*0.68;
            ax.ZColor = [1 1 1]*0.68;
            
            view(ax,-90,90)
            % Field of view plotter
            hPatch = patch(ax,0,0,[0 0 1],'EdgeColor',[0 0 1],...
                'FaceAlpha',0.4,'DisplayName','Vision field of view');
            driving.birdsEyePlot.internal.plotCoverageArea(hPatch, ...
                [1 0 1.1], 150, 0, 38);
            obj.pCoveragePlotter2 = hPatch;
            colorOrder = obj.ColorMap;
            
            obj.pObstaclePlotter2 = plot3(ax,nan,nan,nan,'.','MarkerSize',1,'Color',colorOrder(obj.ObstacleColorIndex,:));
            tp2 = theaterPlot('Parent',ax);
            egoPlotter = platformPlotter(tp2,'DisplayName','Ego Vehicle','MarkerFaceColor',colorOrder(obj.EgoColorIndex,:),'MarkerEdgeColor',colorOrder(obj.EgoColorIndex,:));
            obj.pEgoPlotter = egoPlotter;
            egoPlotter.plotPlatform([0 0 0],struct('Length',4.7,'Width',1.8,'Height',1.4,'OriginOffset',[0 0 0]),eye(3));
            trkPlotter = trackPlotter(tp2,'MarkerFaceColor',colorOrder(obj.TrackColorIndex,:),'MarkerEdgeColor',colorOrder(obj.TrackColorIndex,:),'LabelOffset',[0 -1 0],'MarkerSize',3,'FontSize',10,'Marker','s');
            detPlotter = trackPlotter(tp2,'MarkerSize',3,'MarkerFaceColor',colorOrder(obj.DetectionColorIndex,:),'MarkerEdgeColor',colorOrder(obj.DetectionColorIndex,:),'Marker','o');
            trkPlotter.plotTrack([nan nan nan],struct('Length',1,'Width',1,'Height',1,'OriginOffset',[0 0 0]),eye(3));
            detPlotter.plotTrack([nan nan nan],struct('Length',1,'Width',1,'Height',1,'OriginOffset',[0 0 0]),eye(3));
            
            obj.pTrackPlotter2 = trkPlotter;
            obj.pDetectionPlotter2 = detPlotter;
            l = legend(ax);
            l.Visible = 'off';
            trackingDisplay.ObstaclePlotter = obj.pObstaclePlotter2;
            trackingDisplay.TheaterDisplay = tp2;
            trackingDisplay.ImageDisplay = imdisplay;
            obj.TrackingDisplay = trackingDisplay; 
        end
        
        function createCloseUpDisplay(obj)
            p2 = uipanel('Parent',obj.Figure,'Position',[0.6 0 0.4 0.3],'Title','Tracking Details');
            p2.BackgroundColor = [0.1570 0.1570 0.1570];
            p2.ForegroundColor = [1 1 1];
            ax  = axes('Parent',p2);
            set(ax,'XLim',[-20 20],'YLim',[-5 5],'ZLim',[-0.5 3]);
            ax.NextPlot = 'add';
            ax.Color = [0 0 0];
            grid(ax,'on');
            ax.GridColor = 0.68*[1 1 1];
            ax.GridAlpha = 0.4;
            axis(ax,'equal');
            ax.XLimMode = 'manual';
            ax.YLimMode = 'manual';
            ax.ZLimMode = 'manual';
            ax.XColor = [1 1 1]*0.68;
            ax.YColor = [1 1 1]*0.68;
            ax.ZColor = [1 1 1]*0.68;
            colorOrder = obj.ColorMap;
            obj.pObstaclePlotter3 = plot3(ax,nan,nan,nan,'.','MarkerSize',1,'Color',colorOrder(obj.ObstacleColorIndex,:));
            obj.pRawPCPlotter3 = plot3(ax,nan,nan,nan,'.','MarkerSize',1,'Color',colorOrder(obj.RawCloudColorIndex,:));
            view(ax,3);
            ax.Position = [0.11 0.11 0.8 0.8];
            tp1 = theaterPlot('Parent',ax);
            obj.pTrackingAnalysisTheater = tp1;
            detPlotter = trackPlotter(tp1,'DisplayName','Bounding box detections','MarkerFaceColor',colorOrder(obj.DetectionColorIndex,:),'MarkerEdgeColor',colorOrder(obj.DetectionColorIndex,:),'MarkerSize',3,'Marker','o');
            trkPlotter = trackPlotter(tp1,'DisplayName','Bounding box tracks','MarkerFaceColor',colorOrder(obj.TrackColorIndex,:),'MarkerEdgeColor',colorOrder(obj.TrackColorIndex,:),'LabelOffset',[0 -1 0],'FontSize',10,'Marker','s','MarkerSize',3);
            obj.pDetectionPlotter3 = detPlotter;
            obj.pTrackPlotter3 = trkPlotter;
            trkPlotter.plotTrack([nan nan nan],struct('Length',1,'Width',1,'Height',1,'OriginOffset',[0 0 0]),eye(3));
            detPlotter.plotTrack([nan nan nan],struct('Length',1,'Width',1,'Height',1,'OriginOffset',[0 0 0]),eye(3));
            cpDisp.ObstaclePlotter = obj.pObstaclePlotter3;
            cpDisp.RawPointCloudPlotter = obj.pRawPCPlotter3;
            cpDisp.TheaterDisplay = tp1;
            obj.CloseUpDisplay = cpDisp;
            view(ax,-118,8);
            legend(ax,'off');
        end
    end
end


function [pos,vel,posCov,dimStruct,orientations,labels,labelsMotion] = parseTracks(tracks,posIndex,velIndex,dimIndex,yawIndex,modelProbs)
% Parse tracks for plotting using trackPlotter.
numTracks = numel(tracks);
if numTracks > 0
    stateSize = numel(tracks(1).State);
    allTrackStates = cat(2,tracks.State);
    posSelector = zeros(3,stateSize);
    posSelector(1,posIndex(1)) = 1;
    posSelector(2,posIndex(2)) = 1;
    posSelector(3,posIndex(3)) = 1;
    [pos,posCov] = getTrackPositions(tracks,posSelector);
    velSelector = zeros(3,stateSize);
    velSelector(1,velIndex(1)) = 1;
    velSelector(2,velIndex(2)) = 1;
    velSelector(3,velIndex(3)) = 1;
    vel = getTrackVelocities(tracks,velSelector);
    length = allTrackStates(dimIndex(1),:);
    width = allTrackStates(dimIndex(2),:);
    height = allTrackStates(dimIndex(3),:);
    yaw = allTrackStates(yawIndex,:);
    
    dimStruct = repmat(struct('Length',0,'Width',0,'Height',0,'OriginOffset',[0 0 0]),[numTracks 1]);
    orientations = zeros(3,3,numTracks);
    for i = 1:numTracks
        dimStruct(i).Length = length(i);
        dimStruct(i).Width = width(i);
        dimStruct(i).Height = height(i);
        orientations(:,:,i) = [cosd(yaw(i)) -sind(yaw(i)) 0; sind(yaw(i)) cosd(yaw(i)) 0; 0 0 1]';
    end
    strMotion = string([]);
    strLabels = string([]);
    for i = 1:numTracks
        strLabels(i) = string(sprintf('T%0.0f',tracks(i).TrackID));
        if nargin > 5
            strMotion(i) = string(sprintf('T%0.0f\nct = %0.2f\ncv = %0.2f',tracks(i).TrackID,modelProbs(1,i),modelProbs(2,i)));
        else
            strMotion(i) = strLabels(i);
        end
    end
    labels = strLabels;
    labelsMotion = strMotion;
else
    pos = zeros(0,3);
    posCov = zeros(3,3,0);
    vel = zeros(0,3);
    dimStruct = repmat(struct('Length',0,'Width',0,'Height',0,'OriginOffset',[0 0 0]),[numTracks 1]);
    orientations = zeros(3,3,0);
    labels = {};
    labelsMotion = {};
end
end

function [pos,dimStruct,detOrientations] = parseDetections(detections)
% Parse detections for plotting bounding box measurements using a
% trackPlotter.
numDets = numel(detections);
if numDets > 0
allDetections = [detections{:}];
measurements = cat(2,allDetections.Measurement);
pos = measurements(1:3,:);
pos = pos';
length = measurements(4,:);
width = measurements(5,:);
height = measurements(6,:);
dimStruct = struct('Length',{},'Width',{},'Height',{},'OriginOffset',{});
detOrientations = repmat(eye(3),[1 1 numel(length)]);
for i = 1:numel(length)
    dimStruct(i).Length = length(i);
    dimStruct(i).Width = width(i);
    dimStruct(i).Height = height(i);
    dimStruct(i).OriginOffset = [0 0 0];
end
else
    pos = zeros(0,3);
    dimStruct = repmat(struct('Length',0,'Width',0,'Height',0,'OriginOffset',[0 0 0]),[0 1]);
    detOrientations = zeros(3,3,0);
end
end


