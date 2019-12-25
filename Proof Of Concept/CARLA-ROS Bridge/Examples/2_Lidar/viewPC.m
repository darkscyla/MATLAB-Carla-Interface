  classdef viewPC < matlab.System & matlab.system.mixin.CustomIcon
    % Untitled Add summary here
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    properties
        
    end

    properties(DiscreteState)
        xyz;
        p1;
        p2;
        p3;
    end

    % Pre-computed constants
    properties(Access = private)
        
    end

    methods(Access = protected)
        function stepImpl(obj,xyzpoint)
            % Perform one-time calculations, such as computing constants
            obj.p1 = xyzpoint(:,1);
            obj.p2 = xyzpoint(:,2);
            obj.p3 = xyzpoint(:,3);

            obj.xyz = [obj.p1,obj.p2,obj.p3];
            pcshow(obj.xyz)
        end
        
        function num = getNumInputsImpl(~)
            num = 1;
        end
        function num = getNumOutputsImpl(~)
            num = 0;
        end
        function resetImpl(obj)
            
        end
    end
  end
