% class ProgressBar
%
%	This class provides an easy to use progress bar for the command window.
%
%	An example for a progress bar using all features is shown below between
%	the two lines. The names of the features or elements are written
%	between brackets and are not part of the actual output.
%	=======================================================================
%	permanent text										[display area]
%	permanent text										[display area]
%	permanent text										[display area]
%	|----|----|----|----|----|----|----|----|----|----|	[ruler]
%	***********											[bar]
%	199 / 1000 (19.9%)   elapsed time: 23 sec   remaining time: 1 min 33 sec (est.)
%									[counter, elapsed time, remaining time]
%	text with information to current step				[info]
%	\													[hourglass]
%	>>
%	=======================================================================
% 
%	A (minimum) update interval can be defined (default value is 1 second)
%	to avoid flickering and sacrificing too much processor load just for
%	the display output. The update is not triggered by a timer, but by
%	calls of the UPDATE member function, which is (normaly) called at the
%	beginning of a new step. Therefore the update interval is a minimal
%	interval. At the last step, i.e. if the current step number equals the
%	total step count an update will always performed, even if this violates
%	the update interval.
%
%	To update the information of the progress bar, the old version is
%	deleted from the command window by "typing" a certain number of
%	backspace characters. Therefore, any output of other functions and the
%	progress bar itself will be destroyed by the next progress bar update,
%	which is a little drawback in using it.
%	To cope with this, two possibilities are provided: The progress bar can
%	be hidden, then any output can be written to the command window, and
%	with the next update, the progress bar will be restored. Another
%	possibility is to use the member function DISP of the progress bar.
%	This function prints the given strings in the display area above the
%	progress bar.
%
%	
%	Instantiation
%		pb = ProgressBar(nCount [, value, parameter, ...])
%		pb = ProgressBar(-1 [, value, parameter, ...])
%		pb = ProgressBar(false [, value, parameter, ...])
%
%		Arguments
%			nCount		>0: Total count of steps to perform
%						-1: No total count is known a priori
%						false: Progress bar is deactivated
%						
%			If the count of the steps is not known a priori, nCount has to
%			be set to -1. In this case, the bar doesn't grow, but a marker
%			circles around and no remaining time will be shown, despite the
%			parameter value.
%
%			The value of nCount can be the logical value "false" also. In
%			this case, all further invocations of any member function are
%			simply ignored. This allows for a simple debugging without the
%			progress bar destroying the output of any other function.
%
%		Parameters
%			Ruler			{true}, false
%			Bar				{true}, false
%			Counter			{true}, false
%			ElapsedTime		{true}, false
%			RemainingTime	{true}, false
%			Hourglass		true, {false}
%							If true, the related element is shown.
%			Numbers			{[]}, true, false
%							Unies Counter, ElapsedTime and RemainingTime
%							If not empty, the value for this three
%							Parameters will be overwritten by the value of
%							Numbers.
%			OnlyInfo		true, {false}
%							If true, the paremeters 'Ruler', 'Bar',
%							'Counter', 'ElapsedTime' and 'RemainingTime'
%							are set to FALSE.
%
%			UpdateInterval	{1} (numeric value)
%						This parameter specifies the (minimum) interval
%						between updates of the progress bar (except the
%						hourglass).
%						If UpdateInterval is positive, the value is
%						interpreted as time in seconds. If UpdateInteval is
%						negative, the value is interpreted as (negative) 
%						number of steps. Thus, a value of -1 means that the
%						progress bar will be updated every step.
%						The (minimum) update interval for the hourglass is
%						set to half the value of this parameter.
%	
%	
%	Update
%		pb = obj.update([u] [, info [, display]])
%
%		Arguments
%			u			Integer with number of step, which will be started
%						after this function call.
%			info		String (or cell array of strings) with info text
%						for this step.
%						If no info but display text is given, set info to
%						empty string ''.
%			display		String (or cell array of strings) with text to be
%						displayed in the display area.
%
%	Update with auto-increment of step-counter
%		pb = obj.updateinc([info [, display]])
%
%		Calls the member function
%			UPDATE(pb, u, info, display)
%		with u equals the current step-count incremented by one.
%		N.b.:
%			UPDATEINC(pb, info, display)
%		is not the same as
%			UPDATE(pb, info, display)
%		The first call increments the counter, whereas the second call
%		updates the info-text, add display text and updates the hourglass,
%		if activated, but not the step-count!pb.
%	
%		Arguments
%			display		see member function UPDATE
%			bRedrawNow	see member function UDPATE
%
%	Add text for display area
%		pb = obj.disp(display [, bRedrawNow])
%
%		Arguments
%			display		String (or cell array of strings) with text to be
%						displayed in the display area.
%			bRedrawNow	Logical value, {true}
%						If true, then the display area will be updated
%						immediately. If false, the update will be performed
%						at the next "normal" update.
%
%	Hide
%		pb = pb.hide()
%
%		Removes the progress bar (without the display area) from the
%		command window. It will be restored by the next call of the update
%		function.
%
%	Delete
%		pb = pb.delete();
%
%		Removes the ProgressBar from the command window. After this call,
%		the progress bar is deactivated and all further calls of member
%		functions are ignored.
%
%
%
%	Examples
%		%% Standard bar with default update time of 1 second.
%		pb = ProgressBar(1000);
%		u = 0;
%		while (u < 1000)
%			u = u + 1;
%			pb = pb.update(u);
%			pause(0.001 * randi(10, 1));
%		end
%		pb = pb.delete();
%
%		%% Standard bar with update every step
%		pb = ProgressBar(1000, 'UpdateInterval', -1);
%		u = 0;
%		while (u < 1000)
%			u = u + 1;
%			pb = pb.update(u);
%			pause(0.001 * randi(10, 1));
%		end
%
%		%% With unknown total count, display text and info text
%		pb = ProgressBar(-1, 'UpdateInterval', -1);
%		u = 0;
%		while (u < 100)
%			u = u + 1;
%			if ( mod(u - 1, 10) == 0)
%				text = [num2str(100 - u + 1) ' to go!'];
%			else
%				text = '';
%			end
%			info = ['some redundant information: this is step ' num2str(u)];
%			pb = pb.update(u, info, text);
%			pause(0.25);
%		end
%
%		%% With hourglass to show progress during longer steps
%		pb = ProgressBar(25, 'UpdateInterval', -1, 'Hourglass', true);
%		u = 0;
%		while (u < 25)
%			u = u + 1;
%			pb = pb.update(u);
%			pause(0.25);
%			pb = update(pb);	% If only the hourglass is to be updated,
%			pause(0.25);		% the step number u can be omitted
%			pb = update(pb, u);	% Equivalent to call without u
%			pause(0.25);
%		end
%
%		%% The info text can be changed during the same step
%		pb = ProgressBar(10, 'UpdateInterval', -1, 'Hourglass', true);
%		u = 0;
%		while (u < 10)
%			u = u + 1;
%			pb = pb.updateinc('Substep A');	% Auto-increment of steps!
%			pause(0.25);
%			pb = pb.update();
%			pause(0.25);
%			pb = pb.update();
%			pause(0.25);
%			pb = pb.update('Substep B');
%			pause(0.5);
%		end
%
%		%% Also, the ProgressBar can be used without any bar at all
%		pb = ProgressBar(1, 'UpdateInterval', -1, 'OnlyInfo', true);
%		u = 0;
%		while (u < 10)
%			u = u + 1;
%			info = ['this is step ' num2str(u)];
%			pb = pb.update(info);
%			pause(0.25);
%		end
%		pb.delete();
% 
classdef ProgressBar
    %PROGRESSBAR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties(Hidden)
        bActive
        nCount
        uCurCount
        ticStart
        bShowBar
        bShowRuler
        bShowCount
        bShowElapsedTime
        bShowRemainingTime
        bShowNumbers
        bShowHourglass
        uMinUpdateInterval
        bUpdateIntervalsInTime
        uMinHourglassUpdateInterval
        dispbuffer
        ruler
        szBarChar
        bar
        numbers
        info
        hourglass
        uHourglassState
        bHidden
        ticUpdateHourglass
        ticUpdate
        nextHourglassUpdateCount
        nextUpdateCount
    end
    
    methods
        function obj = ProgressBar(nCount, varargin)
 	        p = inputParser();
 	        p.KeepUnmatched = false;
        
	        p.addParameter('Ruler', true, @islogical);
	        p.addParameter('Bar', true, @islogical);
	        p.addParameter('Counter', true, @islogical);
	        p.addParameter('ElapsedTime', true, @islogical);
	        p.addParameter('RemainingTime', true, @islogical);
	        p.addParameter('Numbers', [], @islogical);
	        p.addParameter('Hourglass', false, @islogical);
	        p.addParameter('OnlyInfo', false, @islogical);
	        
	        p.addParameter('UpdateInterval', 1, @isnumeric);
        
	        p.parse(varargin{:});
	        
	        
	        if (islogical(nCount) && ~nCount)
		        obj.bActive = false;
	        else
		        obj.bActive = true;
		        
		        if ( isempty(nCount) || ~isnumeric(nCount) || p.Results.OnlyInfo )
			        nCount = -1;
		        end
	        end
        
	        
	        obj.nCount = nCount;
	        obj.uCurCount = -1;
	        obj.ticStart = 0;	
        
	        if p.Results.OnlyInfo
		        obj.bShowBar = false;
		        obj.bShowRuler = false;
		        obj.bShowCount = false;
		        obj.bShowElapsedTime = false;
		        obj.bShowRemainingTime = false;
	        else
		        obj.bShowBar = p.Results.Bar;
		        obj.bShowRuler = p.Results.Ruler && obj.bShowBar;
        
		        if isempty(p.Results.Numbers)
			        obj.bShowCount = p.Results.Counter;
			        obj.bShowElapsedTime = p.Results.ElapsedTime;
			        obj.bShowRemainingTime = p.Results.RemainingTime;
		        else
			        obj.bShowCount = p.Results.Numbers;
			        obj.bShowElapsedTime = p.Results.Numbers;
			        obj.bShowRemainingTime = p.Results.Numbers;
		        end
	        end
        
	        if (nCount == -1)
		        obj.bShowRemainingTime = false;
	        end
	        
	        
	        obj.bShowNumbers = obj.bShowCount || obj.bShowElapsedTime ...
												        || obj.bShowRemainingTime;
	        
	        obj.bShowHourglass = p.Results.Hourglass;
	        
        
	        obj.uMinUpdateInterval = p.Results.UpdateInterval;
	        obj.bUpdateIntervalsInTime = (obj.uMinUpdateInterval >= 0);
	        
	        if (obj.bUpdateIntervalsInTime)
		        obj.uMinHourglassUpdateInterval = 0.5;
	        else
		        obj.uMinUpdateInterval = -obj.uMinUpdateInterval;
		        
		        % Fall UpdateInterval == -1 gewählt ist, dann ist damit impliziert,
		        % dass bei jedem Aufruf von UPDATE der ProgressBar aktualisiert
		        % werden soll. Damit dass auch für den Fall sichergestellt ist,
		        % dass Änderungen (Hourglass, Infotext) bei gleichem Schritt u
		        % geschieht, wird in diesem Fall das Interval auf 0 gesetzt.
		        if (obj.uMinUpdateInterval == 1)
			        obj.uMinUpdateInterval = 0;
		        end
		        
		        obj.uMinHourglassUpdateInterval = ceil(obj.uMinUpdateInterval / 2);
	        end
	        
	        
	        obj.dispbuffer = {};
        
	        obj.ruler.szNext = ...
					        '|----|----|----|----|----|----|----|----|----|----|';  obj.bActive = true;
	        obj.ruler.curLen = 0;
	        obj.ruler.bUpdate = obj.bShowRuler;
	        
	        if (obj.bShowRuler)
		        obj.szBarChar = '*';
	        else
		        obj.szBarChar = '.';
	        end
	        
	        if (obj.nCount == -1)
		        obj.bar.szNext = ...
					        '...........                                        ';
		        obj.bar.szNext(obj.bar.szNext ~= ' ') = obj.szBarChar;
		        obj.bar.curLen = 0;
		        obj.bar.bUpdate = obj.bShowBar;
	        else
		        obj.bar.szNext = '';
		        obj.bar.curLen = 0;
		        obj.bar.bUpdate = false;
	        end
	        
	        obj.numbers.szNext = '';
	        obj.numbers.curLen = 0;
	        obj.numbers.bUpdate = false;
	        
	        obj.info.szNext = '';
	        obj.info.curLen = 0;
	        obj.info.bUpdate = false;
	        
	        obj.hourglass.szNext = '|';
	        obj.hourglass.curLen = 0;
	        obj.hourglass.bUpdate = false;
        
	        
	        obj.uHourglassState = 1;
	        
	        
	        obj.bHidden = true;
	        
	        
	        obj.ticUpdateHourglass = uint64(0);
	        obj.ticUpdate = uint64(0);
	        obj.nextHourglassUpdateCount = 0;
	        obj.nextUpdateCount = 0;
        end
        
        function obj = update(obj, u, szInfo, szDisp)
        
	        if (~obj.bActive)
		        return;
	        end
	        
	        bRedraw = obj.bHidden;
	        
	        if (obj.uCurCount == -1)
		        obj.ticStart = tic();
	        else
		        if obj.bShowHourglass
			        if obj.bUpdateIntervalsInTime
				        if (toc(obj.ticUpdateHourglass) >= ...
											        obj.uMinHourglassUpdateInterval)
					        obj = updateHourglass(obj);
					        obj.ticUpdateHourglass = tic();
					        bRedraw = true;
				        end
			        elseif (nargin > 1)
				        if (u >= obj.nextHourglassUpdateCount)
					        obj = updateHourglass(obj);
					        obj.nextHourglassUpdateCount = ...
									        u + obj.uMinHourglassUpdateInterval;
					        bRedraw = true;
				        end
			        elseif (obj.uMinHourglassUpdateInterval == 0)
				        obj = updateHourglass(obj);
				        bRedraw = true;				
			        end
		        end
	        end
        
	        if (nargin > 1)
		        bNewInfo = true;
		        
		        if isnumeric(u)
			        if (nargin < 3), szInfo = ''; bNewInfo = false; end
			        if (nargin < 4), szDisp = ''; end
		        else
			        if (nargin < 3), szInfo = ''; end
			        szDisp = szInfo;
			        szInfo = u;
			        u = obj.uCurCount;
			        
			        bNewInfo = true;
		        end
		        
		        if ~isempty(szDisp)
			        if iscell(szDisp)
				        obj.dispbuffer = [obj.dispbuffer szDisp];
			        else
				        obj.dispbuffer{end+1} = szDisp;
			        end
			        
			        bRedraw = true;
			        
			        if isempty(u)
				        u = obj.uCurCount;
			        end
		        end
        
		        if ( (u ~= obj.uCurCount) || bNewInfo )
			        obj.uCurCount = u;
			        
			        if obj.bUpdateIntervalsInTime
				        if ( (toc(obj.ticUpdate) >= obj.uMinUpdateInterval) || ...
														        (u == obj.nCount) )
				        
					        if obj.bShowBar
						        obj = updateBar(obj);
					        end
					        
					        if obj.bShowNumbers
						        obj = updateNumbers(obj);
					        end
					        
					        obj = updateInfo(obj, szInfo);
        
					        obj.ticUpdate = tic();
					        bRedraw = true;
				        end
			        else
				        if ( (u >= obj.nextUpdateCount) || (u == obj.nCount) )
					        if obj.bShowBar
						        obj = updateBar(obj);
					        end
					        
					        if obj.bShowNumbers
						        obj = updateNumbers(obj);
					        end
					        
					        obj = updateInfo(obj, szInfo);
					        
					        obj.nextUpdateCount = u + obj.uMinUpdateInterval;
					        bRedraw = true;
				        end
			        end
		        end
	        end
	        
	        if bRedraw
		        obj = redraw(obj);
	        end
	        
	        obj.bHidden = false;
	        
        end

        function obj = updateinc(obj, varargin)
        
	        if (~obj.bActive)
		        return;
	        end
	        
	        if (obj.uCurCount == -1)
		        u = 1;
	        else
		        u = obj.uCurCount + 1;
	        end
	        
	        obj = obj.update(u, varargin{:});
		        
        end
           
        function obj = disp(obj, szDisp, bRedrawNow)
        
	        if (~obj.bActive)
		        return;
	        end
	        
	        if (nargin < 3), bRedrawNow = true; end
        
	        if bRedrawNow
		        obj = obj.update([], [], szDisp);
	        else
		        if iscell(szDisp)
			        obj.dispbuffer = [obj.dispbuffer szDisp];
		        else
			        obj.dispbuffer{end+1} = szDisp;
		        end
	        end
	        
        end

        function obj = hide(obj)
        
	        if (~obj.bActive)
		        return;
	        end
        
	        nDel = obj.ruler.curLen ...
				        + obj.bar.curLen ...
				        + obj.numbers.curLen ...
				        + obj.info.curLen ...
				        + obj.hourglass.curLen;
			        
	        if (nDel > 0)
		        obj.ruler.curLen = 0;
		        obj.ruler.bUpdate = true;
		        obj.bar.curLen = 0;
		        obj.numbers.curLen = 0;
		        obj.info.curLen = 0;
		        obj.hourglass.curLen = 0;
		        
		        S = repmat(sprintf('\b'), 1, nDel);
		        fprintf('%s', S);
		        drawnow('update');
	        end	
	        
	        obj.bHidden = true;
        end

        function obj = delete(obj)
        
	        obj = obj.hide();
	        
	        obj.bActive = false;
	        
        end        
    end

    methods(Hidden)
        function obj = updateHourglass(obj)
        
	        S = '|/-\';
	        
	        obj.uHourglassState = obj.uHourglassState + 1;
	        if (obj.uHourglassState > 4)
		        obj.uHourglassState = 1;
	        end
	        
	        obj.hourglass.szNext = S(obj.uHourglassState);
	        obj.hourglass.bUpdate = true;
	        
        end
        
        
        function obj = updateBar(obj)
        
	        if (obj.nCount == -1)
		        obj.bar.szNext = circshift(obj.bar.szNext, [0 1]);
		        obj.bar.bUpdate = true;
	        else
		        n = obj.uCurCount / obj.nCount * 100;
		        n = round(n / 2) + 1;
        
		        bChanged = (n ~= length(obj.bar.szNext));
        
		        if (bChanged)
			        obj.bar.szNext = char(ones(n, 1) * obj.szBarChar);
			        obj.bar.bUpdate = true;
		        end
	        end
	        
        end % function updateBar
        
        
        function obj = updateNumbers(obj)
        
	        S = '';
	        
	        if (obj.bShowCount)
		        if (obj.nCount > 0)
			        S = sprintf('%d / %d (%.1f%%)   ', ...
										        obj.uCurCount, obj.nCount, ...
										        obj.uCurCount / obj.nCount * 100 );
		        else
			        S = [num2str(obj.uCurCount) '   '];
		        end
	        end
		        
	        if (obj.bShowElapsedTime) || (obj.bShowRemainingTime)
		        elapsed = toc(obj.ticStart);
	        end
	        
	        if (obj.bShowElapsedTime)
		        S = [S 'elapsed time: ' formatTime(elapsed) '   '];
	        end
			        
	        if (obj.bShowRemainingTime) && (obj.nCount ~= -1)
		        
		        if (obj.uCurCount < 2)
			        S = [S 'remaining time: ---'];
		        else
			        avg = elapsed / (obj.uCurCount - 1);
			        est = (obj.nCount - obj.uCurCount + 1) * avg;
			        S = [S 'remaining time: ' formatTime(est) ' (est.)'];
		        end
	        end
	        
	        bChanged = ~strcmp(S, obj.numbers.szNext);
	        
	        if (bChanged)
		        obj.numbers.szNext = S;
		        obj.numbers.bUpdate = true;
	        end
        
        end % function updateNumbers
        
        
        function obj = updateInfo(obj, szInfo)
        
	        if iscell(szInfo)
		        S = szInfo{1};
		        
		        for l = 2:length(szInfo)
			        S = [S sprintf('\n%s', szInfo{l})]; %#ok<AGROW>
		        end
		        szInfo = S;
	        end
	        
	        if ~strcmp(obj.info.szNext, szInfo)
		        obj.info.szNext = szInfo;
		        obj.info.bUpdate = true;
	        end
        
        end % function updateInfo

        
        function obj = redraw(obj)
	        
	        bUpdateDisp = ~isempty(obj.dispbuffer);
	        
	        bUpdateRuler = obj.ruler.bUpdate || bUpdateDisp;
	        bUpdateBar = obj.bar.bUpdate || bUpdateRuler;
	        bUpdateNumbers = obj.numbers.bUpdate || bUpdateBar;
	        bUpdateInfo = obj.info.bUpdate || bUpdateNumbers;
	        bUpdateHourglass = obj.hourglass.bUpdate || bUpdateInfo;
	        
	        if ~bUpdateHourglass
		        return;
	        end
	        
	        nDel = 0;
	        
	        if (bUpdateRuler), nDel = nDel + obj.ruler.curLen; end
	        if (bUpdateBar), nDel = nDel + obj.bar.curLen; end
	        if (bUpdateNumbers), nDel = nDel + obj.numbers.curLen; end
	        if (bUpdateInfo), nDel = nDel + obj.info.curLen; end
	        if (bUpdateHourglass), nDel = nDel + obj.hourglass.curLen; end
        
	        if nDel == 0
		        S = '';
	        else
		        S = repmat(sprintf('\b'), 1, nDel);
	        end
	        
	        
	        if (bUpdateDisp)
		        for l = 1:length(obj.dispbuffer)
			        S = [S sprintf('%s\n', obj.dispbuffer{l})]; %#ok<AGROW>
		        end
		        
		        obj.dispbuffer = {};
	        end
	        
	        if (bUpdateRuler && obj.bShowRuler)
		        C = sprintf('%s\n', obj.ruler.szNext);
		        S = [S C];
		        obj.ruler.curLen = length(C);
		        obj.ruler.bUpdate = false;
	        end
	        
	        if (bUpdateBar && obj.bShowBar)
		        C = sprintf('%s\n', obj.bar.szNext);
		        S = [S C];
		        obj.bar.curLen = length(C);
		        obj.bar.bUpdate = false;
	        end
	        
	        if (bUpdateNumbers && obj.bShowNumbers)
		        C = sprintf('%s\n', obj.numbers.szNext);
		        S = [S C];
		        obj.numbers.curLen = length(C);
		        obj.numbers.bUpdate = false;
	        end
		        
	        if (bUpdateInfo)
		        if isempty(obj.info.szNext)
			        obj.info.curLen = 0;
			        obj.info.bUpdate = false;
		        else
			        C = sprintf('%s\n', obj.info.szNext);
			        S = [S C];
			        obj.info.curLen = length(C);
			        obj.info.bUpdate = false;
		        end	
	        end
	        
	        if (bUpdateHourglass && obj.bShowHourglass)
		        C = sprintf('%s\n', obj.hourglass.szNext);
		        S = [S C];
		        obj.hourglass.curLen = length(C);
		        obj.hourglass.bUpdate = false;
	        end
	        
	        fprintf('%s', S);
	        drawnow('update');
	        
        end % function updatedisp
    end
end


function S = formatTime(sec)
    
    min = floor(sec / 60);
    sec = mod(sec, 60);

    h = floor(min / 60);
    min = mod(min, 60);
    
    d = floor(h / 24);
    h = mod(h, 24);

    sec = round(sec);
    
    if (d > 0)
        S = sprintf('%d d %d h %d min %d sec', d, h, min, sec);
    elseif (h > 0)
        S = sprintf('%d h %d min %d sec', h, min, sec);
    elseif (min > 0)
        S = sprintf('%d min %d sec', min, sec);
    else
        S = sprintf('%d sec', sec);
    end

end % function formatTime

