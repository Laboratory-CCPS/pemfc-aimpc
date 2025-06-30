function varargout = figureX(varargin)
    
    if ( (nargin == 1) && isempty(varargin{1}) )
        figure();
        return;        
    elseif ( (nargin ~= 1) || ~ischar(varargin{1}) )
        figure(varargin{:});
        return;
    end
    
    figname = varargin{1};
    
    h = findobj('Type', 'figure', '-and', 'Name', figname);
    
    if isempty(h)
        ret = figure('Name', figname);
    else
        ret = figure(h(1));
    end

    if nargout == 1
        varargout = {ret};
    end
    
end % function figureX
