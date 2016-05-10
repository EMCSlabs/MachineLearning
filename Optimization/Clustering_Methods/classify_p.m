function classify_p
% Classification using perceptron neural network
% Dec 8, 2015
% Sunghah Hwang, EMCS Labs, Korea University

%% Figure
S.fh = figure('uni','pix', 'posi',[350 150 1025 670], 'menub','non',...
    'nam','Classify - Perceptron', 'numbert','off', 'resi','off', 'color', 'k');

%% Axes
S.ax = axes('uni','pix', 'posi',[30 30 600 600], 'nextplot','replacechildren', 'par', S.fh);
title(S.ax,'2D Plane','fontsize', 20, 'fontweight', 'bold')
S.ax_logo = axes('uni','pix', 'position',[646 30 360 140], 'parent', S.fh);
S.im = imread('emcs_logo_s.png'); image(S.im);
set(S.ax_logo,'xtick',[],'ytick',[])

%% Button group & buttons
S.bg = uibuttongroup('units','pix',...
    'position',[674 200 300 420], 'title', 'Parameters', 'fontsize', 20, ...
    'fontweight', 'bold', 'TitlePosition','centertop', 'parent', S.fh, ...
    'highlightcolor', 'w', 'bordertype', 'etchedin', 'borderwidth', 3, ...
    'shadowcolor', 'r', 'backgroundcolor', [.8 .8 .8] );

% # of samples for each category
S.tx_numsamp = uicontrol('style', 'text', ...
    'unit', 'pix', 'position', [45 300 100 40], ...
    'fontsize', 17, 'backgroundcolor', [.8 .8 .8], ...
    'string', '# of samples', 'parent', S.bg);
S.ed_numsamp = uicontrol('style', 'edit', ...
    'units', 'pix', 'position', [195 315 50 30], ...
    'string', '100', 'fontweight', 'bold',...
    'horizontalalign', 'center', ...
    'fontsize', 15, 'parent', S.bg);

% offset
S.tx_offset = uicontrol('style', 'text', 'string', 'offset', ...
    'unit', 'pix', 'position', [45 250 100 40], ...
    'fontsize', 17, 'backgroundcolor', [.8 .8 .8], 'parent', S.bg);
S.ed_offset = uicontrol('style', 'edit', 'units', 'pix', ...
    'position', [195 265 50 30], 'string', '.5', 'fontweight', 'bold', ...
    'horizontalalign', 'center', 'fontsize', 15, 'parent', S.bg);

% max iteration
S.tx_iter = uicontrol('style', 'text', 'backgroundcolor', [.8 .8 .8], ...
    'unit', 'pix', 'position', [45 200 100 40], ...
    'fontsize', 17, 'string', 'iteration', 'parent', S.bg);
S.ed_iter = uicontrol('style', 'edit', 'units', 'pix', 'string', '200', ...
    'position', [195 215 50 30], 'fontweight', 'bold', 'fontsize', 15, ...
    'horizontalalign', 'center', 'parent', S.bg);

% plot interval
S.tx_interval = uicontrol('style', 'text', 'unit', 'pix', ...
    'backgroundcolor', [.8 .8 .8], 'position', [50 150 100 40], ...
    'fontsize', 17, 'string', 'plot invertal', 'parent', S.bg);
S.ed_interval = uicontrol('style', 'edit', 'units', 'pix', ...
    'position', [195 165 50 30], 'string', '.5', 'fontweight', 'bold', ...
    'horizontalalign', 'center', 'fontsize', 15, 'parent', S.bg);

%% push buttons
S.pb_apply = uicontrol('style','push', 'units','pix', 'position',[15 15 120 50], ...
    'fontsize',15, 'string','Apply', 'callback', {@pb_apply,S}, 'parent', S.bg);

S.pb_start = uicontrol('style','push', 'units','pix', 'callback',{@pb_start,S}, ...
    'fontsize',15,'position',[155 15 120 50], 'string','Start', 'parent', S.bg);

%% apply button callback
S.apply_cnt = 0;
    function pb_apply(varargin)
        cla(S.ax)
        pb_press = get(S.pb_start,'string');
        if strcmp(pb_press, 'Stop')
            set(S.pb_start, 'string', 'Start')
        end
        
        S.numsamp = str2num(get(S.ed_numsamp,'string'));
        S.offset = str2num(get(S.ed_offset,'string'));
        S.maxiter = str2num(get(S.ed_iter,'string'));
        S.interval = str2num(get(S.ed_interval,'string'));
        
        S.A = [rand(1,S.numsamp)-S.offset; rand(1,S.numsamp)+S.offset]; % left top
        S.B = [rand(1,S.numsamp)+S.offset; rand(1,S.numsamp)+S.offset]; % right top
        S.C = [rand(1,S.numsamp)-S.offset; rand(1,S.numsamp)-S.offset]; % left bottom
        S.D = [rand(1,S.numsamp)+S.offset; rand(1,S.numsamp)-S.offset]; % right bottom
        
        axes(S.ax)
        plot(S.A(1,:), S.A(2,:), 'ro'); hold on; plot(S.B(1,:), S.B(2,:), 'b*');
        plot(S.C(1,:), S.C(2,:), 'gd'); grid on; plot(S.D(1,:), S.D(2,:), 'ms')
        axis tight
        S.apply_cnt = S.apply_cnt + 1;
        set(S.pb_start,'value',0)
    end

%% start/stop button callback
    function pb_start(varargin)
        
        if ~isfield(S, 'offset')
            errordlg('Apply parameters first', 'Error')
        end
        h = varargin{1};  % Get the pushbutton.
        % S = varargin{3};  % Get the structure
        pb_press = get(h,'string');  
        if strcmp(pb_press,'Start')
            set(h,'string','Stop')
            
            axes(S.ax)
            plot(S.A(1,:), S.A(2,:), 'ro'); hold on; plot(S.B(1,:), S.B(2,:), 'b*');
            plot(S.C(1,:), S.C(2,:), 'gd'); grid on; plot(S.D(1,:), S.D(2,:), 'ms')
            axis tight
            text(.4-S.offset, .38+1.5*S.offset, 'Class A', 'fontsize', 15, 'fontweight', 'bold', 'color', 'r')
            text(.4+S.offset, .38+1.5*S.offset, 'Class B', 'fontsize', 15, 'fontweight', 'bold', 'color', 'b')
            text(.4-S.offset, .6-1.5*S.offset, 'Class C', 'fontsize', 15, 'fontweight', 'bold', 'color', 'g')
            text(.4+S.offset, .6-1.5*S.offset, 'Class D', 'fontsize', 15, 'fontweight', 'bold', 'color', 'm')
            
            % declare inputs
            P = [S.A, S.B, S.C, S.D];
            
            % declare targets
            a = [0;1]; b = [1;1];
            c = [0;0]; d = [1;0];
            
            % target matrix
            T = [ repmat(a,1,length(S.A)), repmat(b,1,length(S.B))...
                repmat(c,1,length(S.C)), repmat(d,1,length(S.D)) ];
            
            % create perceptron neural network
            net = perceptron;
            % train the perceptron
            net.adaptParam.passes = 1;
            linehandle = plotpc(net.IW{1}, net.b{1});
            epoch = 1;
            iter = 0;
            
            while (sse(epoch) && iter < S.maxiter)
                iter = iter+1;
                [net, Y, epoch] = adapt(net, P, T);
                linehandle = plotpc(net.IW{1}, net.b{1}, linehandle); drawnow;
                t = text(.7, .35, sprintf('iteration #:%d', iter), 'fontsize', 15, 'fontweight', 'bold');
                pause(S.interval)
                delete(t)
            end
            text(.6, .42, sprintf('Converged at iteration #:%d', iter), 'fontsize', 18, 'fontweight', 'bold');
            set(h,'string','Start'); S.net = net;
        end
        if strcmp(pb_press,'Stop')
            cla(S.ax); clc
            set(h,'string','Start')
        end
    end
end