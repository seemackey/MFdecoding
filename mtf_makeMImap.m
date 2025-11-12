function mtf_makeMImap(filename)
% Robustly read mixed-type Excel cells, coerce to numeric, blanks -> 0, plot colormap.
% Usage:
%   mtf_makeMImap('winstonleftstgexcel.xlsx')

if nargin < 1
    filename = 'winstonleftstgexcel.xlsx';
end

% --- Read as raw cells to avoid type conflicts ---
C = readcell(filename);   % returns a cell array with numbers, text, empty, etc.

% --- Convert to numeric matrix, treating blanks/empties/non-numerics as 0 ---
M = zeros(size(C));
for r = 1:size(C,1)
    for c = 1:size(C,2)
        v = C{r,c};
        if isnumeric(v)
            if ~isnan(v)
                M(r,c) = v;
            else
                M(r,c) = 0;  % NaN -> 0
            end
        elseif isstring(v) || ischar(v)
            s = strtrim(string(v));
            if strlength(s) == 0
                M(r,c) = 0;  % empty string -> 0
            else
                d = str2double(s);   % converts numeric-like text; non-numeric -> NaN
                if isnan(d), d = 0; end
                M(r,c) = d;
            end
        else
            % logicals, missing, empty cells, etc.
            M(r,c) = 0;
        end
    end
end

% --- Optional: trim fully-zero border rows/cols (uncomment if desired) ---
% keepRows = any(M~=0,2);
% keepCols = any(M~=0,1);
% if any(~keepRows) || any(~keepCols)
%     M = M(keepRows, keepCols);
% end

% --- Plot colormap with zero at the low end (blank cells) ---
figure('Color','w','Position',[100 100 900 650]);
imagesc(M);
axis equal tight
set(gca,'YDir','reverse');          % top row is top
xlabel('Column'); ylabel('Row');
title(sprintf('Values from %s', filename), 'Interpreter','none');

% Choose a colormap; zero will map to the first color
colormap(turbo);                    % try 'parula', 'hot', 'copper', etc.
colorbar;

% Make sure zero is included at the bottom of the scale
mx = max(M(:));
if mx <= 0, mx = 1; end             % avoid [0 0] range
caxis([0 mx]);

% --- Overlay numeric labels (skip zeros for readability) ---
mask = M ~= 0;
[rr, cc] = find(mask);
vals = M(mask);
txt = string(round(vals, 3));
hold on;
for k = 1:numel(vals)
    text(cc(k), rr(k), txt(k), 'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', 'Color','k', 'FontSize',9);
end
hold off;

end
