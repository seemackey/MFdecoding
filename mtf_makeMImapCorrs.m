%% === Spatial correlation across Excel sheets ===
clear; clc; close all;

% --- Configuration ---
dataDir = cd;  % <-- change to your directory
fileList = dir(fullfile(dataDir, '*.xlsx'));

allX = [];
allY = [];
allVals = [];
fileNames = {};

for f = 1:numel(fileList)
    fname = fullfile(fileList(f).folder, fileList(f).name);
    fprintf('Reading %s...\n', fileList(f).name);

    % --- Read as raw cells (handles mixed types safely) ---
    C = readcell(fname);
    M = zeros(size(C));

    % Convert cell contents to numeric, blanks → 0
    for r = 1:size(C,1)
        for c = 1:size(C,2)
            v = C{r,c};
            if isnumeric(v)
                if isnan(v), M(r,c) = 0; else, M(r,c) = v; end
            elseif isstring(v) || ischar(v)
                s = strtrim(string(v));
                if strlength(s)==0
                    M(r,c) = 0;
                else
                    d = str2double(s);
                    if isnan(d), d = 0; else, M(r,c) = d; end
                end
            else
                M(r,c) = 0;
            end
        end
    end

    % --- Gather nonzero entries (exclude blanks) ---
    [rIdx, cIdx, vals] = find(M);
    allX = [allX; cIdx];
    allY = [allY; rIdx];
    allVals = [allVals; vals];
    fileNames = [fileNames; repmat({fileList(f).name}, numel(vals), 1)];
end

fprintf('Total data points (nonzero cells): %d\n', numel(allVals));

%% --- Statistical analysis ---

% Simple correlations
[r_row, p_row] = corr(allY, allVals, 'Type', 'Spearman');
[r_col, p_col] = corr(allX, allVals, 'Type', 'Spearman');

fprintf('Row–Value Spearman rho = %.3f (p = %.3g)\n', r_row, p_row);
fprintf('Col–Value Spearman rho = %.3f (p = %.3g)\n', r_col, p_col);

% Multiple regression (renamed variables to avoid 'Row' conflict)
tbl = table(allX, allY, allVals, 'VariableNames', {'ColIdx','RowIdx','Value'});
lm = fitlm(tbl, 'Value ~ RowIdx + ColIdx');
disp(lm);

%% --- Visualization ---

figure('Color','w','Position',[100 100 900 400]);

subplot(1,2,1);
scatter(allX, allVals, 40, 'filled');
xlabel('Column (X)'); ylabel('Value');
title(sprintf('ρ = %.2f, p = %.3g', r_col, p_col));
grid on; box on;

subplot(1,2,2);
scatter(allY, allVals, 40, 'filled');
xlabel('Row (Y)'); ylabel('Value');
title(sprintf('ρ = %.2f, p = %.3g', r_row, p_row));
grid on; box on;

sgtitle('Spatial Correlation of Cell Values Across Sheets');
