function MTF_visualize_diffISI(diffISI_dir)
% Loads all *_diffISI.mat files and visualizes pairwise Best ISI differences
% between layers (supra vs gran, supra vs infra, gran vs infra),
% performs Wilcoxon signed-rank test, and applies Benjamini-Hochberg correction.

% Load all diffISI files
files = dir(fullfile(diffISI_dir, '*_diffISI.mat'));
all_diffs = table();

for i = 1:length(files)
    fpath = fullfile(diffISI_dir, files(i).name);
    S = load(fpath);
    if isfield(S, 'diffs')
        all_diffs = [all_diffs; S.diffs]; %#ok<AGROW>
    end
end

if isempty(all_diffs)
    error('No diffISI data found in %s.', diffISI_dir);
end

% Prepare output directory
fig_out_dir = fullfile(diffISI_dir, 'figures');
if ~exist(fig_out_dir, 'dir')
    mkdir(fig_out_dir);
end

% Make sure layer info is categorical
all_diffs.Layer1 = categorical(all_diffs.Layer1);
all_diffs.Layer2 = categorical(all_diffs.Layer2);
all_diffs.Comp = strcat(string(all_diffs.Layer1), '-', string(all_diffs.Layer2));

% === Run signed-rank tests and FDR correction ===
group_order = {'supra-granular', 'supra-infra', 'granular-infra'};
pvals = nan(1, numel(group_order));
medians = nan(1, numel(group_order));
n_per_group = nan(1, numel(group_order));

for i = 1:numel(group_order)
    idx = all_diffs.Comp == group_order{i};
    diffs = all_diffs.Diff(idx);
    if numel(diffs) > 2
        [p, ~] = signrank(diffs);
        pvals(i) = p;
        medians(i) = median(diffs);
        n_per_group(i) = numel(diffs);
    end
end

% FDR correction (Benjamini-Hochberg)
[h, crit_p, ~, adj_p] = fdr_bh(pvals);

% Print stats
fprintf('\n=== Signed-Rank Tests (FDR-corrected) ===\n');
for i = 1:numel(group_order)
    fprintf('%s: median = %.2f, n = %d, raw p = %.4f, adj p = %.4f, sig = %d\n', ...
        group_order{i}, medians(i), n_per_group(i), pvals(i), adj_p(i), h(i));
end

% === Plot 1: Violin/strip plot of differences per layer comparison ===
fig1 = figure('Name', 'Best ISI Differences by Layer Pair', 'Position', [100 100 800 500]);
g = all_diffs.Comp;
colors = lines(numel(group_order));
hold on;
for i = 1:numel(group_order)
    idx = g == group_order{i};
    x = i + 0.1*randn(sum(idx),1);
    y = all_diffs.Diff(idx);
    scatter(x, y, 25, 'filled', 'MarkerFaceColor', colors(i,:), 'MarkerFaceAlpha', 0.4);

    % Mean and SEM
    mu = mean(y, 'omitnan');
    se = std(y, 'omitnan') / sqrt(sum(~isnan(y)));
    errorbar(i, mu, se, 'ko', 'MarkerFaceColor', 'k', 'CapSize', 8, 'LineWidth', 1.5);

    % Annotate corrected p-value
    text(i, max(y)+5, sprintf('p = %.3f', adj_p(i)), 'HorizontalAlignment', 'center', 'FontSize', 10);
end

xlim([0.5, numel(group_order) + 0.5]);
xticks(1:numel(group_order));
xticklabels(group_order);
ylabel('Best ISI Difference (ms)');
title('Pairwise Best ISI Differences Between Layers');
grid on;

% Save figure
savefig(fig1, fullfile(fig_out_dir, 'BestISI_Diff_Scatter.fig'));
exportgraphics(fig1, fullfile(fig_out_dir, 'BestISI_Diff_Scatter.jpg'), 'Resolution', 300);

% === Plot 2: Histogram of differences per comparison type ===
fig2 = figure('Name', 'Histogram of Best ISI Differences', 'Position', [100 100 1000 300]);
for i = 1:numel(group_order)
    subplot(1, numel(group_order), i);
    idx = g == group_order{i};
    histogram(all_diffs.Diff(idx), 'BinWidth', 20, 'FaceColor', colors(i,:));
    title(sprintf('%s\np = %.3f', group_order{i}, adj_p(i)));
    xlabel('ISI Difference (ms)');
    ylabel('Count');
    grid on;
end

% Save figure
savefig(fig2, fullfile(fig_out_dir, 'BestISI_Diff_Histogram.fig'));
exportgraphics(fig2, fullfile(fig_out_dir, 'BestISI_Diff_Histogram.jpg'), 'Resolution', 300);

end
