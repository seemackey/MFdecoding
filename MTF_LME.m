clear
close all
dir_configs = struct( ...
    'path', {'E:\MTF\data\click\core\left\imported\decoding_results_LDA2',...
    'E:\MTF\data\click\core\right\imported\decoding_results_LDA',...
    'E:\MTF\data\click\pb\left\imported\decoding_results_LDA',...
    'E:\MTF\data\click\pb\right\imported\decoding_results_LDA'}, ...
    'hemisphere', {'left', 'right', 'left', 'right'}, ...
    'area', {'core', 'core', 'pb', 'pb'} ...
);

ISI_tableCSD = MTF_build_ISI_table(dir_configs, 700, 'oe', true);
ISI_tableMUA = MTF_build_ISI_table(dir_configs, 700, 'om', true);

% "Full" models
lme = fitlme(ISI_tableMUA, 'BestISI ~ Layer*Area*Hemisphere + (1|UnitID)')
anova(lme)

lmecsd = fitlme(ISI_tableCSD, 'BestISI ~ Layer*Area*Hemisphere + (1|UnitID)')
anova(lmecsd)

% very reduced simple models (no int)
lmesimple = fitlme(ISI_tableMUA, 'BestISI ~ Layer+Area+Hemisphere + (1|UnitID)')
anova(lmesimple)

lmecsdsimple = fitlme(ISI_tableCSD, 'BestISI ~ Layer+Area+Hemisphere + (1|UnitID)')
anova(lmecsdsimple)

% moderate realistic models (single interactions)
lmemedium = fitlme(ISI_tableMUA, 'BestISI ~ Layer*Area+Hemisphere + (1|UnitID)')
anova(lmemedium)

lmecsdmedium = fitlme(ISI_tableCSD, 'BestISI ~ Layer*Area+Hemisphere + (1|UnitID)')
anova(lmecsdmedium)

%% plot the predictions
% Ensure categorical variables with desired reference levels
ISI_tableMUA.Layer = categorical(ISI_tableMUA.Layer, {'supra', 'granular', 'infra'});
ISI_tableMUA.Area = categorical(ISI_tableMUA.Area, {'core', 'pb'});
ISI_tableMUA.Hemisphere = categorical(ISI_tableMUA.Hemisphere, {'left', 'right'});

% Predict BestISI for each observation using the fitted model
ISI_tableMUA.Predicted = predict(lme);

% Use grpstats to compute groupwise means
stats = grpstats(ISI_tableMUA, {'Layer', 'Area', 'Hemisphere'}, {'mean'}, 'DataVars', 'Predicted');

% Plot: One subplot per hemisphere, group by Layer, color by Area
figure;
layers = categories(ISI_tableMUA.Layer);
areas = categories(ISI_tableMUA.Area);
hemispheres = categories(ISI_tableMUA.Hemisphere);
colors = lines(numel(areas));

for h = 1:numel(hemispheres)
    subplot(1, numel(hemispheres), h);
    hold on;
    for a = 1:numel(areas)
        vals = NaN(size(layers));
        for l = 1:numel(layers)
            idx = stats.Layer == layers{l} & ...
                  stats.Area == areas{a} & ...
                  stats.Hemisphere == hemispheres{h};
            if any(idx)
                vals(l) = stats.mean_Predicted(idx);
            end
        end
        bar((1:numel(layers)) + (a-1)*0.25, vals, 0.25, 'FaceColor', colors(a,:));
    end
    xticks(1:numel(layers));
    xticklabels(layers);
    xlabel('Layer');
    ylabel('Predicted BestISI (ms)');
    title(['Hemisphere: ', hemispheres{h}]);
    legend(areas, 'Location', 'best');
    grid on;
end

% Ensure categorical reference levels
ISI_tableMUA.Layer = categorical(ISI_tableMUA.Layer, {'supra', 'granular', 'infra'});
ISI_tableMUA.Area = categorical(ISI_tableMUA.Area, {'core', 'pb'});
ISI_tableMUA.Hemisphere = categorical(ISI_tableMUA.Hemisphere, {'left', 'right'});

% Predict fitted values
ISI_tableMUA.Predicted = predict(lme);

% Create combined condition label for plotting
ISI_tableMUA.Group = strcat(string(ISI_tableMUA.Layer), '-', ...
                             string(ISI_tableMUA.Area), '-', ...
                             string(ISI_tableMUA.Hemisphere));

% Plot boxplots by condition
figure('Position', [100 100 1000 500]);
boxplot(ISI_tableMUA.Predicted, ISI_tableMUA.Group, ...
        'Colors', 'k', 'Symbol', 'o', 'Widths', 0.6);
ylabel('Predicted BestISI (ms)');
xlabel('Layer - Area - Hemisphere');
title('Predicted BestISI by Condition');
grid on;
xtickangle(45);
% Ensure categorical types and predictions
ISI_tableMUA.Layer = categorical(ISI_tableMUA.Layer, {'supra', 'granular', 'infra'});
ISI_tableMUA.Area = categorical(ISI_tableMUA.Area, {'core', 'pb'});
ISI_tableMUA.Hemisphere = categorical(ISI_tableMUA.Hemisphere, {'left', 'right'});
ISI_tableMUA.Predicted = predict(lme);

% Combine group labels
ISI_tableMUA.Group = strcat(string(ISI_tableMUA.Layer), '-', ...
                            string(ISI_tableMUA.Area), '-', ...
                            string(ISI_tableMUA.Hemisphere));
group_labels = unique(ISI_tableMUA.Group);

% Preallocate
nGroups = numel(group_labels);
means = zeros(nGroups,1);
sems = zeros(nGroups,1);

% Calculate mean and SEM for each group
for i = 1:nGroups
    this_group = group_labels{i};
    idx = ISI_tableMUA.Group == this_group;
    values = ISI_tableMUA.Predicted(idx);
    means(i) = mean(values, 'omitnan');
    sems(i) = std(values, 'omitnan') / sqrt(sum(~isnan(values)));
end

% Ensure categorical and predictions
ISI_tableMUA.Layer = categorical(ISI_tableMUA.Layer, {'supra', 'granular', 'infra'});
ISI_tableMUA.Area = categorical(ISI_tableMUA.Area, {'core', 'pb'});
ISI_tableMUA.Hemisphere = categorical(ISI_tableMUA.Hemisphere, {'left', 'right'});
ISI_tableMUA.Predicted = predict(lme);

% Combine condition labels
ISI_tableMUA.Group = strcat(string(ISI_tableMUA.Layer), '-', ...
                            string(ISI_tableMUA.Area), '-', ...
                            string(ISI_tableMUA.Hemisphere));
group_labels = unique(ISI_tableMUA.Group, 'stable');
nGroups = numel(group_labels);

% Compute means and SEMs
means = zeros(nGroups,1);
sems = zeros(nGroups,1);
for i = 1:nGroups
    idx = ISI_tableMUA.Group == group_labels(i);
    yvals = ISI_tableMUA.Predicted(idx);
    means(i) = mean(yvals, 'omitnan');
    sems(i) = std(yvals, 'omitnan') / sqrt(sum(~isnan(yvals)));
end

% Plot
figure('Position', [100 100 1000 500]);
hold on;

% Plot individual data points with jitter
for i = 1:nGroups
    idx = ISI_tableMUA.Group == group_labels(i);
    x_jittered = i + (rand(sum(idx),1)-0.5)*0.2;
    scatter(x_jittered, ISI_tableMUA.Predicted(idx), 25, 'filled', ...
            'MarkerFaceColor', [0.6 0.6 0.6], 'MarkerEdgeColor', 'k', 'MarkerFaceAlpha', 0.5);
end

% Plot means and error bars
errorbar(1:nGroups, means, sems, 'ko', 'MarkerFaceColor', 'k', ...
         'LineStyle', 'none', 'CapSize', 10, 'LineWidth', 1.5);

% Axes and labels
xticks(1:nGroups);
xticklabels(group_labels);
xtickangle(45);
ylabel('Predicted BestISI (ms)');
xlabel('Condition (Layer - Area - Hemisphere)');
title('Predicted Best ISI per Group ± SEM');
grid on;
xlim([0.5, nGroups + 0.5]);

% Ensure proper types
ISI_tableMUA.Layer = categorical(ISI_tableMUA.Layer, {'supra', 'granular', 'infra'});
ISI_tableMUA.Area = categorical(ISI_tableMUA.Area, {'core', 'pb'});
ISI_tableMUA.Hemisphere = categorical(ISI_tableMUA.Hemisphere);
ISI_tableMUA.Predicted = predict(lme);

% Get categories
layers = categories(ISI_tableMUA.Layer);
areas = categories(ISI_tableMUA.Area);
colors = lines(numel(areas));
offsets = [-0.15, 0.15];

% Precompute summary stats
figure('Position', [100 100 800 500]); hold on;

for a = 1:numel(areas)
    for l = 1:numel(layers)
        % Logical index
        idx = ISI_tableMUA.Layer == layers{l} & ISI_tableMUA.Area == areas{a};

        % === Raw data ===
        y_data = ISI_tableMUA.BestISI(idx);
        mean_data = mean(y_data, 'omitnan');
        sem_data = std(y_data, 'omitnan') / sqrt(sum(~isnan(y_data)));

        % === Model prediction ===
        y_pred = ISI_tableMUA.Predicted(idx);
        mean_pred = mean(y_pred, 'omitnan');
        sem_pred = std(y_pred, 'omitnan') / sqrt(sum(~isnan(y_pred)));

        % X position
        xpos = l + offsets(a);

        % Plot raw data stats
        errorbar(xpos, mean_data, sem_data, 's', ...
            'Color', colors(a,:), 'LineWidth', 1.5, ...
            'MarkerFaceColor', colors(a,:), 'MarkerEdgeColor', 'k', ...
            'CapSize', 8);

        % Plot model prediction stats
        errorbar(xpos, mean_pred, sem_pred, '^', ...
            'Color', [0 0 0], 'LineWidth', 1.5, ...
            'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', ...
            'CapSize', 8);
    end
end

% Aesthetics
xticks(1:numel(layers));
xticklabels(layers);
xlim([0.5, numel(layers)+0.5]);
ylabel('BestISI (ms)');
xlabel('Layer');
legend({'Raw Mean±SEM', 'Model Mean±SEM'}, 'Location', 'northwest');
title('Raw Data and Model Predictions by Layer and Area');
grid on;


% Ensure types and model predictions
ISI_tableMUA.Layer = categorical(ISI_tableMUA.Layer, {'supra', 'granular', 'infra'});
ISI_tableMUA.Area = categorical(ISI_tableMUA.Area, {'core', 'pb'});
ISI_tableMUA.Hemisphere = categorical(ISI_tableMUA.Hemisphere, {'left', 'right'});
ISI_tableMUA.Predicted = predict(lme);

layers = categories(ISI_tableMUA.Layer);
areas = categories(ISI_tableMUA.Area);
hemispheres = categories(ISI_tableMUA.Hemisphere);
colors = lines(numel(areas));
offsets = [-0.15, 0.15];

% Create subplots for each hemisphere
figure('Position', [100 100 1000 500]);

for h = 1:numel(hemispheres)
    subplot(1, numel(hemispheres), h); hold on;
    title(['Hemisphere: ', hemispheres{h}]);

    for a = 1:numel(areas)
        for l = 1:numel(layers)
            % Logical index for this combo
            idx = ISI_tableMUA.Layer == layers{l} & ...
                  ISI_tableMUA.Area == areas{a} & ...
                  ISI_tableMUA.Hemisphere == hemispheres{h};

            % === Raw data ===
            y_data = ISI_tableMUA.BestISI(idx);
            mean_data = mean(y_data, 'omitnan');
            sem_data = std(y_data, 'omitnan') / sqrt(sum(~isnan(y_data)));

            % === Model prediction ===
            y_pred = ISI_tableMUA.Predicted(idx);
            mean_pred = mean(y_pred, 'omitnan');
            sem_pred = std(y_pred, 'omitnan') / sqrt(sum(~isnan(y_pred)));

            % X position
            xpos = l + offsets(a);

            % Plot raw data mean ± SEM
            errorbar(xpos, mean_data, sem_data, 's', ...
                'Color', colors(a,:), 'LineWidth', 1.5, ...
                'MarkerFaceColor', colors(a,:), 'MarkerEdgeColor', 'k', ...
                'CapSize', 8);

            % Plot model prediction mean ± SEM
            errorbar(xpos, mean_pred, sem_pred, '^', ...
                'Color', [0 0 0], 'LineWidth', 1.5, ...
                'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', ...
                'CapSize', 8);
        end
    end

    % Aesthetics for subplot
    xticks(1:numel(layers));
    xticklabels(layers);
    ylabel('BestISI (ms)');
    xlabel('Layer');
    xlim([0.5, numel(layers)+0.5]);
    grid on;
end

% Shared legend
legend({'Raw Mean±SEM', 'Model Mean±SEM'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
sgtitle('BestISI by Layer, Area, and Hemisphere (Raw vs Model)');

% Ensure types and model predictions
ISI_tableMUA.Layer = categorical(ISI_tableMUA.Layer, {'supra', 'granular', 'infra'});
ISI_tableMUA.Area = categorical(ISI_tableMUA.Area, {'core', 'pb'});
ISI_tableMUA.Hemisphere = categorical(ISI_tableMUA.Hemisphere, {'left', 'right'});
ISI_tableMUA.Predicted = predict(lme);

layers = categories(ISI_tableMUA.Layer);
areas = categories(ISI_tableMUA.Area);
hemispheres = categories(ISI_tableMUA.Hemisphere);
colors = lines(numel(areas));
offsets = [-0.15, 0.15];

% Create subplots for each hemisphere
figure('Position', [100 100 1000 500]);

for h = 1:numel(hemispheres)
    subplot(1, numel(hemispheres), h); hold on;
    title(['Hemisphere: ', hemispheres{h}]);

    for a = 1:numel(areas)
        for l = 1:numel(layers)
            % Logical index for this combo
            idx = ISI_tableMUA.Layer == layers{l} & ...
                  ISI_tableMUA.Area == areas{a} & ...
                  ISI_tableMUA.Hemisphere == hemispheres{h};

            % === Raw data ===
            y_data = ISI_tableMUA.BestISI(idx);
            mean_data = mean(y_data, 'omitnan');
            sem_data = std(y_data, 'omitnan') / sqrt(sum(~isnan(y_data)));

            % === Model prediction ===
            y_pred = ISI_tableMUA.Predicted(idx);
            mean_pred = mean(y_pred, 'omitnan');
            sem_pred = std(y_pred, 'omitnan') / sqrt(sum(~isnan(y_pred)));

            % X position
            xpos = l + offsets(a);

            % Plot raw data mean ± SEM
            errorbar(xpos, mean_data, sem_data, 's', ...
                'Color', colors(a,:), 'LineWidth', 1.5, ...
                'MarkerFaceColor', colors(a,:), 'MarkerEdgeColor', 'k', ...
                'CapSize', 8);


            % Plot model prediction mean ± SEM
            errorbar(xpos, mean_pred, sem_pred, '^', ...
                'Color', [0 0 0], 'LineWidth', 1.5, ...
                'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', ...
                'CapSize', 8);

        end
    end

    % Aesthetics for subplot
    xticks(1:numel(layers));
    xticklabels(layers);
    ylabel('BestISI (ms)');
    xlabel('Layer');
    xlim([0.5, numel(layers)+0.5]);
    grid on;
end

% Shared legend
%legend({'Raw Mean±SEM', 'Model Mean±SEM'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
sgtitle('BestISI by Layer, Area, and Hemisphere (Raw vs Model)');

% Ensure categorical types and model predictions
ISI_tableCSD.Layer = categorical(ISI_tableCSD.Layer, {'supra', 'granular', 'infra'});
ISI_tableCSD.Area = categorical(ISI_tableCSD.Area, {'core', 'pb'});
ISI_tableCSD.Hemisphere = categorical(ISI_tableCSD.Hemisphere, {'left', 'right'});
ISI_tableCSD.Predicted = predict(fitlme(ISI_tableCSD, 'BestISI ~ Layer*Area*Hemisphere + (1|UnitID)'));

% Category settings
layers = categories(ISI_tableCSD.Layer);
areas = categories(ISI_tableCSD.Area);
hemispheres = categories(ISI_tableCSD.Hemisphere);
colors = lines(numel(areas));
offsets = [-0.15, 0.15];

% Create subplots
figure('Position', [100 100 1000 500]);

for h = 1:numel(hemispheres)
    subplot(1, numel(hemispheres), h); hold on;
    title(['Hemisphere: ', hemispheres{h}]);

    plottedArea = containers.Map({'core', 'pb'}, [false, false]);

    for a = 1:numel(areas)
        for l = 1:numel(layers)
            idx = ISI_tableCSD.Layer == layers{l} & ...
                  ISI_tableCSD.Area == areas{a} & ...
                  ISI_tableCSD.Hemisphere == hemispheres{h};

            y_data = ISI_tableCSD.BestISI(idx);
            mean_data = mean(y_data, 'omitnan');
            sem_data = std(y_data, 'omitnan') / sqrt(sum(~isnan(y_data)));

            y_pred = ISI_tableCSD.Predicted(idx);
            mean_pred = mean(y_pred, 'omitnan');
            sem_pred = std(y_pred, 'omitnan') / sqrt(sum(~isnan(y_pred)));

            xpos = l + offsets(a);

            % Raw data: legend only once per area
            if ~plottedArea(areas{a})
                errorbar(xpos, mean_data, sem_data, 's', ...
                    'Color', colors(a,:), 'LineWidth', 1.5, ...
                    'MarkerFaceColor', colors(a,:), 'MarkerEdgeColor', 'k', ...
                    'CapSize', 8, ...
                    'DisplayName', char(areas{a}));
                plottedArea(areas{a}) = true;
            else
                errorbar(xpos, mean_data, sem_data, 's', ...
                    'Color', colors(a,:), 'LineWidth', 1.5, ...
                    'MarkerFaceColor', colors(a,:), 'MarkerEdgeColor', 'k', ...
                    'CapSize', 8, ...
                    'HandleVisibility', 'off');
            end

            % Model predictions: never shown in legend
            errorbar(xpos, mean_pred, sem_pred, '^', ...
                'Color', [0 0 0], 'LineWidth', 1.5, ...
                'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', ...
                'CapSize', 8, ...
                'HandleVisibility', 'off');
        end
    end

    % Aesthetics
    xticks(1:numel(layers));
    xticklabels(layers);
    xlim([0.5, numel(layers)+0.5]);
    ylabel('BestISI (ms)');
    xlabel('Layer');
    grid on;
end

% Shared legend
legend('Location', 'southoutside', 'Orientation', 'horizontal');
sgtitle('CSD BestISI by Layer, Area, and Hemisphere (Raw vs Model)');
