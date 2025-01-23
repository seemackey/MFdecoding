% MTF analysis of core and parabelt data
% chase m 2024
clear;clc;close all;

%% load and epoch data
filedir = 'E:\MTF\hi02\018\';
figuresdir = 'E:\MTF\hi02\018\CSD_decoding\';
bins = [1,5,10,50,150,325,750]; % in ms
csd_flag = 1; % analyze CSD, or if 0, analyze MUA
seeCSD = 0;
selected_channels = [8,11,14]; % for decoding example traces
epoch_tframe = [-50 750];

% Load and epoch data
tic
[epoched_data, srate] = MTF_loadMATfile(filedir, epoch_tframe);
toc
disp('data imported')
%% Plot CSD and MUA

time_axis = epoch_tframe(1):1000/srate:epoch_tframe(2);
numchans = size(epoched_data.LFP_trial_avg{1}, 1); 
selchans = 1:numchans;

% Define number of conditions
num_conditions = length(epoched_data.LFP);

if seeCSD == 1
% Define the number of rows and columns for the subplots
% We can display multiple conditions per figure, splitting them into rows and columns
max_rows_per_figure = 5;  % Maximum number of rows per figure
num_cols_per_condition = 2; % Two subplots per condition (CSD + MUA)
num_plots_per_figure = max_rows_per_figure * num_cols_per_condition; % Maximum number of plots per figure



% Loop through all conditions and plot CSD and MUA using separate figures as needed
for cond_idx = 1:num_conditions

    if mod(cond_idx - 1, max_rows_per_figure) == 0
        % Create new figure if necessary (every max_rows_per_figure conditions)
        figure;
    end

    % Calculate the row index for the subplot
    row_in_fig = mod(cond_idx - 1, max_rows_per_figure) + 1;
    col_offset = (row_in_fig - 1) * num_cols_per_condition;

    % Extract data for the current condition
    csd = epoched_data.CSD_trial_avg{cond_idx};
    mua = epoched_data.MUA_trial_avg{cond_idx};

   

    % Determine Color Axis limits
    csd_min = min(min(csd));
    csd_max = max(max(csd));
    csd_caxis = [-max(abs([csd_min, csd_max])) * 0.7, max(abs([csd_min, csd_max])) * 0.7];

    mua_min = min(min(mua));
    mua_max = max(max(mua));
    mua_caxis = [-max(abs([mua_min, mua_max])) * 0.7, max(abs([mua_min, mua_max])) * 0.7];

    % Plot CSD (left column for current condition)
    subplot(max_rows_per_figure, num_cols_per_condition, col_offset + 1);
    imagesc(time_axis, 1:size(csd, 1), -csd); % Inverted CSD
    caxis(csd_caxis); % Set symmetric color limits
    title(['CSD for Condition ' num2str(cond_idx)]);
    xlabel('Time (ms)');
    ylabel('Channels');
    colormap(gca, 'jet'); % Set jet colormap
    colorbar;

    % Plot MUA (right column for current condition)
    subplot(max_rows_per_figure, num_cols_per_condition, col_offset + 2);
    imagesc(time_axis, 1:size(mua, 1), mua);
    caxis(mua_caxis);
    title(['MUA for Condition ' num2str(cond_idx)]);
    xlabel('Time (ms)');
    ylabel('Channels');
    colormap(gca, 'hot'); % Set hot colormap
    colorbar;
end
end


%% Decoding stimulus conditions over time
if ~exist(figuresdir, 'dir')
    mkdir(figuresdir);
end

% Initialize progress bar for the main decoding loop
h = waitbar(0, 'Decoding in progress...');
total_steps = length(selchans) * length(bins); % Total number of iterations
current_step = 0;

for bin_idx = 1:length(bins)
    window_size = bins(bin_idx);
    window_size_samples = window_size / 1000 * srate;
    %num_windows = floor(length(time_axis_decoding) / window_size_samples);
    % Create a new time axis based on bin size
    %window_time_axis = 0:window_size:epoch_tframe(2);

    positive_duration = epoch_tframe(2); % Only positive portion, 0 to end
    num_windows = floor(positive_duration / window_size);
    window_time_axis = 0 + (0:num_windows-1) * window_size;



    decoding_accuracy_windows = zeros(length(selchans), num_windows);
    decoding_accuracy_conditions = zeros(length(selchans), num_conditions, num_windows);
    decoding_accuracy_cis = zeros(length(selchans), num_windows, 2); % 95% CIs
    confusion_matrices_all = zeros(length(selchans), num_windows, num_conditions, num_conditions);

    % Prepare the data for decoding
    for ch_idx = 1:length(selchans)
        ch = selchans(ch_idx);
            % Increment step counter
            current_step = current_step + 1;

            % Update waitbar
            progress = current_step / total_steps;
            waitbar(progress, h, sprintf('Decoding in progress... %.1f%%', progress * 100));
        for w = 1:num_windows
            window_start = (w - 1) * window_size_samples + 1;
           window_end = min(window_start + window_size_samples - 1, length(time_axis)); % force window end to fit?


            flattened_data_window = [];
            flattened_labels_window = [];

            for cond_idx = 1:num_conditions
                if csd_flag == 1
                    csd = epoched_data.CSD{cond_idx}(ch, :, :);
                else
                    csd = epoched_data.MUA{cond_idx}(ch, :, :);
                end
                num_trials = size(csd, 2);

                for trial = 1:num_trials
                    trial_data_window = csd(:, trial, window_start:window_end);
                    flattened_data_window = [flattened_data_window; trial_data_window(:)'];
                    flattened_labels_window = [flattened_labels_window; cond_idx];
                end
            end

            if ~isa(flattened_data_window, 'double')
                flattened_data_window = double(flattened_data_window);
            end

            decoding_accuracy = zeros(4, 1);
            confusion_matrices_folds = zeros(4, num_conditions, num_conditions);

            cv = cvpartition(flattened_labels_window, 'KFold', 4);
            for fold = 1:cv.NumTestSets
                train_idx = cv.training(fold);
                test_idx = cv.test(fold);

                train_data = flattened_data_window(train_idx, :);
                test_data = flattened_data_window(test_idx, :);
                train_labels = flattened_labels_window(train_idx);
                test_labels = flattened_labels_window(test_idx);

                template = templateSVM('Standardize', true);
                model = fitcecoc(train_data, train_labels, 'Learners', template);

                predicted_labels = predict(model, test_data);
                decoding_accuracy(fold) = mean(predicted_labels == test_labels);

                for i = 1:length(test_labels)
                    confusion_matrices_folds(fold, test_labels(i), predicted_labels(i)) = ...
                        confusion_matrices_folds(fold, test_labels(i), predicted_labels(i)) + 1;
                end
            end

            avg_confusion_matrix = squeeze(mean(confusion_matrices_folds, 1));
            confusion_matrices_all(ch_idx, w, :, :) = avg_confusion_matrix;
            decoding_accuracy_windows(ch_idx, w) = mean(decoding_accuracy);

            for cond_idx = 1:num_conditions
                decoding_accuracy_conditions(ch_idx, cond_idx, w) = ...
                    avg_confusion_matrix(cond_idx, cond_idx) / sum(avg_confusion_matrix(cond_idx, :));
            end

            decoding_accuracy_cis(ch_idx, w, :) = prctile(decoding_accuracy, [2.5 97.5]);
        end
    end

    %% Empirically calculate chance performance via shuffling labels
    tic
    if bin_idx == 1
        % Define a fixed window size for the permutation analysis
        perm_window_size = 10; % ms
        perm_window_size_samples = perm_window_size / 1000 * srate; % Convert to samples
        perm_num_windows = floor(length(time_axis) / perm_window_size_samples); % Number of windows 
        perm_window_time_axis = epoch_tframe(1) + (0:perm_num_windows-1) * perm_window_size;

        num_permutations = 100;
        permutation_accuracies = zeros(num_permutations,numchans,perm_num_windows);
        
        % reinitialize flattened windows for perm analysis
        perm_flattened_labels_window = [];
        for cond_idx = 1:num_conditions
            if csd_flag == 1
                csd = epoched_data.CSD{cond_idx};
            else
                csd = epoched_data.MUA{cond_idx};
            end
            num_trials = size(csd, 2);
            perm_flattened_labels_window = [perm_flattened_labels_window; repmat(cond_idx, num_trials, 1)];
        end

        for perm = 1:num_permutations
            % Shuffle the labels
            shuffled_labels_window = perm_flattened_labels_window(randperm(length(perm_flattened_labels_window)));
            decoding_accuracy_permuted = zeros(length(selchans), perm_num_windows);
    
            % Loop through each channel
            for ch_idx = 1:length(selchans)
                ch = selchans(ch_idx);
    
                % Loop through each time window 
                for w = 1:perm_num_windows
                    perm_window_start = (w - 1) * perm_window_size_samples + 1;
                    
                    perm_window_end = min(perm_window_start + perm_window_size_samples - 1, length(time_axis));

                    flattened_data_window = [];
                    % Extract and flatten data for the current window and channel
                    for cond_idx = 1:num_conditions
                        if csd_flag == 1
                            csd = epoched_data.CSD{cond_idx}(ch, :, :); % Use CSD data
                        else
                            csd = epoched_data.MUA{cond_idx}(ch, :, :); % Use MUA data
                        end
                        num_trials = size(csd, 2);
    
                        for trial = 1:num_trials
                            trial_data_window = csd(:, trial, perm_window_start:perm_window_end); % Extract data for window
                            flattened_data_window = [flattened_data_window; trial_data_window(:)']; % Flatten
                        end
                    end
    
                    % Ensure data is double for model training
                    if ~isa(flattened_data_window, 'double')
                        flattened_data_window = double(flattened_data_window);
                    end
    
                    % Perform 4-fold cross-validation decoding
                    decoding_accuracy = zeros(4, 1); % Initialize accuracy for each fold
                    cv = cvpartition(shuffled_labels_window, 'KFold', 4);
    
                    for fold = 1:cv.NumTestSets
                        train_idx = cv.training(fold);
                        test_idx = cv.test(fold);
    
                        train_data = flattened_data_window(train_idx, :);
                        test_data = flattened_data_window(test_idx, :);
                        train_labels = shuffled_labels_window(train_idx);
                        test_labels = shuffled_labels_window(test_idx);
    
                        % Train and test the model
                        template = templateSVM('Standardize', true);
                        model = fitcecoc(train_data, train_labels, 'Learners', template);
    
                        predicted_labels = predict(model, test_data);
                        decoding_accuracy(fold) = mean(predicted_labels == test_labels); % Store fold accuracy
                    end
    
                    decoding_accuracy_permuted(ch_idx, w) = mean(decoding_accuracy); % Store accuracy for this channel/window
                end

            end
    
            % Store the average decoding accuracy across 
            permutation_accuracies(perm,:, :) = decoding_accuracy_permuted;
        end
    
        % Calculate mean and confidence intervals for permutation accuracies
        
        mean_permutation_accuracy(:,:) = squeeze(mean(permutation_accuracies, 1));
        std_permutation_accuracy(:,:) = std(permutation_accuracies, [], 1);
        
    
        toc
    end



% Plotting Results
    f = figure;
    f.Position = [150 150 1800 700];

    % Compute bin centers for accurate x-axis representation
    num_windows = size(decoding_accuracy_windows, 2);
    bin_centers = (0:num_windows-1) * bins(bin_idx) + bins(bin_idx) / 2;

    % Plot Decoding Accuracy Over Time
    subplot(1, 3, 1);
    imagesc(bin_centers, 1:length(selchans), squeeze(decoding_accuracy_windows(:, :)));
    xlabel('Time (ms)');
    ylabel('Channels');
    title(['Decoding Accuracy over Time (Bin: ' num2str(bins(bin_idx)) ' ms)']);
    colorbar;

    % Plot Decoding Accuracy by Condition and Channel
    subplot(1, 3, 2);
    imagesc(1:num_conditions, 1:length(selchans), squeeze(mean(decoding_accuracy_conditions, 3)));
    xlabel('Conditions');
    ylabel('Channels');
    title(['Decoding Accuracy by Condition and Channel (Bin: ' num2str(bins(bin_idx)) ' ms)']);
    colorbar;
    
    % Plot Individual Channel Decoding
    if all(selected_channels <= numchans) % Validate selected channels
        colors = lines(length(selected_channels));
        for i = 1:length(selected_channels)
            subplot(length(selected_channels), 3, 3*i);  % Stack vertically

            ch = selected_channels(i);
            mean_accuracy = squeeze(decoding_accuracy_windows(ch, :));
            ci_lower = squeeze(decoding_accuracy_cis(ch, :, 1));
            ci_upper = squeeze(decoding_accuracy_cis(ch, :, 2));

            hold on;
            % Decoding accuracy and confidence intervals centered on bins
            plot(bin_centers, mean_accuracy, 'LineWidth', 1, 'Color', colors(i, :), 'HandleVisibility', 'off');
            plot(bin_centers, ci_lower, 'LineWidth', 1, 'LineStyle', ':', 'Color', colors(i, :), 'HandleVisibility', 'off');
            plot(bin_centers, ci_upper, 'LineWidth', 1, 'LineStyle', ':', 'Color', colors(i, :), 'HandleVisibility', 'off');
            
            % Permutation accuracy and confidence intervals
            plot(perm_window_time_axis, mean_permutation_accuracy(ch, :), 'r--', 'LineWidth', 2, 'DisplayName', 'Chance Level');
            fill([perm_window_time_axis, fliplr(perm_window_time_axis)], ...
                 [mean_permutation_accuracy(ch, :) + 2*std_permutation_accuracy(ch, :), ...
                  fliplr(mean_permutation_accuracy(ch, :) - 2*std_permutation_accuracy(ch, :))], ...
                 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            legend('Chance');


            % Add labels and titles
            xlabel('Time (ms)');
            ylabel('Decoding Accuracy');
            title(['Decoding - Channel ' num2str(ch) ' (Bin: ' num2str(bins(bin_idx)) ' ms)']);
            grid on;
            hold off;
        end
    else
        error('Selected channels exceed the number of available channels (%d).', numchans);
    end

% Save Figures and Results
saveas(f, fullfile(figuresdir, sprintf('decoding_bins_%dms.fig', bins(bin_idx))));
saveas(f, fullfile(figuresdir, sprintf('decoding_bins_%dms.jpg', bins(bin_idx))));
save(fullfile(figuresdir, sprintf('decoding_results_bin_%dms.mat', bins(bin_idx))), ...
    'decoding_accuracy_windows', 'decoding_accuracy_cis', ...
    'decoding_accuracy_conditions', 'decoding_accuracy_permuted', ...
    'epoched_data', 'confusion_matrices_all','mean_permutation_accuracy',...
    'std_permutation_accuracy', 'bins');



end