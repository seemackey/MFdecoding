% MTF analysis of core and parabelt data
% chase m 2024
clear;clc;close all;
 
%% load and epoch data
filedir = 'E:\MTF\hi02\017\';
figuresdir = '';
bins = [1,5,10,50,100,500]; % in ms
selected_channels = [8,11,14]; % for decoding example traces
epoch_tframe = [-30 750];
tic
[epoched_data,srate] = MTF_loadMATfile(filedir, epoch_tframe);
toc
%
%function [] = MTF_DecodingFxn(epoched_data,window_size,epoch_tframe,selected_channels,figuresdir)
%% plot csd and mua
% Define time axis for plotting
time_axis = epoch_tframe(1):1000/srate:epoch_tframe(2);

% select channels
tmp = size(epoched_data.LFP_trial_avg{1}(:,1,1));
numchans = tmp(1);
selchans = numchans;

% Define the number of rows and columns for the subplots
num_conditions = length(epoched_data.LFP);
num_rows = num_conditions; % One row per condition
num_cols = 2; % One column for CSD and one for MUA

% Define outlier maximum values
outlier_max_csd = 2500;
outlier_max_mua = 100;

% Define maximum number of rows and columns per figure
max_rows_per_figure = 5;  % Maximum number of rows per figure
num_cols_per_subplot = 2;  % Number of columns per condition (one for CSD and one for MUA)

% Calculate the total number of rows and columns needed
num_conditions = length(epoched_data.LFP);
num_rows = min(max_rows_per_figure, num_conditions);
num_cols = ceil(num_conditions / num_rows) * num_cols_per_subplot;


    figure;
    % Loop through conditions and plot CSD and MUA with imagesc
    for cond_idx = 1:num_conditions
        % Extract data for the current condition
        csd = epoched_data.CSD_trial_avg{cond_idx}(:, :, :);
        mua = epoched_data.MUA_trial_avg{cond_idx}(:, :, :);

        % Determine the scaling limits based on the absolute maximum value
        csd_max_val = max(abs(csd(:)));
        mua_max_val = max(mua(:));
        mua_min_val = min(mua(:));

        % Adjust CSD scaling limits if they exceed the outlier threshold
        if csd_max_val > outlier_max_csd
            csd_max_val = outlier_max_csd;
        end

        % Adjust MUA scaling limits if they exceed the outlier threshold
        if mua_max_val > outlier_max_mua
            mua_max_val = outlier_max_mua;
        end

        if mua_min_val < -outlier_max_mua
            mua_min_val = -outlier_max_mua * 0.5;
        end

        % Calculate the subplot index for CSD and MUA
        subplot_idx_csd = (cond_idx - 1) * num_cols_per_subplot + 1;
        subplot_idx_mua = (cond_idx - 1) * num_cols_per_subplot + 2;

        % Plot CSD
        subplot(num_rows, num_cols, subplot_idx_csd);
        imagesc(time_axis, 1:size(csd, 1), -csd); % Average across trials
        caxis([-csd_max_val csd_max_val]); % Set color limits to be symmetric
        title(['CSD for Condition ' num2str(cond_idx)]);
        xlabel('Time (ms)');
        ylabel('Channels');
        ax1 = gca; % Get the current axes
        colormap(ax1, 'jet'); % Set the colormap for this subplot
        colorbar;

        % Plot MUA
        subplot(num_rows, num_cols, subplot_idx_mua);
        imagesc(time_axis, 1:size(mua, 1), mua); % Average across trials
        caxis([mua_min_val mua_max_val]); % Set color limits to be symmetric
        title(['MUA for Condition ' num2str(cond_idx)]);
        xlabel('Time (ms)');
        ylabel('Channels');
        ax1 = gca; % Get the current axes
        colormap(ax1, 'hot'); % Set the colormap for this subplot
        colorbar;
    end

% save the figure to the newly created figures directory as
% "filename_profiles%


%% Decoding stimulus conditions over time
bins = [1,5,10,50,100,500]; % in ms
for bin_idx = 1:length(bins)
% Define the time window size (50 ms)
window_size = bins(bin_idx);
window_size_samples = window_size / 1000 * srate; % Convert to samples
num_windows = floor(length(time_axis) / window_size_samples); % Number of windows

% Initialize variables
decoding_accuracy_windows = zeros(length(bins),length(selchans), num_windows);
decoding_accuracy_conditions = zeros(length(bins),length(selchans), num_conditions, num_windows);
decoding_accuracy_cis = zeros(length(bins),length(selchans), num_windows, 2); % 95% CIs

% Initialize storage for confusion matrices
confusion_matrices_all = zeros(length(bins),length(selchans), num_windows, num_conditions, num_conditions); % 4D array

    % Prepare the data for decoding
    for ch_idx = 1:length(selchans)
        ch = selchans(ch_idx);
        
        % Loop through each time window
        for w = 1:num_windows
            window_start = (w-1) * window_size_samples + 1;
            window_end = window_start + window_size_samples - 1;
            
            flattened_data_window = [];
            flattened_labels_window = [];
            
            % Extract and flatten data for the current window and channel
            for cond_idx = 1:num_conditions
                csd = epoched_data.CSD{cond_idx}(ch, :, :); % Select data for the current channel
                num_trials = size(csd, 2);
                
                for trial = 1:num_trials
                    trial_data_window = csd(:, trial, window_start:window_end); % Extract data for each window
                    flattened_data_window = [flattened_data_window; trial_data_window(:)'];
                    flattened_labels_window = [flattened_labels_window; cond_idx];
                end
            end
            
            % Convert flattened_data_window to double if it's not
            if ~isa(flattened_data_window, 'double')
                flattened_data_window = double(flattened_data_window);
            end
            
            % Initialize variables for decoding accuracy and confusion matrices
            decoding_accuracy = zeros(4, 1); % 4-fold cross-validation
            confusion_matrices_folds = zeros(4, num_conditions, num_conditions); % Initialize confusion matrix for each fold
            
            % Perform cross-validation and decoding
            cv = cvpartition(flattened_labels_window, 'KFold', 4);
            for fold = 1:cv.NumTestSets
                train_idx = cv.training(fold);
                test_idx = cv.test(fold);
                
                train_data = flattened_data_window(train_idx, :);
                test_data = flattened_data_window(test_idx, :);
                train_labels = flattened_labels_window(train_idx);
                test_labels = flattened_labels_window(test_idx);
                
                % Create and train the model
                template = templateSVM('Standardize', true);
                model = fitcecoc(train_data, train_labels, 'Learners', template);
                
                % Test the model
                predicted_labels = predict(model, test_data);
                
                % Calculate decoding accuracy
                decoding_accuracy(fold) = mean(predicted_labels == test_labels);
                
                % Update confusion matrix for current fold
                fold_confusion_matrix = zeros(num_conditions, num_conditions);
                for i = 1:length(test_labels)
                    fold_confusion_matrix(test_labels(i), predicted_labels(i)) = ...
                        fold_confusion_matrix(test_labels(i), predicted_labels(i)) + 1;
                end
                
                % Store confusion matrix for the current fold
                confusion_matrices_folds(fold, :, :) = fold_confusion_matrix;
            end
            
            % Average confusion matrix across folds for the current window and channel
            avg_confusion_matrix = squeeze(mean(confusion_matrices_folds, 1)); 
            confusion_matrices_all(ch_idx, w, :, :) = avg_confusion_matrix; % Store the averaged confusion matrix
            
            % Calculate and store the average decoding accuracy for the current window
            decoding_accuracy_windows(ch_idx, w) = mean(decoding_accuracy);
            
            % Calculate accuracy for each condition and store it for the current window
            for cond_idx = 1:num_conditions
                decoding_accuracy_conditions(ch_idx, cond_idx, w) = ...
                    avg_confusion_matrix(cond_idx, cond_idx) / sum(avg_confusion_matrix(cond_idx, :));
            end
            
            % Calculate the 95% CI using the percentile method
            decoding_accuracy_cis(ch_idx, w, :) = prctile(decoding_accuracy, [2.5 97.5]);
        end
    end
end


% Identify the best condition for each channel considering all windows
best_conditions = zeros(length(selchans), 1);
for ch_idx = 1:length(selchans)
    avg_accuracy_conditions = mean(decoding_accuracy_conditions(ch_idx, :, :), 3);
    [~, best_conditions(ch_idx)] = max(avg_accuracy_conditions);
end

% Create a new time axis for the windows
window_time_axis = linspace(epoch_tframe(1), epoch_tframe(2), num_windows);



%% empirically calculate chance via shuffling decoding labels
num_permutations = 5; 

window_size = bins(bin_idx);
window_size_samples = window_size / 1000 * srate; % Convert to samples
num_windows = floor(length(time_axis) / window_size_samples); % Number of windows

% Initialize a matrix to store decoding accuracies for each permutation
permutation_accuracies = zeros(num_permutations, 50);

% Perform the permutation test
for perm = 1:num_permutations
    % Shuffle the labels
    shuffled_labels_window = flattened_labels_window(randperm(length(flattened_labels_window)));
    
    % Initialize variables for decoding accuracy across channels
    decoding_accuracy_permuted = zeros(length(selchans), 50);
    
    % Loop through channels and windows
    for ch_idx = 1:length(selchans)
        ch = selchans(ch_idx);
        
        for w = 1:num_windows
            window_start = (w-1) * window_size_samples + 1;
            window_end = window_start + window_size_samples - 1;
            
            flattened_data_window = [];
            
            % Extract and flatten data for the current window and channel
            for cond_idx = 1:num_conditions
                csd = epoched_data.CSD{cond_idx}(ch, :, :); % Select data for the current channel
                num_trials = size(csd, 2);
                
                for trial = 1:num_trials
                    trial_data_window = csd(:, trial, window_start:window_end); % Extract data for each window
                    flattened_data_window = [flattened_data_window; trial_data_window(:)'];
                end
            end
            
            % Convert to double if necessary
            if ~isa(flattened_data_window, 'double')
                flattened_data_window = double(flattened_data_window);
            end
            
            % Initialize variables for decoding accuracy
            decoding_accuracy = zeros(4, 1); % 4-fold cross-validation
            
            % Perform cross-validation and decoding
            cv = cvpartition(shuffled_labels_window, 'KFold', 4);
            for fold = 1:cv.NumTestSets
                train_idx = cv.training(fold);
                test_idx = cv.test(fold);
                
                train_data = flattened_data_window(train_idx, :);
                test_data = flattened_data_window(test_idx, :);
                train_labels = shuffled_labels_window(train_idx);
                test_labels = shuffled_labels_window(test_idx);
                
                % Create and train the model
                template = templateSVM('Standardize', true);
                model = fitcecoc(train_data, train_labels, 'Learners', template);
                
                % Test the model
                predicted_labels = predict(model, test_data);
                
                % Calculate decoding accuracy
                decoding_accuracy(fold) = mean(predicted_labels == test_labels);
            end
            
            % Calculate and store the average decoding accuracy for the current window
            decoding_accuracy_permuted(ch_idx, w) = mean(decoding_accuracy);
        end
    end
    
    % Store the average decoding accuracy across channels for this permutation
    permutation_accuracies(perm, :) = mean(decoding_accuracy_permuted, 1);
end

% Calculate the mean and confidence intervals for the permutation accuracies
mean_permutation_accuracy = mean(permutation_accuracies, 1);
std_permutation_accuracy = std(permutation_accuracies, [], 1);

% Determine the 95th percentile as the empirical chance level
chance_level = prctile(permutation_accuracies, 95);



%% plots

% Display decoding accuracy over time
f = figure;
f.Position = [150 150 1800 500];
subplot(1,3,1)
imagesc(window_time_axis, 1:length(selchans), decoding_accuracy_windows);
xlabel('Time (ms)');
ylabel('Channels');
title('Decoding Accuracy over Time');
colorbar;

% Display decoding accuracy by condition and channel
subplot(1,3,2)
imagesc(1:num_conditions, 1:length(selchans), mean(decoding_accuracy_conditions, 3));
xlabel('Conditions');
ylabel('Channels');
title('Decoding Accuracy by Condition and Channel');
colorbar;


% Subplot for decoding accuracy over time with CIs for selected channels
subplot(1, 3, 3);
hold on;


colors = lines(length(selected_channels)); % Color scheme for different channels
% Loop through each selected channel
for i = 1:length(selected_channels)
    ch = selected_channels(i);
    
    % Calculate 95% confidence intervals for the decoding accuracy over time
    mean_accuracy = decoding_accuracy_windows(ch, :);
    ci_lower = decoding_accuracy_cis(ch, :, 1); % Lower 95% CI
    ci_upper = decoding_accuracy_cis(ch, :, 2); % Upper 95% CI
    
    % Plot mean decoding accuracy
    plot(window_time_axis, mean_accuracy, 'LineWidth', 2, 'Color', colors(i, :), 'DisplayName', ['Channel ' num2str(ch)]);
    
    % Plot shaded area for 95% CIs
    fill([window_time_axis, fliplr(window_time_axis)], [ci_lower, fliplr(ci_upper)], colors(i, :), 'FaceAlpha', 0.2, 'EdgeColor', 'none','HandleVisibility', 'off');
end

% Plot the permuted chance level line
plot(window_time_axis, mean_permutation_accuracy, 'r--', 'LineWidth', 2, 'DisplayName', 'Chance Level');

% Plot shaded area for permutation chance level 95% CIs
fill([window_time_axis, fliplr(window_time_axis)], ...
     [mean_permutation_accuracy + 2*std_permutation_accuracy, fliplr(mean_permutation_accuracy - 2*std_permutation_accuracy)], ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none','HandleVisibility', 'off');


xlabel('Time (ms)');
ylabel('Decoding Accuracy');
title('Decoding Accuracy for Selected Channels');
legend('show');
grid on;
hold off;

% save decoding figure as .fig and .jpg in newly created figures directory as
% filename_decoding

% save decoding results
% decoding_accuracy, decoding_accuracy_cis, decoding_accuracy_conditions
% decoding_accuracy_permuted, decoding_accuracy_windows, epoched_data,