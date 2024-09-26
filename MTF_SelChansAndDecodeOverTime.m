% MTF analysis of core and parabelt data
% chase m 2024
clear;clc;close all;
 
%% load and epoch data
filedir = 'E:\MTF\hi02\017\';
epoch_tframe = [-30 500];
tic
[epoched_data,srate] = MTF_loadMATfile(filedir, epoch_tframe);
toc
%
%% plot csd and mua
% Define time axis for plotting
time_axis = epoch_tframe(1):1000/srate:epoch_tframe(2);

% select channels
tmp = size(epoched_data.LFP_trial_avg{1}(:,1,1));
numchans = tmp(1);
selchans = 1:21;

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

while true
    figure;
    % Loop through conditions and plot CSD and MUA with imagesc
    for cond_idx = 1:num_conditions
        % Extract data for the current condition
        csd = epoched_data.CSD_trial_avg{cond_idx}(selchans, :, :);
        mua = epoched_data.MUA_trial_avg{cond_idx}(selchans, :, :);

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

    % Prompt for user input
    new_selchans = input('Enter new channel selection as a vector (e.g., [2:18]), or press Enter to keep current selection and exit: ');

    % Check if the user wants to replot with a new selection or exit
    if isempty(new_selchans)
        break; % Exit the loop if no input is given
    else
        selchans = new_selchans; % Update selchans with the new selection
        close(gcf); % Close the current figure before replotting
    end
end


%% decoding stimulus conditions over time
% Define the time window size (50 ms)
window_size = 50; % in ms
window_size_samples = window_size / 1000 * srate; % convert to samples
num_windows = floor(length(time_axis) / window_size_samples); % number of windows


% Initialize variables
decoding_accuracy_windows = zeros(length(selchans), num_windows);

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
        
        % Initialize variables for decoding accuracy
        decoding_accuracy = zeros(4, 1); % 4-fold cross-validation
        
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
        end
        
        % Calculate and store the average decoding accuracy for the current window
        decoding_accuracy_windows(ch_idx, w) = mean(decoding_accuracy);
    end
end

% Create a new time axis for the windows
window_time_axis = linspace(epoch_tframe(1), epoch_tframe(2), num_windows);

% Display results
figure;
imagesc(window_time_axis, 1:length(selchans), decoding_accuracy_windows);
xlabel('Time (ms)');
ylabel('Channels');
title('Decoding Accuracy over Time');
colorbar;

%% decoding of individual conditions
% Initialize variables to store decoding accuracy for each condition and channel
decoding_accuracy_conditions = zeros(length(selchans), num_conditions);

% Perform decoding analysis as before, but also record confusion matrices
for ch_idx = 1:length(selchans)
    ch = selchans(ch_idx);
    for w = 1:num_windows
        window_start = (w-1) * window_size_samples + 1;
        window_end = window_start + window_size_samples - 1;
        flattened_data_window = [];
        flattened_labels_window = [];
        
        for cond_idx = 1:num_conditions
            csd = epoched_data.CSD{cond_idx}(ch, :, :);
            num_trials = size(csd, 2);
            for trial = 1:num_trials
                trial_data_window = csd(:, trial, window_start:window_end);
                flattened_data_window = [flattened_data_window; trial_data_window(:)'];
                flattened_labels_window = [flattened_labels_window; cond_idx];
            end
        end
        
        % Convert to double
        if ~isa(flattened_data_window, 'double')
            flattened_data_window = double(flattened_data_window);
        end
        
        cv = cvpartition(flattened_labels_window, 'KFold', 4);
        confusion_matrix = zeros(num_conditions, num_conditions);
        
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
            
            for i = 1:length(test_labels)
                confusion_matrix(test_labels(i), predicted_labels(i)) = ...
                    confusion_matrix(test_labels(i), predicted_labels(i)) + 1;
            end
        end
        
        % Calculate accuracy for each condition
        for cond_idx = 1:num_conditions
            decoding_accuracy_conditions(ch_idx, cond_idx) = ...
                confusion_matrix(cond_idx, cond_idx) / sum(confusion_matrix(cond_idx, :));
        end
    end
end

% Identify the best condition for each channel
best_conditions = zeros(length(selchans), 1);
for ch_idx = 1:length(selchans)
    [~, best_conditions(ch_idx)] = max(decoding_accuracy_conditions(ch_idx, :));
end

% Visualization
figure;
imagesc(1:length(selchans), 1:num_conditions, decoding_accuracy_conditions');
xlabel('Channels');
ylabel('Conditions');
title('Decoding Accuracy by Condition and Channel');
colorbar;

% Plot the best condition for each channel
figure;
bar(best_conditions);
xlabel('Channels');
ylabel('Best Decoded Condition');
title('Best Decoded Condition by Channel');
