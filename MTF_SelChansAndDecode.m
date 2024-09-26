% MTF analysis of core and parabelt data
% chase m 2024
clear;clc;close all;
 
%% load and epoch data
filedir = 'E:\MTF\hi02\017\';
epoch_tframe = [-50 500];
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



%%
% Prepare the data for decoding
flattened_data = [];
flattened_labels = [];

for cond_idx = 1:num_conditions
    csd = epoched_data.CSD{cond_idx}(selchans, :, :); % Select only the channels in selchans
    num_trials = size(csd, 2);

    % Flatten the data and construct labels
    for trial = 1:num_trials
        trial_data = reshape(csd(:, trial, :), [], 1); % Flatten the data for each trial
        flattened_data = [flattened_data; trial_data'];
        flattened_labels = [flattened_labels; cond_idx];
    end
end

% Convert flattened_data to double if it's not
if ~isa(flattened_data, 'double')
    flattened_data = double(flattened_data);
end

% Initialize variables for decoding accuracy
decoding_accuracy = zeros(num_conditions, 1);

% Perform cross-validation and decoding
cv = cvpartition(flattened_labels, 'KFold', 4);
for fold = 1:cv.NumTestSets
    train_idx = cv.training(fold);
    test_idx = cv.test(fold);

    train_data = flattened_data(train_idx, :);
    test_data = flattened_data(test_idx, :);
    train_labels = flattened_labels(train_idx);
    test_labels = flattened_labels(test_idx);

    % Create and train the model
    template = templateSVM('Standardize', true);
    model = fitcecoc(train_data, train_labels, 'Learners', template);

    % Test the model
    predicted_labels = predict(model, test_data);

    % Calculate decoding accuracy
    decoding_accuracy(fold) = mean(predicted_labels == test_labels);
end

% Calculate and display the average decoding accuracy
mean_decoding_accuracy = mean(decoding_accuracy);
disp(['Mean Decoding Accuracy: ', num2str(mean_decoding_accuracy)]);


%% Plot Decoding Accuracy
% Average accuracy across conditions for visualization
avg_decoding_accuracy = decoding_accuracy;

figure;
imagesc(time_axis, selchans, avg_decoding_accuracy);
colorbar;
xlabel('Time (ms)');
ylabel('Channels');
title('Average Decoding Accuracy Over Time');
