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




%% calc CIs and plot timing of sig activation


% Define the baseline indices (time points) for calculating baseline CIs
baseline_indices = time_axis < 0;

% Initialize variables for significant activation times
csd_sig = cell(num_conditions, 1);
mua_sig = cell(num_conditions, 1);

% Loop through each condition to calculate and compare CIs
for cond_idx = 1:num_conditions
    % Extract data for the current condition
    csd = epoched_data.CSD{cond_idx}(selchans, :, :);
    mua = epoched_data.MUA{cond_idx}(selchans, :, :);
    
    % Calculate percentile CIs for CSD and MUA
    csd_ci = MTF_percentile_CI(csd, 0.90);
    mua_ci = MTF_percentile_CI(mua, 0.90);
    
    % Calculate baseline mean and CIs for the pre-stimulus period 
    baseline_csd_upper = mean(csd_ci.upper(:,baseline_indices),2);
    baseline_csd_lower = mean(csd_ci.lower(:,baseline_indices),2);
    
    baseline_mua_upper = mean(mua_ci.upper(:,baseline_indices),2);
    baseline_mua_lower = mean(mua_ci.lower(:,baseline_indices),2);
    
    % Determine significant activation times for CSD
%    csd_sig{cond_idx} = (csd_ci.upper < 0 | csd_ci.lower > 0);
    
    % Determine significant activation times for MUA
    mua_sig{cond_idx} = (mua_ci.upper < 0 | mua_ci.lower > 0);
    
%         % Determine significant activation times for CSD
     csd_sig{cond_idx} = (csd_ci.upper < baseline_csd_lower | csd_ci.lower > baseline_csd_upper);
%     
%     % Determine significant activation times for MUA
%     mua_sig{cond_idx} = (mua_ci.upper < baseline_mua_lower | mua_ci.lower > baseline_mua_upper);
end

%%
% Plotting the significant activation times
figure;
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
    
    % Overlay significant activation times
    hold on;
    [sig_ch, sig_time] = find(csd_sig{cond_idx});
    plot(time_axis(sig_time), sig_ch, 'o', 'MarkerSize', 4,'Color',[0.5 0 0.8]);
    hold off;
    
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
    
    % Overlay significant activation times
    hold on;
    [sig_ch, sig_time] = find(mua_sig{cond_idx});
    plot(time_axis(sig_time), sig_ch, 'o', 'MarkerSize', 4,'Color',[1 1 1]);
    hold off;
end

%% Calculate new significant times for Vector Strength analysis

% Initialize the vector strength matrices
vs_csd = nan(num_conditions, numchans);
vs_mua = nan(num_conditions, numchans);

% Loop through each condition to calculate and compare CIs
for cond_idx = 1:num_conditions
    % Extract data for the current condition
    csd = epoched_data.CSD{cond_idx}(selchans, :, :);
    mua = epoched_data.MUA{cond_idx}(selchans, :, :);
    
    % Calculate percentile CIs for CSD and MUA
    csd_ci = MTF_percentile_CI(csd, 0.99);
    mua_ci = MTF_percentile_CI(mua, 0.99);
    
    % Calculate baseline mean and CIs for the pre-stimulus period 
    baseline_csd_upper = mean(csd_ci.upper(:,baseline_indices),2);
    baseline_csd_lower = mean(csd_ci.lower(:,baseline_indices),2);
    
    baseline_mua_upper = mean(mua_ci.upper(:,baseline_indices),2);
    baseline_mua_lower = mean(mua_ci.lower(:,baseline_indices),2);
    
    % Initialize variables to store significant times for CSD and MUA
    sig_times_csd = [];
    sig_times_mua = [];

    % Determine significant activation times for CSD and store them
    for ch = 1:size(csd, 1)
        for trial = 1:size(csd, 2)
            for t = 1:size(csd, 3)
                if csd(ch, trial, t) < baseline_csd_lower(ch) || csd(ch, trial, t) > baseline_csd_upper(ch)
                    sig_times_csd = [sig_times_csd; [ch, trial, time_axis(t)]];
                end
            end
        end
    end

    % Determine significant activation times for MUA and store them
    for ch = 1:size(mua, 1)
        for trial = 1:size(mua, 2)
            for t = 1:size(mua, 3)
                if mua(ch, trial, t) < baseline_mua_lower(ch) || mua(ch, trial, t) > baseline_mua_upper(ch)
                    sig_times_mua = [sig_times_mua; [ch, trial, time_axis(t)]];
                end
            end
        end
    end

    % Calculate vector strength for CSD
    mod_freq = epoched_data.modfreqs(cond_idx);
    for ch = 1:numchans
        ch_sig_times_csd = sig_times_csd(sig_times_csd(:,1) == ch, :);
        if ~isempty(ch_sig_times_csd)
            phase_csd = 2 * pi * mod_freq * ch_sig_times_csd(:, 3);
            x_csd = cos(phase_csd);
            y_csd = sin(phase_csd);
            vs_csd(cond_idx, ch) = sqrt(sum(x_csd)^2 + sum(y_csd)^2) / size(ch_sig_times_csd, 1);
        else
            vs_csd(cond_idx, ch) = NaN; % Assign NaN if no significant times
        end
    end
    
    % Calculate vector strength for MUA
    for ch = 1:numchans
        ch_sig_times_mua = sig_times_mua(sig_times_mua(:,1) == ch, :);
        if ~isempty(ch_sig_times_mua)
            phase_mua = 2 * pi * mod_freq * ch_sig_times_mua(:, 3);
            x_mua = cos(phase_mua);
            y_mua = sin(phase_mua);
            vs_mua(cond_idx, ch) = sqrt(sum(x_mua)^2 + sum(y_mua)^2) / size(ch_sig_times_mua, 1);
        else
            vs_mua(cond_idx, ch) = NaN; % Assign NaN if no significant times
        end
    end
end

% Plotting the vector strength results
figure;
plotchans = [8:2:14];
for ch = 1:numchans
    % Plot CSD Vector Strength
    subplot(1, 2, 1);
    hold on;
    semilogx(epoched_data.modfreqs, vs_csd(:, ch), '-o', 'DisplayName', ['Channel ' num2str(ch)]);
    title('Vector Strength for CSD');
    xlabel('Modulation Frequency (Hz)');
    ylabel('Vector Strength');
    grid on;

    % Plot MUA Vector Strength
    subplot(1, 2, 2);
    hold on;
    semilogx(epoched_data.modfreqs, vs_mua(:, ch), '-o', 'DisplayName', ['Channel ' num2str(ch)]);
    title('Vector Strength for MUA');
    xlabel('Modulation Frequency (Hz)');
    ylabel('Vector Strength');
    grid on;
end
hold off;
legend;
