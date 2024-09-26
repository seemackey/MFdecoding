% MTF analysis of core and parabelt data
% chase m 2024
clear;clc;close all;
 
%% load and epoch data
filedir = 'E:\MTF\hi01\016';
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
selchans = 1:numchans;

% Define the number of rows and columns for the subplots
num_conditions = length(epoched_data.LFP);
num_rows = num_conditions; % One row per condition
num_cols = 2; % One column for CSD and one for MUA

% Define outlier maximum values
outlier_max_csd = 2500;
outlier_max_mua = 70;


while true
    figure
    % Loop through conditions and plot CSD and MUA with imagesc
    for cond_idx = 1:num_conditions
        % Extract data for the current condition
        csd = epoched_data.CSD_trial_avg{cond_idx}(selchans,:,:);
        mua = epoched_data.MUA_trial_avg{cond_idx}(selchans,:,:);

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
            mua_min_val = -outlier_max_mua*0.5;
        end

        % Plot CSD
        subplot(num_rows, num_cols, (cond_idx - 1) * num_cols + 1);
        imagesc(time_axis, 1:size(csd, 1), csd); % Average across trials
        caxis([-csd_max_val csd_max_val]); % Set color limits to be symmetric
        title(['CSD for Condition ' num2str(cond_idx)]);
        xlabel('Time (ms)');
        ylabel('Channels');
        ax1 = gca; % Get the current axes
        colormap(ax1, 'jet'); % Set the colormap for this subplot
        colorbar;

        % Plot MUA
        subplot(num_rows, num_cols, (cond_idx - 1) * num_cols + 2);
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
