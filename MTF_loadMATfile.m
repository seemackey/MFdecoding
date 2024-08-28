function [epoched_data,srate] = MTF_loadMATfile(filedir, epoch_tframe)
    % input is a path to a directory with files that have @e (lfp), @m 
    % (mua) and .ev2 (stimulus) file extensions
    % 2nd input is the time frame in milliseconds we wish to epoch with
    
    %% Initialize variables
    lfp_data = [];
    mua_data = [];
    srate = [];
    
    % Load LFP data (@e)
    lfp_file = dir(fullfile(filedir, '*@e.mat'));
    if ~isempty(lfp_file)
        lfp_data_struct = load(fullfile(filedir, lfp_file.name));
        lfp_data = lfp_data_struct.data;
        srate = lfp_data_struct.srate;
    else
        error('LFP data file not found.');
    end

    % Load MUA data (@m)
    mua_file = dir(fullfile(filedir, '*@m.mat'));
    if ~isempty(mua_file)
        mua_data_struct = load(fullfile(filedir, mua_file.name));
        mua_data = mua_data_struct.data;
        % Assume srate is the same for both LFP and MUA
    else
        error('MUA data file not found.');
    end

    % Load EV2 file
    ev2_file = dir(fullfile(filedir, '*.ev2'));
    if isempty(ev2_file)
        error('EV2 file not found in the specified directory.');
    end

    % 
    ev2_data = load_ev2(fullfile(filedir, ev2_file.name));
    ev2_filename = ev2_file.name;

    % Extract event times and stimulus conditions
    event_times = ev2_data(:, 6) / srate;  % Convert from samples to seconds
    stimulus_conditions = ev2_data(:, 2);

    % Get unique stimulus conditions
    unique_conditions = unique(stimulus_conditions);
    
    %% Load the spreadsheet data
    % clumsy but seems the only way to do this
    spreadsheet_path = 'E:\MTF\ev2_subset.xlsx';  % Replace with actual path
    spreadsheet_data = readtable(spreadsheet_path);

    % Find modulation frequencies based on the EV2 filename
    row_idx = strcmp(spreadsheet_data.filename, ev2_filename);
    if any(row_idx)
        row_idx = find(row_idx, 1);  % Get the index of the matching row
        if ~isempty(spreadsheet_data.clicktrain{row_idx})
            ISIs = eval(spreadsheet_data.clicktrain{row_idx});
        elseif ~isempty(spreadsheet_data.samN{row_idx})
            ISIs = eval(spreadsheet_data.samN{row_idx});
        elseif ~isempty(spreadsheet_data.samT{row_idx})
            ISIs = eval(spreadsheet_data.samT{row_idx});
        else
            error('Modulation frequency information not found for the specified EV2 file.');
        end
    else
        error('Modulation frequency information not found for the specified EV2 file.');
    end

%%
    % Time window for epoching
    pre_event_samples = round(epoch_tframe(1) / 1000 * srate);
    post_event_samples = round(epoch_tframe(2) / 1000 * srate);
    epoch_length = post_event_samples - pre_event_samples;

    % Initialize output cell arrays
    epoched_data.LFP = cell(length(unique_conditions), 1);
    epoched_data.MUA = cell(length(unique_conditions), 1);
    epoched_data.CSD = cell(length(unique_conditions), 1);
    epoched_data.LFP_trial_avg = cell(length(unique_conditions), 1);
    epoched_data.MUA_trial_avg = cell(length(unique_conditions), 1);
    epoched_data.CSD_trial_avg = cell(length(unique_conditions), 1);
    epoched_data.srate = srate;
    
    % Loop through each unique condition and epoch data
    for cond_idx = 1:length(unique_conditions)
        cond = unique_conditions(cond_idx);
        condition_event_times = event_times(stimulus_conditions == cond);
        
        % Epoch LFP and MUA data
        epochs_lfp = [];
        epochs_mua = [];
        for event_idx = 1:length(condition_event_times)
            event_sample = round(condition_event_times(event_idx) * srate);
            epoch_start = event_sample + pre_event_samples;
            epoch_end = event_sample + post_event_samples;

            % Check bounds
            if epoch_start > 0 && epoch_end <= size(lfp_data, 2)
                epochs_lfp(:, event_idx, :) = lfp_data(:, epoch_start:epoch_end);
                epochs_mua(:, event_idx, :) = mua_data(:, epoch_start:epoch_end);
            end
        end

        % Baseline correction
        baseline_start = 1;
        baseline_end = abs(pre_event_samples);
        baseline_lfp = mean(epochs_lfp(:, :, baseline_start:baseline_end), 3);
        baseline_mua = mean(epochs_mua(:, :, baseline_start:baseline_end), 3);

        epochs_lfp = epochs_lfp - baseline_lfp;
        epochs_mua = epochs_mua - baseline_mua;

        % Store in cell arrays
        epoched_data.LFP{cond_idx} = epochs_lfp;
        epoched_data.MUA{cond_idx} = epochs_mua;

        % Compute CSD 
        epoched_data.CSD{cond_idx} = -diff(epochs_lfp, 2, 1); % CSD


        % Reject artifacts separately for LFP, CSD, and MUA
        % 
        [epoched_data.LFP{cond_idx}, ~] = MTF_rejectartifacts(epoched_data.LFP{cond_idx}, 'median', 3);
        [epoched_data.CSD{cond_idx}, ~] = MTF_rejectartifacts(epoched_data.CSD{cond_idx}, 'median', 3);
        [epoched_data.MUA{cond_idx}, ~] = MTF_rejectartifacts(epoched_data.MUA{cond_idx}, 'median', 3);



        %  trial averages 

        epoched_data.CSD_trial_avg{cond_idx} = squeeze(mean(epoched_data.CSD{cond_idx}, 2)); % mean CSD
        epoched_data.LFP_trial_avg{cond_idx} = squeeze(mean(epoched_data.LFP{cond_idx}(2:end-1,:,:), 2)); % mean LFP
        epoched_data.MUA_trial_avg{cond_idx} = squeeze(mean(epoched_data.MUA{cond_idx}(2:end-1,:,:), 2)); % mean MUA

        % make channels match CSD
        epoched_data.LFP{cond_idx} = epoched_data.LFP{cond_idx}(2:end-1,:,:); %  LFP
        epoched_data.MUA{cond_idx} = epoched_data.MUA{cond_idx}(2:end-1,:,:); %  MUA


    end
    ISI_map = [640, 320, 226, 160, 113, 80, 57, 40, 28, 20, 14, 10, 5];
    ISI_ms =ISI_map(ISIs);
    epoched_data.ISI_ms = ISI_ms;
end



function ev2_data = load_ev2(filename)
    % Load .ev2 file and return as matrix
    % Assuming .ev2 is a text file with space-separated values
    ev2_data = dlmread(filename);
end

function [cleaned_data, outlier_idx_unique] = MTF_rejectartifacts(data, method, threshold)
    % Artifact rejection function
    % data: Input data (channels x trials x time)
    % method: Method to identify outliers ('median' or 'mean')
    % threshold: Threshold for identifying outliers

    if nargin < 2
        method = 'median';  % Default method
    end
    
    if nargin < 3
        threshold = 3;  % Default threshold (z-score)
    end

    num_trials = size(data, 2);

    max_vals = zeros(1, num_trials);

    % Calculate the maximum mean absolute value for each trial
    for trial = 1:num_trials
        max_vals(trial) = max(mean(abs(squeeze(data(:, trial, :))), 1));
    end

    % Identify outliers based on the chosen method
    switch method
        case 'median'
            outlier_idx = abs(max_vals - median(max_vals)) > threshold * std(max_vals);
        case 'mean'
            outlier_idx = abs(max_vals - mean(max_vals)) > threshold * std(max_vals);
        otherwise
            error('Unsupported method. Use "median" or "mean".');
    end

    outlier_idx_unique = find(outlier_idx);

    % Remove outliers
    cleaned_data = data;
    cleaned_data(:, outlier_idx_unique, :) = [];

end

