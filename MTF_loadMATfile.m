function [epoched_data,srate] = MTF_loadMATfile(filedir, epoch_tframe)
    % input is a path to a directory with files that have @e (lfp), @m 
    % (mua) and .ev2 (stimulus) file extensions
    % 2nd input is the time frame in milliseconds we wish to epoch with
    

    %% Initialize
    lfp_data = [];
    mua_data = [];
    srate = [];

    % Load LFP (@oe)
    lfp_file = dir(fullfile(filedir, '*@oe.mat'));
    if ~isempty(lfp_file)
        lfp_struct = load(fullfile(filedir, lfp_file.name));
        lfp_data = lfp_struct.cnt;
        srate = lfp_struct.adrate;
    end

    % Load MUA (@om)
    mua_file = dir(fullfile(filedir, '*@om.mat'));
    if ~isempty(mua_file)
        mua_struct = load(fullfile(filedir, mua_file.name));
        mua_data = mua_struct.cnt;
        if isempty(srate)
            srate = mua_struct.adrate;
        end
    end

    if isempty(lfp_data) && isempty(mua_data)
        error('No LFP or MUA data found in directory: %s', filedir);
    end

    % Load EV2
    ev2_file = dir(fullfile(filedir, '*.ev2'));
    if isempty(ev2_file)
        error('No EV2 file found in directory: %s', filedir);
    end
    ev2_data = load_ev2(fullfile(filedir, ev2_file.name));
    event_times = ev2_data(:, 6) / srate;
    stimulus_conditions = ev2_data(:, 2);
    unique_conditions = unique(stimulus_conditions);

    %% Epoching
    pre_samp = round(epoch_tframe(1) / 1000 * srate);
    post_samp = round(epoch_tframe(2) / 1000 * srate);
    epoched_data.srate = srate;

    % Initialize output
    for cond_idx = 1:length(unique_conditions)
        cond = unique_conditions(cond_idx);
        event_times_cond = event_times(stimulus_conditions == cond);

        epochs_lfp = [];
        epochs_mua = [];

        for e = 1:length(event_times_cond)
            sample_idx = round(event_times_cond(e) * srate);
            epoch_start = sample_idx + pre_samp;
            epoch_end = sample_idx + post_samp;

            if epoch_start <= 0
                continue;
            end

            % LFP
            if ~isempty(lfp_data) && epoch_end <= size(lfp_data, 2)
                epochs_lfp(:, end+1, :) = lfp_data(:, epoch_start:epoch_end); %#ok<AGROW>
            end

            % MUA
            if ~isempty(mua_data) && epoch_end <= size(mua_data, 2)
                epochs_mua(:, end+1, :) = mua_data(:, epoch_start:epoch_end); %#ok<AGROW>
            end
        end

        % Store and baseline-correct
        if ~isempty(epochs_lfp)
            baseline = mean(epochs_lfp(:, :, 1:abs(pre_samp)), 3);
            epochs_lfp = epochs_lfp - baseline;
            epoched_data.LFP{cond_idx} = epochs_lfp;
            epoched_data.CSD{cond_idx} = -diff(epochs_lfp, 2, 1);
        end

        if ~isempty(epochs_mua)
            baseline = mean(epochs_mua(:, :, 1:abs(pre_samp)), 3);
            epochs_mua = epochs_mua - baseline;
            epoched_data.MUA{cond_idx} = epochs_mua;
        end
    end

    
    %% Trial averages + artifact rejection
    for cond_idx = 1:length(unique_conditions)
        % ----- LFP -----
        has_lfp = isfield(epoched_data, 'LFP') && ...
                  cond_idx <= length(epoched_data.LFP) && ...
                  ~isempty(epoched_data.LFP{cond_idx});
        if has_lfp
            data = epoched_data.LFP{cond_idx};
            [data, ~] = MTF_rejectartifacts(data);
            if size(data,1) > 2
                data = data(2:end-1,:,:);
            end
            epoched_data.LFP{cond_idx} = data;
            epoched_data.LFP_trial_avg{cond_idx} = squeeze(mean(data, 2));
        else
            epoched_data.LFP{cond_idx} = [];
            epoched_data.LFP_trial_avg{cond_idx} = [];
        end
    
        % ----- CSD -----
        has_csd = isfield(epoched_data, 'CSD') && ...
                  cond_idx <= length(epoched_data.CSD) && ...
                  ~isempty(epoched_data.CSD{cond_idx});
        if has_csd
            data = epoched_data.CSD{cond_idx};
            [data, ~] = MTF_rejectartifacts(data);
            epoched_data.CSD{cond_idx} = data;
            epoched_data.CSD_trial_avg{cond_idx} = squeeze(mean(data, 2));
        else
            epoched_data.CSD{cond_idx} = [];
            epoched_data.CSD_trial_avg{cond_idx} = [];
        end
    
        % ----- MUA -----
        has_mua = isfield(epoched_data, 'MUA') && ...
                  cond_idx <= length(epoched_data.MUA) && ...
                  ~isempty(epoched_data.MUA{cond_idx});
        if has_mua
            data = epoched_data.MUA{cond_idx};
            [data, ~] = MTF_rejectartifacts(data);
            if size(data,1) > 2
                data = data(2:end-1,:,:);
            end
            epoched_data.MUA{cond_idx} = data;
            epoched_data.MUA_trial_avg{cond_idx} = squeeze(mean(data, 2));
        else
            epoched_data.MUA{cond_idx} = [];
            epoched_data.MUA_trial_avg{cond_idx} = [];
        end
    end
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

