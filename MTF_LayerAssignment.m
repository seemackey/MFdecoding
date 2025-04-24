%% MTF Assign Layers Script
% chase m 2025
clear; clc; close all;

% === CONFIGURATION ===
parent_dir = 'E:\MTF\data\noise\pb\right\imported\subset\';    
epoch_tframe = [-10 80];

% === Get all 1-*@oe.mat files (we will find matching om automatically) ===
oe_files = dir(fullfile(parent_dir, '1-*@oe*.mat'));
oe_files = [oe_files; dir(fullfile(parent_dir, '2-*@oe*.mat'))];

if isempty(oe_files)
    error('No @oe files found in directory!');
end

for i = 1:length(oe_files)
    oe_file = oe_files(i);
    [~, base_name, ~] = fileparts(oe_file.name);  % '1-filename@oe'

    fprintf('\n>> Processing %s\n', base_name);

    % Match the corresponding om file
    om_name = strrep(oe_file.name, '@oe', '@om');
    om_file = dir(fullfile(parent_dir, om_name));
    
    if isempty(om_file)
        warning('Matching @om file not found for %s. Skipping...', base_name);
        continue;
    end

    % Match the ev2 file (remove prefix and @oe suffix)


    ev2_base = regexprep(base_name, '^[12]-(.*)@oe$', '$1');
    ev2_file = dir(fullfile(parent_dir, [ev2_base '.ev2']));
    
    if isempty(ev2_file)
        warning('EV2 file not found for %s. Skipping...', base_name);
        continue;
    end

    % === Setup temp dir for isolated processing ===
    temp_dir = fullfile(parent_dir, ['temp_' ev2_base]);
    if exist(temp_dir, 'dir')
        rmdir(temp_dir, 's');
    end
    mkdir(temp_dir);

    % Copy files into temp dir
    copyfile(fullfile(oe_file.folder, oe_file.name), fullfile(temp_dir, oe_file.name));
    copyfile(fullfile(om_file.folder, om_file.name), fullfile(temp_dir, om_file.name));
    copyfile(fullfile(ev2_file.folder, ev2_file.name), fullfile(temp_dir, ev2_file.name));

    % === Load Data ===
    [epoched_data, srate] = MTF_loadMATfile(temp_dir, epoch_tframe);

    % === Plot CSD + MUA for each condition ===
    time_axis = epoch_tframe(1):1000/srate:epoch_tframe(2);
    numchans = size(epoched_data.LFP_trial_avg{1}, 1);
    num_conditions = length(epoched_data.LFP);

    figure('Position', [100 100 1800 1000]);
    max_rows = min(5, num_conditions);
    num_cols = ceil(num_conditions / max_rows) * 2;

    outlier_max_csd = 50;
    outlier_max_mua = 10;

    for cond_idx = 1:num_conditions
        csd = epoched_data.CSD_trial_avg{cond_idx};
        mua = epoched_data.MUA_trial_avg{cond_idx};

        csd_max = min(max(abs(csd(:)))*0.8, outlier_max_csd);
        mua_max = min(max(mua(:))*0.8, outlier_max_mua);
        mua_min = max(min(mua(:)), -outlier_max_mua * 0.5);

        subplot_idx_csd = (cond_idx - 1) * 2 + 1;
        subplot_idx_mua = (cond_idx - 1) * 2 + 2;

        subplot(max_rows, num_cols, subplot_idx_csd);
        imagesc(time_axis, 1:numchans, -csd);
        caxis([-csd_max csd_max]);
        colormap(gca, 'jet');
        colorbar;
        xlabel('Time (ms)'); ylabel('Channels');
        title(['CSD Cond ' num2str(cond_idx)]);

        subplot(max_rows, num_cols, subplot_idx_mua);
        imagesc(time_axis, 1:numchans, mua);
        caxis([mua_min mua_max]);
        colormap(gca, 'hot');
        colorbar;
        xlabel('Time (ms)'); ylabel('Channels');
        title(['MUA Cond ' num2str(cond_idx)]);
    end

    % 
    % === Ask user to input layer groupings with re-entry option ===
    confirmed = false;
    while ~confirmed
        disp('----------------------------------')
        disp('Please assign channels into layers:');
        supra_chans = input('Enter SUPRA channels (e.g., [2 3 4]): ');
        granular_chans = input('Enter GRANULAR channels (e.g., [5 6 7]): ');
        infra_chans = input('Enter INFRA channels (e.g., [8 9 10]): ');

        disp('You entered:');
        fprintf('SUPRA: [%s]\n', num2str(supra_chans));
        fprintf('GRANULAR: [%s]\n', num2str(granular_chans));
        fprintf('INFRA: [%s]\n', num2str(infra_chans));

        response = input('Is this correct? (y/n): ', 's');
        if strcmpi(response, 'y')
            confirmed = true;
        else
            disp('Re-entering layer assignment...');
        end
        disp('----------------------------------')
    end


    % === Save layer info ===
    layers = struct();
    layers.file = base_name; % Save full base_name (including 1- or 2- prefix)
    layers.datetime = datetime('now');
    layers.supra = supra_chans;
    layers.granular = granular_chans;
    layers.infra = infra_chans;
    
    % Extract prefix (1- or 2-) and ev2 base name
    prefix = base_name(1); % '1' or '2'
    ev2_base = regexprep(base_name, '^[12]-(.*)@oe$', '$1');

    % Save using prefix to differentiate 1- vs 2- files
    layer_filename = sprintf('%s-%s_layers.mat', prefix, ev2_base);
    layer_filepath = fullfile(parent_dir, layer_filename);

    % === Check for existing layer file ===
    if exist(layer_filepath, 'file')
        fprintf('WARNING: Layer file "%s" already exists.\n', layer_filename);
        choice = input('Do you want to overwrite it? (y/n): ', 's');
        if ~strcmpi(choice, 'y')
            fprintf('Skipping save. Existing layer file retained.\n');
            close all;
            rmdir(temp_dir, 's');
            continue;
        end
    end

    save(fullfile(parent_dir, layer_filename), 'layers');
    
    fprintf('Saved layer info to %s\n', fullfile(parent_dir, layer_filename));

    
    close all; % Close figure
    rmdir(temp_dir, 's'); % Clean up temp directory



end

fprintf('\n=== Layer assignment complete ===\n');
