%% MTF Batch Plot and Save Script
% Chase M 2025 - Save .fig and .jpg for each dataset

clear; clc; close all;

% === CONFIGURATION ===
parent_dir = 'E:\MTF\data\noise\pb\left\imported\subset_cnt\imported\';    
output_dir = fullfile(parent_dir, 'Profiles');  % Customize name/location
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

epoch_tframe = [-10 500];

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

    % Match the ev2 file
    ev2_base = regexprep(base_name, '^[12]-(.*)@oe$', '$1');
    ev2_file = dir(fullfile(parent_dir, [ev2_base '.ev2']));
    if isempty(ev2_file)
        warning('EV2 file not found for %s. Skipping...', base_name);
        continue;
    end

    % === Setup temp dir for isolated processing ===
    temp_dir = fullfile(parent_dir, ['temp_' ev2_base]);
    if exist(temp_dir, 'dir'); rmdir(temp_dir, 's'); end
    mkdir(temp_dir);

    % Copy files into temp dir
    copyfile(fullfile(oe_file.folder, oe_file.name), fullfile(temp_dir, oe_file.name));
    copyfile(fullfile(om_file.folder, om_file.name), fullfile(temp_dir, om_file.name));
    copyfile(fullfile(ev2_file.folder, ev2_file.name), fullfile(temp_dir, ev2_file.name));

    % === Load Data ===
    [epoched_data, srate] = MTF_loadMATfile(temp_dir, epoch_tframe);

    % === Plot CSD + MUA ===
    time_axis = epoch_tframe(1):1000/srate:epoch_tframe(2);
    numchans = size(epoched_data.LFP_trial_avg{1}, 1);
    num_conditions = length(epoched_data.LFP);

    fig = figure('Position', [100 100 1800 1000]);
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

    % === Save the figure ===
    out_name = fullfile(output_dir, [base_name '_csdmua']);
    savefig(fig, [out_name '.fig']);
    exportgraphics(fig, [out_name '.jpg'], 'Resolution', 300);

    fprintf('Saved .fig and .jpg to %s\n', out_name);

    % === Cleanup ===
    close(fig);
    rmdir(temp_dir, 's');
end

fprintf('\n=== Batch plot export complete ===\n');
