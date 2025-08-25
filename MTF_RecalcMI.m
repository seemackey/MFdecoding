%% MTF batch script for removing 226 ms ISI condition from saved decoding results

clear; clc;

% === CONFIGURATION ===
results_dir = 'E:\MTF\data\noise\core\right\imported\decoding_results_MI';
epoched_dir = 'E:\MTF\data\noise\core\right\imported\decoding_results_LDA';
output_dir = fullfile(results_dir, 'cleaned_MI_exclude226');
if ~exist(output_dir, 'dir'); mkdir(output_dir); end

bin_size = 700; % Assume we're operating on 700 ms bin
ISI_to_remove = 226; % ISI in ms to exclude

result_files = dir(fullfile(results_dir, sprintf('*decoding_%dms.mat', bin_size)));

for f = 1:length(result_files)
    result_file = result_files(f);
    result_path = fullfile(result_file.folder, result_file.name);
    [~, name_no_ext, ~] = fileparts(result_file.name);

    % Match to epoched data file
    subj_id = regexprep(name_no_ext, '_decoding_\d+ms', '');
    mat_file = dir(fullfile(epoched_dir, [subj_id '*.mat']));
    if isempty(mat_file)
        warning('No matching epoched data for %s', subj_id);
        continue;
    end
    epoched_data = load(fullfile(mat_file(1).folder, mat_file(1).name));
    close all
    % Run cleaning function and save output
    try
        fprintf('Processing: %s\n', result_file.name);
        cleaned_results = remove_bad_condition(epoched_data, result_path, ISI_to_remove);

        save_name = sprintf(name_no_ext);
        save_path = fullfile(output_dir, name_no_ext);
        save(save_path, '-struct', 'cleaned_results');
    catch err
        warning('Error processing %s: %s', result_file.name, err.message);
        disp(err.stack);
    end
end

fprintf('=== Done excluding ISI = %d ms from decoding results ===\n', ISI_to_remove);


function R = remove_bad_condition(epoched_data, result_path, bad_ISI_ms)
% Removes a bad condition (e.g., 226 ms ISI) from confusion matrices and recalculates MI

if ~isfield(epoched_data, 'epoched_data') || ~isfield(epoched_data.epoched_data, 'ISI_ms')
    error('Missing required field (epoched_data.ISI_ms)');
end

ISIs = epoched_data.epoched_data.ISI_ms;
bad_idx = find(ISIs == bad_ISI_ms);
if isempty(bad_idx)
    error('Bad ISI (%d ms) not found in ISI_ms', bad_ISI_ms);
end

R = load(result_path);

% Remove bad condition from confusion matrices
R.confusion_matrices_true(:, :, :, bad_idx, :) = [];
R.confusion_matrices_true(:, :, :, :, bad_idx) = [];
R.confusion_matrices_shuffled(:, :, :, bad_idx, :) = [];
R.confusion_matrices_shuffled(:, :, :, :, bad_idx) = [];

% Recalculate MI
[num_perm, num_chans, num_wins, ~, ~] = size(R.confusion_matrices_true);
mi_true_all = NaN(num_perm, num_chans, num_wins);
mi_shuff_all = NaN(num_perm, num_chans, num_wins);

for p = 1:num_perm
    for ch = 1:num_chans
        for w = 1:num_wins
            cm_true = squeeze(R.confusion_matrices_true(p, ch, w, :, :));
            cm_shuff = squeeze(R.confusion_matrices_shuffled(p, ch, w, :, :));
            mi_true_all(p, ch, w) = compute_mutual_information_single(cm_true);
            mi_shuff_all(p, ch, w) = compute_mutual_information_single(cm_shuff);
        end
    end
end

R.mi_true_mean = squeeze(mean(mi_true_all, 1));
R.mi_true_loCI = squeeze(prctile(mi_true_all, 2.5, 1));
R.mi_true_hiCI = squeeze(prctile(mi_true_all, 97.5, 1));
R.mi_shuffled_mean = squeeze(mean(mi_shuff_all, 1));
R.mi_shuffled_loCI = squeeze(prctile(mi_shuff_all, 2.5, 1));
R.mi_shuffled_hiCI = squeeze(prctile(mi_shuff_all, 97.5, 1));
end

function mi = compute_mutual_information_single(cm)
if sum(cm(:)) == 0
    mi = NaN;
    return;
end
Pxy = cm / sum(cm(:));
Px = sum(Pxy, 2);
Py = sum(Pxy, 1);
PxPy = Px * Py;
mask = Pxy > 0;
mi = sum(Pxy(mask) .* log2(Pxy(mask) ./ PxPy(mask)), 'all');
end
