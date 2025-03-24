%% MTF decoding batch script

clear; clc; close all;

% === CONFIGURATION ===
parent_dir = 'E:\MTF\data\click\core\left\imported\';    % directory containing oe/om/ev2 files
figures_dir = fullfile(parent_dir, 'decoding_results');
epoch_tframe = [-50 750];                   % ms
bins = [1, 5, 10, 50, 150, 325, 750];       % decoding window sizes (ms)
selected_channels = [8,11,14];

% === Get all 1-/2- files ===
all_files = dir(fullfile(parent_dir, '1-*@o*.mat'));
all_files = [all_files; dir(fullfile(parent_dir, '2-*@o*.mat'))];

for i = 1:length(all_files)
    file = all_files(i);
    [~, name_no_ext, ~] = fileparts(file.name);  % e.g., '1-wnc6022@oe'
    fprintf('\n>> Processing %s\n', name_no_ext);

    % Determine modality and flag
    if contains(name_no_ext, '@oe')
        csd_flag = 1;
    elseif contains(name_no_ext, '@om')
        csd_flag = 0;
    else
        warning('Skipping unknown file type: %s', name_no_ext);
        continue;
    end

    % Find corresponding .ev2 file (strip prefix like 1- or 2-)
    ev2_base = regexprep(name_no_ext, '^[12]-(.*)@o[em]$', '$1');

    if isempty(ev2_base)
        warning('Unable to extract EV2 base for %s. Skipping...', name_no_ext);
        continue;
    end

    ev2_file = dir(fullfile(parent_dir, [ev2_base '.ev2']));
    if isempty(ev2_file)
        warning('EV2 file not found for %s. Skipping...', name_no_ext);
        continue;
    end

    % Setup temp dir
    temp_dir = fullfile(parent_dir, ['temp_' name_no_ext]);
    if ~exist(temp_dir, 'dir'); mkdir(temp_dir); end

    % Copy necessary files to temp dir
    copyfile(fullfile(file.folder, file.name), fullfile(temp_dir, file.name));
    copyfile(fullfile(ev2_file.folder, ev2_file.name), fullfile(temp_dir, ev2_file.name));

    % Epoch and decode
    try
        [epoched_data, srate] = MTF_loadMATfile(temp_dir, epoch_tframe);
        numchans = size(epoched_data.LFP_trial_avg{1}, 1);
        selchans = 1:numchans;
        num_conditions = length(epoched_data.LFP);

        for bin_idx = 1:length(bins)
            bin = bins(bin_idx);
            fprintf('  - Bin %d ms...\n', bin);
            results = MTF_run_decoding(epoched_data, bin, srate, csd_flag, selchans, selected_channels, epoch_tframe);

            % Save results using full filename
            tag = name_no_ext;
            savefile = fullfile(figures_dir, sprintf('%s_decoding_%dms.mat', tag, bin));
            figfile  = fullfile(figures_dir, sprintf('%s_decoding_%dms.fig', tag, bin));
            save(savefile, '-struct', 'results');
            saveas(results.figure, figfile);
            close(results.figure);
        end
    catch err
        warning('Error decoding %s: %s', name_no_ext, err.message);
    end

    rmdir(temp_dir, 's');
end


fprintf('\n=== Batch decoding complete ===\n');

function results = MTF_run_decoding(epoched_data, bin_size_ms, srate, csd_flag, selchans, selected_channels, epoch_tframe)
% Run decoding for a given bin size and epoched data
% Returns results struct with accuracy matrices, CIs, confusion matrices, and plot

% -----------------------------
% === Setup and Definitions ===
% -----------------------------
num_conditions = length(epoched_data.LFP);
numchans = length(selchans);
window_size_samples = bin_size_ms / 1000 * srate;

positive_duration = epoch_tframe(2);  % e.g., 750 ms
num_windows = floor(positive_duration / bin_size_ms);
bin_centers = (0:num_windows-1) * bin_size_ms + bin_size_ms/2;

% Preallocate
decoding_accuracy_windows = zeros(numchans, num_windows);
decoding_accuracy_conditions = zeros(numchans, num_conditions, num_windows);
decoding_accuracy_cis = zeros(numchans, num_windows, 2);
confusion_matrices_all = zeros(numchans, num_windows, num_conditions, num_conditions);

% ----------------------
% === Main Decoding ===
% ----------------------
time_axis = epoch_tframe(1):1000/srate:epoch_tframe(2);
for ch_idx = 1:numchans
    ch = selchans(ch_idx);

    for w = 1:num_windows
        window_start = (w - 1) * window_size_samples + 1;
        window_end = min(window_start + window_size_samples - 1, length(time_axis));

        flattened_data_window = [];
        flattened_labels_window = [];

        for cond_idx = 1:num_conditions
            if csd_flag == 1
                data = epoched_data.CSD{cond_idx}(ch, :, :);
            else
                data = epoched_data.MUA{cond_idx}(ch, :, :);
            end

            num_trials = size(data, 2);
            for trial = 1:num_trials
                trial_data = data(:, trial, window_start:window_end);
                flattened_data_window = [flattened_data_window; trial_data(:)'];
                flattened_labels_window = [flattened_labels_window; cond_idx];
            end
        end

        flattened_data_window = double(flattened_data_window);
        decoding_accuracy = zeros(4, 1);
        confusion_matrices_folds = zeros(4, num_conditions, num_conditions);

        cv = cvpartition(flattened_labels_window, 'KFold', 4);
        for fold = 1:cv.NumTestSets
            train_idx = cv.training(fold);
            test_idx = cv.test(fold);

            model = fitcecoc(flattened_data_window(train_idx, :), ...
                             flattened_labels_window(train_idx), ...
                             'Learners', templateSVM('Standardize', true));

            preds = predict(model, flattened_data_window(test_idx, :));
            truth = flattened_labels_window(test_idx);
            decoding_accuracy(fold) = mean(preds == truth);

            for i = 1:length(truth)
                confusion_matrices_folds(fold, truth(i), preds(i)) = ...
                    confusion_matrices_folds(fold, truth(i), preds(i)) + 1;
            end
        end

        decoding_accuracy_windows(ch_idx, w) = mean(decoding_accuracy);
        confusion_matrices_all(ch_idx, w, :, :) = squeeze(mean(confusion_matrices_folds, 1));
        decoding_accuracy_cis(ch_idx, w, :) = prctile(decoding_accuracy, [2.5 97.5]);

        for cond_idx = 1:num_conditions
            total = sum(confusion_matrices_all(ch_idx, w, cond_idx, :));
            correct = confusion_matrices_all(ch_idx, w, cond_idx, cond_idx);
            decoding_accuracy_conditions(ch_idx, cond_idx, w) = correct / total;
        end
    end
end

% ----------------------------------------
% === Permutation-Based Chance Accuracy ===
% ----------------------------------------
perm_window_size = 10; % ms
perm_window_size_samples = perm_window_size / 1000 * srate;
perm_num_windows = floor(length(time_axis) / perm_window_size_samples);
perm_window_time_axis = epoch_tframe(1) + (0:perm_num_windows-1) * perm_window_size;
num_permutations = 50;

perm_flattened_labels = [];
for cond_idx = 1:num_conditions
    if csd_flag == 1
        trials = size(epoched_data.CSD{cond_idx}, 2);
    else
        trials = size(epoched_data.MUA{cond_idx}, 2);
    end
    perm_flattened_labels = [perm_flattened_labels; repmat(cond_idx, trials, 1)];
end

permutation_accuracies = zeros(num_permutations, numchans, perm_num_windows);

for perm = 1:num_permutations
    shuffled_labels = perm_flattened_labels(randperm(length(perm_flattened_labels)));

    for ch_idx = 1:numchans
        ch = selchans(ch_idx);
        for w = 1:perm_num_windows
            w_start = (w - 1) * perm_window_size_samples + 1;
            w_end = min(w_start + perm_window_size_samples - 1, length(time_axis));

            flattened_data = [];
            for cond_idx = 1:num_conditions
                if csd_flag == 1
                    data = epoched_data.CSD{cond_idx}(ch, :, :);
                else
                    data = epoched_data.MUA{cond_idx}(ch, :, :);
                end
                for trial = 1:size(data, 2)
                    trial_data_window = data(:, trial, w_start:w_end);
                    flattened_data = [flattened_data; trial_data_window(:)'];
                end
            end

            flattened_data = double(flattened_data);
            cv = cvpartition(shuffled_labels, 'KFold', 4);
            fold_acc = zeros(4, 1);
            for f = 1:4
                model = fitcecoc(flattened_data(cv.training(f), :), shuffled_labels(cv.training(f)), ...
                                 'Learners', templateSVM('Standardize', true));
                preds = predict(model, flattened_data(cv.test(f), :));
                fold_acc(f) = mean(preds == shuffled_labels(cv.test(f)));
            end
            permutation_accuracies(perm, ch_idx, w) = mean(fold_acc);
        end
    end
end

mean_perm_acc = squeeze(mean(permutation_accuracies, 1));
std_perm_acc = squeeze(std(permutation_accuracies, 0, 1));

% ------------------
% === Visualization ===
% ------------------
f = figure;
f.Position = [150 150 1800 700];

% Plot decoding accuracy over time
subplot(1, 3, 1);
imagesc(bin_centers, 1:numchans, decoding_accuracy_windows);
xlabel('Time (ms)'); ylabel('Channels');
title(['Decoding Accuracy (Bin: ' num2str(bin_size_ms) ' ms)']);
colorbar;

% Condition-wise decoding
subplot(1, 3, 2);
imagesc(1:num_conditions, 1:numchans, squeeze(mean(decoding_accuracy_conditions, 3)));
xlabel('Condition'); ylabel('Channels');
title('Accuracy by Condition'); colorbar;

% Channel traces
colors = lines(length(selected_channels));
for i = 1:length(selected_channels)
    subplot(length(selected_channels), 3, 3*i);
    ch = selected_channels(i);

    plot(bin_centers, decoding_accuracy_windows(ch,:), '-', 'Color', colors(i,:), 'LineWidth', 1.5, 'HandleVisibility', 'off'); hold on;
    plot(bin_centers, decoding_accuracy_cis(ch,:,1), ':', 'Color', colors(i,:), 'HandleVisibility', 'off');
    plot(bin_centers, decoding_accuracy_cis(ch,:,2), ':', 'Color', colors(i,:), 'HandleVisibility', 'off');

    plot(perm_window_time_axis, mean_perm_acc(ch,:), 'r--', 'LineWidth', 2, 'DisplayName', 'Chance');
    fill([perm_window_time_axis, fliplr(perm_window_time_axis)], ...
         [mean_perm_acc(ch,:) + 2*std_perm_acc(ch,:), fliplr(mean_perm_acc(ch,:) - 2*std_perm_acc(ch,:))], ...
         'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');

    xlabel('Time (ms)'); ylabel('Accuracy');
    title(['Channel ' num2str(ch)]); grid on;
    legend('Chance');
end

% ----------------
% === RETURN ===
% ----------------
results.decoding_accuracy_windows = decoding_accuracy_windows;
results.decoding_accuracy_cis = decoding_accuracy_cis;
results.decoding_accuracy_conditions = decoding_accuracy_conditions;
results.decoding_accuracy_permuted = permutation_accuracies;
results.mean_permutation_accuracy = mean_perm_acc;
results.std_permutation_accuracy = std_perm_acc;
results.confusion_matrices_all = confusion_matrices_all;
results.figure = f;

end
