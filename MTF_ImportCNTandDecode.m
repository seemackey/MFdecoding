clear; close all;

% Set paths
dirin = 'E:\MTF\data\click\pb\right\';  % your current working directory
dirout = fullfile(dirin, 'imported\');

% Read spreadsheet
ev2_table = readtable('E:\MTF\ev2_TFR.xlsx', 'Sheet', 'ev2');
all_ev2_names = ev2_table.filename;
col1 = ev2_table{:,2};
col2 = ev2_table{:,3};

% Determine current region from dirin path
if contains(lower(dirin), 'core')
    current_region = 'core';
elseif contains(lower(dirin), 'pb')
    current_region = 'pb';
else
    error('Cannot determine region from dirin path: must contain "core" or "pb"');
end

% Get CNT files in dirin (both e and m)
cnt_files_in_dir = dir(fullfile(dirin, '*.cnt'));
cnt_filenames = {cnt_files_in_dir.name};

% Set constants
trigchan = [24];
threshold = 50;
chan_group_1 = [1 23];   % for 1-
chan_group_2 = [25 47];  % for 2-

% Loop through all CNT files
for i = 1:length(cnt_filenames)
    cnt_file = cnt_filenames{i};

    % Strip down to base name: e.g., 'wnc6022@e.cnt' -> 'wnc6022.ev2'
    base_name = strtok(cnt_file, '@');
    ev2_name = [base_name '.ev2'];

    % Match to spreadsheet
    match_idx = find(strcmpi(ev2_name, all_ev2_names));
    if isempty(match_idx)
        continue;  % skip if not in spreadsheet
    end

    % Normalize labels
    normalize_label = @(x) lower(regexprep(strtrim(x), '^(left|right)\s*', ''));

    label1 = normalize_label(col1{match_idx});
    label2 = normalize_label(col2{match_idx});

    label1_has_region = contains(label1, current_region);
    label2_has_region = contains(label2, current_region);

    % Import each channel group only if its label matches region
    if label1_has_region
        disp(['Importing 1- (ch 1–23) for: ' cnt_file]);
        dirimp_cnt03(dirin, chan_group_1, dirout, trigchan, {cnt_file}, threshold, '');
    end

    if label2_has_region
        disp(['Importing 2- (ch 25–47) for: ' cnt_file]);
        dirimp_cnt03(dirin, chan_group_2, dirout, trigchan, {cnt_file}, threshold, '');
    end
end
