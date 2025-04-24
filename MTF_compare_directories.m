function [best_ISIs1,best_ISIs2] = MTF_compare_directories(dir1, dir2, bin_size, file_type)
% Compare decoding performance across two directories for a specific bin size
% INPUTS:
%   dir1 - First directory containing decoding results
%   dir2 - Second directory containing decoding results
%   bin_size - Bin size (ms) to analyze
%   file_type - 'oe' for LFP/CSD or 'om' for MUA

if ~ismember(file_type, {'oe', 'om'})
    error('Invalid file_type. Must be ''oe'' or ''om''.');
end

% === Get .mat files for the specified bin size ===
files1 = dir(fullfile(dir1, sprintf('*%s_decoding_%dms.mat', file_type, bin_size)));
files2 = dir(fullfile(dir2, sprintf('*%s_decoding_%dms.mat', file_type, bin_size)));

if isempty(files1)
    error('No files found in %s for the specified file type (%s) and bin size (%d ms).', dir1, file_type, bin_size);
end
if isempty(files2)
    error('No files found in %s for the specified file type (%s) and bin size (%d ms).', dir2, file_type, bin_size);
end

% === Collect best condition ISIs for both directories ===
best_ISIs1 = extract_best_ISIs(files1, dir1);
best_ISIs2 = extract_best_ISIs(files2, dir2);

% === Plotting CDFs ===
figure;
hold on;
cdfplot(best_ISIs1);
cdfplot(best_ISIs2);
xlabel('Preferred ISI (ms)');
ylabel('Cumulative Probability');
title(sprintf('Comparison of Best Condition ISIs (Bin Size = %d ms, %s)', bin_size, file_type));
legend({extract_dir_name(dir1), extract_dir_name(dir2)});
grid on;
hold off;
[k,p] = kstest2(best_ISIs1,best_ISIs2)
%text(sprintf('p = %.3g', p), 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');

end

function best_ISIs = extract_best_ISIs(files, dir_path)
% Extracts the best condition ISI for each channel in all files
% INPUTS:
%   files - List of .mat files to process
%   dir_path - Path to the directory containing the files
% OUTPUT:
%   best_ISIs - Array of best ISIs pooled across files and channels

best_ISIs = []; % Initialize storage

for i = 1:length(files)
    file_name = files(i).name;
    fprintf('Loading %s...\n', file_name);
    
    filepath = fullfile(dir_path, file_name);
    data = matfile(filepath);

    try
        % Extract relevant variables without fully loading the file
        accuracies = data.decoding_accuracy_conditions;
        chance_hiCI = mean(data.permutation_accuracy_hiCI,2);
        
        


        if size(accuracies, 1) > 0
            num_chans = size(accuracies, 1);
            best_ISIs_for_file = [];
            temp = load(filepath, 'epoched_data');  % Load only 'epoched_data'
            ISIs = temp.epoched_data.ISI_ms;       % Extract ISI_ms
            clear temp;                              % Free memory immediately

            for ch = 1:num_chans
                % Find the best condition index (the highest decoding accuracy across conditions and time)
                if size(accuracies,3) == 1
                    [best_accuracy, best_accuracy_idx] = max(accuracies(ch, :));
                    % Only keep if the best accuracy is above the threshold 
                    if best_accuracy > chance_hiCI(ch)
    
                        % Store the corresponding ISI
                        best_ISIs_for_file = [best_ISIs_for_file; ISIs(best_accuracy_idx)]; %#ok<AGROW>
                    end
                else
                    % Multiple windows case
                    best_accuracy = 0;
                    best_ISI = NaN;
            
                    for window = 1:size(accuracies, 3)
                        % Find the best condition for this window
                        [current_max_accuracy, current_best_idx] = max(accuracies(ch, :, window));
            
                        % Check if this is the best accuracy so far across all windows
                        if current_max_accuracy > best_accuracy
                            best_accuracy = current_max_accuracy;
                            best_ISI = ISIs(current_best_idx);
                        end
                    end
                    
                    % Only add if the best accuracy found is above threshold
                    if best_accuracy > chance_hiCI(ch)
                        best_ISIs_for_file = [best_ISIs_for_file; best_ISI]; %#ok<AGROW>
                    end
                end
                

            end
            
            % Append results for the current file to the pooled list
            best_ISIs = [best_ISIs; best_ISIs_for_file]; %#ok<AGROW>
        else
            disp('Could not extract data');
            disp(file_name);
        end

    catch ME
        warning('Error reading %s: %s', file_name, ME.message);
    end
end
end


function name = extract_dir_name(dir_path)
% Extracts the name of the directory for labeling plots
[~, name] = fileparts(dir_path);
end
