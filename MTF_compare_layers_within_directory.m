function layer_ISIs = MTF_compare_layers_within_directory(dir_path, bin_size, file_type, verbose)
% Compare best decoding ISI distributions across supra, granular, and infra layers within a single directory
% Inputs:
%   dir_path  - Directory containing decoding results
%   bin_size  - Bin size (in ms) to match decoding results
%   file_type - 'oe' or 'om'
%   verbose   - Optional: Set true for logging
%
% Output:
%   layer_ISIs - Struct with fields 'supra', 'granular', 'infra', each containing best ISIs across channels/files
%
% Example:
%   layer_ISIs = MTF_compare_layers_within_directory('E:\MTF\core\right', 700, 'oe', true);

if nargin < 4
    verbose = false;
end

% === Find all matching decoding result files ===
files = dir(fullfile(dir_path, sprintf('*%s_decoding_%dms.mat', file_type, bin_size)));
if isempty(files)
    error('No matching decoding result files found in %s for bin size %dms and type %s.', dir_path, bin_size, file_type);
end

% === Extract ISIs grouped by layer ===
layer_ISIs = extract_best_ISIs_by_layer(files, dir_path, verbose);

% === Plot CDFs comparing layers ===
layer_names = {'supra', 'granular', 'infra'};
colors = lines(3);
figure('Name', sprintf('Layer ISI CDFs (%s, %dms)', file_type, bin_size), 'Position', [100 100 600 500]);

hold on;
for i = 1:length(layer_names)
    layer = layer_names{i};
    if ~isempty(layer_ISIs.(layer))
        cdfplot(layer_ISIs.(layer));
    end
end
hold off;

legend(upper(layer_names), 'Location', 'best');
xlabel('Best ISI (ms)');
ylabel('Cumulative Probability');
title(sprintf('CDF of Best ISI by Layer\n(%s, %d ms)', file_type, bin_size));
set(gca, 'XScale', 'log');
grid on;

% === Perform pairwise KS tests ===
fprintf('\n--- KS Test Results (within %s) ---\n', extract_dir_name(dir_path));
for i = 1:2
    for j = i+1:3
        layer1 = layer_names{i};
        layer2 = layer_names{j};
        if ~isempty(layer_ISIs.(layer1)) && ~isempty(layer_ISIs.(layer2))
            [~, p] = kstest2(layer_ISIs.(layer1), layer_ISIs.(layer2));
            fprintf('%s vs. %s: p = %.4f\n', upper(layer1), upper(layer2), p);
        end
    end
end


function layer_ISIs = extract_best_ISIs_by_layer(files, dir_path, verbose)
layer_ISIs = struct('supra', [], 'granular', [], 'infra', []);
    for i = 1:length(files)
        filename = files(i).name;
        filepath = fullfile(dir_path, filename);
        [~, decode_base, ~] = fileparts(filename);  % e.g. '1-hi01018@oe_decoding_700ms'
    
        % === Extract root base name for matching layer file ===
        % Strip '@oe' or '@om' and '_decoding_###ms'
        layer_base = regexprep(decode_base, '@o[em]_decoding_\d+ms$', '');
        layer_filename = [layer_base '_layers.mat'];
        layer_file_path = fullfile(dir_path, layer_filename);
    
        if ~isfile(layer_file_path)
            warning('Layer file not found for %s. Expected: %s', decode_base, layer_filename);
            continue;
        elseif ~contains(decode_base,layer_base)
            warning('mismatch b/w decoding results and layer file')
            disp(decode_base)
            continue
        end
    
        try
            % Load decoding data
            matdata = matfile(filepath);
            accuracies = matdata.decoding_accuracy_conditions;
            chance_hiCI = mean(matdata.permutation_accuracy_hiCI, 2);
    
            % Load ISIs
            temp = load(filepath, 'epoched_data');
            ISIs = temp.epoched_data.ISI_ms;
            clear temp;
    
            % Load layer file
            layer_data = load(layer_file_path);
            layers = layer_data.layers;
    
            num_chans = size(accuracies, 1);
    
            % Process each layer group
            for layer_name = {'supra', 'granular', 'infra'}
                layer = layer_name{1};
    
                % Skip empty layers
                if ~isfield(layers, layer) || isempty(layers.(layer))
                    if verbose
                        fprintf('Skipping empty layer "%s" in %s\n', layer, filename);
                    end
                    continue;
                end
    
                chans = layers.(layer);
                new_ISIs = [];
    
                for ch = chans
                    if ch > num_chans
                        warning('Channel %d exceeds available channels in %s. Skipping...', ch, filename);
                        continue;
                    end
                    % if we don't have time windows 
                    if size(accuracies, 3) == 1
                        [val, idx] = max(accuracies(ch, :));
                        if val > chance_hiCI(ch)
                            new_ISIs = [new_ISIs; ISIs(idx)];
                        end
                    else % we decoded over time
                        max_val = 0; best_ISI = NaN;
                        for w = 1:size(accuracies, 3)
                            [val, idx] = max(accuracies(ch, :, w));
                            if val > max_val
                                max_val = val;
                                best_ISI = ISIs(idx);
                            end
                        end
                        if max_val > chance_hiCI(ch)
                            new_ISIs = [new_ISIs; best_ISI];
                        end
                    end
                end
    
                % Append valid ISIs
                layer_ISIs.(layer) = [layer_ISIs.(layer); new_ISIs];
    
                if verbose && ~isempty(new_ISIs)
                    fprintf('Added %d ISIs to layer %s from %s\n', length(new_ISIs), layer, filename);
                end
            end
    
        catch ME
            warning('Error processing %s: %s', filename, ME.message);
            continue;
        end
    end
end

function name = extract_dir_name(dir_path)
[~, name] = fileparts(dir_path);
end

end
