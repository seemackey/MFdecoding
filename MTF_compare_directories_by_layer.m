function MTF_compare_directories_by_layer(dir1, dir2, bin_size, file_type, verbose)
% Compare best decoding ISI distributions across supra, granular, and infra layers between two directories.
% Inputs:
%   dir1      - Directory 1 path
%   dir2      - Directory 2 path
%   bin_size  - Bin size (in ms) to match decoding results
%   file_type - 'oe' for CSD/LFP or 'om' for MUA
%   verbose   - (Optional) Set true to enable detailed logging

if nargin < 5
    verbose = false;
end

% Get files for each directory
files1 = dir(fullfile(dir1, sprintf('*%s_decoding_%dms.mat', file_type, bin_size)));
files2 = dir(fullfile(dir2, sprintf('*%s_decoding_%dms.mat', file_type, bin_size)));

% Get ISIs grouped by layer
layers1 = extract_best_ISIs_by_layer(files1, dir1, verbose);
layers2 = extract_best_ISIs_by_layer(files2, dir2, verbose);

% Plot comparisons by layer
layer_names = {'supra', 'granular', 'infra'};
figure('Name', sprintf('ISI Layer Comparison (%s, %dms)', file_type, bin_size), 'Position', [100 100 1200 400]);
for i = 1:length(layer_names)
    layer = layer_names{i};
    subplot(1, 3, i);
    
    if ~isempty(layers1.(layer)) && ~isempty(layers2.(layer))
        cdfplot(layers1.(layer)); hold on;
        cdfplot(layers2.(layer));
        legend(extract_dir_name(dir1), extract_dir_name(dir2), 'Location', 'best');
        [~, p] = kstest2(layers1.(layer), layers2.(layer));
        title(sprintf('%s Layer\np = %.4f', upper(layer), p));
        xlabel('Best ISI (ms)');
        ylabel('Cumulative Probability');
        grid on;
    else
        title(sprintf('%s Layer\n(No data)', upper(layer)));
        axis off;
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
    
                    if size(accuracies, 3) == 1
                        [val, idx] = max(accuracies(ch, :));
                        if val > chance_hiCI(ch)
                            new_ISIs = [new_ISIs; ISIs(idx)];
                        end
                    else
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
