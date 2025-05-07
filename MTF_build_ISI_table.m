function ISI_table = MTF_build_ISI_table(dir_configs, bin_size, file_type, verbose)
% Collects best ISI values across layer, hemisphere, and area for all recordings.
%
% INPUTS:
%   dir_configs - Struct array with fields:
%                 .path        - Directory path
%                 .hemisphere  - 'left' or 'right'
%                 .area        - 'core' or 'pb'
%   bin_size    - e.g. 700
%   file_type   - 'oe' or 'om'
%   verbose     - optional (default = false)
%
% OUTPUT:
%   ISI_table   - Long-format table of ISIs with metadata for stats
%
% EXAMPLE:
%   dirs = struct( ...
%       'path', {'E:\MTF\core\left', 'E:\MTF\core\right'}, ...
%       'hemisphere', {'left', 'right'}, ...
%       'area', {'core', 'core'} ...
%   );
%   T = MTF_build_ISI_table(dirs, 700, 'oe', true);

if nargin < 4
    verbose = false;
end

all_rows = [];

for i = 1:length(dir_configs)
    dir_path = dir_configs(i).path;
    hemisphere = dir_configs(i).hemisphere;
    area = dir_configs(i).area;

    files = dir(fullfile(dir_path, sprintf('*%s_decoding_%dms.mat', file_type, bin_size)));
    if isempty(files)
        warning('No decoding files found in %s', dir_path);
        continue;
    end

    layer_ISIs = extract_best_ISIs_by_layer(files, dir_path, 1);

    % Convert each layer’s ISIs into table rows
    for layer_name = {'supra', 'granular', 'infra'}
        layer = layer_name{1};
        ISIs = layer_ISIs.(layer);
        if isempty(ISIs); continue; end

        n = length(ISIs);
        T = table(ISIs, ...
                  repmat({layer}, n, 1), ...
                  repmat({hemisphere}, n, 1), ...
                  repmat({area}, n, 1), ...
                  (1:n)', ...  % This adds the UnitID column
                  'VariableNames', {'BestISI', 'Layer', 'Hemisphere', 'Area', 'UnitID'});


        all_rows = [all_rows; T]; %#ok<AGROW>
    end
end

% Combine into final table
ISI_table = all_rows;

% Optional: show preview
disp(head(ISI_table, 10));

% Optional: run LME model
% lme = fitlme(ISI_table, 'log(BestISI) ~ Layer*Area*Hemisphere + (1|Area)');
% disp(anova(lme));
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

end
