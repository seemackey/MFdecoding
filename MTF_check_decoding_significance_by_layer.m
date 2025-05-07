function summary = MTF_check_decoding_significance_by_layer(dir_path, bin_size, file_type, verbose)
% Check proportion of significantly decoding channels by layer
% Inputs:
%   dir_path  - directory containing decoding result .mat files
%   bin_size  - decoding bin size in ms
%   file_type - 'oe' (LFP/CSD) or 'om' (MUA)
%   verbose   - true/false, print per-file summaries

if nargin < 4
    verbose = false;
end

files = dir(fullfile(dir_path, sprintf('*%s_decoding_%dms.mat', file_type, bin_size)));
if isempty(files)
    error('No decoding files found for bin size %d ms and file type %s', bin_size, file_type);
end

layers_list = {'supra', 'granular', 'infra'};
summary = struct();
for l = layers_list
    summary.(l{1}) = struct('total_chans', 0, 'sig_chans', 0);
end

for i = 1:length(files)
    filename = files(i).name;
    filepath = fullfile(dir_path, filename);
    matdata = matfile(filepath);
    
    try
        accuracies = matdata.decoding_accuracy_windows;
        hiCI = matdata.permutation_accuracy_hiCI;
        
        % Load ISI mapping to layer
        decode_base = erase(filename, sprintf('_decoding_%dms.mat', bin_size));
        decode_base = regexprep(decode_base, '@o[em]$', ''); % remove @oe/@om
        layer_file = fullfile(dir_path, [decode_base '_layers.mat']);
        
        if ~isfile(layer_file)
            warning('Missing layer file for %s', filename);
            continue;
        end
        
        layers = load(layer_file);
        layers = layers.layers;
        
        for l = layers_list
            layer_name = l{1};
            if ~isfield(layers, layer_name) || isempty(layers.(layer_name))
                continue;
            end
            
            chans = layers.(layer_name);
            chans = chans(chans <= size(accuracies, 1)); % safe indexing
            summary.(layer_name).total_chans = summary.(layer_name).total_chans + numel(chans);
            
            for ch = chans
                if size(accuracies, 2) == 1  % Single bin
                    is_sig = accuracies(ch) > hiCI(ch);
                else  % Multiple bins
                    is_sig = any(accuracies(ch, :) > hiCI(ch));
                end
                if is_sig
                    summary.(layer_name).sig_chans = summary.(layer_name).sig_chans + 1;
                end
            end
        end

        if verbose
            fprintf('File: %s\n', filename);
            for l = layers_list
                n = summary.(l{1}).total_chans;
                s = summary.(l{1}).sig_chans;
                fprintf('  %s: %d/%d channels significant\n', l{1}, s, n);
            end
        end

    catch ME
        warning('Error processing %s: %s', filename, ME.message);
    end
end

% === Summary Printout ===
fprintf('\n=== Decoding Significance Summary (%s, %d ms) ===\n', file_type, bin_size);
for l = layers_list
    layer = l{1};
    n = summary.(layer).total_chans;
    s = summary.(layer).sig_chans;
    pct = 100 * s / max(n, 1);
    fprintf('%s: %d/%d channels (%.1f%% significant)\n', upper(layer), s, n, pct);
end

end
