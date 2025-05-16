function MTF_compute_pairwise_diffISI(dir_path, bin_size, file_type)
% For each decoding file + associated layer file, computes all pairwise differences
% in Best ISI between channels in different layers (supra, granular, infra).
% Saves result to a .mat file in a "diffISI" subdirectory.

% Create output subdirectory
out_dir = fullfile(dir_path, 'diffISI');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Get decoding files
files = dir(fullfile(dir_path, sprintf('*%s_decoding_%dms.mat', file_type, bin_size)));
if isempty(files)
    error('No matching decoding result files found in %s for bin size %dms and type %s.', dir_path, bin_size, file_type);
end

for i = 1:length(files)
    try
        filename = files(i).name;
        filepath = fullfile(dir_path, filename);
        [~, decode_base, ~] = fileparts(filename);

        % Derive layer file path
        layer_base = regexprep(decode_base, '@o[em]_decoding_\d+ms$', '');
        layer_filename = [layer_base '_layers.mat'];
        layer_file_path = fullfile(dir_path, layer_filename);

        if ~isfile(layer_file_path)
            warning('Layer file not found for %s. Skipping.', decode_base);
            continue;
        end

        % Load data
        matdata = matfile(filepath);
        accuracies = matdata.decoding_accuracy_conditions;
        chance_hiCI = mean(matdata.permutation_accuracy_hiCI, 2);

        temp = load(filepath, 'epoched_data');
        ISIs = temp.epoched_data.ISI_ms;
        clear temp;

        layer_data = load(layer_file_path);
        layers = layer_data.layers;

        num_chans = size(accuracies, 1);
        bestISI_per_chan = nan(num_chans, 1);

        % Extract Best ISI per channel
        for ch = 1:num_chans
            if size(accuracies, 3) == 1
                [val, idx] = max(accuracies(ch, :));
                if val > chance_hiCI(ch)
                    bestISI_per_chan(ch) = ISIs(idx);
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
                    bestISI_per_chan(ch) = best_ISI;
                end
            end
        end

        % Create all pairwise differences
        diffs = [];
        pairs = {@(a,b,n1,n2) deal(a,b,n1,n2), {'supra','granular'}, {'supra','infra'}, {'granular','infra'}};
        for p = 2:length(pairs)
            name1 = pairs{p}{1}; name2 = pairs{p}{2};
            if isfield(layers, name1) && isfield(layers, name2)
                ch1 = layers.(name1); ch2 = layers.(name2);
                [C1, C2] = ndgrid(ch1, ch2);
                C1 = C1(:); C2 = C2(:);
                ISI1 = bestISI_per_chan(C1);
                ISI2 = bestISI_per_chan(C2);
                valid = ~isnan(ISI1) & ~isnan(ISI2);
                tbl = table(repmat({decode_base}, sum(valid),1), repmat({name1}, sum(valid),1), repmat({name2}, sum(valid),1),...
                            C1(valid), C2(valid), ISI1(valid), ISI2(valid), ISI1(valid) - ISI2(valid),...
                            'VariableNames', {'File','Layer1','Layer2','Ch1','Ch2','BestISI1','BestISI2','Diff'});
                diffs = [diffs; tbl];
            end
        end

        % Save result
        save(fullfile(out_dir, [decode_base '_diffISI.mat']), 'diffs');
        fprintf('Saved diffISI for %s\n', decode_base);

    catch ME
        warning('Failed to process %s: %s', files(i).name, ME.message);
    end
end

end
