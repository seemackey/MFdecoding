%% MTF Batch: Phase-Locking Value (PLV) with Random-Phase Null (Post Window)
% Requires: MTF_loadMATfile.m on MATLAB path (your provided loader)
% For each recording this script:
%   1) Loads & epochs signals (LFP/CSD/MUA) via MTF_loadMATfile
%   2) Saves epoched data to /epoched (provenance)
%   3) Computes PLV at f0 = 1000/ISI (and optional harmonics) in the POST window
%      using complex demodulation per trial, then:
%         - PLV (vector strength) across trials per channel
%         - Analytic Rayleigh p-value for uniform phases (optional)
%         - Amplitude-weighted random-phase Monte-Carlo p-value (default)
%         - Benjamini–Hochberg FDR across channels within each condition/harmonic
%   4) Saves results to /phase
%
% Notes:
% - Works with many cycles per trial: demodulation fits a sinusoid at f0 across the whole
%   post window; PLV measures cross-trial phase concentration (steady-state tagging).
% - Demodulation uses an optional taper to reduce edge effects; phases are invariant to
%   constant time offsets, so alignment only affects mean phase, not PLV magnitude.
% - You can enable harmonics (e.g., 2*f0, 3*f0) below.

clear; clc; close all;

%% ===================== CONFIG =====================
parent_dir   = 'E:\MTF\data\noise\core\right\imported\';  % root dir containing 1-/2- files

% Epoch & analysis windows (ms relative to event)
epoch_tframe = [-50 700];   % must encompass the post window
post_win_ms  = [0 650];      % POST window for PLV

% Channel selection (empty = use all available channels per file/signal)
selected_channels = [];

% Demodulation / windowing
use_taper     = true;        % apply Hann taper within post window for stability

% Hypothesis testing / FDR
alpha         = 0.05;        % significance level
n_mc          = 1000;        % Monte-Carlo draws for weighted random-phase p-value
use_fdr       = true;        % apply BH-FDR across channels within each condition/harmonic

% Harmonics (set to [1] to test only the fundamental f0)
harmonics     = [1 2 3];     % test f0, 2*f0, 3*f0

% Output subdirectories
out_epoched = fullfile(parent_dir, 'epoched');
out_phase   = fullfile(parent_dir, 'phaseITPC');
if ~exist(out_epoched, 'dir'); mkdir(out_epoched); end
if ~exist(out_phase,   'dir'); mkdir(out_phase);   end

% Discover candidate files
all_files = dir(fullfile(parent_dir, '1-*@o*.mat'));
all_files = [all_files; dir(fullfile(parent_dir, '2-*@o*.mat'))];
fprintf('Found %d candidate files in %s\n', numel(all_files), parent_dir);

%% ===================== MAIN LOOP =====================
for i = 1:numel(all_files)
    file = all_files(i);
    [~, name_no_ext, ~] = fileparts(file.name);  % e.g., '1-xxx@oe'
    fprintf('\n>> Processing %s\n', name_no_ext);

    % Determine modality (for logs only)
    if contains(name_no_ext, '@oe')
        modality_flag = 'oe'; %#ok<NASGU>
    elseif contains(name_no_ext, '@om')
        modality_flag = 'om'; %#ok<NASGU>
    else
        warning('Skipping unknown file type: %s', name_no_ext);
        continue;
    end

    % Match EV2 (strip leading 1-/2- and trailing @o[em])
    ev2_base = regexprep(name_no_ext, '^[12]-(.*)@o[em]$', '$1');
    if isempty(ev2_base)
        warning('Unable to extract EV2 base for %s. Skipping...', name_no_ext);
        continue;
    end
    ev2_file = dir(fullfile(parent_dir, [ev2_base '.ev2']));
    if isempty(ev2_file)
        warning('EV2 not found for %s. Skipping...', name_no_ext);
        continue;
    end

    % Stage in temp dir
    temp_dir = fullfile(parent_dir, ['temp_' name_no_ext]);
    if ~exist(temp_dir, 'dir'); mkdir(temp_dir); end
    copyfile(fullfile(file.folder, file.name), fullfile(temp_dir, file.name));
    copyfile(fullfile(ev2_file.folder, ev2_file.name), fullfile(temp_dir, ev2_file.name));

    % ==== Load & epoch ====
    try
        [epoched_data, srate] = MTF_loadMATfile(temp_dir, epoch_tframe);
    catch err
        warning('Load/epoch failed for %s: %s', name_no_ext, err.message);
        disp(err.stack);
        rmdir(temp_dir, 's');
        continue;
    end

    % Save epoched (provenance)
    try
        save(fullfile(out_epoched, sprintf('%s_epoched.mat', name_no_ext)), ...
             'epoched_data', 'srate', 'epoch_tframe', '-v7.3');
    catch err
        warning('Failed saving epoched for %s: %s', name_no_ext, err.message);
    end

    % Channel selection per signal type from first nonempty condition
    sel = struct();
    sel.LFP = pick_channels_from(epoched_data, 'LFP', selected_channels);
    sel.CSD = pick_channels_from(epoched_data, 'CSD', selected_channels);
    sel.MUA = pick_channels_from(epoched_data, 'MUA', selected_channels);

    % Convert post window to indices
    try
        post_idx = ms2idx(post_win_ms, srate, epoch_tframe);
    catch err
        warning('Window->index conversion failed for %s: %s', name_no_ext, err.message);
        rmdir(temp_dir, 's');
        continue;
    end

    % ==== PLV analysis (random-phase null) ====
    try
        phase_out = struct();
        phase_out.meta = struct('file', name_no_ext, 'srate', srate, ...
                                'epoch_tframe', epoch_tframe, 'post_ms', post_win_ms, ...
                                'alpha', alpha, 'n_mc', n_mc, 'use_fdr', use_fdr, ...
                                'harmonics', harmonics, 'use_taper', use_taper);
        if isfield(epoched_data, 'ISI_ms'); phase_out.meta.ISI_ms = epoched_data.ISI_ms; end

        if isfield(epoched_data,'LFP') && ~isempty(epoched_data.LFP)
            phase_out.LFP = plv_by_condition(epoched_data.LFP, sel.LFP, srate, post_idx, ...
                                             epoched_data.ISI_ms, harmonics, use_taper, alpha, n_mc, use_fdr);
        end
        if isfield(epoched_data,'CSD') && ~isempty(epoched_data.CSD)
            phase_out.CSD = plv_by_condition(epoched_data.CSD, sel.CSD, srate, post_idx, ...
                                             epoched_data.ISI_ms, harmonics, use_taper, alpha, n_mc, use_fdr);
        end
        if isfield(epoched_data,'MUA') && ~isempty(epoched_data.MUA)
            phase_out.MUA = plv_by_condition(epoched_data.MUA, sel.MUA, srate, post_idx, ...
                                             epoched_data.ISI_ms, harmonics, use_taper, alpha, n_mc, use_fdr);
        end

        save(fullfile(out_phase, sprintf('%s_PLV.mat', name_no_ext)), 'phase_out', '-v7.3');
    catch err
        warning('PLV analysis failed for %s: %s', name_no_ext, err.message);
        disp(err.stack);
    end

    % Cleanup temp
    rmdir(temp_dir, 's');
end

fprintf('\nAll done.\n');

%% ===================== HELPERS =====================
function chans = pick_channels_from(epoched_data, fieldName, selected_channels)
% Return channel indices for this signal type from the first nonempty condition
    chans = [];
    if ~isfield(epoched_data, fieldName) || isempty(epoched_data.(fieldName))
        return;
    end
    idx = find(cellfun(@(x) ~isempty(x), epoched_data.(fieldName)), 1, 'first');
    if isempty(idx); return; end
    nCh = size(epoched_data.(fieldName){idx},1);
    if isempty(selected_channels)
        chans = 1:nCh;
    else
        chans = selected_channels(selected_channels>=1 & selected_channels<=nCh);
    end
end

function idx = ms2idx(win_ms, srate, epoch_tframe)
% Convert [t1 t2] ms (relative to event) into 1-based sample indices within epoch arrays
    if win_ms(1) < epoch_tframe(1) || win_ms(2) > epoch_tframe(2)
        error('Window [%d %d] ms lies outside epoch_tframe [%d %d] ms.', ...
               win_ms(1), win_ms(2), epoch_tframe(1), epoch_tframe(2));
    end
    rel0 = epoch_tframe(1);  % ms at sample index 1
    t1 = win_ms(1) - rel0;   % ms from epoch start
    t2 = win_ms(2) - rel0;
    s1 = max(1, round(t1/1000*srate));
    s2 = round(t2/1000*srate);
    if s2 <= s1
        error('Invalid window: computed indices [%d %d].', s1, s2);
    end
    idx = [s1 s2];
end

function out = plv_by_condition(dataCell, chans, srate, post_idx, ISI_ms, harmonics, use_taper, alpha, n_mc, use_fdr)
% Compute PLV per condition and per requested harmonic.
% Returns a 1xNcond cell; each cell is a struct array (length = numel(harmonics)) with fields:
%   freq, PLV [ch x 1], mean_phase [ch x 1], p_rayleigh [ch x 1],
%   p_weighted [ch x 1], q [ch x 1], signif [ch x 1]
    nCond = numel(dataCell);
    out = cell(1, nCond);

    for c = 1:nCond
        X = dataCell{c};
        if isempty(X), out{c} = struct([]); continue; end
        chAvail = size(X,1);
        useCh = chans(chans>=1 & chans<=chAvail);
        if isempty(useCh), out{c} = struct([]); continue; end

        seg = X(useCh,:,post_idx(1):post_idx(2));   % [ch x tr x time]
        base_f0 = 1000 / ISI_ms(c);

        % Prepare outputs per harmonic
        H = numel(harmonics);
        S(H) = struct('freq',[], 'PLV',[], 'mean_phase',[], ...
                      'p_rayleigh',[], 'p_weighted',[], 'q',[], 'signif',[]); %#ok<AGROW>

        for h = 1:H
            f = base_f0 * harmonics(h);
            if f <= 0 || f >= (srate/2)
                % Out of band: mark as empty
                S(h).freq = f; S(h).PLV = []; S(h).mean_phase = [];
                S(h).p_rayleigh = []; S(h).p_weighted = []; S(h).q = []; S(h).signif = [];
                continue;
            end
            % PLV OR ITPC
            [PLV, mean_phase, p_ray, p_w] = plv_and_randomphase(seg, srate, f, use_taper, n_mc);
            % Normalize PLV for trial count: Rayleigh's Z (aka ITPCz)
            nTr_local = size(seg, 2);                 % number of trials in this condition
            ITPCz     = nTr_local .* (PLV.^2);        % per-channel ITPCz
            PLV = ITPCz/nTr_local;

            % FDR across channels for this harmonic
            signif = p_w < alpha; q = [];
            if use_fdr
                q = fdr_bh(p_w);
                signif = q < alpha;
            end

            S(h).freq        = f;
            S(h).PLV         = PLV;
            S(h).mean_phase  = mean_phase;
            S(h).p_rayleigh  = p_ray;
            S(h).p_weighted  = p_w;
            S(h).q           = q;
            S(h).signif      = signif;
        end

        out{c} = S;
    end
end

function [PLV, mean_phase, p_rayleigh, p_weighted] = plv_and_randomphase(seg, srate, f0, use_taper, n_mc)
% seg: [ch x trials x time] post window
% Returns per-channel PLV and p-values for random-phase nulls
    [nCh,nTr,nT] = size(seg);
    t = (0:nT-1)/srate;                % seconds, relative to window start

    % Window/taper
    if use_taper
        if exist('hann','file')
            win = hann(nT).';
        else
            win = 0.5 - 0.5*cos(2*pi*(0:nT-1)/nT); % fallback Hann
        end
    else
        win = ones(1,nT);
    end

    kern = exp(-1i*2*pi*f0*t);         % complex demod kernel

    Z = zeros(nCh,nTr) + 1i*zeros(nCh,nTr);
    for tr = 1:nTr
        x = double(squeeze(seg(:,tr,:)));      % [ch x time]
        xw = x .* win;                          % taper
        a = xw * kern.';                        % complex dot against sinusoid
        Z(:,tr) = a / nT;                       % scale to be length-agnostic
    end

    A   = abs(Z);                               % trial amplitudes [ch x trials]
    phi = angle(Z);                             % trial phases [ch x trials]

    % PLV and mean phase per channel
    C = mean(exp(1i*phi), 2);                   % complex mean over trials
    PLV = abs(C);
    mean_phase = angle(C);

    % Analytic Rayleigh p-value (uniform phase null)
    z = nTr .* (PLV.^2);
    p_rayleigh = exp(-z) .* (1 + (2*z - z.^2)/(4*nTr));  % small-sample correction

    % Amplitude-weighted random-phase Monte-Carlo p-value
    p_weighted = nan(nCh,1);
    for ch = 1:nCh
        w = A(ch,:);                    % fix amplitudes; randomize phases
        robj = abs(sum(w .* exp(1i*phi(ch,:)))) / sum(w + eps);
        Rmc  = zeros(n_mc,1);
        for b = 1:n_mc
            Rmc(b) = abs(sum(w .* exp(1i*(2*pi*rand(1,nTr))))) / sum(w + eps);
        end
        p_weighted(ch) = mean(Rmc >= robj);
    end
end

function q = fdr_bh(p)
% Benjamini–Hochberg FDR-adjusted p-values (q-values)
    p = p(:); m = numel(p);
    [ps, idx] = sort(p);
    qtmp = m ./ (1:m)' .* ps;
    qtmp = flipud(cummin(flipud(qtmp)));
    q = nan(m,1); q(idx) = qtmp;
end
