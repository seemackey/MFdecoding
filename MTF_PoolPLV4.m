%% Pool PLV by Cortical Layer across files in imported/phase (REV C)
% Changes in REV C:
%   • Buckets by ISI (snaps f0 to nearest of your canonical ISIs) to avoid float-splitting.
%   • Adds an AUDIT mode that prints per-file alignment and pooling stats.
%   • Keeps OE→CSD and OM→MUA routing; silences cross-modality warnings.
%   • Saves PNG + FIG; log-scaled x-axis (from REV B).

clear; clc; close all;

%% ===================== CONFIG =====================
parent_dir       = 'E:/MTF/data/noise/core/left/imported/';
phase_dir        = fullfile(parent_dir, 'phaseITPC');
harmonic_to_use  = 1;      % 1 = f0, 2 = 2*f0, etc.
AUDIT            = true;   % print per-file diagnostics

save_output      = true;
make_quick_plots = true;   % quick visualization per layer

% Canonical ISIs used in your loader (ms) — order important for mapping
CANON_ISI_MS = [640, 320, 226, 160, 113, 80, 57, 40, 28, 20, 14, 10, 5];

%% ===================== FIND PHASE FILES =====================
if ~exist(phase_dir, 'dir')
    error('Phase directory not found: %s', phase_dir);
end
plv_files = dir(fullfile(phase_dir, '*_PLV.mat'));
if isempty(plv_files)
    error('No *_PLV.mat files found in %s. Run MTF_batch_PLV_randomphase.m first.', phase_dir);
end
disp('found plv files')

%% ===================== AGGREGATION SETUP =====================
% We will pool separately for CSD (from OE files) and MUA (from OM files)
pool = struct(); raw = struct();
pool.CSD = containers.Map('KeyType','double','ValueType','any');
raw.CSD  = containers.Map('KeyType','double','ValueType','any');
pool.MUA = containers.Map('KeyType','double','ValueType','any');
raw.MUA  = containers.Map('KeyType','double','ValueType','any');

%% ===================== ITERATE FILES =====================
imported_dir = parent_dir; % phase_dir is a subfolder of this
for i = 1:numel(plv_files)
    fpath = fullfile(plv_files(i).folder, plv_files(i).name);
    S = load(fpath);
    if ~isfield(S, 'phase_out'), continue; end
    P = S.phase_out;

    if ~isfield(P,'meta') || ~isfield(P.meta,'file')
        % Cannot locate layer file; skip silently
        if AUDIT
            disp('cannot locate layer file')
        continue;
        end
    end

    rec_tag = P.meta.file;   % e.g., '1-wn35013@oe' or '2-wn35013@om'
    base_core = regexprep(rec_tag, '^[12]-(.*)@o[em]$', '$1'); % e.g., 'wn35013'

    % ---- Find matching layer file in imported/ ----
    layer_fp = find_layer_file(imported_dir, rec_tag, base_core);
    if isempty(layer_fp)
        if AUDIT
        continue;
        end
    end

    L = load(layer_fp);
    lyr = extract_layers_struct(L); % struct with fields: supra, granular, infra; and optional .file

    % Build a layer vector for base hardware channels (length = max channel index)
    [layer_vec, base_nCh] = layer_vector_from_struct(lyr);

    % Decide which signal to process for this file
    sigs_for_file = signals_for_file(rec_tag); % {'CSD'} for @oe, {'MUA'} for @om

    for s = 1:numel(sigs_for_file)
        sig = sigs_for_file{s};
        if ~isfield(P, sig) || isempty(P.(sig)), continue; end
        C = P.(sig);   % 1xNcond cell; each cell is an array over harmonics
        nCond = numel(C); if nCond==0, continue; end

        % Determine nCh for this signal from the first available harmonic
        [nCh_sig, ~] = first_harmonic_size(C);
        if nCh_sig == 0, continue; end

        % Align layer vector to the signal's channel count
        layer_sig = align_layer_vec(layer_vec, base_nCh, nCh_sig);

        if AUDIT
            % Count per layer for this file/signal
            lstr = string(layer_sig(:));
            nSG = sum(startsWith(lstr,'supragran', 'IgnoreCase',true));
            nG  = sum(startsWith(lstr,'gran',      'IgnoreCase',true));
            nIG = sum(startsWith(lstr,'infra',     'IgnoreCase',true));
            
        end

        for cnd = 1:nCond
            if isempty(C{cnd}), continue; end
            H = C{cnd};
            if harmonic_to_use > numel(H), continue; end
            h = H(harmonic_to_use);
            % --- frequency key (snap to canonical ISI) ---
            % Prefer the freq saved in the PLV struct; otherwise derive from ISI list.
            if isfield(h,'freq') && ~isempty(h.freq)
                f0_raw = h.freq;                       % Hz
            elseif isfield(P,'meta') && isfield(P.meta,'ISI_ms') && cnd <= numel(P.meta.ISI_ms)
                f0_raw = 1000 / P.meta.ISI_ms(cnd);    % Hz
            else
                if AUDIT, fprintf('   (no freq info) SKIP %s cond%02d\n', sig, cnd); end
                continue;
            end
            keyF = make_key(f0_raw, CANON_ISI_MS);     % snapped key (Hz)
            
            % --- PLV vector and significance mask ---
            if ~isfield(h,'PLV') || isempty(h.PLV), continue; end
            plv_vec = h.PLV(:);                        % [ch x 1]
            
            % Use per-channel significance for this condition/harmonic
            if isfield(h,'signif') && ~isempty(h.signif)
                sigMask = logical(h.signif(:));
            elseif isfield(h,'sig') && ~isempty(h.sig)
                sigMask = logical(h.sig(:));           % fallback field name, if present
            else
                if AUDIT, fprintf('   (no signif mask) SKIP %s cond%02d f=%.4f\n', sig, cnd, f0_raw); end
                continue;
            end
            
            % --- reconcile lengths & apply mask ---
            m = min([numel(plv_vec), numel(layer_sig), numel(sigMask)]);
            if m == 0, continue; end
            plv_vec   = plv_vec(1:m);
            layer_use = layer_sig(1:m);
            sigMask   = sigMask(1:m);
            
            % Keep only significant channels
            plv_vec   = plv_vec(sigMask);
            layer_use = layer_use(sigMask);
            if isempty(plv_vec), continue; end
            
            % Optional audit for this condition/frequency
            if AUDIT
                lstr = string(layer_use(:));
                nSGs = sum(startsWith(lstr,'supragran','IgnoreCase',true));
                nGs  = sum(startsWith(lstr,'gran','IgnoreCase',true));
                nIGs = sum(startsWith(lstr,'infra','IgnoreCase',true));
                fprintf('   %s cond%02d f=%.4f Hz | sig chans: SG=%d G=%d IG=%d (total %d)\n', ...
                        sig, cnd, keyF, nSGs, nGs, nIGs, numel(plv_vec));
            end
            
            % --- pool only significant channels ---
            [pool, raw] = append_plv(pool, raw, sig, keyF, plv_vec, layer_use);

        end
    end
end

%% ===================== SUMMARIZE TO TABLES =====================
summary = struct();
[s_CSD, r_CSD] = summarize_signal(pool.CSD, raw.CSD); summary.CSD = s_CSD; summary.CSD_raw = r_CSD;
[s_MUA, r_MUA] = summarize_signal(pool.MUA, raw.MUA); summary.MUA = s_MUA; summary.MUA_raw = r_MUA;

%% ===================== SAVE =====================
if save_output
    out_file = fullfile(phase_dir, 'PLV_layer_pool.mat');
    save(out_file, 'summary', 'pool', 'raw', 'parent_dir', 'phase_dir', 'CANON_ISI_MS');
    disp('saved data')
end

%% ===================== QUICK PLOTS =====================
if make_quick_plots
    fig_dir = fullfile(phase_dir, 'figures');
    if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
    quick_plot(summary.CSD,  'CSD',  fig_dir);
    quick_plot(summary.MUA,  'MUA',  fig_dir);
end

disp('plots done')

%% ===================== HELPERS =====================
function sigs = signals_for_file(rec_tag)
% Return which signal(s) to process for a given file tag
    if contains(rec_tag, '@oe')
        sigs = {'CSD'};   % oe -> CSD only
    elseif contains(rec_tag, '@om')
        sigs = {'MUA'};   % om -> MUA only
    else
        sigs = {};        % unknown
    end
end

function layer_fp = find_layer_file(imported_dir, rec_tag, base_core)
% Try several filename patterns; fall back to searching all *layers.mat and matching .file field
    cand = {};
    cand{end+1} = fullfile(imported_dir, [base_core '_layers.mat']);
    cand{end+1} = fullfile(imported_dir, [rec_tag '_layers.mat']);

    for i = 1:numel(cand)
        if exist(cand{i}, 'file'), layer_fp = cand{i}; return; end
    end

    % Search all *layers.mat and match 'layers.file' if present
    Ls = dir(fullfile(imported_dir, '*layers.mat'));
    for i = 1:numel(Ls)
        try
            tmp = load(fullfile(Ls(i).folder, Ls(i).name));
            lyr = extract_layers_struct(tmp);
            if isfield(lyr, 'file')
                f = string(lyr.file);
                if f == string(base_core) || f == string(rec_tag)
                    layer_fp = fullfile(Ls(i).folder, Ls(i).name); return;
                end
            end
        catch
            % skip invalid files
        end
    end
    layer_fp = '';
end

function lyr = extract_layers_struct(S)
% Accept either S.layers.supra/..., or top-level fields supra/... (as in your screenshot)
    if isfield(S, 'layers') && isstruct(S.layers)
        lyr = S.layers;
    else
        % find a struct that has supra/granular/infra
        fn = fieldnames(S);
        lyr = struct();
        for k = 1:numel(fn)
            val = S.(fn{k});
            if isstruct(val) && all(isfield(val, {'supra','granular','infra'}))
                lyr = val; return;
            end
        end
        % maybe S itself has them
        if all(isfield(S, {'supra','granular','infra'}))
            lyr = S; return;
        end
        error('No layer struct with fields supra/granular/infra found in layer file.');
    end
end

function [layer_vec, base_nCh] = layer_vector_from_struct(lyr)
% Build a per-channel categorical vector from channel index lists in lyr
    supra = get_index_vec(lyr, 'supra');
    granular = get_index_vec(lyr, 'granular');
    infra = get_index_vec(lyr, 'infra');
    base_nCh = max([supra(:); granular(:); infra(:)]);
    lab = strings(base_nCh,1); lab(:) = missing;
    lab(supra)   = "supragranular";
    lab(granular)= "granular";
    lab(infra)   = "infragranular";
    layer_vec = categorical(lower(lab), ["supragranular","granular","infragranular"], 'Ordinal', true);
end

function idx = get_index_vec(lyr, field)
    if isfield(lyr, field)
        idx = lyr.(field);
        if iscell(idx), idx = cell2mat(idx); end
        idx = idx(:)';
    else
        idx = [];
    end
end

function [nCh_sig, nCond] = first_harmonic_size(C)
    nCh_sig = 0; nCond = numel(C);
    for c = 1:nCond
        if isempty(C{c}), continue; end
        H = C{c};
        for h = 1:numel(H)
            if isfield(H(h),'PLV') && ~isempty(H(h).PLV)
                nCh_sig = numel(H(h).PLV); return;
            end
        end
    end
end

function layer_sig = align_layer_vec(layer_vec, base_nCh, nCh_sig)
% Map hardware-channel layer vector to the signal's channel count.
% Both MUA and CSD have length base_nCh-2 in your pipeline (edges removed).
    v = layer_vec;
    if nCh_sig == base_nCh-2
        layer_sig = v(2:end-1); return;
    elseif nCh_sig == base_nCh
        layer_sig = v; return;
    end
    % Fallbacks: prefer truncation over padding to avoid categorical issues
    layer_sig = v(1:min(numel(v), nCh_sig));
end

function keyF = make_key(f0_raw, CANON_ISI_MS)
% Snap frequency to the nearest canonical ISI (in Hz) to avoid float splitting
    f_list = 1000 ./ CANON_ISI_MS;          % Hz
    [~, idx] = min(abs(f_list - f0_raw));
    keyF = round(f_list(idx), 6);           % quantize to 1e-6 Hz
end

function [pool, raw] = append_plv(pool, raw, sig, keyF, plv_vec, layer_sig)
    key = keyF;  % numeric key for containers.Map
    if ~isKey(pool.(sig), key)
        pool.(sig)(key) = struct('supragranular', [], 'granular', [], 'infragranular', []);
        raw.(sig)(key)  = struct('supragranular', [], 'granular', [], 'infragranular', []);
    end
    S = pool.(sig)(key); R = raw.(sig)(key);

    l = lower(string(layer_sig(:)'));
    is_sg = (l=="supragranular" | l=="sg" | startsWith(l,"supra"));
    is_g  = (l=="granular"      | l=="g"  | startsWith(l,"gran"));
    is_ig = (l=="infragranular" | l=="ig" | startsWith(l,"infra"));

    S.supragranular = [S.supragranular; plv_vec(is_sg)];
    S.granular      = [S.granular;      plv_vec(is_g) ];
    S.infragranular = [S.infragranular; plv_vec(is_ig)];

    R.supragranular = [R.supragranular; plv_vec(is_sg)];
    R.granular      = [R.granular;      plv_vec(is_g) ];
    R.infragranular = [R.infragranular; plv_vec(is_ig)];

    pool.(sig)(key) = S; raw.(sig)(key)  = R;
end

function [tbl, raw_tbl] = summarize_signal(poolMap, rawMap)
% Convert pooled containers into tidy tables with mean and bootstrap CIs
    if isempty(poolMap) || poolMap.Count==0
        tbl = table(); raw_tbl = table(); return;
    end

    keys = cell2mat(poolMap.keys)'; keys = sort(keys);
    freq = []; layer = []; n = []; meanPLV = []; ci_lo = []; ci_hi = [];

    for k = 1:numel(keys)
        f0 = keys(k); S = poolMap(f0); cats = fieldnames(S);
        for ci = 1:numel(cats)
            catName = cats{ci};
            if ~isfield(S, catName) || isempty(S.(catName)), continue; end
            vals = S.(catName);
            vals = vals(~isnan(vals));

            if isempty(vals), continue; end
            freq   = [freq;   f0];
            layer  = [layer;  string(catName)];
            n      = [n;      numel(vals)];
            meanPLV= [meanPLV; mean(vals)];
            % bootstrap CI on mean
            B = 1000; boots = zeros(B,1);
            for b = 1:B
                idx = randi(numel(vals), [numel(vals) 1]);
                boots(b) = mean(vals(idx));
            end
            ci_lo = [ci_lo; prctile(boots, 2.5)];
            ci_hi = [ci_hi; prctile(boots,97.5)];
        end
    end

    tbl = table(freq, categorical(layer), n, meanPLV, ci_lo, ci_hi, ...
                'VariableNames', {'freq','layer','N','mean_PLV','ci_lo','ci_hi'});

    % Raw table (long form)
    rf = []; rl = []; rv = [];
    for k = 1:numel(keys)
        f0 = keys(k); S = rawMap(f0); cats = fieldnames(S);
        for ci = 1:numel(cats)
            cat = cats{ci}; vals = S.(cat); vals = vals(~isnan(vals));
            rf = [rf; f0*ones(numel(vals),1)];
            rl = [rl; repmat(string(cat), numel(vals),1)];
            rv = [rv; vals(:)];
        end
    end
    raw_tbl = table(rf, categorical(rl), rv, 'VariableNames', {'freq','layer','PLV'});
end

function quick_plot(T, sig, fig_dir)
    if isempty(T) || height(T)==0, return; end
    layers = unique(T.layer);
    figure('Color','w','Position',[100 100 1100 700]); hold on; grid on; box on;
    set(gca,'FontSize',11,'LineWidth',1.2);
    title(sprintf('PLV by Layer — %s', sig)); xlabel('Frequency (Hz)'); ylabel('PLV');
    set(gca,'XScale','log','XMinorTick','on');

    % Ensure consistent order in legend
    order = ["supragranular","granular","infragranular"];
    T.layer = categorical(string(T.layer), order, 'Ordinal', true);
    layers = order(ismember(order, categories(T.layer)));

    cmap = lines(numel(layers));
    for li = 1:numel(layers)
        Lcat = layers(li);
        idx = T.layer == Lcat;
        f = T.freq(idx); [f, ord] = sort(f);
        m = T.mean_PLV(idx); m = m(ord);
        lo = T.ci_lo(idx); lo = lo(ord);
        hi = T.ci_hi(idx); hi = hi(ord);
        fill_between(f, lo, hi, cmap(li,:), 0.15);
        plot(f, m, 'LineWidth', 1.8, 'Color', cmap(li,:));
    end

    allf = T.freq(T.freq>0);
    if ~isempty(allf), xlim([min(allf)*0.9, max(allf)*1.1]); end

    legend(cellstr(layers), 'Location','northeastoutside');
    outbase = fullfile(fig_dir, sprintf('PLV_by_layer_%s', sig));
    exportgraphics(gcf, [outbase '.png'], 'Resolution', 200);
    savefig(gcf, [outbase '.fig']);
    close(gcf);
end

function fill_between(x, y1, y2, colorRGB, alpha)
% shaded patch helper for quick plots
    xv = [x(:); flipud(x(:))]; yv = [y1(:); flipud(y2(:))];
    p = patch(xv, yv, colorRGB, 'EdgeColor','none'); set(p, 'FaceAlpha', alpha);
end
