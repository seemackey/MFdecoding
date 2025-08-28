plv_vec = h.PLV(:);           % [ch x 1]

% --- significance mask (per-channel) ---
if isfield(h,'signif') && ~isempty(h.signif)
    sigMask = logical(h.signif(:));
elseif isfield(h,'sig') && ~isempty(h.sig)
    sigMask = logical(h.sig(:));   % fallback name if used elsewhere
else
    % Conservative: skip pooling for this condition if no mask present
    if AUDIT
        fprintf('   (no signif mask present) SKIP %s f=%.4f\n', sig, f0_raw);
    end
    continue;
end

% Reconcile lengths and apply mask
m = min([numel(plv_vec), numel(layer_sig), numel(sigMask)]);
if m == 0, continue; end
plv_vec   = plv_vec(1:m);
layer_use = layer_sig(1:m);
sigMask   = sigMask(1:m);

% Keep only significant channels
plv_vec   = plv_vec(sigMask);
layer_use = layer_use(sigMask);
if isempty(plv_vec), continue; end

% Optional audit: show how many sig channels per layer for this freq
if AUDIT
    lstr = string(layer_use(:));
    nSGs = sum(startsWith(lstr,'supragran','IgnoreCase',true));
    nGs  = sum(startsWith(lstr,'gran','IgnoreCase',true));
    nIGs = sum(startsWith(lstr,'infra','IgnoreCase',true));
    fprintf('   %s cond%02d f=%.4f Hz | sig chans: SG=%d G=%d IG=%d (total %d)\n', ...
            sig, cnd, keyF, nSGs, nGs, nIGs, numel(plv_vec));
end

[pool, raw] = append_plv(pool, raw, sig, keyF, plv_vec, layer_use);
