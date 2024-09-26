function ci = MTF_percentile_CI(data, ci_level)
    % Calculate the percentile confidence interval for each channel and time point
    % Data is expected to be in the format: channels x trials x time points
    num_channels = size(data, 1);
    num_time_points = size(data, 3);
    
    % Initialize ci structure
    ci_lower = zeros(num_channels, num_time_points);
    ci_upper = zeros(num_channels, num_time_points);
    
    % Calculate percentiles
    lower_percentile = (1 - ci_level) / 2 * 100;
    upper_percentile = (1 - (1 - ci_level) / 2) * 100;

    for ch = 1:num_channels
        for t = 1:num_time_points
            trial_data = squeeze(data(ch, :, t));
            ci_lower(ch, t) = prctile(trial_data, lower_percentile);
            ci_upper(ch, t) = prctile(trial_data, upper_percentile);
        end
    end
    
    % Assign to ci structure
    ci.lower = ci_lower;
    ci.upper = ci_upper;
end