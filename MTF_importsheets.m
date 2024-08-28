

% Load the spreadsheet data
spreadsheet_data = readtable('E:\MTF\ev2_subset.xlsx');

% Initialize a struct to store all the information
data_info = struct();

% Loop through each row in the spreadsheet and store information in the struct
for i = 1:height(spreadsheet_data)
    filename = spreadsheet_data.filename{i};
    
    % Store electrode information
    data_info(i).filename = filename;
    data_info(i).electrode1 = spreadsheet_data.electrode1{i};
    data_info(i).electrode2 = spreadsheet_data.electrode2{i};
    
    % Parse the modulation frequency indices for click train, samN, and samT
    if ~isempty(spreadsheet_data.clicktrain{i})
        data_info(i).clicktrain = eval(spreadsheet_data.clicktrain{i});
    else
        data_info(i).clicktrain = [];
    end
    
    if ~isempty(spreadsheet_data.samN{i})
        data_info(i).samN = eval(spreadsheet_data.samN{i});
    else
        data_info(i).samN = [];
    end
    
    if ~isempty(spreadsheet_data.samT{i})
        data_info(i).samT = eval(spreadsheet_data.samT{i});
    else
        data_info(i).samT = [];
    end
end

% Function to find modulation frequencies based on the EV2 filename
function mod_freqs = get_modulation_frequencies(ev2_filename, data_info)
    mod_freqs = [];
    for i = 1:length(data_info)
        if strcmp(data_info(i).filename, ev2_filename)
            mod_freqs.clicktrain = data_info(i).clicktrain;
            mod_freqs.samN = data_info(i).samN;
            mod_freqs.samT = data_info(i).samT;
            break;
        end
    end
end

% Example of how to use the function
ev2_filename = 'qu07019.ev2';
mod_freqs = get_modulation_frequencies(ev2_filename, data_info);
disp(mod_freqs);
