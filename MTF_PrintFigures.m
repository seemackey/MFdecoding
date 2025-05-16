inputDir = 'E:\MTF\data\noise\pb\left\imported\subset_cnt\imported\decoding_results_LDA\';  
outputDir = 'E:\MTF\data\noise\pb\left\imported\subset_cnt\imported\decoding_results_LDA\';  
figFiles = dir(fullfile(inputDir, '*.fig'));
if ~exist(outputDir)
    mkdir(outputDir)
end

for i = 1:length(figFiles)
    figPath = fullfile(inputDir, figFiles(i).name);
    [~, baseName] = fileparts(figFiles(i).name);

    % Open the .fig file
    fig = openfig(figPath, 'invisible');

    % Save as .jpg (high resolution)
    %saveas(fig, fullfile(outputDir, [baseName, '.jpg']));

    % Alternatively: exportgraphics for higher quality
     exportgraphics(fig, fullfile(outputDir, [baseName, '.jpg']), 'Resolution', 300);

    close(fig);
end

disp('All .fig files converted to .jpg.');
