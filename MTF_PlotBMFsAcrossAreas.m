%clear
% dir1 = 'E:\MTF\data\click\core\left\imported\decoding_results_LDA2';
% dir2 = 'E:\MTF\data\click\core\right\imported\decoding_results_LDA';
% bin_size = 700;
% 
% [BMF1cMUA,BMF2cMUA] = MTF_compare_directories(dir1, dir2, bin_size,'om');
% 
% dir1 = 'E:\MTF\data\click\pb\left\imported\decoding_results_LDA';
% dir2 = 'E:\MTF\data\click\pb\right\imported\decoding_results_LDA';
% bin_size = 700;
% 
% [BMF1pbMUA,BMF2pbMUA] = MTF_compare_directories(dir1, dir2, bin_size,'om');

dir1 = 'E:\MTF\data\noise\pb\left\imported\decoding_results_LDA';
dir2 = 'E:\MTF\data\noise\pb\right\imported\decoding_results_LDA';
bin_size = 700;

MTF_compare_directories_by_layer(dir1, dir2, bin_size, 'om', true)




