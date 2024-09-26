% plot Yoshi's way
 
% load data
 
 
 
%% CSD
 
csd=squeeze(mean(eeg,2));
csd=csd-repmat(mean(csd(:,50:100)')',1,length(time));
%
figure,imagesc(time,[],-csd)
set(gca,'XLim',[-50 150])
colormap('jet')
set(gca,'CLim',[-7 7])

