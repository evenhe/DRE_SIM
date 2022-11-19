%% 
clear;
imgs=imread('sim_target17.tiff');
out_file='test';
imgs=double(imgs)/65535;
[x,y]=size(imgs);
wavelets='db6';
temp = imgs(:,:);
sacle_factor=1;
[m,n] = wavedec2(temp,7,wavelets);
vector = zeros(size(m));
vector(1:n(1)*n(1)*1) = m(1:n(1)*n(1)*1);
Background(:,:) = waverec2(vec,n,wavelets);
out_img=imgs-Background/sacle_factor;
out_img(out_img<0)=0;
imwrite(out_img,[out_file,'/SIM_no_background.tiff']);
imwrite(Background,[out_file,'/Background.tiff']);
imwrite(uint16(Background*65535*sacle_factor),[out_file,'/Background_65535.tiff']);
imwrite(uint16(out_img*65535*sacle_factor),[out_file,'/SIM_no_background_65535.tiff'])
subplot(2,2,1);
imshow(imgs);
title('SIM')
subplot(2,2,2);
imshow(Background);
title("Bachground")
subplot(2,2,3);
imshow(out_img);
title('Remove background');
