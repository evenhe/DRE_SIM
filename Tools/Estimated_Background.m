%% 
clear;
imgs=imread('sim_target17.tiff');
out_file='background';
imgs=double(imgs)/65535;
[x,y]=size(imgs);
wavename='db6';
initial = imgs(:,:);
sacle_factor=1;
[m,n] = wavedec2(initial,7,wavename);
vec = zeros(size(m));
vec(1:n(1)*n(1)*1) = m(1:n(1)*n(1)*1);
Biter =  waverec2(vec,n,wavename);
Background(:,:) = Biter;
Background=Background(1:x,1:y);
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