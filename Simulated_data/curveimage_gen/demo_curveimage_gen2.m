clc;clear;
num_curve=25;
size_img=512;
for i=1:num_curve
img(i,:,:)=fun_curveimage_gen2(size_img);
end
out=sum(img,1);
outimage=squeeze(out);
outimage=sqrt(sqrt(outimage));
if 1
    outimage=outimage/(max(max(outimage)));
    outimage=uint16(outimage*45000);
end
%outimage=imresize(outimage,[800 800]);
%imgout=imnoise(outimage,'gaussian',0.003,0.0000001);
imgout=outimage;
noise_=imgout-outimage;
imagesc(imgout);hold
imwrite(imgout,strcat('curveimage8.tif'));
clear;