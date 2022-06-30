point=480;
[x,y]=meshgrid(linspace(-240,240,point));
%width=40;%%%%%%%%%%%小孔滤波，孔径大小确定
width=10;
map=zeros(480,480);
for i=1:20
width=width+i;
ring=hole(x,y,width,point);
map=map+ring;
end
imshow(map)
function [ring] = hole(x,y,width,point)
ring=zeros(point,point);
D=((x).^2+(y).^2).^(0.5);
ring(find((D>=width-1)&(D<=width)))=1;
end
