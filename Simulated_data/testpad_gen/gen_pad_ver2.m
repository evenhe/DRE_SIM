string=zeros(1,480);
k=16;
%%
count=1;
for i=1:k
    for j=1:2
        string(count)=1;
        count=count+i;
    end
    count=count+20;
end
size_block=150;
temp_im = repmat(string, [size_block, 1]);
temp_im=temp_im';
temp_im=temp_im(1:480,:);
%imshow(temp_im)
%%
ball_map=zeros(60,480);
batch=zeros(40,40);
[x,y]=meshgrid(linspace(-20,20,40));
for m =1:10
    get=hole2(x,y,m,40);
    ball_map(10:50-1,m*40:m*40+40-1)=ball_map(10:50-1,m*40:m*40+40-1)+get;
end
dist_map=zeros(480,150);
for g=1:12
    dist_ball=ball2(g+1);
    dist_map(g*30-30+1+g*10:g*30+g*10,1:150)=dist_ball;
end
%imshow(dist_map);
ball_map=ball_map';
map=zeros(512,512);
map(16:496-1,16:166-1)=temp_im;
map(16:496-1,167-1:226-1)=zeros(480,60);
map(16:496-1,227-1:286-1)=ball_map;
map(16:496-1,287-1:346-1)=zeros(480,60);
map(16:496-1,347-1:496-1)=dist_map;
map=uint16(65535*map);
imwrite(map,'new_testpad.tiff');
imshow(map);
function[map]=ball2(k)
map=zeros(30,150);
for i =1:30
    for j=1:150
    if (mod(i,k)==0)&&(mod(j,k)==0)
        map(i,j)=1;
    end
    end
end
end
function [ring] = hole2(x,y,width,point)
ring=zeros(point,point);
D=((x).^2+(y).^2).^(0.5);
ring(find((D<=width)))=1;
end