function [result] = fun_curveimage_gen2(size_img)
%% Random curve image generation
%% [result] = fun_curveimage_gen(size_img)


% size_img=512; % image size
var=0.04; % acceleration change weight
a_t=25;% acceleration  change period


    
p0=round(size_img*rand(1,2));
%moving
num_run=200000;
p(1,:)=p0;

v(1,:)=[size_img/2-p(1,1),size_img/2-p(1,2)]/size_img*2;%initial velocity towards center point
v(1,:)=1/sqrt(v(1,1).^2+v(1,2).^2)*v(1,:);% normalization
a=0;

for i=2:num_run
% cceleration change
if mod(i,a_t)==0
a=rand()-0.5;
end

v(i,:)=v(i-1,:)+var*a*[-v(i-1,2),v(i-1,1)];%velocity update 
temp=1/sqrt(v(i,1).^2+v(i,2).^2).*v(i,:) ;
v(i,:)=temp;% normalization
p(i,:)=p(i-1,:)+v(i,:);

% Edge cutting
if p(i,1)>size_img-1
    p(i,1)=size_img-1;
    break
end
if p(i,1)<1
    p(i,1)=1;
    break
end
 if p(i,2)> size_img-1
    p(i,2)=size_img-1;
    break
 end
if p(i,2)<1
    p(i,2)=1;
    break
end

end
    rp=round(p);
    space(:,:)=zeros(size_img);   
    for j=1:size(p,1)
        space(rp(j,1),rp(j,2))=1;
    end

result=space;


end

