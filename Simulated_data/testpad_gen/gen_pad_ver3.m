string=zeros(1,440);
k=17;
%%
map=zeros(512,512);
count=1;
for i=6:k
    for j=1:2
        string(count)=1;
       % string(count+1)=1;
        %string(count+2)=1;
        count=count+i;
    end
    count=count+16;
end
size_block=512;
temp_im = repmat(string, [size_block, 1]);
temp_im=temp_im';
%map(:,6:505)=temp_im;
map(46:485,:)=temp_im;
%map(:,1:20)=1;
%map(:,492:512)=1;
map=uint16(map*65535);
imshow(map);
imwrite(map,'test_xian_1_6.tiff');