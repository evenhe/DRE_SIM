% scripy for creating testpad image
size_block = 103;
ntest = 5^2;
testset = zeros(size_block, size_block, ntest);
for ii=1:ntest
    temp = 1:size_block;
    temp = mod(temp, ii);
    temp(temp <  ii/2) = 0;
    temp(temp >= ii/2) = 1;
    if  mod(ii,2)==1
        temp_im = repmat(temp, [size_block, 1]);
    else
        temp_im = repmat(temp, [size_block, 1]);
        temp_im = temp_im';
    end
    
    testset(:,:,ii) = temp_im;
end

ns = sqrt(ntest);
testpad = zeros(size_block*ns, size_block*ns);
testpad(0*size_block+1: 1*size_block,0*size_block+1: 1*size_block)=testset(:,:,2);
testpad(0*size_block+1: 1*size_block,1*size_block+1: 2*size_block)=testset(:,:,3);
testpad(0*size_block+1: 1*size_block,2*size_block+1: 3*size_block)=testset(:,:,4);
testpad(0*size_block+1: 1*size_block,3*size_block+1: 4*size_block)=testset(:,:,5);
testpad(0*size_block+1: 1*size_block,4*size_block+1: 5*size_block)=testset(:,:,6);%
%%%%%%%%
testpad(1*size_block+1: 2*size_block,0*size_block+1: 1*size_block)=testset(:,:,7);
testpad(1*size_block+1: 2*size_block,4*size_block+1: 5*size_block)=testset(:,:,8);
testpad(2*size_block+1: 3*size_block,0*size_block+1: 1*size_block)=testset(:,:,10);
testpad(2*size_block+1: 3*size_block,4*size_block+1: 5*size_block)=testset(:,:,9);
testpad(3*size_block+1: 4*size_block,0*size_block+1: 1*size_block)=testset(:,:,11);
testpad(3*size_block+1: 4*size_block,4*size_block+1: 5*size_block)=testset(:,:,12);
%%%%%
testpad(4*size_block+1: 5*size_block,0*size_block+1: 1*size_block)=testset(:,:,13);
testpad(4*size_block+1: 5*size_block,1*size_block+1: 2*size_block)=testset(:,:,14);
testpad(4*size_block+1: 5*size_block,2*size_block+1: 3*size_block)=testset(:,:,15);
testpad(4*size_block+1: 5*size_block,3*size_block+1: 4*size_block)=testset(:,:,16);
testpad(4*size_block+1: 5*size_block,4*size_block+1: 5*size_block)=testset(:,:,17);%
%testpad(4*size_block+1: 5*size_block,0*size_block+1: 1*size_block)=testset(:,:,13);
%testpad(4*size_block+1: 5*size_block,4*size_block+1: 5*size_block)=testset(:,:,14);
% for ii=1:ns
%     for jj = 1:ns
%         testpad( (ii-1)*size_block+1: ii*size_block, ...
%         (jj-1)*size_block+1: jj*size_block) = testset(:,:,(ii-1)*ns+jj);
%     end
% end
% 
point=306;
[x,y]=meshgrid(linspace(-240,240,point));
width=10;
map=zeros(point,point);
count=1;
for i=1:18
width=width+i+2;
ring=hole(x,y,width,point);
map=map+ring;
end
ring2=hole2(x,y,1,point);
map=map+ring2;
testpad(1*size_block+2: 4*size_block-2,1*size_block+2: 4*size_block-2)= map;
testpad=uint16(testpad*65535);
imshow(testpad)
testpad=testpad(1:512,1:512);
imwrite(testpad, 'testpad.tiff');
function [ring] = hole(x,y,width,point)
ring=zeros(point,point);
D=((x).^2+(y).^2).^(0.5);
ring(find((D>=width-2)&(D<=width)))=1;
end
function [ring] = hole2(x,y,width,point)
ring=zeros(point,point);
D=((x).^2+(y).^2).^(0.5);
ring(find((D<=width)))=1;
end
%%%%%%



