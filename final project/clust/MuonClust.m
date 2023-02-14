for HH=-5:5
filename=['./C' num2str(HH) '00.txt'];%文件名对应经过初步处理的atoms系列文件
[x1,y1,z1,e1,x2,y2,z2,e2]=textread(filename,'%f %f %f %f %f %f %f %f','delimiter', ' ');


str0='C:\Users\25345\Desktop\final project\classification\generator\test\'; %这个改成你想要存放图片的位置
newPath=strrep(str0,'\','/');
for I=100:125
DD=zeros(16);
for J=0:15

    DD(J+1,1)=(x1(I*32+J*2+1)+70.4)/140.8*256;
    DD(J+1,2)=(y1(I*32+J*2+1)+70.4)/140.8*256;
    DD(J+1,3)=(z1(I*32+J*2+1)-50)*128;
    DD(J+1,4)=e1(I*32+J*2+1)/0.0595*256;
    DD(J+1,5)=(70.4+x2(I*32+J*2+1))/140.8*256;
    DD(J+1,6)=(70.4+y2(I*32+J*2+1))/140.8*256;
    DD(J+1,7)=(z2(I*32+J*2+1)-50)*128;
    DD(J+1,8)=e2(I*32+J*2+1)/0.0595*256;
    DD(J+1,9)=(x1(I*32+J*2+2)+70.4)/140.8*256;
    DD(J+1,10)=(y1(I*32+J*2+2)+70.4)/140.8*256;
    DD(J+1,11)=(z1(I*32+J*2+2)-50)*128;
    DD(J+1,12)=e1(I*32+J*2+2)/0.0595*256;
    DD(J+1,13)=(x2(I*32+J*2+2)+70.4)/140.8*256;
    DD(J+1,14)=(x2(I*32+J*2+2)+70.4)/140.8*256;
    DD(J+1,15)=(z2(I*32+J*2+2)-50)*128;
    DD(J+1,16)=e2(I*32+J*2+2)/0.0595*256;
    
end
MM=mat2gray(DD);

imwrite(MM,[newPath num2str(HH) '00/' num2str(I)  '.png'])
end
end


