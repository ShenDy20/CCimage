for HH=-5:5
for GG=-3:0
filename=['./C' num2str(HH) num2str(GG) '0.txt'];
[x1,y1,z1,e1,x2,y2,z2,e2]=textread(filename,'%f %f %f %f %f %f %f %f','delimiter', ' ');
EVE=[(x1+70.4)/140.8*256,(y1+70.4)/140.8*256,(z1-50)*128,e1/0.0595*256,(x2+70.4)/140.8*256,(y2+70.4)/140.8*256,(z2-50)*128,e2/0.0595*256];

str0='C:\Users\25345\Desktop\final project\classification\generator\test\'; 
newPath=strrep(str0,'\','/');
for I=0:49
rdm=randperm(900)+3000;
od1=rdm(1:128);
DD=zeros(32);
for J=0:31
  
    DD(J+1,:)=[EVE(od1(1+J*4),:),EVE(od1(2+J*4),:),EVE(od1(3+J*4),:),EVE(od1(4+J*4),:)];
    
end
MM=mat2gray(DD);

imwrite(MM,[newPath num2str(HH) num2str(GG) '0/' num2str(I)  '.png'])
end
end
end
