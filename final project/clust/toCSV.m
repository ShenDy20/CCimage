cnt=0;
lbls=[];
for HH=-5:5
for GG=-3:0
filename=['./C' num2str(HH) num2str(GG) '0.txt'];
[x1,y1,z1,e1,x2,y2,z2,e2]=textread(filename,'%f %f %f %f %f %f %f %f','delimiter', ' ');
EVE=[x1+70.4,y1+70.4,z1,e1,x2+70.4,y2+70.4,z2,e2];
label=(HH+5)*4+GG+4;
str0='C:\Users\25345\Desktop\final project\classification\vit\test\'; 
newPath=strrep(str0,'\','/');
for I=0:49
rdm=randperm(900)+3000;
od1=rdm(1:128);
DD=zeros(32);
for J=0:31
  
    DD(J+1,:)=[EVE(od1(1+J*4),:),EVE(od1(2+J*4),:),EVE(od1(3+J*4),:),EVE(od1(4+J*4),:)];
    
end
cnt=cnt+1;
lbls=[lbls;label];
save([newPath num2str(cnt)  '.txt'],'DD','-ascii')

end
end
end
save([newPath 'test.txt'],'lbls','-ascii')