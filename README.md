
data: from GATE simulation   
 

Primary process for 2D situation：Using shapegene to generate 2D connected pixel graph. Codegene2D to generate corresponding GATE code. After GATE simulation, using datagene2DlowQ for dataset with format: [X1 Y1 Z1 E1 X2 Y2 Z2 E2] 
  
  
In 'final project', traditional methods like coneprojection, MLEM and OSEM are available. U can also use extractlowQ to transform vec dataset to 32*32 pixel graph to use RESNET and VIT. or u can directly launch RESNET4.py

The point accuracy refers to whether actual locations are in the picked 20 locations, while accuracy refers to the overall accuracy. 
    

          
