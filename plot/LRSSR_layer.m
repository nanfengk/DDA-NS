
%% 数据准备
x = [1 3 5 7 9];    
dataset = [0.9454 0.9340 0.8905 0.9014 0.7793 ;
           0.9483 0.9373 0.8967  0.9161 0.7909 ;
            0.9371 0.9335 0.8756 0.8778 0.752 ;
            0.9317 0.9343 0.8705 0.8705 0.7412 ;
           0.9277 0.9335 0.8692 0.8679 0.7394 ]'; 
%   'AUC',' ','AUPR',' ','F1',' ','Rec',' ','MCC'},...

 % 误差矩阵
% AVG = dataset/50; % 下方长度
AVG = [0.0024 0.0016 0.0033 0.0029 0.0029;
       0.0035 0.0041 0.0019 0.0034 0.0029;
       0.0045 0.0051 0.0023 0.0043 0.0029;
       0.0034 0.0026 0.0051 0.0038 0.0029;
       0.0034 0.0026 0.0051 0.0038 0.0029];
% STD = dataset/100; % 上方长度
STD = [0.0024 0.0016 0.0033 0.0029 0.0029;
       0.0035 0.0041 0.0019 0.0034 0.0029;
       0.0045 0.0051 0.0023 0.0043 0.0029;
       0.0034 0.0026 0.0051 0.0038 0.0029;
       0.0034 0.0026 0.0051 0.0038 0.0029];
%% 颜色提取
close
% C1 = C(1,:);
% C2 = C(2,:);
% C3 = C(3,:);
C1=[0.5529 0.6275 0.7961];
C2=[0.9882 0.5529 0.3843];
C3=[0.4000 0.7608 0.6588];
C4=[0.6588 0.8196 0.5608];
C5=[1 1 0.55];

% C1=[0.7725 0.8706 0.7059];
% C2=[0.6588 0.8196 0.5608];
% C3=[0.3255 0.5098 0.1961];



%% 图窗设定
figureUnits = 'centimeters';
figureWidth = 20;
figureHeight = 15;
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on

%% 多组柱状图绘制
GO = bar(x,dataset,0.9,'EdgeColor','k');
hYLabel = ylabel(' ');
hXLabel = xlabel('(d)');
% 赋色
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;
GO(4).FaceColor = C4;
GO(5).FaceColor = C5;
% 文字注释
high=0.025
for ii = 1
    text(ii-0.65,dataset(ii,1)+STD(ii,1)+high,strcat(num2str(dataset(ii,1),'%.3f'),'±',num2str(STD(ii,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.33,dataset(ii,2)+STD(ii,2)+high,strcat(num2str(dataset(ii,2),'%.3f'),'±',num2str(STD(ii,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii,dataset(ii,3)+STD(ii,3)+high,strcat(num2str(dataset(ii,3),'%.3f'),'±',num2str(STD(ii,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.33,dataset(ii,4)+STD(ii,4)+high,strcat(num2str(dataset(ii,4),'%.3f'),'±',num2str(STD(ii,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.63,dataset(ii,5)+STD(ii,5)+high,strcat(num2str(dataset(ii,5),'%.3f'),'±',num2str(STD(ii,5),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center'); 
end
for ii = 3
    text(ii-0.65,dataset(ii-1,1)+STD(ii-1,1)+high,strcat(num2str(dataset(ii-1,1),'%.3f'),'±',num2str(STD(ii-1,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.33,dataset(ii-1,2)+STD(ii-1,2)+high,strcat(num2str(dataset(ii-1,2),'%.3f'),'±',num2str(STD(ii-1,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii,dataset(ii-1,3)+STD(ii-1,3)+high,strcat(num2str(dataset(ii-1,3),'%.3f'),'±',num2str(STD(ii-1,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.33,dataset(ii-1,4)+STD(ii-1,4)+high,strcat(num2str(dataset(ii-1,4),'%.3f'),'±',num2str(STD(ii-1,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.63,dataset(ii-1,5)+STD(ii-1,5)+high,strcat(num2str(dataset(ii-1,5),'%.3f'),'±',num2str(STD(ii-1,5),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center'); 
end
for ii = 5
    text(ii-0.65,dataset(ii-2,1)+STD(ii-2,1)+high,strcat(num2str(dataset(ii-2,1),'%.3f'),'±',num2str(STD(ii-2,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.33,dataset(ii-2,2)+STD(ii-2,2)+high,strcat(num2str(dataset(ii-2,2),'%.3f'),'±',num2str(STD(ii-2,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii,dataset(ii-2,3)+STD(ii-2,3)+high,strcat(num2str(dataset(ii-2,3),'%.3f'),'±',num2str(STD(ii-2,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.33,dataset(ii-2,4)+STD(ii-2,4)+high,strcat(num2str(dataset(ii-2,4),'%.3f'),'±',num2str(STD(ii-2,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.63,dataset(ii-2,5)+STD(ii-2,5)+high,strcat(num2str(dataset(ii-2,5),'%.3f'),'±',num2str(STD(ii-2,5),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center'); 
end
for ii = 7
    text(ii-0.65,dataset(ii-3,1)+STD(ii-3,1)+high,strcat(num2str(dataset(ii-3,1),'%.3f'),'±',num2str(STD(ii-3,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.33,dataset(ii-3,2)+STD(ii-3,2)+high,strcat(num2str(dataset(ii-3,2),'%.3f'),'±',num2str(STD(ii-3,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii,dataset(ii-3,3)+STD(ii-3,3)+high,strcat(num2str(dataset(ii-3,3),'%.3f'),'±',num2str(STD(ii-3,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.33,dataset(ii-3,4)+STD(ii-3,4)+high,strcat(num2str(dataset(ii-3,4),'%.3f'),'±',num2str(STD(ii-3,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.63,dataset(ii-3,5)+STD(ii-3,5)+high,strcat(num2str(dataset(ii-3,5),'%.3f'),'±',num2str(STD(ii-3,5),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center'); 
end
for ii = 9
    text(ii-0.65,dataset(ii-4,1)+STD(ii-4,1)+high,strcat(num2str(dataset(ii-4,1),'%.3f'),'±',num2str(STD(ii-4,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.33,dataset(ii-4,2)+STD(ii-4,2)+high,strcat(num2str(dataset(ii-4,2),'%.3f'),'±',num2str(STD(ii-4,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii,dataset(ii-4,3)+STD(ii-4,3)+high,strcat(num2str(dataset(ii-4,3),'%.3f'),'±',num2str(STD(ii-4,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.33,dataset(ii-4,4)+STD(ii-4,4)+high,strcat(num2str(dataset(ii-4,4),'%.3f'),'±',num2str(STD(ii-4,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.63,dataset(ii-4,5)+STD(ii-4,5)+high,strcat(num2str(dataset(ii-4,5),'%.3f'),'±',num2str(STD(ii-4,5),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center'); 
end


[M,N] = size(dataset);
xpos = zeros(M,N);
xpos(:,1) = GO(1,1).XEndPoints';
xpos(:,2) = GO(1,2).XEndPoints';
xpos(:,3) = GO(1,3).XEndPoints';
xpos(:,4) = GO(1,4).XEndPoints';
xpos(:,5) = GO(1,5).XEndPoints';


hE = errorbar(xpos, dataset, AVG, STD);
set(hE, 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 0.7)


%% 坐标区细节调整
% 坐标轴参数调整
set(gca, 'Box', 'off', ...                                         
         'XGrid', 'off', 'YGrid', 'off', ...                       
         'TickDir', 'out', 'TickLength', [.005 .005], ...           
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             
         'XColor', [0 0 0],  'YColor', [0 0 0 ],...           
         'YTick', 0.7:0.02:1,...                                      
         'Ylim' , [0.7 1], ...                                   
         'Xlim' , [0 10], ...
         'Xtick', [0:10], ... 
         'Xticklabel',{' ', 'AUC',' ','AUPR',' ','F1',' ','Rec',' ','MCC'},...
         'Yticklabel',{num2str([0.7:0.02:1]','%.2f')})
% legend
hLegend = legend([GO(1),GO(2),GO(3),GO(4),GO(5)], ...
                 '1 layer','2 layers', '3 layers', '4 layers','5 layers', ...
                 'Location', 'northoutside','Orientation','horizontal');
hLegend.ItemTokenSize = [5 5];
legend('boxoff');
% 字体字号
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10)
set(hLegend, 'FontName',  'Times New Roman', 'FontSize', 11)
set(hYLabel, 'FontName',  'Times New Roman', 'FontSize', 14)
set(hXLabel, 'FontName',  'Times New Roman', 'FontSize', 20)
set(gcf,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
% print(figureHandle,[fileout,'.png'],'-r300','-dpng');

xc = get(gca,'XColor');
yc = get(gca,'YColor');
unit = get(gca,'units');
ax = axes( 'Units', unit,...
           'Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor',xc,...
           'YColor',yc);
set(ax, 'linewidth',0.7,...
        'XTick', [],...
        'YTick', []);