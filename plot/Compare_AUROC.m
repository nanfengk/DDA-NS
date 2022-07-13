% Nature论文插图复刻第1期
% 公众号：阿昆的科研日常

%   'NIMCGCN', 'LAGCN', 'BNNR','DRRS', 'DeepDR','NSGNN',...
%% 数据准备
x = [1 3 5 7];    
dataset = [0.8553 0.9205 0.9482 0.9481 0.9331 0.9563;
           0.9118 0.9341 0.9550 0.9340 0.9421 0.9782;
           0.8325 0.8910 0.9327 0.9284 0.8993 0.9575;
           0.8334 0.8870 0.8945 0.9273 0.9138 0.9483]; 
 % 误差矩阵
% AVG = dataset/50; % 下方长度
AVG = [0.0032 0.0024 0.0045 0.0051 0.0037 0.0012;
       0.0055 0.0022 0.0046 0.0024 0.0031 0.0027;
       0.0039 0.0037 0.0021 0.0037 0.0029 0.0012;
       0.0031 0.0041 0.0034 0.0027 0.0022 0.0036];
% STD = dataset/100; % 上方长度
STD = [0.0032 0.0024 0.0045 0.0051 0.0037 0.0012;
       0.0055 0.0022 0.0046 0.0024 0.0031 0.0027;
       0.0039 0.0037 0.0021 0.0037 0.0029 0.0012;
       0.0031 0.0041 0.0034 0.0027 0.0022 0.0036];
      
%% 颜色提取
% ColorCopy函数获取方式：
% 公众号后台回复：复制
% C = ColorCopy;
close
% C1 = C(1,:);
% C2 = C(2,:);
% C3 = C(3,:);
C1=[0.4 0.4 0.55];
C2=[0.3020 0.7333 0.9392];
C3=[0.0 0.6275 0.5255];
C4=[0.2392 0.3294 0.5333];
C5=[1 1 0.55];
C6=[0.9098 0.2941 0.2078];

% C1=[0.7725 0.8706 0.7059];
% C2=[0.6588 0.8196 0.5608];
% C3=[0.3255 0.5098 0.1961];


%% 图窗设定
figureUnits = 'centimeters';
figureWidth = 20;
figureHeight = 17;
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on

%% 多组柱状图绘制
GO = bar(x,dataset,0.9,'EdgeColor','k');
hYLabel = ylabel('AUC');
hXLabel = xlabel('(a)');
% 赋色
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;
GO(4).FaceColor = C4;
GO(5).FaceColor = C5;
GO(6).FaceColor = C6;
%文字注释
% high=0.025;
high=0.017;
for ii = 1
    text(ii-0.67,dataset(ii,1)+STD(ii,1)+high,strcat(num2str(dataset(ii,1),'%.3f'),'±',num2str(STD(ii,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.42,dataset(ii,2)+STD(ii,2)+high,strcat(num2str(dataset(ii,2),'%.3f'),'±',num2str(STD(ii,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii-0.15,dataset(ii,3)+STD(ii,3)+high,strcat(num2str(dataset(ii,3),'%.3f'),'±',num2str(STD(ii,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.13,dataset(ii,4)+STD(ii,4)+high,strcat(num2str(dataset(ii,4),'%.3f'),'±',num2str(STD(ii,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii+0.40,dataset(ii,5)+STD(ii,5)+high,strcat(num2str(dataset(ii,5),'%.3f'),'±',num2str(STD(ii,5),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii+0.66,dataset(ii,6)+STD(ii,6)+high,strcat(num2str(dataset(ii,6),'%.3f'),'±',num2str(STD(ii,6),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
end
for ii = 3
    text(ii-0.67,dataset(ii-1,1)+STD(ii-1,1)+high,strcat(num2str(dataset(ii-1,1),'%.3f'),'±',num2str(STD(ii-1,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.42,dataset(ii-1,2)+STD(ii-1,2)+high,strcat(num2str(dataset(ii-1,2),'%.3f'),'±',num2str(STD(ii-1,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii-0.15,dataset(ii-1,3)+STD(ii-1,3)+high,strcat(num2str(dataset(ii-1,3),'%.3f'),'±',num2str(STD(ii-1,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.13,dataset(ii-1,4)+STD(ii-1,4)+high,strcat(num2str(dataset(ii-1,4),'%.3f'),'±',num2str(STD(ii-1,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii+0.40,dataset(ii-1,5)+STD(ii-1,5)+high,strcat(num2str(dataset(ii-1,5),'%.3f'),'±',num2str(STD(ii-1,5),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii+0.66,dataset(ii-1,6)+STD(ii-1,6)+high,strcat(num2str(dataset(ii-1,6),'%.3f'),'±',num2str(STD(ii-1,6),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
end
for ii = 5
    text(ii-0.67,dataset(ii-2,1)+STD(ii-2,1)+high,strcat(num2str(dataset(ii-2,1),'%.3f'),'±',num2str(STD(ii-2,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.42,dataset(ii-2,2)+STD(ii-2,2)+high,strcat(num2str(dataset(ii-2,2),'%.3f'),'±',num2str(STD(ii-2,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii-0.15,dataset(ii-2,3)+STD(ii-2,3)+high,strcat(num2str(dataset(ii-2,3),'%.3f'),'±',num2str(STD(ii-2,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.13,dataset(ii-2,4)+STD(ii-2,4)+high,strcat(num2str(dataset(ii-2,4),'%.3f'),'±',num2str(STD(ii-2,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii+0.40,dataset(ii-2,5)+STD(ii-2,5)+high,strcat(num2str(dataset(ii-2,5),'%.3f'),'±',num2str(STD(ii-2,5),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii+0.66,dataset(ii-2,6)+STD(ii-2,6)+high,strcat(num2str(dataset(ii-2,6),'%.3f'),'±',num2str(STD(ii-2,6),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
end
for ii = 7
    text(ii-0.67,dataset(ii-3,1)+STD(ii-3,1)+high,strcat(num2str(dataset(ii-3,1),'%.3f'),'±',num2str(STD(ii-3,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.42,dataset(ii-3,2)+STD(ii-3,2)+high,strcat(num2str(dataset(ii-3,2),'%.3f'),'±',num2str(STD(ii-3,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii-0.15,dataset(ii-3,3)+STD(ii-3,3)+high,strcat(num2str(dataset(ii-3,3),'%.3f'),'±',num2str(STD(ii-3,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.13,dataset(ii-3,4)+STD(ii-3,4)+high,strcat(num2str(dataset(ii-3,4),'%.3f'),'±',num2str(STD(ii-3,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii+0.40,dataset(ii-3,5)+STD(ii-3,5)+high,strcat(num2str(dataset(ii-3,5),'%.3f'),'±',num2str(STD(ii-3,5),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii+0.66,dataset(ii-3,6)+STD(ii-3,6)+high,strcat(num2str(dataset(ii-3,6),'%.3f'),'±',num2str(STD(ii-3,6),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
end



[M,N] = size(dataset);
xpos = zeros(M,N);
xpos(:,1) = GO(1,1).XEndPoints';
xpos(:,2) = GO(1,2).XEndPoints';
xpos(:,3) = GO(1,3).XEndPoints';
xpos(:,4) = GO(1,4).XEndPoints';
xpos(:,5) = GO(1,5).XEndPoints';
xpos(:,6) = GO(1,6).XEndPoints';

hE = errorbar(xpos, dataset, AVG, STD);
set(hE, 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 0.7)


%% 坐标区细节调整
% 坐标轴参数调整
set(gca, 'Box', 'off', ...                                         
         'XGrid', 'off', 'YGrid', 'off', ...                       
         'TickDir', 'out', 'TickLength', [.005 .005], ...           
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             
         'XColor', [0 0 0],  'YColor', [0 0 0 ],...           
         'YTick', 0.8:0.02:1,...                                      
         'Ylim' , [0.8 1.015], ...                                   
         'Xlim' , [0 8], ...
         'Xtick', [0:8], ... 
         'Xticklabel',{' ', 'Cdataset',' ','DNdataset',' ','Fdataset',' ','LSRRL'},...
         'Yticklabel',{num2str([0.8:0.02:1]','%.2f')})
% legend
hLegend = legend([GO(1),GO(2),GO(3),GO(4),GO(5),GO(6)], ...
                 'NIMCGCN', 'LAGCN', 'BNNR','DRRS', 'DeepDR','NSGNN',...
                 'Location', 'northoutside','Orientation','horizontal');
hLegend.ItemTokenSize = [5 5];
legend('boxoff');
% 字体字号
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10)
set(hLegend, 'FontName',  'Times New Roman', 'FontSize', 11)
set(hYLabel, 'FontName',  'Times New Roman', 'FontSize', 14)
set(hXLabel, 'FontName',  'Times New Roman', 'FontSize', 20)
set(gcf,'Color',[1 1 1])
set(gca,'LineWidth',0.7)


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
% 输出
% print('test2.png','-r300','-dpng')
