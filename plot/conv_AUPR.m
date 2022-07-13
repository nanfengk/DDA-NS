

%% 数据准备
x = [1 3 5 7];    
dataset = [0.9552 0.9570 0.9489 0.9468;
           0.9746 0.9649 0.9611 0.9626 ;
            0.9601 0.9553 0.9507 0.9475 ;   
           0.9335 0.9364 0.9243 0.9163]; 
%   'GCN', 'GIN', 'GraphSAGE','GAT', ...
      
 % 误差矩阵
% AVG = dataset/50; % 下方长度
AVG = [0.0024 0.0016 0.0033 0.0029;
       0.0035 0.0041 0.0019 0.0034;
       0.0045 0.0051 0.0023 0.0043;
       0.0034 0.0026 0.0051 0.0038];
% STD = dataset/100; % 上方长度
STD = [0.0024 0.0016 0.0033 0.0029;
       0.0035 0.0041 0.0019 0.0034;
       0.0045 0.0051 0.0023 0.0043;
       0.0034 0.0026 0.0051 0.0038];

%% 颜色提取

close
% C1 = C(1,:);
% C2 = C(2,:);
% C3 = C(3,:);
C1=[0.9098 0.2941 0.2078];
C2=[0.3020 0.7333 0.9392];
C3=[0.0 0.6275 0.5255];
C4=[0.2392 0.3294 0.5333];
% C5=[1 1 0.55];

% C1=[0.7725 0.8706 0.7059];
% C2=[0.6588 0.8196 0.5608];
% C3=[0.3255 0.5098 0.1961];

% ax1 = subplot(1,2,1);
% subplot(1, 2, 1)
%% 图窗设定
figureUnits = 'centimeters';
figureWidth = 20;
figureHeight = 16;
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on


%% 多组柱状图绘制
GO = bar(x,dataset,0.9,'EdgeColor','k');
hYLabel = ylabel('AUPR');
hXLabel = xlabel('(b)');
% 赋色
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;
GO(4).FaceColor = C4;
% GO(5).FaceColor = C5;
% 文字注释
high=0.017
for ii = 1
    text(ii-0.56,dataset(ii,1)+STD(ii,1)+high,strcat(num2str(dataset(ii,1),'%.3f'),'±',num2str(STD(ii,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.18,dataset(ii,2)+STD(ii,2)+high,strcat(num2str(dataset(ii,2),'%.3f'),'±',num2str(STD(ii,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii+0.18,dataset(ii,3)+STD(ii,3)+high,strcat(num2str(dataset(ii,3),'%.3f'),'±',num2str(STD(ii,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.54,dataset(ii,4)+STD(ii,4)+high,strcat(num2str(dataset(ii,4),'%.3f'),'±',num2str(STD(ii,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
end
for ii = 3
    text(ii-0.56,dataset(ii-1,1)+STD(ii-1,1)+high,strcat(num2str(dataset(ii-1,1),'%.3f'),'±',num2str(STD(ii-1,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.18,dataset(ii-1,2)+STD(ii-1,2)+high,strcat(num2str(dataset(ii-1,2),'%.3f'),'±',num2str(STD(ii-1,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii+0.18,dataset(ii-1,3)+STD(ii-1,3)+high,strcat(num2str(dataset(ii-1,3),'%.3f'),'±',num2str(STD(ii-1,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.54,dataset(ii-1,4)+STD(ii-1,4)+high,strcat(num2str(dataset(ii-1,4),'%.3f'),'±',num2str(STD(ii-1,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
end
for ii = 5
    text(ii-0.56,dataset(ii-2,1)+STD(ii-2,1)+high,strcat(num2str(dataset(ii-2,1),'%.3f'),'±',num2str(STD(ii-2,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.18,dataset(ii-2,2)+STD(ii-2,2)+high,strcat(num2str(dataset(ii-2,2),'%.3f'),'±',num2str(STD(ii-2,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii+0.18,dataset(ii-2,3)+STD(ii-2,3)+high,strcat(num2str(dataset(ii-2,3),'%.3f'),'±',num2str(STD(ii-2,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.54,dataset(ii-2,4)+STD(ii-2,4)+high,strcat(num2str(dataset(ii-2,4),'%.3f'),'±',num2str(STD(ii-2,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
end
for ii = 7
    text(ii-0.56,dataset(ii-3,1)+STD(ii-3,1)+high,strcat(num2str(dataset(ii-3,1),'%.3f'),'±',num2str(STD(ii-3,1),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');
    text(ii-0.18,dataset(ii-3,2)+STD(ii-3,2)+high,strcat(num2str(dataset(ii-3,2),'%.3f'),'±',num2str(STD(ii-3,2),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');     
    text(ii+0.18,dataset(ii-3,3)+STD(ii-3,3)+high,strcat(num2str(dataset(ii-3,3),'%.3f'),'±',num2str(STD(ii-3,3),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
    text(ii+0.54,dataset(ii-3,4)+STD(ii-3,4)+high,strcat(num2str(dataset(ii-3,4),'%.3f'),'±',num2str(STD(ii-3,4),'%.3f')),...
         'ROtation',90,'color','k','FontSize',9,'FontName',  'Times New Roman', 'HorizontalAlignment','center');  
end



[M,N] = size(dataset);
xpos = zeros(M,N);
xpos(:,1) = GO(1,1).XEndPoints';
xpos(:,2) = GO(1,2).XEndPoints';
xpos(:,3) = GO(1,3).XEndPoints';
xpos(:,4) = GO(1,4).XEndPoints';


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
         'Ylim' , [0.8 1.013], ...                                   
         'Xlim' , [0 8], ...
         'Xtick', [0:8], ... 
         'Xticklabel',{' ', 'Cdataset',' ','DNdataset',' ','Fdataset',' ','LSRRL'},...
         'Yticklabel',{num2str([0.8:0.02:1]','%.2f')})
% legend
hLegend = legend([GO(1),GO(2),GO(3),GO(4)], ...
                 'GCN', 'GIN', 'GraphSAGE','GAT', ...
                 'Location', 'northeast');
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
