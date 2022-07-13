% Nature论文插图复刻第1期
% 公众号：阿昆的科研日常

%% 数据准备
x = [1 3 5 7];    
dataset = [0.9522 0.947 0.9483 0.9481;
           0.9756 0.9690 0.9670 0.9465 ;
            0.9570 0.9464 0.9421 0.9495 ;   
           0.9364 0.9273 0.9184 0.9222 ]; 
  
%% 颜色提取
% ColorCopy函数获取方式：
% 公众号后台回复：复制
% C = ColorCopy;
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



%% 图窗设定
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 11;
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on

%% 多组柱状图绘制
GO = bar(x,dataset,0.9,'EdgeColor','k');
hYLabel = ylabel('AUC');
hXLabel = xlabel(' ');
% 赋色
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;
GO(4).FaceColor = C4;
% GO(5).FaceColor = C5;
% 文字注释
% for ii = 1
%     text(ii-0.56,dataset(ii,1)+2,num2str(dataset(ii,1),'%.2f'),...
%          'ROtation',0,'color','k','FontSize',9,'FontName',  'Helvetica', 'HorizontalAlignment','center');
%     text(ii-0.18,dataset(ii,2)+2,num2str(dataset(ii,2)),...
%          'ROtation',0,'color','k','FontSize',9,'FontName',  'Helvetica', 'HorizontalAlignment','center');     
%     text(ii+0.18,dataset(ii,3)+2,num2str(dataset(ii,3)),...
%          'ROtation',0,'color','k','FontSize',9,'FontName',  'Helvetica', 'HorizontalAlignment','center');  
%     text(ii+0.56,dataset(ii,4)+2,num2str(dataset(ii,4)),...
%          'ROtation',0,'color','k','FontSize',9,'FontName',  'Helvetica', 'HorizontalAlignment','center');  
% end
% for ii = 3
%     text(ii-0.56,dataset(ii-1,1)+2,num2str(dataset(ii-1,1)),...
%          'ROtation',0,'color','k','FontSize',9,'FontName',  'Helvetica', 'HorizontalAlignment','center');
%     text(ii-0.18,dataset(ii-1,2)+2,num2str(dataset(ii-1,2)),...
%          'ROtation',0,'color','k','FontSize',9,'FontName',  'Helvetica', 'HorizontalAlignment','center');     
%     text(ii+0.18,dataset(ii-1,3)+2,num2str(dataset(ii-1,3)),...
%          'ROtation',0,'color','k','FontSize',9,'FontName',  'Helvetica', 'HorizontalAlignment','center');  
%     text(ii+0.56,dataset(ii-1,4)+2,num2str(dataset(ii-1,4)),...
%          'ROtation',0,'color','k','FontSize',9,'FontName',  'Helvetica', 'HorizontalAlignment','center')
% end

%% 坐标区细节调整
% 坐标轴参数调整
set(gca, 'Box', 'off', ...                                         
         'XGrid', 'off', 'YGrid', 'off', ...                       
         'TickDir', 'out', 'TickLength', [.005 .005], ...           
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             
         'XColor', [0 0 0],  'YColor', [0 0 0 ],...           
         'YTick', 0.8:0.02:1,...                                      
         'Ylim' , [0.8 1], ...                                   
         'Xlim' , [0 8], ...
         'Xtick', [0:8], ... 
         'Xticklabel',{' ', 'Cdataset',' ','DNdataset',' ','Fdataset',' ','LSRRL'},...
         'Yticklabel',{num2str([0.8:0.02:1]','%.2f')})
% legend
hLegend = legend([GO(1),GO(2),GO(3),GO(4)], ...
                 'GCN', 'GAT', 'GIN','SAGE', ...
                 'Location', 'northeast');
hLegend.ItemTokenSize = [5 5];
legend('boxoff');
% 字体字号
set(gca, 'FontName', 'Arial', 'FontSize', 10)
set(hLegend, 'FontName',  'Arial', 'FontSize', 10)
set(hYLabel, 'FontName',  'Arial', 'FontSize', 11)
set(gcf,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');