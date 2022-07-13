% 横向柱状图绘制模板
% By：阿昆的科研日常

%% 数据准备
% 读取数据
% 自变量x
x = [1 3 5 7];
% 因变量y
y = [0.9468 0.9481 0.9490; 
     0.9486 0.9425 0.9491; 
     0.9502 0.9513 0.9527
     0.9355 0.9396 0.9425];

%% 颜色定义
% addcolorplus函数获取方式：
% 公众号后台回复：配色强化
% C1 = addcolorplus(193);
% C2 = addcolorplus(194);
% C3 = addcolorplus(195);

% C1=[0.5529 0.6275 0.7961];
% C2=[0.9882 0.5529 0.3843];
% C3=[0.4000 0.7608 0.6588];

C1=[0.7725 0.8706 0.7059];
C2=[0.6588 0.8196 0.5608];
C3=[0.3255 0.5098 0.1961];

%% 图片尺寸设置（单位：厘米）
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 11;


%% 窗口设置
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 11;
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on


%% 横向柱状图绘制
% 原始横向柱状图
GO = barh(x,y,0.9,'EdgeColor','k');

xlabel('AUC')
ylabel('Hidden num')

% 赋色
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;
% GO(4).FaceColor = C4;


%% 细节优化
set(gca, 'Box', 'off', ...                                         % 边框
         'XGrid', 'off', 'YGrid', 'off', ...                       % 网格
         'TickDir', 'out', 'TickLength', [.01 .01], ...            % 刻度
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % 小刻度
         'XColor', [0 0 0],  'YColor', [0 0 0],...           % 坐标轴颜色
         'YTick', 0:1:8,...                                 % 刻度位置、间隔
         'Ylim' , [0 8], ...
         'Xlim' , [0.88 1], ...
         'XTick', 0.88:0.01:1,...
         'Xticklabel',{0.88:0.01:1},...                                % X坐标轴刻度标签
         'Yticklabel',{' ', '64',' ','128',' ','256',' ','512'})                            % Y坐标轴刻度标签 

 
     
hLegend = legend([GO(1),GO(2),GO(3)], ...
                 'NSGNN-center','NSGNN-sum','NSGNN', ...
                 'Location', 'northoutside','Orientation','horizontal');

% 字体和字号
set(gca, 'FontName', 'Arial', 'FontSize', 10)

% 背景颜色
set(gca,'Color',[1 1 1])

%% 图片输出
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');