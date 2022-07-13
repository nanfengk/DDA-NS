% ������״ͼ����ģ��
% By�������Ŀ����ճ�

%% ����׼��
% ��ȡ����
% �Ա���x
x = [1 3 5 7];
% �����y
y = [0.9468 0.9481 0.9490; 
     0.9486 0.9425 0.9491; 
     0.9502 0.9513 0.9527
     0.9355 0.9396 0.9425];

%% ��ɫ����
% addcolorplus������ȡ��ʽ��
% ���ںź�̨�ظ�����ɫǿ��
% C1 = addcolorplus(193);
% C2 = addcolorplus(194);
% C3 = addcolorplus(195);

% C1=[0.5529 0.6275 0.7961];
% C2=[0.9882 0.5529 0.3843];
% C3=[0.4000 0.7608 0.6588];

C1=[0.7725 0.8706 0.7059];
C2=[0.6588 0.8196 0.5608];
C3=[0.3255 0.5098 0.1961];

%% ͼƬ�ߴ����ã���λ�����ף�
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 11;


%% ��������
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 11;
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on


%% ������״ͼ����
% ԭʼ������״ͼ
GO = barh(x,y,0.9,'EdgeColor','k');

xlabel('AUC')
ylabel('Hidden num')

% ��ɫ
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;
% GO(4).FaceColor = C4;


%% ϸ���Ż�
set(gca, 'Box', 'off', ...                                         % �߿�
         'XGrid', 'off', 'YGrid', 'off', ...                       % ����
         'TickDir', 'out', 'TickLength', [.01 .01], ...            % �̶�
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % С�̶�
         'XColor', [0 0 0],  'YColor', [0 0 0],...           % ��������ɫ
         'YTick', 0:1:8,...                                 % �̶�λ�á����
         'Ylim' , [0 8], ...
         'Xlim' , [0.88 1], ...
         'XTick', 0.88:0.01:1,...
         'Xticklabel',{0.88:0.01:1},...                                % X������̶ȱ�ǩ
         'Yticklabel',{' ', '64',' ','128',' ','256',' ','512'})                            % Y������̶ȱ�ǩ 

 
     
hLegend = legend([GO(1),GO(2),GO(3)], ...
                 'NSGNN-center','NSGNN-sum','NSGNN', ...
                 'Location', 'northoutside','Orientation','horizontal');

% ������ֺ�
set(gca, 'FontName', 'Arial', 'FontSize', 10)

% ������ɫ
set(gca,'Color',[1 1 1])

%% ͼƬ���
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');