% ��״ͼ(������)����ģ��
% By�������Ŀ����ճ�

%% ����׼��
% ��ȡ����
% �Ա���
x = 1:5;
% �����
% datasetΪ5*3�ľ���һ��3��Ϊһ�飬��5��
dataset = [0.241,0.393,0.294;
           0.219,0.254,0.238;
           0.238,0.262,0.272;
           0.198,0.329,0.287;
           0.201,0.197,0.185];
% ������
AVG = dataset/5; % �·�����
STD = dataset/7; % �Ϸ�����
       
%% ��ɫ����
% colorplus������ȡ��ʽ��
% ���ںź�̨�ظ���450
% C1 = colorplus(66);
% C2 = colorplus(374);
% C3 = colorplus(357);
C1=[0.7725 0.8706 0.7059];
C2=[0.6588 0.8196 0.5608];
C3=[0.3255 0.5098 0.1961];

%% ͼƬ�ߴ����ã���λ�����ף�
figureUnits = 'centimeters';
figureWidth = 12;
figureHeight = 9;

%% ��������
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]); % define the new figure dimensions
hold on

%% ����������״ͼ����
% ���Ƴ�ʼ��״ͼ
GO = bar(x,dataset,1,'EdgeColor','k');
% �������
[M,N] = size(dataset);
xpos = zeros(M,N);
% for i = 1:N
%     xpos(:,i) = GO(1,i).XEndPoints'; % v2019b
% end

xpos(:,1) = GO(1,1).XEndPoints';
xpos(:,2) = GO(1,2).XEndPoints';
xpos(:,3) = GO(1,3).XEndPoints';


hE = errorbar(xpos, dataset, AVG, STD);
% ����
hTitle = title('Bar with errorbar');
hXLabel = xlabel('Samples');
hYLabel = ylabel('RMSE (m)');

%% ϸ���Ż�
% ��״ͼ��ɫ
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;
% ��������
set(hE, 'LineStyle', 'none', 'Color', 'k', 'LineWidth', 1.2)
% ����������
set(gca, 'Box', 'off', ...                                         % �߿�
         'XGrid', 'off', 'YGrid', 'on', ...                        % ����
         'TickDir', 'out', 'TickLength', [.01 .01], ...            % �̶�
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             % С�̶�
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1],...           % ��������ɫ
         'YTick', 0:0.1:1,...                                      % �̶�λ�á����
         'Ylim' , [0 0.5], ...                                     % �����᷶Χ
         'Xticklabel',{'samp1' 'samp2' 'samp3' 'samp4' 'samp5'},...% X������̶ȱ�ǩ
         'Yticklabel',{[0:0.1:1]})                                 % Y������̶ȱ�ǩ

% Legend ����    
hLegend = legend([GO(1),GO(2),GO(3)], ...
                 'A', 'B', 'C', ...
                 'Location', 'northeast');
% Legendλ��΢�� 
P = hLegend.Position;
hLegend.Position = P + [0.015 0.03 0 0];

% ������ֺ�
set(gca, 'FontName', 'Helvetica')
set([hXLabel, hYLabel], 'FontName', 'AvantGarde')
set(gca, 'FontSize', 10)
set([hXLabel, hYLabel], 'FontSize', 11)
set(hTitle, 'FontSize', 11, 'FontWeight' , 'bold')

% ������ɫ
set(gcf,'Color',[1 1 1])

%% ͼƬ���
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');