% Nature���Ĳ�ͼ���̵�1��
% ���ںţ������Ŀ����ճ�

%% ����׼��
x = [1 3 5 7];    
dataset = [0.9414 0.947 0.9543;
           0.9502 0.9510 0.9570;
            0.8721 0.8764 0.8861;
           0.8955 0.8973 0.9084]; 
  
%% ��ɫ��ȡ
% ColorCopy������ȡ��ʽ��
% ���ںź�̨�ظ�������
% C = ColorCopy;
close
% C1 = C(1,:);
% C2 = C(2,:);
% C3 = C(3,:);

% C1=[0.5529 0.6275 0.7961];
% C2=[0.9882 0.5529 0.3843];
% C3=[0.4000 0.7608 0.6588];
% C4=[0.6588 0.8196 0.5608];
% C5=[1 1 0.55];

C1=[0.7725 0.8706 0.7059];
C2=[0.6588 0.8196 0.5608];
C3=[0.3255 0.5098 0.1961];



%% ͼ���趨
figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 11;
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on

%% ������״ͼ����
GO = bar(x,dataset,0.9,'EdgeColor','k');
hYLabel = ylabel(' ');
hXLabel = xlabel('Cdataset');
% ��ɫ
GO(1).FaceColor = C1;
GO(2).FaceColor = C2;
GO(3).FaceColor = C3;
% GO(4).FaceColor = C4;
% GO(5).FaceColor = C5;
% ����ע��
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

%% ������ϸ�ڵ���
% �������������
set(gca, 'Box', 'off', ...                                         
         'XGrid', 'off', 'YGrid', 'off', ...                       
         'TickDir', 'out', 'TickLength', [.005 .005], ...           a
         'XMinorTick', 'off', 'YMinorTick', 'off', ...             
         'XColor', [0 0 0],  'YColor', [0 0 0 ],...           
         'YTick', 0.8:0.02:1,...                                      
         'Ylim' , [0.8 1], ...                                   
         'Xlim' , [0 8], ...
         'Xtick', [0:8], ... 
         'Xticklabel',{' ', '64',' ','128',' ','256',' ','512'},...
         'Yticklabel',{num2str([0.8:0.02:1]','%.2f')})
% legend
hLegend = legend([GO(1),GO(2),GO(3)], ...
                 'NSGNN', 'NSGNN-center', 'NSGNN-sum', ...
                 'Location', 'northoutside','Orientation','horizontal');
hLegend.ItemTokenSize = [5 5];
legend('boxoff');
% �����ֺ�
set(gca, 'FontName', 'Arial', 'FontSize', 10)
set(hLegend, 'FontName',  'Arial', 'FontSize', 10)
set(hYLabel, 'FontName',  'Arial', 'FontSize', 11)
set(gcf,'Color',[1 1 1])

%% ͼƬ���
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'test';
print(figureHandle,[fileout,'.png'],'-r300','-dpng');