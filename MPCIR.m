
set(0,'DefaultFigureColormap',jet);% 绘图的默认颜色图
% DTV重建图像： D:\Dr\four\208\test\DTV\50views\915\915_50views_object_DTV.mat
% HDTV重建图像：D:\Dr\four\208\test\HDTV\50views\915\915_50views_object_HDTV.mat
% 深度先验：D:\Dr\four\208\test\Deep_learning\Wavelet-INR\50views\915\915_50views.mat
%clear all; 
clc;
load('D:\Dr\four\208\test\DTV\50views\1017\1017_50views_object_DTV.mat');
load('D:\Dr\four\208\test\HDTV\50views\1017\1017_50views_object_HDTV.mat');


%Projection_data
n_proj = 208; % 投影数 828
n_iter = 400;  % 迭代数 50
N = 64;       % 图像大小
n_polar = 18; % 投影角度数量

% t_factor= 0.45; % Try 0.2:0.1:0.6     0.15  0.45(原)  高阶TV设0.6好一点

len_proj=3*sqrt(2);
bandwidth=0.08; % 0.12, 0.085(原)  (单瓶)0.08  (双瓶,三瓶)0.35

tx_factor=0.5;
ty_factor=0.85; %0.85
tz_factor=0.75;
t_factor=0.45;

[GX,GY,GZ] = GetAngleVector(n_polar,n_polar);%投影角度

% load('sparse_symmetric-arc-1-plus-3.mat');
% GX = GX(idx(1:n_proj)); %取前几个
% GY = GY(idx(1:n_proj));
% GZ = GZ(idx(1:n_proj));


%Parameters
LA = compute_tau_3D_fun_POCS_MSPS(N,GX,GY,GZ); %相当于是A %v_1 = compute_L_for_A_fun_2D(N,n_proj); 系统矩阵A的最大奇异值
[L1,L2,L3,L] = compute_L_for_D_3D_fun(N); %求最大奇异值 %相当于是D %[D_x,D_y,D] = compute_L_for_D_fun_2D(N); 矩阵D的最大奇异值，即物体的离散梯度变换
v_x = LA/L1;
v_y = LA/L2;
v_z = LA/L3;
v = LA/L;

c=1.5; % 0.075 0.5 1.5

lamda = 0.000001; % you may let it to be 0.1, 1 and 10，影响收敛速率
factor = 2.5;  % V的factor，you may let it to be 0.1, 1 and 10，影响收敛速率 0.01(高阶有限 good)；飞了的时候，减小factor的值
v_x = factor*v_x;
v_y = factor*v_y;
v_z = factor*v_z;
v = factor*v;

% high_order prior image
projection_HDTV= FP_3D_GPU(object_HDTV,GX,GY,GZ);

projection_DTV=FP_3D_GPU(object_DTV,GX,GY,GZ);

% load('D:\Dr\four\208\test\projection\915_projection_64_208.mat'); % 投影放这 
% 
% projection=denoise_proj(my_projection,bandwidth); % denoise the projecitons
% 
% projection=projection(:,idx(1:n_proj)); % projection=projection(:,idx(1:n_proj));
% 
% projection=imresize(projection,[N,n_proj]); % projection=imresize(projection,[N,n_proj]); projection interpolation to let the number of projection-bin be N. 
% 
% projection=projection*((N/(len_proj))^2);  % normalize the projection values.


% set up the model parameter

load('D:\Dr\four\208\test\object_truth\1017_reference.mat'); % load('D:\Dr\方向TV\方向TV\实际重建\data\object_truth_0.08_80_828.mat');

object_truth = mat_recFXD;
% win_1=[0 0.00085];
projection= FP_3D_GPU(object_truth,GX,GY,GZ);

tau=1/(n_proj+6*v);
delta = 1/N/N; % delta = 1/(factor*L);
delta1 = 1/2/v_x; %
delta2 = 1/2/v_y; %
delta3 = 1/2/v_z; % 
delta4 = 1/2/v;   %   1/2/v;
delta_all = 1/2/v;


TV_truth_x=D1f_3D(permute(flip(object_truth, 2), [3,2,1])); % flip(object_truth, 2): 将 object_truth 沿第二个维度（即y轴）翻转
TV_truth_x=sum(sum(sum(abs(TV_truth_x)))); % permute(..., [3, 2, 1]): 将结果的维度进行重新排列，使其变为 [z, y, x] 的顺序。
tx=tx_factor*TV_truth_x;% D1f_3D(...): 对重新排列后的图像进行三维梯度计算

TV_truth_y=D2f_3D(permute(flip(object_truth, 2), [3,2,1]));
TV_truth_y=sum(sum(sum(abs(TV_truth_y))));
ty=ty_factor*TV_truth_y;

TV_truth_z=D3f_3D(permute(flip(object_truth, 2), [3,2,1]));
TV_truth_z=sum(sum(sum(abs(TV_truth_z))));
tz=tz_factor*TV_truth_z;

TV_truth=qiao_TV_3D(permute(flip(object_truth, 2), [3,2,1]));
TV_truth=sum(sum(sum(abs(TV_truth))));
t=t_factor*TV_truth;

load('D:\Dr\pycharm\NerSP\src\model\EWT_500\The_fourth_paper\Wavelet-INR\50views\third\1017\denoised_volume.mat') % 加载object_DL数据

%Initial value
p=zeros(N,n_proj);
q1=zeros(N,N,N);
q2=q1;
q3=q1;
q4=q1;

object=q1;
object0=q1;
t1 = q1;

w = 20; % 1/2
beta = 0.47; % 调整深度先验的权重大小
k = beta * norm(object_DL(:)); %object_DL为深度先验图像 norm(object_DL(:))

one_object=ones(N,N,N);
u1 = LA / norm(one_object(:)); 

% prepare the iteration observation variable
error=0;
data_error=error;
TV_x=error;
nRMSE= error;
NDE= error;
PCC= error;

figure;
tic;
for ii=1:n_iter
        
       % p=(p+delta*(FP_3D_GPU(object0,GX,GY,GZ)-projection))/(1+delta/lamda);
       p=(p+delta*(FP_3D_GPU(object0,GX,GY,GZ)-(c*projection_DTV + (1-c)*projection_HDTV)))/(1+delta/lamda);  %
       
       a1=q1+0.5*D1f_3D(object0);% a1=q1+0.5*D1f_3D(object0);a1=q1+0.5*D1;
       m=abs(a1)*2*v_x; % m=abs(a1)/delta2;
       s=ProjectOntoL1Ball(m,v_x*tx);   %  s=ProjectOntoL1Ball(m,v*tx);
       q1(m~=0)=a1(m~=0).*(one_object(m~=0)-delta1*s(m~=0)./((m(m~=0)*delta1)));

       a2=q2+0.5*D2f_3D(object0);% a2=q2+0.5*D2f_3D(object0);a2=q2+0.5*D2;
       m=abs(a2)*2*v_y;  
       s=ProjectOntoL1Ball(m,v_y*ty);   
       q2(m~=0)=a2(m~=0).*(one_object(m~=0)-delta2*s(m~=0)./((m(m~=0)*delta2)));
    
       a3=q3+0.5*D3f_3D(object0); % a=q+delta*q   q=vDu  a3=q3+0.5*D3f_3D(object0);a3=q3+0.5*D3;
       m=abs(a3)*2*v_z; 
       s=ProjectOntoL1Ball(m,v_z*tz); % 
       q3(m~=0)=a3(m~=0).*(one_object(m~=0)-delta3*s(m~=0)./((m(m~=0)*delta3)));

       a4=q4+w*delta4*(object0-object_DL);
       q4=max(norm(reshape(a4,[],1),2)-w*delta4*k,0)*a4./norm(reshape(a4,[],1),2); 

       % a4 = q4 + delta4 * w * (object_DL-object0);
       % m=abs(a4)*2*v; 
       % s=ProjectOntoL1Ball(m,w*k);  
       % q4(m~=0)=a4(m~=0).*(one_object(m~=0)-delta4*s(m~=0)./((m(m~=0)*delta4)));
       
       % a4 = q4 + delta4 * w * (object_DL - object0);
       % norm_a4 = norm(a4(:));
       % if norm_a4 > w * k
       %     q4 = a4 * (w * k) / norm_a4;  % L2球投影
       % else
       %     q4 = a4;
       % end

       t1 = min((t1 + delta_all*u1*object0),0);%Step9
       object1=object-tau*BP_3D_GPU_full(p,GX,GY,GZ) - tau*v_x*D1Tf_3D(q1)- tau*v_y*D2Tf_3D(q2) - tau*v_z*D3Tf_3D(q3)- tau*w*q4 -tau*u1*t1; %Step10 %update strength
       %object1=object-tau*BP_3D_GPU_full(p,GX,GY,GZ) - tau*v_x*D1Tf_3D(q1)- tau*v_y*D2Tf_3D(q2) - tau*v_z*D3Tf_3D(q3)-tau*u1*t1; %Step10 %update strength
       %object1 = object1 + tau * 0.1 * (object_DL - object); %额外添加：直接将object推向object_DL
 
   object_step=object1-object;
   object=object1; % object 
   object0=object+object_step;% TVcDM 
   
   %Check-----------------------
  
   % data_error(ii)=norm(b(:))/sqrt(N*n_proj); % RMSE of data  % data_error(ii)=norm(b(:))/sqrt(N*n_proj);

    TV_x(ii)=sum(sum(sum(abs(D1f_3D(object))))); % TV at x direction.

%
  b=FP_3D_GPU(object,GX,GY,GZ)-projection;
  NDE(ii) = norm(b(:))/norm(projection(:));
  data_error(ii)=norm(b(:))/sqrt(N*n_proj); % RMSE of data
  
  object_old=object;

  % object = flip(object, 2); % flip y axis  % 第76行对 object_truth 处理了，所以这里也要对object作同样的处理
  % object = permute(object, [3,2,1]); % change x and z
  % 
        % nRMSE(ii)=sqrt(mean(mean(mean((object-object_truth).^2))))/mean(mean(mean(object_truth)));
  nRMSE(ii)=sqrt(mean(mean(mean((object-object_truth).^2))))/sqrt(mean(mean(mean((object_truth).^2))));
        %-----------------------PCC-------------------------------------
  coverance=cov(object(:),object_truth(:));
  PCC(ii)=abs(coverance(1,2))/(sqrt(coverance(1,1))*sqrt(coverance(2,2)));

   subplot(2,4,1)
   imagesc(flip(squeeze(object_truth(:,N/2,:)),1));
   % caxis(win_1);          % 或 clim(win_1)
   % colorbar;              % 如果需要在旁边显示颜色条


   subplot(2,4,2)
   imagesc(flip(squeeze(object(:,N/2,:)),1));
   title(num2str(ii));

   subplot(2,4,3)
   set(gca,'FontSize',15);
   plot(1:N,squeeze(object_truth(:,N/2,N/2)),'k',1:N,squeeze(object(:,N/2,N/2)),'--r');
   xlabel('The voxel position')
   ylabel('The voxel value')
  
   subplot(2,4,4)
    set(gca,'FontSize',15);
    plot(1:N,squeeze(object_truth(:,N/4,N/2)),'k',1:N,squeeze(object(:,N/4,N/2)),'--r');
    xlabel('The voxel position')
    ylabel('The voxel value')

    subplot(2,4,5)
    set(gca,'FontSize',15);
    semilogy(nRMSE);
    % title('object error')
    xlabel('The iteration index');
    ylabel('nRMSE')

    subplot(2,4,6)
    set(gca,'FontSize',15);
    semilogy(NDE);
    xlabel('The iteration index');
    ylabel('NDE');
    
    subplot(2,4,7)
    set(gca,'FontSize',15);
    semilogy(PCC);
    xlabel('The iteration index');
    ylabel('PCC');

    subplot(2,4,8)
    set(gca,'FontSize',15);
    semilogy(tx*ones(length(TV_x),1),'r'); 
    hold on;
    plot(TV_x,'b'); 
    hold off;
    % title('TV trend')
    xlabel('The iteration index');
    ylabel('The TV_x value');

   object=object_old;

   drawnow;
end

toc;
% fprintf('the reconstruction used %f seconds\n',toc);
% object = flip(object, 2); % flip y axis
% object = permute(object, [3,2,1]); % change x and z
%b=FP_3D_GPU(object,GX,GY,GZ)-projection;

NDE1= NDE(1,n_iter);
nRMSE1=nRMSE(1,n_iter);
PCC1=PCC(1,n_iter);

fprintf('the value of nRMSE is%8.6f\n',nRMSE1);
fprintf('the value of PCC is%8.6f\n',PCC1);
fprintf('the value of NDE is%8.6f\n',NDE1);

save (['w_',num2str(w),'_beta_',num2str(beta),'_N_',num2str(N),'_n_proj_',num2str(n_proj),'_ii_',num2str(ii),'_lamda_',num2str(lamda),'_factor_',num2str(factor),'_tx_factor_',num2str(tx_factor),'_ty_factor_',num2str(ty_factor),'_tz_factor_',num2str(tz_factor),'_c_',num2str(c),'.mat']);
saveas(gcf,['w_',num2str(w),'_beta_',num2str(beta),'_N_',num2str(N),'_n_proj_',num2str(n_proj),'_ii_',num2str(ii),'_lamda_',num2str(lamda),'_factor_',num2str(factor),'_tx_factor_',num2str(tx_factor),'_ty_factor_',num2str(ty_factor),'_tz_factor_',num2str(tz_factor),'_c_',num2str(c),'.fig']);








