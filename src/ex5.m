close all; clear all; clc

TargetChar=[];
StimulusType=[];

fprintf(1, 'Collecting Responses and Performing classification... \n\n' );
load 'Subject_B_Train.mat' % load data file
window=240; % window after stimulus (1s)
channel=[4,5,6,10,11,12,17,18]; % only using Cz for analysis and plots

% convert to double precision
Signal=double(Signal);
Flashing=double(Flashing);
StimulusCode=double(StimulusCode);
StimulusType=double(StimulusType);

for epoch=1:size(Signal,1)
    
    % get reponse samples at start of each Flash
    rowcolcnt=ones(1,12);
    for n=2:size(Signal,2)
        if Flashing(epoch,n)==0 & Flashing(epoch,n-1)==1
            rowcol=StimulusCode(epoch,n-1);
            responses(rowcol,rowcolcnt(rowcol),:,:)=Signal(epoch,n-24:n+window-25,:);
            rowcolcnt(rowcol)=rowcolcnt(rowcol)+1;
        end
    end
    responses_8=responses(:,:,:,channel);
    avgresp8=mean(responses_8,2);
    avgresp8=squeeze(avgresp8);
    
    if isempty(StimulusType)==0
        a=randperm(12);
        label=unique(StimulusCode(epoch,:).*StimulusType(epoch,:));
        targetlabel=(6*(label(3)-7))+label(2);
        a(find(a==label(2)))=[];
        a(find(a==label(3)))=[];
        Target(epoch,1,:,:)=avgresp8(label(2),:,:);
        Target(epoch,2,:,:)=avgresp8(label(3),:,:);
        NonTarget(epoch,1:10,:,:)=avgresp8(a,:,:);
    end
end

for i=1:size(Target,1)
    for j=1:size(Target,2)
        for k=1:size(Target,4)
        Target_reshape((i-1)*2+j,k,:)=Target(i,j,:,k);
        end
    end
end
for i=1:size(NonTarget,1)
    for j=1:size(NonTarget,2)
       for k=1:size(NonTarget,4)
        NonTarget_reshape((i-1)*10+j,k,:)=NonTarget(i,j,:,k);
       end
    end
end


% abb=squeeze(NonTarget_reshape(:,7,:))';
% figure
% plot(abb)
% Target_reshape1=Target_reshape;
% NonTarget_reshape1=NonTarget_reshape;


for i=1:size(Target_reshape,2)
    Tshape=squeeze(Target_reshape(:,i,:));
    cc=mean(Tshape(:));
    dd=std(Tshape(:));
    Tshape=(Tshape-cc)/dd;
    Target_reshape1(:,i,:)=Tshape;
end
for i=1:size(NonTarget_reshape,2)
    NTshape=squeeze(NonTarget_reshape(:,i,:));
    cc=mean(NTshape(:));
    dd=std(NTshape(:));
    NTshape=(NTshape-cc)/dd;
    NonTarget_reshape1(:,i,:)=NTshape;
end

% abb=squeeze(NonTarget_reshape1(:,7,:))';
% figure
% plot(abb)

fs=240;
wp =15/(fs/2);  %通带截止频率,取50~100中间的值,并对其归一化
ws =20/(fs/2);  %阻带截止频率,取50~100中间的值,并对其归一化
alpha_p = 3; %通带允许最大衰减为  db
alpha_s = 20;%阻带允许最小衰减为  db
%获取阶数和截止频率
[ N1,wc1 ] = buttord( wp , ws , alpha_p , alpha_s);
%获得转移函数系数 
[b1,a1]= butter(N1,wc1);


for i=1:size(Target_reshape1,1)
    for j=1:size(Target_reshape1,2)
        
        bb=Target_reshape1(i,j,:);
        bbfilt = filter(b1,a1,bb);
%         bbfilt=dyaddown(bbfilt);
        Targetfilt(i,j,:)=bbfilt;
        
    end
end

for i=1:size(NonTarget_reshape1,1)
     for j=1:size(NonTarget_reshape1,2)
        bb=NonTarget_reshape1(i,j,:);
        bbfilt = filter(b1,a1,bb);
%         bbfilt=dyaddown(bbfilt);
        NonTargetfilt(i,j,:)=bbfilt;
     end
end

% aab=squeeze(NonTargetfilt(:,7,:))';
% figure
% plot(aab)
% Targetfilt=Target_reshape;
% NonTargetfilt=NonTarget_reshape;

Allsignal0=[Target_reshape1;NonTarget_reshape1];
Allsignal=[Targetfilt;NonTargetfilt];
% for i=1:size(Allsignal,2)
%     Allsignal0=squeeze(Allsignal(:,i,:));
%     cc=mean(Allsignal0(:));
%     dd=std(Allsignal0(:));
%     Allnormal0=(Allsignal0-cc)/dd;
%     Allnormal(:,i,:)=Allnormal0;
% end
Targetout=Allsignal(1:170,:,:);
NonTargetout=Allsignal(171:1020,:,:);

% Targetfinal=Targetout;
% filename='C:\Users\lenovo\Desktop\data\targetoutB_8channels.txt';
% dlmwrite(filename,Targetfinal);
% NonTargetfinal=NonTargetout;
% filename1='C:\Users\lenovo\Desktop\data\NontargetoutB_8channels.txt';
% dlmwrite(filename1,NonTargetfinal);


figure
plot(mean(squeeze(Target_reshape1(:,1,10:215)),1),'linewidth',1);
hold on
plot(mean(squeeze(Targetfilt(:,1,25:230)),1),'linewidth',1.5);
axis([1,205,-0.4,0.8]);

% 
% ab=squeeze(NonTargetout(:,7,:));



% figure
% plot(NonTargetout(:,:)');
% resha=reshape(Targetout,85*2*15,72);
% isequal(resha(2,:),squeeze(Targetout(2,1,1,:))')
% ccc=squeeze(Targetout(1,1,:,:));
% plot(ccc');
% figure
% cccmean=mean(ccc);
% plot(cccmean);


% f=fftshift(fft(bb));                  %b表示信号值data
% w=linspace(-512,512,length(bb)); 
% figure  %根据奈奎斯特采样定理，512/2为最大频率
% plot(w,abs(f));                      %Hz为单位
% %设计一个巴特沃夫低通滤波器,要求把50Hz的频率分量保留,其他分量滤掉



% ff=fftshift(fft(filter_lp_s1));                  %b表示信号值data
% ww=linspace(-512,512,length(filter_lp_s1));  %根据奈奎斯特采样定理，512/2为最大频率
% figure
% plot(ww,abs(ff));  
% 
% ff2=fftshift(fft(filter_lp_s2));                  %b表示信号值data
% ww2=linspace(-512,512,length(filter_lp_s2));  %根据奈奎斯特采样定理，512/2为最大频率
% figure
% plot(ww2,abs(ff2));  
% 
% figure
% plot(bb(:,1:144));
% hold on
% plot(filter_lp_s1(:,1:144));
% hold on
% plot(filter_lp_s2(:,1:144));
% responses_cz=responses(:,:,:,11);
% response_1=squeeze(responses_cz(1,:,:));
% t=1:1:240;
% plot(t,response_1(:,:));

% avgresp=mean(responses,2);
% avgresp=reshape(avgresp,12,window,64);
% avgresp_cz=avgresp(:,:,11);
% plot(t,avgresp_cz(1,:));
