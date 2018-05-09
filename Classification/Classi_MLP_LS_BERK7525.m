% Program for  MLP..........................................
% Update weights for a given epoch

clear all
close all
clc
%maxhid=-1;
%maxAA=-1;

bestEff = zeros(2,1);
weight_in = zeros(750,391,2);       % hidden , features , set
weight_out = zeros(11,750,2);        % Out , hidden , set 
errors = zeros(2,1);
for set=1:2
    
    switch set
        % Set 1
        case 1
            inp_rows = [1:50];
            out_rows = [51:99];
            
            % Set 2
        case 2
            inp_rows = [50:99];
            out_rows = [1:49];
            
%             % Set 3
%         case 3
%             inp_rows = [1:50 75:99];
%             out_rows = [51:74];
%             
%             % Set 4
%         case 4
%             inp_rows = [25:99];
%             out_rows = [1:24];
    end
    
    % Load the training data..................................................
    file=xlsread('BERK7525_19.xlsx');
    for i=1:391
        minval = min(file(:,i)) ;
        maxval = max(file(:,i));
        minmat = ones(size(file,1),1).*minval;
        maxmat = ones(size(file,1),1).*maxval;
        tp =ones(size(file,1),1);
        file(:,i)= ((file(:,i) - minmat) ./ (maxmat -minmat)).* 2 - tp ;
    end
    Ntrain=file(inp_rows,:);
    [NTD,inp] = size(Ntrain);
    %NTD1=NTD;
    epo=500;
    %NTD=floor(0.9*NTD1);
    % Initialize the Algorithm Parameters.....................................
    inp = inp-11;          % No. of input neurons
    hid = 750;       % No. of hidden neurons
    out = 11;          % No. of Output Neurons
    lam = 0.001;     % Learning rate
    
    % Initialize the weights..................................................
    Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
    Wo = 0.001*(rand(out,hid)*2.0-1.0);  % Output weights
    cnt=0;
    % Train the network.......................................................
    for ep = 1 : epo
        sumerr = 0;
        DWi = zeros(hid,inp);
        DWo = zeros(out,hid);
        for sa = 1 : NTD
            xx = Ntrain(sa,1:inp)';     % Current Sample
            cno = Ntrain(sa,inp+1:end)' ;% Current Target
            [maxi index] = max(cno);
            tt(1:out,1)=0;
            if maxi > 0
                tt(index,1)=1;
            end
            Yh = 1./(1+exp(-Wi*xx));    % Hidden output
            Yo = Wo*Yh;                 % Predicted output
            er = tt - Yo   ;           % Error
            DWo = DWo + lam * (er * Yh'); % update rule for output weight
            DWi = DWi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';    %update for input weight
            sumerr = sumerr + er(1,:).^2 + er(2,:).^2 + er(3,:).^2 + er(4,:).^2 + er(5,:).^2 + er(6,:).^2 + er(7,:).^2 + er(8,:).^2 + er(9,:).^2+ er(10,:).^2+ er(11,:).^2 ; ;
%             if( ~rem(cnt,20))
%                disp(sqrt(sumerr/NTD)); 
%             end
            Wi = Wi + DWi;
            Wo = Wo + DWo;
            cnt=cnt+1;
        end
         %disp(sqrt(sumerr/NTD)); 
    end
    weight_in(:,:,set) = Wi;
    weight_out(:,:,set) = Wo;
    
    % Validate the network.....................................................
    Nval=file(out_rows,:);
    [NTD,~] = size(Nval);
    val_conf=zeros(out,out);
    rmstra = zeros(out,1);
    res_tra = zeros(NTD,1);
    for sa = 1: NTD
        xx = Nval(sa,1:inp)' ;  % Current Sample
        tt = Nval(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx)) ; % Hidden output
        Yo = Wo*Yh ;% Predicted output
        [actual_max actual_index]= max(tt);
        actual_class=actual_index;
        [predicted_max predicted] = max(Yo);% Predicted class
        predicted_max;
        predicted_class=predicted;
        val_conf(actual_class,predicted_class) = val_conf(actual_class,predicted_class) + 1;
        res_tra(sa,:) = [predicted_class];
    end
    
    disp(val_conf)
    no_elements_per_class=zeros(out,1);
    total_eles=0;
    for i= 1:out
        sum_ele=0;
        for j = 1:NTD
            ele = Nval(j,inp+i);
            if ele == 1
                sum_ele=sum_ele+1;
                total_eles=total_eles+1;
            end
        end
        no_elements_per_class(i,1)=sum_ele;
    end
    %no_elements_per_class
    sum=0;
    ga=1;
    individual_efficiency = zeros(out,1);
    for i = 1:size(val_conf,1)
        sum=sum+val_conf(i,i);
        ga=ga*(val_conf(i,i)/no_elements_per_class(i));
        individual_efficiency(i,1)=(val_conf(i,i))/no_elements_per_class(i);
    end
    for i=1:size(val_conf,1)
        fprintf('Individual Efficieny for all the classes are given as follows : Class %d is %f',i,individual_efficiency(i,1));
        fprintf('\n');
    end
    fprintf('\n');
    overalleff=100*((sum)/total_eles);         %Computing Overall Efficiency
    fprintf('Overall Efficiency of the classes is given as : %f\n',overalleff);
    fprintf('\n');
    averageff=0;
    for i=1:out
        averageff=averageff+individual_efficiency(i,1);
    end
    averageff=100*(averageff/out);
    fprintf('Average Efficiency for the classes is given as : %f\n',averageff);
    geometricmean= 100*(ga.^(1/out));
    fprintf('Geometric Mean Accuracy for the classes is given as : %f\n\n',geometricmean);
    bestEff(set,:) = overalleff;
    % if bestEff<averageff
    %     bestEff(set,:) = averageff;
    %
    % end
    
end
fprintf('Best efficiency out of all 2 sets is %d',max(bestEff(:,1)));
bestEff
[best,index] = max(bestEff(:,1));
Wo = weight_out(:,:,index);
Wi = weight_in(:,:,index);
NFeature=xlsread('BERK_test_s19.xlsx');

for i=1:391
    minval = min(NFeature(:,i)) ;
    maxval = max(NFeature(:,i));
    minmat = ones(size(NFeature,1),1).*minval;
    maxmat = ones(size(NFeature,1),1).*maxval;
    tp =ones(size(NFeature,1),1);
    NFeature(:,i)= ((NFeature(:,i) - minmat) ./ (maxmat -minmat)).* 2 - tp ;
end
[NTD,~]=size(NFeature);
op = zeros(NTD , 1);
for sa = 1: NTD
    xx = NFeature(sa,1:inp)';   % Current Sample
    ca = NFeature(sa,end);      % Actual Output
    Yh = 1./(1+exp(-Wi*xx));    % Hidden output
    Yo = Wo*Yh;                 % Predicted output
    [maxil,bestop]=max(Yo);
    op(sa,:)=bestop;
    %disp(sqrt(rmstes/NTD));
end
save -ascii mlp_BERK19.dat op;