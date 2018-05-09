% Program for  MLP..........................................
% Update weights for a given epoch

clear all;
close all;
clc;

weights = zeros(1,30,4);
weights_in = zeros(30,10,4);
errors = zeros(4,1);
for set=1:4

switch set    
% Set 1
case 1
    inp_rows = [251:1000];
    out_rows = [1:250];

% Set 2
case 2
    inp_rows = [1:250 501:1000];
    out_rows = [251:500];

% Set 3
case 3
    inp_rows = [1:500 751:1000];
    out_rows = [501:750];

% Set 4
case 4
    inp_rows = [1:750];
    out_rows = [751:1000];
end

% Load the training data..................................................
file=xlsread('fin_19.xlsx');
for i=1:11
   minval = min(file(:,i)) ;
   maxval = max(file(:,i));
   minmat = ones(size(file,1),1).*minval;
   maxmat = ones(size(file,1),1).*maxval;
   tp =ones(size(file,1),1);
   file(:,i)= ((file(:,i) - minmat) ./ (maxmat -minmat)).* 2 - tp ;
end
Ntrain = file(inp_rows,:);
[NTD,~] = size(Ntrain);
% Initialize the Algorithm Parameters.....................................
inp = 10;          % No. of input neurons
hid = 30;        % No. of hidden neurons
out = 1;            % No. of Output Neurons
lam = 0.0001;       % Learning rate
epo = 2000;

% Initialize the weights..................................................
Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0);  % Output weights

% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    DWi = zeros(hid,inp);
    DWo = zeros(out,hid);
    for sa = 1 : NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        er = tt - Yo;               % Error
        DWo = DWo + lam * (er * Yh'); % update rule for output weight
        DWi = DWi + lam * ((Wo'*er).*Yh.*(1-Yh))*xx';    %update for input weight
        sumerr = sumerr + sum(er.^2);
    end
    Wi = Wi + DWi;
    Wo = Wo + DWo;
%    disp(sqrt(sumerr/(NTD)))
%     save -ascii Wi.dat Wi;
%     save -ascii Wo.d at Wo;
end
weights(:,:,set) = Wo;
weights_in(:,:,set) = Wi;

% Validate the network.....................................................
Nval = file(out_rows,:);
[NTD,~] = size(Nval);
rmstra = zeros(out,1);
res_tra = zeros(NTD,2);
for sa = 1: NTD
        xx = Nval(sa,1:inp)';     % Current Sample
        tt = Nval(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        rmstra = rmstra + (tt-Yo).^2;
        res_tra(sa,:) = [tt Yo];
end
fprintf('Error After Validation for set %d',set);
disp(sqrt(rmstra/(NTD)));
errors(set,:) = sqrt(rmstra/(NTD));
subplot(2,2,set);
plot(res_tra(:,1),'r');
title(['Actual and predicted values for set #',int2str(set)]);
hold on;
plot(res_tra(:,2),'g');
end

%% Test the network.........................................................

[best,index] = min(errors(:,1));
best
Wo = weights(:,:,index);
Wi = weights_in(:,:,index);
NFeature=xlsread('fin_test_s19.xlsx');
for i=1:10
   minval = min(NFeature(:,i)) ;
   maxval = max(NFeature(:,i));
   minmat = ones(size(NFeature,1),1).*minval;
   maxmat = ones(size(NFeature,1),1).*maxval;
   tp =ones(size(NFeature,1),1);
   NFeature(:,i)= ((NFeature(:,i) - minmat) ./ (maxmat -minmat)).* 2 - tp ;
end
[NTD,~]=size(NFeature);
%rmstes = zeros(out,1);
%res_tes = zeros(NTD,2);
op = zeros(NTD, 1);
for sa = 1: NTD
        xx = NFeature(sa,1:inp)';   % Current Sample
        %ca = NFeature(sa,end);      % Actual Output
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        op(sa, :) = Yo; 
        %rmstes = rmstes + (ca-Yo).^2;
        %res_tes(sa,:) = [ca Yo];
        %disp(sqrt(rmstes/NTD));
end
save -ascii mlpBatch_fin19.dat op;