% Program for  MLP..........................................
% Update weights for a given epoch

clear all;
close all;
clc;

weights = zeros(1,5,4);
weights_in = zeros(5,2,4);
errors = zeros(4,1);
for set=1:4

switch set    
% Set 1
case 1
    inp_rows = [12501:50000];
    out_rows = [1:12500];

% Set 2
case 2
    inp_rows = [1:12500 25001:50000];
    out_rows = [12501:25000];

% Set 3
case 3
    inp_rows = [1:25000 37501:50000];
    out_rows = [25001:37500];

% Set 4
case 4
    inp_rows = [1:37500];
    out_rows = [37501:50000];
end

% Load the training data..................................................
file=xlsread('SI_19.xlsx');
for i=1:3
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
inp = 2;          % No. of input neurons
hid = 5;        % No. of hidden neurons
out = 1;            % No. of Output Neurons
lam = 0.01;       % Learning rate
epo = 100;
gamma=.5;
input_update=0;
output_update=0;

% Initialize the weights..................................................
Wi = 0.001*(rand(hid,inp)*2.0-1.0);  % Input weights
Wo = 0.001*(rand(out,hid)*2.0-1.0);  % Output weights

% Train the network.......................................................
for ep = 1 : epo
    sumerr = 0;
    input_update=0;
    output_update=0;
    for sa = 1 : NTD
        xx = Ntrain(sa,1:inp)';     % Current Sample
        tt = Ntrain(sa,inp+1:end)'; % Current Target
        Yh = 1./(1+exp(-Wi*xx));    % Hidden output
        Yo = Wo*Yh;                 % Predicted output
        er = tt - Yo;               % Error
        input_update = gamma*input_update + lam * (er * Yh');
        output_update = gamma*output_update + ((Wo'*er).*Yh.*(1-Yh))*xx';
        Wo = Wo + input_update; % update rule for output weight
        Wi = Wi + output_update; %update for input weight
        sumerr = sumerr + sum(er.^2);
    end    
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
errors(set,:) = sqrt(rmstra/(NTD));
subplot(2,2,set);
disp(sqrt(rmstra/(NTD)))
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
NFeature=xlsread('SI_test_s19.xlsx');
for i=1:2
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
save -ascii mlpMLSMomentum_SI19.dat op;