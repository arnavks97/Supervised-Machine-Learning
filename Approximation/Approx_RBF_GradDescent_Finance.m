clc;
clear all;
close all;

final_weights = zeros(1,20,4);           %out , hidden , set 
final_centers = zeros(20,10,4);           % hidden, input , set
final_sigma = zeros(20,1,4);            % hidden , output , set

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

trainingError = 0;
testingError = 0;

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

[~, tmp] = size(Ntrain(1, :));
inp = tmp - 1;      % No. of input neurons
hid = 20;            % No. of hidden neurons
out = 1;            % No. of Output Neurons
epo = 300;
lrw = 1e-02;        % Learning rate for weights
lrs = 1e-04;        % Learning rate for sigma
lrc = 1e-01;        % Learning rate for centres
u = zeros(hid,inp);             %centre  
xx = randperm(NTD);
centres = xx(1:hid);
for i = 1:hid
   u(i,:) = Ntrain(centres(i),1:inp);
end
d = dist(u');
dmax = max(d(:));
sig = zeros(hid,1);
for i=1:hid
    sig(i,1) = dmax/(sqrt(hid));
end 
for ep = 1:epo
    sumerr = 0;
    for sa = 1:NTD
        t = Ntrain(sa,1:inp)';
        xx = repmat(t',hid,1);
        tt = Ntrain(sa,inp+1:end)';
        if sa == 1&& ep == 1
              w = (pinv(exp(-sum((xx - u).^2,2)./(2*sig.^2))')*tt')';   
        end
        tmp = abs(xx - u);
        tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
        phi = exp(-sum(tmp1,2));
        Yo = w*phi;
        er = tt - Yo;
        dW = er * phi';
        w = w + lrw * dW;
        tmp2 = (xx - u);
        tmp3 = bsxfun(@rdivide,tmp2,sig.^2);
        dC = bsxfun(@times,-2*(er'*w).*phi',tmp3');
        u = u - lrc * dC';
        tmp4 = bsxfun(@rdivide,tmp.^2,sig.^3);
        tmp5 = sum(tmp4,2);
        dS = -2*(er'*w).*phi'.*tmp5';
        sig = sig - lrs*dS';
        sumerr = sumerr + sum(er.^2);   
    end
    %disp(sqrt(sumerr/NTD));
end
final_centers(:,:,set) = u;
final_weights(:,:,set) = w;
final_sigma(:,:,set) = sig;

Nval = file(out_rows,:);
[NTD,~] = size(Nval);
rmstra = zeros(out,1);
res_tra = zeros(NTD,2);
pre1 = zeros(1, NTD);
ac1 = zeros(1, NTD);
x1 = zeros(1, NTD);
for sa = 1: NTD
        x1(1, sa) = sa;
        t = Nval(sa,1:inp)';   % Current Sample
        xx = repmat(t',hid,1);
        tt = Nval(sa,end);      % Actual Output
        tmp = abs(xx - u);
        tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
        phi = exp(-sum(tmp1,2));
        Yo = w*phi;
        pre1(1, sa) = Yo;
        ac1(1, sa) = tt;
        er = tt - Yo;
        rmstra = rmstra + sum(er.^2);
        res_tra(sa,:) = [tt Yo];
end
NTD1 = NTD;
errors(set,:) = sqrt(rmstra/(NTD));

fprintf('Validation Error for set %d',set);
trainingError=(sqrt(rmstra/NTD));
disp(sqrt(rmstra/NTD))
subplot(2,2,set);
plot(res_tra(:,1),'r');
title(['Actual and predicted output of set #',int2str(set)]);
hold on;
plot(res_tra(:,2),'g');
axis([0 100 -2 2]);
end
file=xlsread('fin_test_s19.xlsx');
for i=1:10
   minval = min(file(:,i)) ;
   maxval = max(file(:,i));
   minmat = ones(size(file,1),1).*minval;
   maxmat = ones(size(file,1),1).*maxval;
   tp =ones(size(file,1),1);
   file(:,i)= ((file(:,i) - minmat) ./ (maxmat -minmat)).* 2 - tp ;
end

NFeature=file(:,1:inp);
NFeature2=file(:,inp+1:end);
[NTD,~]=size(NFeature);
[best index] = min(errors(:,1));
best
w=final_weights(:,:,index);
u=final_centers(:,:,index);
sig=final_sigma(:,:,index);
rmstes = zeros(out,1);
res_tes = zeros(NTD,1);
pre2 = zeros(1, NTD);
ac2 = zeros(1, NTD);
x2 = zeros(1, NTD);
for sa = 1: NTD
        x2(1,sa) = sa;
        t = NFeature(sa,1:inp)';   % Current Sample
        xx = repmat(t',hid,1);
        %tt = NFeature2(sa,end);      % Actual Output
        tmp = abs(xx - u);
        tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
        phi = exp(-sum(tmp1,2));
        Yo = w*phi;
%         er = tt - Yo;
%         pre2(1, sa) = Yo;
%         ac2(1, sa) = tt;
%         rmstes = rmstes + sum(er.^2);
        res_tes(sa,:) = Yo;
end
save -ascii rbfSimple_fin19.dat res_tes;
% disp('Testing Error');
% disp(sqrt(rmstes/NTD));
% testingError=(sqrt(rmstes/NTD));
% figure;
% plot(x1, pre1, x1, ac1);
% legend('Predicted Output', 'Actual Output'),title('Validation graph');
% figure;
% plot(x2, pre2, x2, ac2);
% legend('Predicted Output', 'Actual Output');