clear all
close all
clc
Ntrain = xlsread('BERK7525_19.xlsx');
[NTD,~] = size(Ntrain);
%Nfeature = load('Set 1/ION.tes');
Ntrain;
% Initialize the Algorithm Parameters.....................................
[~, tmp] = size(Ntrain(1, :));
inp = tmp - 11;                          % No. of input neurons
hid = 29;                               % No. of hidden neurons
out = 11;           % No. of Output Neurons
for i=1:inp
  min_col = min(Ntrain(:,i));
  max_col = max(Ntrain(:,i));
  Ntrain(:,i)=(Ntrain(:,i)-min_col)/(2*(max_col-min_col)-1);
 end
% Train the network.......................................................
xx = randperm(size(Ntrain,1));
u = Ntrain(xx,: );
u = u(1:hid,1:inp);
sigma = zeros(hid,1);
dist = zeros(hid,hid);
x_train = Ntrain(:,1:inp);
y_train = Ntrain(:,inp+1:end);
%x_tes = Nfeature(:,1:inp)
%y_tes = load('Results/Group 1/ION.cla');
for i = 1 : hid
    for j = 1 : hid
        dist(i,j) = sqrt(sum((u(j,:) - u(i,:)).^2));
    end
end
dmax = max(max(dist));
sigma = sigma + (dmax/sqrt(hid));
phi = zeros(NTD, hid);
for i = 1 : NTD
    for j = 1 : hid
        phi(i, j) = exp(-(hid./(2.*dmax.*dmax)).*(norm(x_train(i,1:inp)-u(j,1:inp)).^2)); 
    end
end
A = ones(NTD, out)*-1;
for i = 1 : NTD
  hey = y_train(i,:);
  [maxi index] = max(y_train(i,:));
    A(i, index)=1;
end
weights = pinv(phi)*A;
y_cross = phi * weights;
mis = 0;
valid_op = zeros(NTD, 1);
conf_cross = zeros(out, out);
for i = 1 : NTD
    t = find(y_cross(i,:) == max(y_cross(i, :)));
    valid_op(i) = t;
    [ maxi index] = max(y_train(i,:));
    conf_cross(index, t) = conf_cross(index, t) + 1;  
end
no = 0;
ng = 1;
na = 0;
ni = 0;
for i = 1 : out
    no = no + conf_cross(i, i);
    ni = sum(conf_cross(i, :));
    na = na + conf_cross(i, i) / ni;
    ng = (100 * ng * conf_cross(i, i)) / ni;
end
no = (100 * no) / NTD
na = (100 * na) / out
ng = ng ^ (1/out)
conf_cross

% TESTING THE NETWORK 
Ntrain = xlsread('BERK_test_s19.xlsx');
[NTD,~] = size(Ntrain);
%Nfeature = load('Set 1/ION.tes');
Ntrain;
% Initialize the Algorithm Parameters.....................................
[~, tmp] = size(Ntrain(1, :));
inp = tmp - 11;                          % No. of input neurons
%hid = 200;                               % No. of hidden neurons
out = 11;           % No. of Output Neurons
for i=1:inp      %Normalization 
  min_col = min(Ntrain(:,i));
  max_col = max(Ntrain(:,i));
  Ntrain(:,i)=(Ntrain(:,i)-min_col)/(2*(max_col-min_col)-1);
 end
% Train the network.......................................................
xx = randperm(size(Ntrain,1));
u = Ntrain(xx,: );
u = u(1:hid,1:inp);
x_train = Ntrain(:,1:inp);
y_train = Ntrain(:,inp+1:end);
%x_tes = Nfeature(:,1:inp);
%y_tes = load('Results/Group 1/ION.cla');
for i = 1 : hid
    for j = 1 : hid
        dist(i,j) = sqrt(sum((u(j,:) - u(i,:)).^2));
    end
end
dmax = max(max(dist));
sigma = sigma + (dmax/sqrt(hid));
phi = zeros(NTD, hid);
for i = 1 : NTD
    for j = 1 : hid
        phi(i, j) = exp(-(hid./(2.*dmax.*dmax)).*(norm(x_train(i,1:inp)-u(j,1:inp)).^2)); 
    end
end
A = ones(NTD, out)*-1;
for i = 1 : NTD
  hey = y_train(i,:);
  [maxi index] = max(y_train(i,:));
    A(i, index)=1;
end
weights = pinv(phi)*A;
y_cross = phi * weights;
%valid_op = zeros(NTD, 1);
op=zeros(NTD,1);
for i = 1 : NTD
    [ maxi index] = max(y_cross(i,:));
                   % Predicted class
        op(i,1)= index;
end
%save -ascii rbf_pseudoINV_BERK19.dat op;