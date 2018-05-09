clear all
close all
clc
Ntrain = xlsread('SI_19.xlsx');
[NTD,inp] = size(Ntrain);
for i=1:inp
  min_col = min(Ntrain(:,i));
  max_col = max(Ntrain(:,i));
  Ntrain(:,i)=(Ntrain(:,i)-min_col)/(2*(max_col-min_col)-1);
 end
% Initialize the Algorithm Parameters.....................................
[~, tmp] = size(Ntrain(1, :));
inp = tmp - 1;                          % No. of input neurons
hid = 10;                               % No. of hidden neurons
out = 1;           % No. of Output Neurons


% Train the network.......................................................
xx = randperm(size(Ntrain,1))
u = Ntrain(xx,: );
u = u(1:hid,1:inp);
sigma = zeros(hid,1);
dist = zeros(hid,hid);
x_train = Ntrain(:,1:inp);
y_train = Ntrain(:,inp+1);
% x_tes = Nfeature(:,1:inp);
% y_tes = load('Results/Group 1/ION.cla');
for i = 1 : hid
    for j = 1 : hid
        dist(i,j) = sqrt(sum((u(j,:) - u(i,:)).^2));
    end
end
dmax = max(max(dist));
sigma = sigma + (dmax/sqrt(hid));
phi = zeros(NTD, hid);
for i = 1 :NTD
    for j = 1 : hid
        phi(i, j) = exp(-(hid./(2.*dmax.*dmax)).*(norm(x_train(i,1:inp)-u(j,1:inp)).^2)); 
    end
end
% A = ones(NTD, out)*-1;
% for i = 1 : NTD
%     A(i, y_train(i))=1;
% end
weights = pinv(phi)*y_train;    
y_cross = phi * weights;
mis = 0;
out = [y_train(:,1) y_cross(:,1) ]
plot(out(:,1),'r')
hold on 
plot(out(:,2),'g')
axis([0 100 0 inf])
err= y_train - y_cross;
err;
sumerr = sum(err .^2 );

%VALIDATION ___________________________________________________-----
x_val=Ntrain(:,1:inp);

y_val=Ntrain(:,inp+1);

for i = 1 : hid
    for j = 1 : hid
        dist(i,j) = sqrt(sum((u(j,:) - u(i,:)).^2));
    end
end
dmax = max(max(dist));
sigma = sigma + (dmax/sqrt(hid));
phi = zeros(NTD, hid);
for i = 1 :NTD
    for j = 1 : hid
        phi(i, j) = exp(-(hid./(2.*dmax.*dmax)).*(norm(x_val(i,1:inp)-u(j,1:inp)).^2)); 
    end
end
disp('VAlidation output');
y_out= phi * weights;
out = [y_val(:,1) y_out(:,1) ]
plot(out(:,1),'r')
hold on 
plot(out(:,2),'g')
%axis([0 100 0 inf])
err= y_val - y_out;
err;
disp('Validation Error');
sumerr = sum(err .^2 );
rmse= sqrt(sumerr/NTD)

%End of validation --------------------------
%TESTING 
file=xlsread('SI_test_s19.xlsx');
for i=1:2
   minval = min(file(:,i)) ;
   maxval = max(file(:,i));
   minmat = ones(size(file,1),1).*minval;
   maxmat = ones(size(file,1),1).*maxval;
   tp =ones(size(file,1),1);
   file(:,i)= ((file(:,i) - minmat) ./ (maxmat -minmat)).* 2 - tp ;
end
x_val=file(:,1:inp);
xx = randperm(size(file,1));
u = file(xx,: );
u = u(1:hid,1:inp);
sigma = zeros(hid,1);
dist = zeros(hid,hid);
[NTD,inp] = size(file);
for i = 1 : hid
    for j = 1 : hid
        dist(i,j) = sqrt(sum((u(j,:) - u(i,:)).^2));
    end
end
dmax = max(max(dist));
sigma = sigma + (dmax/sqrt(hid));
phi = zeros(NTD, hid);
for i = 1 :NTD
    for j = 1 : hid
        phi(i, j) = exp(-(hid./(2.*dmax.*dmax)).*(norm(x_val(i,1:inp)-u(j,1:inp)).^2)); 
    end
end
disp('Testing Done');
y_out= phi * weights;
out =y_out(:,1);
% plot(out(:,1),'r')
% hold on 
% plot(out(:,2),'g')
%axis([0 100 0 inf])
% err= y_val - y_out;
% err;
% disp('Validation Error');
% sumerr = sum(err .^2 );
% rmse= sqrt(sumerr/NTD)
save -ascii rbfPseudo_SI19.dat out;