% Program for MLP..........................................
% Update weights for a given epoch

%function op1 = RBF_MLS(set_no, file_name, EPO, INP, HID, OUT,LAM)
clc;
clear all;
close all;
bestEff= zeros(2,1);
final_weights = zeros(11,50,2);            %outputs , hidden , set
final_centers = zeros(50,391,2);           % hidden , features , set   
final_sigma = zeros(50,1,2);               %hidden , 1 , set 
errors = zeros(2,1);
for set=1:2
    
    switch set
        % Set 1
        case 1
            inp_rows = [1:99];
            out_rows = [1:99];
            
            % Set 2
        case 2
            inp_rows = [50:99];
            out_rows = [1:49];
            
            % Set 3
%         case 3
%             inp_rows = [1:1216 1857:2432];
%             out_rows = [1217:1856];
%             
%             % Set 4
%         case 4
%             inp_rows = [1:1856];
%             out_rows = [1857:2432];
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
    
    
    
    minAA=0;
    minhid=0;
    minepo=0;
    minGA=0;
    minOA=0;
    % test_inp = load(s2);
    % opp = load(s3);
    
    inp = 391; % No. of input neurons
    hid = 50; % No. of hidden neurons
    out = 11; % No. of Output Neurons
    lam = 0.01;% Learning rate
    epo = 100 ;
    
    Mu = zeros(hid,inp);    %centre  // will have to change for bipolar input
    perm = randperm(NTD);
    centres = perm(1:hid);
    %get centres:
    for i = 1:hid
        Mu(i,:) =  Ntrain(centres(i),1:inp);
    end
    
    % now find sigma: dmax/sqrt(k)
    d = dist(Mu');
    dmax= max(d(:));
    sig = zeros(hid,1);
    for i=1:hid
        sig(i,1) = dmax/sqrt(hid);
    end
    
    lrSig = 1e-1;
    lrCentre = 1e-1;
    lrWeight = 1e-1;
    ;
    %[trI,valI,testI] = dividerand(NTD,.9,.1,0);
    %[~,NumTr] = size(trI);
    %[~,NumVal] = size(valI);
    
    for iter = 1:epo
        sumerr = 0;
        miscla = 0;
        for sa = 1:NTD
            %input_index = trI(1,sa);
            x = Ntrain(sa,1:inp)';
            xx = repmat(x',hid,1);
            tt=zeros(1,out);
            [maxi class] = max(Ntrain(sa,inp+1:end));
            for i = 1:out
                if i == class
                    tt(1,i)=1;
                else
                    tt(1,i)=-1;
                end
            end
            tt=tt';
            
            if sa == 1&& iter==1
                w = (pinv(exp(-sum(  (xx - Mu).^2,2 )  ./(2*sig.^2))')*tt')';   %out x hid
            end
            tmp = abs(xx - Mu);
            tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
            phi = exp(-sum(tmp1,2));
            Yo = w*phi;
            er = tt - Yo;
            extra=tt.*Yo;
            for i=1:out
                if(extra(i)>1)
                    er(i)=0;
                end
            end;
            deltaW = er * phi';
            w = w + lrWeight * deltaW;
            
            tmp2 = (xx - Mu);
            tmp3 = bsxfun(@rdivide,tmp2,sig.^2);
            deltaC = bsxfun(@times,-2*(er'*w).*phi',tmp3');
            Mu = Mu - lrCentre * deltaC';
            
            tmp4 = bsxfun(@rdivide,tmp.^2,sig.^3);
            tmp5 = sum(tmp4,2);
            deltaSigma = -2*(er'*w).*phi'.*tmp5';
            sig = sig - lrSig*deltaSigma';
            
            
            sumerr = sumerr + sum(er.^2);
            ca = find(tt==1);           % actual class
            [~,cp] = max(Yo);           % Predicted class
            if ca~=cp
                miscla = miscla + 1;
            end
        end
        if(rem(iter,150)==0)
            iter
        if(rem(iter,20)==0)
            sumerr
        end
    end
    
    final_centers(:,:,set) = Mu;
    final_weights(:,:,set) = w;
    final_sigma(:,:,set) = sig;
    
    Nval = file(out_rows,:);
    [NTD,~] = size(Nval);
    confusion = zeros(out,out);
    miscla_val = 0;
    pre_tr = zeros(NTD,1);
    for sa = 1 : NTD
        % input_index = valI(1,sa);
        x = Nval(sa,1:inp)';
        xx = repmat(x',hid,1);
        tt=zeros(1,out);
        [maxi class] = max(Nval(sa,inp+1:end));
        for i = 1:out
            if i == class
                tt(1,i)=1;
            else
                tt(1,i)=-1;
            end
        end
        tt=tt';
        
        if sa == 1&& iter==1
            w = (pinv(exp(-sum((xx - Mu).^2,2)./(2*sig.^2))')*tt')';   %out x hid
        end
        tmp = abs(xx - Mu);
        tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
        phi = exp(-sum(tmp1,2));
        Yo = w*phi;
        er = tt - Yo;
        
        sumerr = sumerr + sum(er.^2);
        ca = find(tt==1);           % actual class
        [~,cp] = max(Yo);           % Predicted class
        pre_tr(sa) = cp;
        if ca~=cp
            miscla_val = miscla_val + 1;
        end
        confusion(ca,cp) = confusion(ca,cp) + 1;
    end
    end
    miscla_val
    confusion
    %%
    Geo_acc=1;
    overall_acc=0;
    avg_acc =0;
    for var = 1 : out
        Ni = sum(confusion(var,:));
        overall_acc = overall_acc + confusion(var,var);
        Geo_acc = Geo_acc * ((100*confusion(var,var))/Ni);
        avg_acc = avg_acc + (confusion(var,var) / Ni);
    end
    overall_acc = 100* (overall_acc / NTD)
    avg_acc =( avg_acc / out) * 100
    Geo_acc = Geo_acc^ (1/out)
    
    bestEff(set,:) = overall_acc;
end

fprintf('Best efficiency out of all four set is %d', max(bestEff(:,1)));
bestEff
[best_val, index] = max(bestEff(:,1));
Mu=final_centers(:,:,index);
   w= final_weights(:,:,index) ;
    sig=final_sigma(:,:,index);
    
 
%Testing Network
file=xlsread('BERK_test_s19.xlsx');
for i=1:391
        minval = min(file(:,i)) ;
        maxval = max(file(:,i));
        minmat = ones(size(file,1),1).*minval;
        maxmat = ones(size(file,1),1).*maxval;
        tp =ones(size(file,1),1);
        file(:,i)= ((file(:,i) - minmat) ./ (maxmat -minmat)).* 2 - tp ;
    end
    % test_inp = load(s2);
    % opp = load(s3);
    [NTD,inp] = size(file);
     confusion = zeros(out,out);
    miscla_val = 0;
    pre_tr = zeros(NTD,1);
    Nval=file;
    op= zeros(NTD,1);
    for sa = 1 : NTD
        % input_index = valI(1,sa);
        x = Nval(sa,1:inp)';
        xx = repmat(x',hid,1);
        tt=zeros(1,out);
        [maxi class] = max(Nval(sa,inp+1:end));
        for i = 1:out
            if i == class
                tt(1,i)=1;
            else
                tt(1,i)=-1;
            end
        end
        tt=tt';
        
        if sa == 1&& iter==1
            w = (pinv(exp(-sum((xx - Mu).^2,2)./(2*sig.^2))')*tt')';   %out x hid
        end
        tmp = abs(xx - Mu);
        tmp1 = bsxfun(@rdivide,tmp.^2,2*sig.^2);
        phi = exp(-sum(tmp1,2));
        Yo = w*phi;
       % er = tt - Yo;
        
       % sumerr = sumerr + sum(er.^2);
       % ca = find(tt==1);           % actual class
        [~,cp] = max(Yo);           % Predicted class
        op(sa,1)= cp;
    end
    
    save -ascii rbf_BERK19.dat op;