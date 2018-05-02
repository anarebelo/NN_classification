function main

file=strcat('DB');

d=dir(file);
d=struct2cell(d);
names=d(1,3:end,:);
a=ones(1,size(d,2)-2);
b=1:size(d,2)-2;
target=diag(a);
TRG=[];
TRG_aux=[];
k=1;
for i=1:size(names,2)
    str=strcat(file,delimiter,char(names(i)))
    structure=dir(str);
    structure=struct2cell(structure);
    images=structure(1,3:end,:);
        for j=1:size(images,2)
            str=strcat(file,delimiter,char(names(i)),delimiter,char(images(j)));
            img=imread(str);
            if islogical(img)==0
                t=graythresh(img);
                img=im2bw(img,t);
            end
            img = imresize(img,[20,20]);
            
            geomproperties=geometricproperties(1-img);
            valueMax=max(geomproperties);
            valueMin=min(geomproperties);
            geomproperties=(geomproperties-valueMin)/(valueMax-valueMin);
            bsm=blurred_shape_model(img);
            DATASYMBOLS(k,:)=[img(:)' geomproperties bsm];
            
            k=k+1;
        end
    TRG=[TRG; repmat(target(i,:),j,1)];
    TRG_aux=[TRG_aux; repmat(b(i),j,1)];
end
DATASYMBOLS=double(DATASYMBOLS)';
TRG=TRG'

size(DATASYMBOLS)
size(TRG)
N=size(DATASYMBOLS,2)


ERROGLOBAL=[];
numNeuronsGLOBAL=[];
CLASSGLOBAL={};
TRGBLOGAL={};
netGLOBAL=struct();

for numberTimes=1:10
    data = {};
    trg = {};
    for j=min(TRG_aux):max(TRG_aux)
        aux = find(TRG_aux == j);
        data1 = DATASYMBOLS(:,aux);
        trg1 = TRG(:,aux);
        
        n = size(trg1,2);
        
        data = [data data1(:,randperm(n))];
        trg = [trg trg1(:,randperm(n))];
    end

    dadosTestT={};
    classStruct={};
    classStructT={};
    netSTRUCT=struct();
    for i=1:4
        sprintf('Constructing the dataset')
        %Dataset
        datasymbolstrain = [];
        datasymbolstest = [];
        
        if i == 1
            numberprevious = zeros(1,size(data,2));
            [datasymbolstest,trgsymbolstest, ~,~,datasetTrain, trgsetTrain,numberprevious1] = constdataset(data,trg,numberprevious);
            [datasymbolsval,trgsymbolsval, datasymbolstrain,trgsymbolstrain,~, ~,~] = constdataset(datasetTrain,trgsetTrain,numberprevious);
        elseif i==2
            numberprevious = numberprevious1;
            [datasymbolstest,trgsymbolstest, ~,~,datasetTrain, trgsetTrain,numberprevious1] = constdataset(data,trg,numberprevious);
            [datasymbolsval,trgsymbolsval, datasymbolstrain,trgsymbolstrain,~, ~,~] = constdataset(datasetTrain,trgsetTrain,numberprevious);
        elseif i==3
            numberprevious = numberprevious1;
            [datasymbolstest,trgsymbolstest, ~,~,datasetTrain, trgsetTrain,numberprevious1] = constdataset(data,trg,numberprevious);
            [datasymbolsval,trgsymbolsval, datasymbolstrain,trgsymbolstrain,~, ~,~] = constdataset(datasetTrain,trgsetTrain,numberprevious);
        elseif i==4
            numberprevious = numberprevious1;
            [datasymbolstest,trgsymbolstest, ~,~,datasetTrain, trgsetTrain,numberprevious1] = constdataset(data,trg,numberprevious);
            [datasymbolsval,trgsymbolsval, datasymbolstrain,trgsymbolstrain,~, ~,~] = constdataset(datasetTrain,trgsetTrain,numberprevious);
        end
        
       sprintf('Train size %d: ', size(datasymbolstrain,2))
       sprintf('Test size %d: ', size(datasymbolstest,2))
       sprintf('Validation size %d: ', size(datasymbolsval,2))
        
       sprintf('Choosing number of neurons')
       vecNeuronios = 40:5:70;
       kk=1;
       netaux=struct();
       for k=1:length(vecNeuronios)
           net = newff(minmax(datasymbolstrain),[vecNeuronios(k) size(trgsymbolstrain,1)],{'logsig' 'logsig'},'trainrp');

           net.trainParam.epochs = 2000;
           net.trainParam.show = NaN;
           net.trainParam.showWindow = false;
           net.trainParam.showCommandLine = true;
           net=train(net,datasymbolstrain,trgsymbolstrain);
           fieldname = sprintf('network%d',k);
           netaux=setfield(netaux, fieldname, net);
           
           a=sim(net,datasymbolsval);
  
           [v1 ind_a]=max(a);
           [v2 ind_t]=max(trgsymbolsval);

           correct=length(find(ind_a==ind_t));
           NN = size(datasymbolsval,2);
           classific_correct=correct/NN;
           vector_classif_incorrect(kk)=1-classific_correct;
           
           kk=kk+1;   
       end
       
       [v1 IND]=min(vector_classif_incorrect);

       datasymbolstrainval = [datasymbolstrain datasymbolsval];
       trgsymbolstrainval = [trgsymbolstrain trgsymbolsval];
       
       sprintf('Train + Val size: %d ', size(datasymbolstrainval,2))
       sprintf('Test size: %d ', size(datasymbolstest,2))
       sprintf('Number of neurons: %d ', vecNeuronios(IND))
       
       field_names = fieldnames(netaux);
       net=train(getfield(netaux, field_names{IND}),datasymbolstrainval,trgsymbolstrainval);
       a=sim(net,datasymbolstest);
       
       [v1 ind_a]=max(a);
       [v2 ind_t]=max(trgsymbolstest);
       
       classif_correct(i)=length(find(ind_a==ind_t));
       classif_incorrect(i)=size(datasymbolstest,2)-length(find(ind_a==ind_t));
       erro(i)=classif_incorrect(i)/size(datasymbolstest,2);
       
       numNeurons(i)=vecNeuronios(IND);
       
       dadosTestT=[dadosTestT trgsymbolstest];
       classStruct=[classStruct ind_a];
       classStructT=[classStructT ind_t];
       
       fieldname = sprintf('network%d',i);
       netSTRUCT=setfield(netSTRUCT, fieldname, net);
       
       sprintf('i: %d', i)
       dlmwrite('test.txt', i, '-append', 'delimiter', ';');
    end

    [erro value]=min(erro);
    
    field_names = fieldnames(netSTRUCT);
    net=getfield(netSTRUCT, field_names{value});
    fieldname = sprintf('network%d',numberTimes);
    netGLOBAL=setfield(netGLOBAL, fieldname, net);
    
    numNeurons=numNeurons(value);
    classif_incorrect=classif_incorrect(value);
    classif_correct=classif_correct(value);
    class=cell2mat(classStruct(value));
    trg=cell2mat(dadosTestT(value));
    
    MatrixConfusion=MatrixConf(trg,class,cell2mat(classStructT(value)));
    filenameM=sprintf('MatrixConfusionNeuralNetwork%d.csv',numberTimes);
    dlmwrite(filenameM, MatrixConfusion, ';');

    ERROGLOBAL=[ERROGLOBAL erro];
    numNeuronsGLOBAL=[numNeuronsGLOBAL numNeurons];

    disp(['numberTimes ', num2str(numberTimes)]) 
end

[erro value]=min(ERROGLOBAL);
numNeurons=round(mean(numNeuronsGLOBAL));
perfom = (1-ERROGLOBAL)*100;
[mu,sigma,muci,sigmaci] = normfit(perfom,0.01);

disp(['Performance =', num2str(perfom)])

field_names = fieldnames(netGLOBAL)
net=getfield(netGLOBAL, field_names{value});

save 'networkspace_rests' net


%%
function [datasymbolstest,trgsymbolstest, datasymbolstrain,trgsymbolstrain, datasetTrain, trgsetTrain, numberprevious1] = constdataset(data,trg,numberprevious)
datasymbolstest  = [];
datasymbolstrain = [];
trgsymbolstest   = [];
trgsymbolstrain  = [];
datasetTrain     = {};
trgsetTrain      = {};

numberprevious1 =[];
for j=1:size(data,2)
    dataset = cell2mat(data(j));
    trgset  = cell2mat(trg(j));
    n = size(dataset,2);

    number = round(n*0.25);
    
    datasymbolstest = [datasymbolstest dataset(:,numberprevious(j)+1:number+numberprevious(j))];    
    trgsymbolstest = [trgsymbolstest trgset(:,numberprevious(j)+1:number+numberprevious(j))];
    
    dataset(:,numberprevious(j)+1:number+numberprevious(j)) = [];
    trgset(:,numberprevious(j)+1:number+numberprevious(j)) = [];
    
    datasymbolstrain = [datasymbolstrain dataset];
    trgsymbolstrain  = [trgsymbolstrain trgset];
    
    datasetTrain = [datasetTrain dataset];
    trgsetTrain  = [trgsetTrain  trgset];
    
    numberprevious1 = [numberprevious1 number];
end

%%
function matrix_conf=MatrixConf(trg,ind_a,ind_t)

k=1;
a=[];
for i=1:size(trg,1)
    for j=1:size(trg,1)
        a(k)=i-size(trg,1)*j;
        k=k+1;
    end
end

matriz_conf=[];
diff=[ind_t-size(trg,1)*ind_a];
b=[];

for i=1:length(a)
    b(k)=length(find(diff==a(i)));
    k=k+1;
end

matrix_conf=[];
matrix_conf(1,:)=b((size(trg,1)*size(trg,1)+1):(size(trg,1)*size(trg,1)+size(trg,1)));
k=2;
for i=(size(trg,1)*size(trg,1)+size(trg,1)):size(trg,1):length(b)-size(trg,1)
    matrix_conf(k,:)=b(i+1:i+size(trg,1));
    k=k+1;
end