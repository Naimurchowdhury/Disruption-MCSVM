normal = Normal(:,2:5);
%importing data for normal operation
nrm = table2array(normal);
pod1 = nrm(1:500,1);
itr1 = nrm(1:500,2);
csl1 = nrm(1:500,3);
std1 = nrm(1:500,4);
% features for normal operation
normalpattern = [pod1, itr1, csl1, std1];

%importing data for demand disruption
Demand = demand(:,2:5);
dmd = table2array(Demand);
pod2 = dmd(1:500,1);
itr2 = dmd(1:500,2);
csl2 = dmd(1:500,3);
std2 = dmd(1:500,4);
% features for demand disruption
demanddisruption = [pod2, itr2, csl2, std2];

%importing data for inventory disruption
Inventory = inventory(:,2:5);
inv = table2array(Inventory);
pod3 = inv(1:500,1);
itr3 = inv(1:500,2);
csl3 = inv(1:500,3);
std3 = inv(1:500,4);
% features for inventory disruption
invdisruption = [pod3, itr3, csl3, std3];


%importing data for distribution disruption
Distribution = distribution(:,2:5);
dst = table2array(Distribution);
pod4 = dst(1:500,1);
itr4 = dst(1:500,2);
csl4 = dst(1:500,3);
std4 = dst(1:500,4);
% features for distribution disruption
dstdisruption = [pod4, itr4, csl4, std4];

% Model Training 
X=[normalpattern;demanddisruption;invdisruption;dstdisruption];
A=ones(500,1);
B=1+ones(500,1);
C=2+ones(500,1);
D=3+ones(500,1);
Y=[A;B;C;D];

%Model Testing 
test_set=[0.944305373279395,1.06299783444491,0.951712586661964,0.867271839584499;
    0.717787771052552,0.91,0.957286577139459,2.20206272479997;
    0.884577664233758,0.909364632787397,0.622481519230472,1.43141776820304;
    0.716958556714294,0.917523749272700,0.872495440106260,1.11604051079763];

result=multisvm(X,Y,test_set);


