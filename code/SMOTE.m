function res=SMOTE(Sample,T,N,k)
% Sample: minority class samples
% T: Number of minority class samples
% N: SMOTE N%
% k: Number of nearest neighbors
[size1 size2]=size(Sample);
rownum=find(Sample(:,size2)==1);
Sample=Sample(rownum,:); % Sample -> Minority Class Samples ONLY
if N<100
     Sample=Sample(randperm(T),:); % Shuffles Minority Class Samples
     T=T*N/100;
     Sample=Sample(1:T,:); % Takes a subsample of them
     N=100;
end
% SampleT=Sample';
N=N/100;
Distances = pdist2(Sample(:,1:size2-1),Sample(:,1:size2-1)); % Distances between all samples
Distances = (eye(size(Distances))*max(Distances(:))*10)+Distances; % So that a sample does not get picked as the nearest neighbor of itself
for i=1:T
    Temp = sortrows([Distances(:,i),[1:T]']);
    if size(Temp,1) < 5
        k = 3;
    end
    NN_Array = Sample(Temp(1:k,2),:); % Temp(1:k,2) is the location address of the k nearest neighbors
    Res(((i-1)*N)+1:i*N,:)=Populate(Sample,N,i,NN_Array);
end
res=Res;
end
