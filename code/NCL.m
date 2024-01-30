function res = NCL(Sample)
warning off
[size1 size2]=size(Sample);
Group = Sample(:,size2);
adrN = find(Group==0);
adrP = find(Group==1);
SampleN = Sample(adrN,:); % Negative Training Data
SampleP = Sample(adrP,:); % Positive Training Data
size1N=size(SampleN,1);
size1P=size(SampleP,1);
% Classify negative samples
for j=1:size1N
    SubSample=Sample;
    SubSample(j,:)=[];
    Mdl = fitcknn(SubSample(:,1:size2-1),SubSample(:,size2),'NumNeighbors',5); %TD = training dataset, Gr = ground truth
    ClassN(j,1) = predict(Mdl, SampleN(j,1:size2-1)); % knnclassify(Sample,TrainData,TrainDataGroup, k)
end
% Find and remove noise
NoiseAdrNSamp = find(ClassN~=0);
% % Remove negative data that cause misclassification of positive data

% Classify Positive samples
for j=1:size1P
    SubSample = Sample;
    SubSample(j,:) = [];
    Mdl = fitcknn(SubSample(:,1:size2-1),SubSample(:,size2),'NumNeighbors',5); %TD = training dataset, Gr = ground truth
    ClassP(j,1) = predict(Mdl, SampleP(j,1:size2-1));
end
missclass = find(ClassP~=1);
Distances = pdist2(Sample(:,1:size2-1),SampleP(missclass,1:size2-1)); % As many rows as Sample's, As many columns as SampleP's
Loc_Misc=[];
for i=1:size(Distances,2)
    Temp = sortrows([Distances(:,i),Sample(:,size2),[1:size(Distances,1)]']);
    Class = Temp(1:6,2); % k=4 because the point itself is also considered, so actually its the 5 nearest neighbors
    Temp_Loc = Temp(find(Class==0),3);
    Loc_Misc = [Loc_Misc,Temp_Loc(:)'];
end
MiscAdr=Loc_Misc;
% REMOVAL of noisey negatives and negatives that cause misclassification
% ---------------------------
% CALCULATE (Address of noisy data in SampleN -> Address of noisy data in
% Sample)
NoiseAdr=0;
n=1;
for i=1:size(NoiseAdrNSamp,1)
    for j=1:size(Sample,1)
    Check=isequal(Sample(j,:),SampleN(NoiseAdrNSamp(i,1),:));
    if Check==1
    NoiseAdr(n,1)=j;
    n=n+1;
    end
    end
end
% Put together address of Samples that need to be removed
if size(MiscAdr,1)==1
    MiscAdr=MiscAdr';
end
DeleteAddress=unique([NoiseAdr;MiscAdr]);
res = DeleteAddress;
%res=MiscAdr;
end