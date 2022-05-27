function res=Populate(Sample,N,i,nnarray)
index=1;
nattrs=size(nnarray,2); % number of attributes
while N~=0
nn=randi(size(nnarray,1)); % random number between 1 and k
for attr=1:nattrs
    dif=nnarray(nn,attr)-Sample(i,attr);
    gap=rand(1);
    Synthetic(index,attr)=Sample(i,attr)+gap*dif;
end
index=index+1;
N=N-1;
end
res=Synthetic;
end