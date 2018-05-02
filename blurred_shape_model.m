function data=blurred_shape_model(image)

%Skeleton
image_skel = bwmorph(1-image,'thin',Inf);
% figure, imshow(image_skel)

%Dividir a imagem em 16 regioes
nrows=size(image_skel,1)/4;
ncols=size(image_skel,2)/4;

% image16regioes={};
centroids = [];
for i=1:nrows:size(image_skel,1)
    row=min(nrows+i-1,size(image_skel,1));
    for j=1:ncols:size(image_skel,2)
        column=min(ncols+j-1,size(image_skel,2));
        img = image_skel(i:row,j:column);
        centroids = [centroids; size(img,1)/2+i-1 size(img,2)/+2+j-1];
%         image16regioes=[image16regioes img];
    end
end

[row, col] = find(image_skel == 1);

v = zeros(1,16);
for i=1:size(row,1)
    x = [row(i) col(i)];    
    d = sqrt((x(1)-centroids(:,1)).^2 + (x(2)-centroids(:,2)).^2);   
    N = find(d < 2*size(img,1));    
    D = sum(1 ./ d(N));    
    value = 1./(d(N)*D);
    
    for j=1:size(N,1)
        v(N(j)) = v(N(j)) + value(j);
    end 
end

%Normalize v
data = v./sum(v);