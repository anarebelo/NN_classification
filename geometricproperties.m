function data = geometricproperties(B)

%Número de colunas/linhas
%number_columns=size(B,2);
%number_rows=size(B,1);

%Momentos
p=[0 1 2];
c=combvec(p,p);
c(:,6)=[];
c(:,7:8)=[];
p=c(1,:);
q=c(2,:);
m=[];
%M00, M10, M20, M01, M11, M02
for i=1:length(p)
    soma_moment=moment(p(i),q(i),B);
    m=[m soma_moment];
end

%Orientação do objecto
aux=(2*(m(1)*m(5)-m(4)*m(2)))/(m(1)*(m(3)-m(6))+m(4)^2-m(2)^2+eps);
orientation=1/2*atan(aux);

%Volume do objecto: area/ncols*nrows
VolumeObj=m(1)/(size(B,1)*size(B,2));

%Compactness
%Perimetro do objecto/volume do objecto
%Em vez de calcular o perimetro faço uma dilação do objecto e calculo o seu
%volume; 
%compactness = volume(dilated) - volume(original) / volume(original)
se=strel('rectangle',[10 10]);
Bdilate = imdilate(B, se);
area=moment(0,0,Bdilate);
VolumeObjDilate=area/(size(B,1)*size(B,2));
Compactness=(VolumeObjDilate-VolumeObj)/VolumeObj;

%ncols/nrows
%aspect_ratio=size(B,2)/size(B,1);

%Volume16regions
%Dividir a imagem em 16 regioes e calcular o volume para cada uma delas
% nrows=size(B,1)/4;
% ncols=size(B,2)/4;

% volume16regioes=[];
% for i=1:nrows:size(B,1)
%     row=min(nrows+i,size(B,1));
%     for j=1:ncols:size(B,2)
%         column=min(ncols+j,size(B,2));
%         imagem=B(i:row,j:column);
%         area=moment(0,0,imagem);
%         volumeaux=area/(size(imagem,1)*size(imagem,2));
%         volume16regioes=[volume16regioes volumeaux];
%     end
% end

%Volume64regions
%Dividir a imagem em 64 regioes e calcular o volume para cada uma delas
% nrows=size(B,1)/8;
% ncols=size(B,2)/8;
% aux=length(1:ncols:size(B,2));
% if aux~=8
%     ncols=ncols-0.03;
% end
% volume64regioes=[];
% for i=1:nrows:size(B,1)
%     row=min(nrows+i,size(B,1));
%     for j=1:ncols:size(B,2)
%         column=min(ncols+j,size(B,2));
%         imagem=B(i:row,j:column);
%         area=moment(0,0,imagem);
%         volumeaux=area/(size(imagem,1)*size(imagem,2));
%         volume64regioes=[volume64regioes volumeaux];
%     end
% end

%Número de buracos na imagem
%Contagem horizontal
nholesHoriz=0;
nholesH=0;
for i=1:size(B,1)
    for j=1:size(B,2)-1
        if (B(i,j)==0 && B(i,j+1)==1)
            nholesHoriz=nholesHoriz+1;
        end
    end
    nholesH=nholesHoriz/size(B,2)+nholesH;
    nholesHoriz=0;
end

%Contagem vertical
nholesVert=0;
nholesV=0;
for j=1:size(B,2)
    for i=1:size(B,1)-1
        if (B(i,j)==0 && B(i+1,j)==1)
            nholesVert=nholesVert+1;
        end
    end
    nholesV=nholesVert/size(B,1)+nholesV;
    nholesVert=0;
end

%Primeira linha que contém um pixel branco e ultima linha que contém um
%pixel branco

% for i=1:size(B,1)
%     %Encontrou um pixel branco
%     if size(find(B(i,:)==1),1)~=0
%         row_top=i;
%         break;
%     end
% end
% for i=size(B,1):-1:1
%     %Encontrou um pixel branco
%     if size(find(B(i,:)==1),1)~=0
%         row_bottom=i;
%         break;
%     end
% end

%Momentos de Zernike
%centroid=[m(4)/m(1); m(2)/m(1)]
%[newB]=imgscale(B,centroid);
% [newB]=lans_imgscale(B);
% %[newB]=momentZernik(B,centroid);
% [momentosZernike]=lans_zmoment(newB,[0 1 2],[0 1 2]);


%Skeleton
skeleton_B = bwmorph(B,'skel',inf);
%Numero de pontos finais
terminating_pts = find_skel_ends(skeleton_B);
%Número de intersecções
intersecting_pts = find_skel_intersection(skeleton_B);

data=[VolumeObj orientation Compactness nholesH nholesV size(terminating_pts,1) size(intersecting_pts,1)];


if any(isnan(data(:)))
    isnan(data)
    kakakakkaak
end

%%
function soma=moment(p,q,B)
soma=0;
for i=1:size(B,1)
    for j=1:size(B,2)
        soma = soma + double((i^p)*(j^q)*B(i,j));
    end
    
end
return;