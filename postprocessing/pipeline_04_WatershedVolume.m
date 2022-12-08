clear all;clc

%%% vars
path_file_input   = "./pipeline_output/3-denoised.dream3d";
path_file_output  = "./pipeline_output/4-watershed.dream3d";
path_celldata     = "/DataContainers/ImageDataContainer/CellData";
name_errormask    = "Error_Mask";
name_orientations = "EulerAngles";
gradient_threshold = 0;
contrast = 64;
sigma = 2; %for gaussian blur
tolerance = 0.10; %for re-inserting zero data %as a percentage

%%% create the output file
if ~exist(fileparts(path_file_output), 'dir')
    mkdir(fileparts(path_file_output))
end
copyfile(path_file_input, path_file_output)

%%% import arrays
% get eulerangles from dream3d file
orientations = read_dream3d_dataset(path_file_output,path_celldata,name_orientations);
%get the error mask from the dream3d file
mask =  read_dream3d_dataset(path_file_output,path_celldata,name_errormask);

%%% find the (inverse) image gradient
% find the image gradient
gradient = zeros(size(orientations(:,:,:,1)));
for component = 1:size(orientations,4)
    gradient_i = imgradient3( orientations(:,:,:,component) );
    gradient_i = normalize_0_1( gradient_i );
    gradient_i(gradient_i < gradient_threshold) = 0;
    gradient = gradient + gradient_i;
end
% normalize the gradient
gradient = normalize_0_1( gradient );
% remove the error mask data from the gradient
% we set to 1 so it acts as a hard boundary to the watershed
gradient(mask==0) = 1;
%plot_volume(gradient, "Gradient "+name_dataset);
create_dataset(path_file_output, path_celldata, 'Gradient', cast(gradient, 'single'))


%%% contrast the gradient
gradient_contrast = histeq(gradient, contrast);
%plot_volume(gradient_contrast, "Gradient Contrast "+name_dataset);
create_dataset(path_file_output, path_celldata, 'GradientContrast', cast(gradient_contrast, 'single'))

% %%% erode/dilate the gradient
% se = strel("cube",3);
% gradient_erode_dilate = imclose(gradient_contrast,se);
% %plot_volume(gradient_erode_dilate, "Gradient Erode/Dilate "+name_dataset);
% create_dataset(path_file_output, path_celldata, 'GradientErodeDilate', cast(gradient_erode_dilate, 'single'))

%%% perform gaussian blur on the image gradient to smooth out local noise
gradient_gauss = imgaussfilt3(gradient_contrast, sigma);
%plot_volume(gradient_gauss, "Gradient Gauss "+name_dataset);
create_dataset(path_file_output, path_celldata, 'GradientGauss', cast(gradient_gauss,    'single'))

%invert smoothed eulerangles
%%%minima likely correspond to grain boundaries
%%%maxima likely correspond to grain centers
%%%watershed nucleates from minima
%watershed the inverted, smoothed gradient
basins = cast( watershed( gradient_gauss ),'single' );
%re-introduce void spaces
basins(mask==0) = 0;
%plot_volume(basins, "Watershed Basins of "+name_dataset);
create_dataset(path_file_output, path_celldata, 'WatershedBasins', cast(basins, 'int32'))

%find the average euler angles corresponding to the watershed basins
n_basins = max(basins,[],'all');
mu    = zeros(size(orientations));
sigma = zeros(size(orientations));
delta = zeros(size(orientations));
for id_basin = 1:n_basins
    for component = 1:size(orientations,4)
        indecies = false(size(orientations));
        indecies(:,:,:,component) = (basins == id_basin);
        mu   (indecies) = mean( orientations(indecies) );
        delta(indecies) = abs ( mu(indecies) - orientations(indecies) );
        sigma(indecies) = std ( orientations(indecies) );
    end
end
%plot_volume(mu, "mu")
%plot_volume(sigma, "sigma")
create_dataset(path_file_output, path_celldata, 'WatershedBasinMean'             , cast(mu   , 'single'))
create_dataset(path_file_output, path_celldata, 'WatershedBasinMeanDifference'   , cast(delta, 'single'))
create_dataset(path_file_output, path_celldata, 'WatershedBasinStandardDeviation', cast(sigma, 'single'))

function result = normalize_0_1(matrix)
    result = matrix - min(matrix(:));
    result = result ./ max(result(:));
end

function dataset = read_dream3d_dataset(path_file,path_group,name_dataset)
    dataset = h5read(path_file,path_group+"/"+name_dataset);
    dataset = permute(dataset,length(size(dataset)):-1:1);
end

function write_dream3d_dataset(path_file,path_dataset, data)
    data = permute(data, length(size(data)):-1:1);
    h5write(path_file, path_dataset, single(data));
end

%rescale 0-2*pi to 0-255
function x = radians_to_colors(x)
    x = x*(255)/(2*pi);
end

%rescale 0-255 to 0-2*pi
function x = colors_to_radians(x)
    x = x*(2*pi)/(255);
end

function display_stats_dataset(dataset,description)
    spacing = "   ";
    disp( ...
        "Dataset("+description+"): "+newline+ ...
        spacing+"Shape="+join(string(size(dataset)))+newline+ ...
        spacing+"Min  ="+string(min(dataset,[],"all"))+newline+ ...
        spacing+"Max  ="+string(max(dataset,[],"all"))+newline+ ...
        spacing+"Ave  ="+string(mean(dataset,"all"))+newline ...
        )
end

%1-0^x = {x>0 => 1, otherwise => 0
function A = is_greater_than_zero(varargin)
    if nargin == 0
        error('is_greater_than_zero requires input arguments!')
    end
    if nargin >= 1
        A = varargin{1};
        tolerance = 0;
    end
    if nargin >= 2
        tolerance=varargin{2};
    end
    A = 1-0.^max(A-tolerance,0);
end

function color_map = get_color_map()
    % %use custom made colormap
    % intensity = [0, 127, 255];
    % color = ...
    %     [  0,  0,255;
    %      255,  0,255;
    %      255,  0,  0] ...
    %      ./ 255;
    % querypoints = 0:255;
    % color_map = interp1(intensity,color,querypoints);

    %use pre-built colormap
    %colormap = colormap(gca,parula);
    color_map = colormap(gca,jet);
end

function plot_volume(dataset, title_name)
    figures = findall(groot,'Type','figure');
    figure(length(figures)+1)
    %volumeViewer(dataset)
    volshow(dataset,'Colormap',get_color_map(),'BackgroundColor','w')
    title(title_name)
    
    %testing camera (2022-10-31)
    %view([1,1,1])
    %set(gca, 'CameraPosition', [500,500,500]);
end

function datatype_dream3d = get_datatype_matlab_to_dream3d(datatype_matlab)
    if strcmp(datatype_matlab, 'single')
        datatype_dream3d = "DataArray<float>";
    elseif strcmp(datatype_matlab, 'int32')
        datatype_dream3d = "DataArray<int32_t>";
    elseif strcmp(datatype_matlab, 'uint32')
        datatype_dream3d = "DataArray<uint32_t>";
    elseif strcmp(datatype_matlab, 'uint8')
        datatype_dream3d = "DataArray<bool>";
    elseif strcmp(datatype_matlab, 'string')
        datatype_dream3d = "StringDataArray";
    end
end

function attribute_details = create_attribute_details(name, path, type)
    attribute_details.Name       = name;
    attribute_details.AttachedTo = path;
    attribute_details.AttachType = type;
end

function create_dataset(path_file, group, name, data)
    
    % create path to new dataset
    path_dataset = group+"/"+name;

    % matlab uses fortran type arrays, but dream3d uses c type arrays
    % these two arrays are dimensionally reversed
    data = permute(data, length(size(data)):-1:1);

    % add a trailing singleton to arrays with one component
    if numel(size(data)) < 4
        data = permute(data, [length(size(data))+1, 1:length(size(data))]);
    end
    
    % if the array already exists, just overwrite the data inside it
    try

        h5info(path_file, path_dataset);
        fprintf('Dataset ''%s'' exists, modifying dataset\n', name)

        % write data to dataset
        h5write(path_file, path_dataset, data);

        return
    
    % if the array does not exist, create it and the required attributes
    catch

        fprintf('Dataset ''%s'' does not exist, creating dataset\n', name)

        % write dataset attributes
        dims = flip(size(data));
        axes = ['x='; 'y='; 'z='];

        ComponentDimensions = cast( dims(end), 'uint64' );
        DataArrayVersion = cast( 2, 'int32' );
        ObjectType = get_datatype_matlab_to_dream3d(class(data));
        TupleAxisDimensions = join( join( [ [axes(1:numel(dims)-1, :), string(flip(dims(1:end-1))')]' ]', '', 2 ), ',', 1 );
        TupleDimensions = cast( flip(dims(1:end-1)), 'uint64' );

%         % create and write data to dataset
%         h5create(path_file, path_dataset, size(data));
%         h5write(path_file, path_dataset, data);
        
%         % create dataset attributes
%         h5writeatt(path_file, path_dataset, 'ComponentDimensions'  , ComponentDimensions);
%         h5writeatt(path_file, path_dataset, 'DataArrayVersion'     , DataArrayVersion   );
%         h5writeatt(path_file, path_dataset, 'ObjectType'           , ObjectType         );
%         h5writeatt(path_file, path_dataset, 'Tuple Axis Dimensions', TupleAxisDimensions);
%         h5writeatt(path_file, path_dataset, 'TupleDimensions'      , TupleDimensions    );
        
        data_details.Location = group;
        data_details.Name = name;
        
        ComponentDimensions_details = create_attribute_details('ComponentDimensions'  , path_dataset, 'dataset');
        DataArrayVersion_details    = create_attribute_details('DataArrayVersion'     , path_dataset, 'dataset');
        ObjectType_details          = create_attribute_details('ObjectType'           , path_dataset, 'dataset');
        TupleAxisDimensions_details = create_attribute_details('Tuple Axis Dimensions', path_dataset, 'dataset');
        TupleDimensions_details     = create_attribute_details('TupleDimensions'      , path_dataset, 'dataset');
        
        hdf5write( ...
            path_file                                       , ...
            data_details               , data               , ...
            ComponentDimensions_details, ComponentDimensions, ...
            DataArrayVersion_details   , DataArrayVersion   , ...
            ObjectType_details         , ObjectType         , ...
            TupleAxisDimensions_details, TupleAxisDimensions, ...
            TupleDimensions_details    , TupleDimensions    , ...
            'WriteMode'                , 'append'             ...
        )

        return

    end
    
end