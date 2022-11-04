import compare1d

# compare equivalent diameters 
files = [
    "../pipeline_output/6-feature_attributes.dream3d",
    "../pipeline_input/6-yz_small_cleaned_grains_feature_attributes.dream3d"
    ] 
path_data = "/DataContainers/ImageDataContainer/CellFeatureData/EquivalentDiameters"
OutName = "./compare_output/EquivalentDiameters"
nbBins = 15
labels = [
    "SliceGAN",
    "EBSD"
    ]
xLabel = "Grain size [$\mu m$]"
yLabel = "Fraction [$\%$]"
compare1d.plot_comparison(files, path_data, OutName, nbBins, labels, xLabel, yLabel)

# compare aspect ratios
files = [
    "../pipeline_output/6-feature_attributes.dream3d",
    "../pipeline_input/6-yz_small_cleaned_grains_feature_attributes.dream3d"
    ] 
path_data = "/DataContainers/ImageDataContainer/CellFeatureData/AspectRatios"
OutName = "./compare_output/AspectRatios"
nbBins = 15
labels = [
    "SliceGAN",
    "EBSD"
    ]
xLabel = "Aspect Ratio [$\mu m$]"
yLabel = "Fraction [$\%$]"
compare1d.plot_comparison(files, path_data, OutName, nbBins, labels, xLabel, yLabel)

# compare disorientation
files = [
    "../pipeline_output/6-feature_attributes.dream3d",
    "../pipeline_input/6-yz_small_cleaned_grains_feature_attributes.dream3d"
    ] 
path_data = "/DataContainers/ImageDataContainer/CellFeatureData/MisorientationList"
OutName = "./compare_output/MisorientationList"
nbBins = 50
labels = [
    "SliceGAN",
    "EBSD"
    ]
xLabel = "Disorientation [$^{\circ}$]"
yLabel = "Fraction [$\%$]"
compare1d.plot_comparison(files, path_data, OutName, nbBins, labels, xLabel, yLabel)

quit()

# compare area fractions
for axis in ["X","Y","Z"]:
    ### Paths
    files = [
        f"../pipeline_output/13-2d_ebsd_{axis}.txt",
        f"../pipeline_output/13-3d_original_{axis}.txt",
        f"../pipeline_output/13-3d_modified_{axis}.txt"
        ] 
    path_data = "/DataContainers/ImageDataContainer/CellFeatureData/EquivalentDiameters"
    OutName = f"./compare_output/AreaFractions_{axis}"

    nbBins = 25
    labels = [
        "EBSD",
        "SliceGAN",
        "Improved"
        ]
    xLabel = f"Area Fraction ({'XYZ'.replace(axis, '')})"
    yLabel = "Fraction [$\%$]"
    compare1d.plot_comparison(files, path_data, OutName, nbBins, labels, xLabel, yLabel)