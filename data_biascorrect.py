import netCDF4 as nc
import numpy as np
import numbers
import matplotlib.pyplot as plt
import xarray as xr


# Define Year Range for Historical and for Predicted
yearrange = list(range(2007, 2015+1))
scenarioyearrange = list(range(2092, 2100+1))


# Define Latitude and Longitude
latitude_s = input("Latitude Start: ")
latitude_e = input("Latitude End: ")
longitude_s = input("Longitude Start: ")
longitude_e = input("Longitude End: ")
locationname = input("Region Examined: ")
lat_fs = float(latitude_s)
long_fs = float(longitude_s)
lat_fe = float(latitude_e)
long_fe = float(longitude_e)

counter = 0
mean = np.zeros((len(yearrange), 6))
std = np.zeros((len(yearrange), 6))
             
    
# Loop over each year in the selected period
for year in yearrange:
    
    #Define the Files being used for the bias correction: ERA5 and the Historical Runs
    path_1 = "/gws/nopw/j04/cpdn_rapidwatch/luke_scratch/ERA5_historical/ERA_" + str(year) + ".nc"
    path_2 = "/gws/nopw/j04/cpdn_rapidwatch/luke_scratch/batch_892_ensemble/ensmean_" + str(year) + ".nc"


    # Extract Datasets
    ds1 = nc.Dataset(path_1)
    ds2 = nc.Dataset(path_2)  


    # Extract latitude/longitude
    lat1 = ds1.variables['latitude'][:]
    lon1 = ds1.variables['longitude'][:]
    lat2 = ds2.variables['lat'][:]
    lon2 = ds2.variables['lon'][:]


    # find closest index to specified value
    def near(array,value):
        idx=(abs(array-value)).argmin()
        return idx


    # Find index of nearest point to desired location
    ix1s = near(lon1, long_fs)
    iy1s = near(lat1, lat_fs)
    ix1e = near(lon1, long_fe)
    iy1e = near(lat1, lat_fe)
    ix2s = near(lon2, long_fs)
    iy2s = near(lat2, lat_fs)
    ix2e = near(lon2, long_fe)
    iy2e = near(lat2, lat_fe)


    #Extract Wind Data from 
    v10 = ds1['v10'][:, iy1e:iy1s, ix1s:ix1e]
    u10 = ds1['u10'][:, iy1e:iy1s, ix1s:ix1e]
    WindS_1 = [(u ** 2 + v ** 2) ** 0.5 for u, v in zip(u10, v10)]
    WindS_2 = ds2['item3249_1hrly_mean'][:, :, iy2s:iy2e, ix2s:ix2e]
    WindS_ds1 = xr.DataArray(WindS_1)
    WindS_ds2 = xr.DataArray(WindS_2)

    #Analyse the datasets
    mean_ds1 = xr.DataArray.mean(WindS_ds1).values
    mean_ds2 = xr.DataArray.mean(WindS_ds2).values
    std_ds1 = xr.DataArray.std(WindS_ds1).values
    std_ds2 = xr.DataArray.std(WindS_ds2).values
    print(str(year) +' - Mean: ERA = {mean_ERA}, CPDN = {mean_CPDN}'.format(mean_ERA=str(mean_ds1), mean_CPDN = str(mean_ds2)))
    print(str(year) +' - Standard Deviation: ERA = {std_ERA}, CPDN = {std_CPDN}'.format(std_ERA=str(std_ds1), std_CPDN = str(std_ds2)))
    
    
    #Read .nc as a dataset
    WindS_ToAdjust = xr.open_dataset(path_2)
    
    
    #Bias Correction approach
    mask_lon1 = (WindS_ToAdjust.lon >= long_fs ) & (WindS_ToAdjust.lon <= long_fe)
    mask_lat1 = (WindS_ToAdjust.lat >= lat_fs ) & (WindS_ToAdjust.lat <= lat_fe)
    WindS_ToAdjust_region1 = WindS_ToAdjust.where(mask_lon1 & mask_lat1, drop=True) #get subset of data
    WindS_adjust_1 = mean_ds1 + (std_ds1 / std_ds2 * (WindS_ToAdjust_region1 - mean_ds2)) #Adjust subset of dataset
    WindS_adjusted_1 = WindS_adjust_1["item3249_1hrly_mean"][:, :, :, :]
    print(str(year) + "- Mean of adjusted dataset:", WindS_adjusted_1.mean().values)
    print(str(year) + "- STD of adjusted dataset:", WindS_adjusted_1.std().values)
    
    #Store mean and std before and after bias correction
    mean[counter, 0] = mean_ds2
    mean[counter, 1] = WindS_adjusted_1.mean().values
    std[counter, 0] = std_ds2
    std[counter, 1] = WindS_adjusted_1.std().values
    
    year_future = scenarioyearrange[counter]
    
    #Read Path for 1.5C and 2C scenario
    path_3 = "/gws/nopw/j04/cpdn_rapidwatch/luke_scratch/batch_893_ensemble/ensmean_" + str(year_future) + ".nc"
    path_4 = "/gws/nopw/j04/cpdn_rapidwatch/luke_scratch/batch_894_ensemble/ensmean_" + str(year_future) + ".nc"
    
    #Read .nc as a dataset
    WindS_ToAdjust_15 = xr.open_dataset(path_3)
    WindS_ToAdjust_2 = xr.open_dataset(path_4)
    
    #Apply Bias Correction to 1.5C scenario
    mask_lon_15 = (WindS_ToAdjust_15.lon >= long_fs ) & (WindS_ToAdjust_15.lon <= long_fe)
    mask_lat_15 = (WindS_ToAdjust_15.lat >= lat_fs ) & (WindS_ToAdjust_15.lat <= lat_fe)
    WindS_ToAdjust_region_15 = WindS_ToAdjust_15.where(mask_lon_15 & mask_lat_15, drop=True) #get subset of data
    WindS_adjust_15 = mean_ds1 + (std_ds1 / std_ds2 * (WindS_ToAdjust_region_15 - mean_ds2))
    WindS_adjusted_15 = WindS_adjust_15["item3249_1hrly_mean"][:, :, :, :]

    #Calculate mean and std for 1.5C
    mean_init_1_5 = WindS_ToAdjust_15["item3249_1hrly_mean"][:, :, iy2s:iy2e, ix2s:ix2e].mean().values
    mean_adjust_1_5 = WindS_adjusted_15.mean().values
    std_init_1_5 = WindS_ToAdjust_15["item3249_1hrly_mean"][:, :, iy2s:iy2e, ix2s:ix2e].std().values
    std_adjust_1_5 = WindS_adjusted_15.std().values
    print(str(year_future) +' (1.5C) - Mean: Initial = {mean_init}, Adjusted = {mean_adjust}'.format(mean_init=str(mean_init_1_5), mean_adjust = str(mean_adjust_1_5)))
    print(str(year_future) +' (1.5C) - Standard Deviation: Initial = {std_init}, Adjusted = {std_adjust}'.format(std_init=str(std_init_1_5), std_adjust = str(std_adjust_1_5)))
    
    #Store mean and std for 1.5C
    mean[counter, 2] = mean_init_1_5
    mean[counter, 3] = mean_adjust_1_5
    std[counter, 2] = std_init_1_5
    std[counter, 3] = std_adjust_1_5
    
    #Apply Bias Correction to 2C scenario
    mask_lon_2 = (WindS_ToAdjust_2.lon >= long_fs ) & (WindS_ToAdjust_2.lon <= long_fe)
    mask_lat_2 = (WindS_ToAdjust_2.lat >= lat_fs ) & (WindS_ToAdjust_2.lat <= lat_fe)
    WindS_ToAdjust_region_2 = WindS_ToAdjust_2.where(mask_lon_2 & mask_lat_2, drop=True) #get subset of data
    WindS_adjust_2 = mean_ds1 + (std_ds1 / std_ds2 * (WindS_ToAdjust_region_2 - mean_ds2))
    WindS_adjusted_2 = WindS_adjust_2["item3249_1hrly_mean"][:, :, :, :]
    
    #Calculate mean and std for 2C
    mean_init_2 = WindS_ToAdjust_2["item3249_1hrly_mean"][:, :, iy2s:iy2e, ix2s:ix2e].mean().values
    mean_adjust_2 = WindS_adjusted_2.mean().values
    std_init_2 = WindS_ToAdjust_2["item3249_1hrly_mean"][:, :, iy2s:iy2e, ix2s:ix2e].std().values
    std_adjust_2 = WindS_adjusted_2.std().values
    print(str(year_future) +' (2C) - Mean: Initial = {mean_init}, Adjusted = {mean_adjust}'.format(mean_init=str(mean_init_2), mean_adjust = str(mean_adjust_2)))
    print(str(year_future) +' (2C) - Standard Deviation: Initial = {std_init}, Adjusted = {std_adjust}'.format(std_init=str(std_init_2), std_adjust = str(std_adjust_2)))
    
    #Store mean and std for 2C
    mean[counter, 4] = mean_init_2
    mean[counter, 5] = mean_adjust_2
    std[counter, 4] = std_init_2
    std[counter, 5] = std_adjust_2
    

    #Save the Bias Corrected Files
    new_filename_1 = 'WindS_' + str(year) + locationname + ".nc" 
    new_filename_2 = 'WindS_' + str(year_future) + locationname + "_1.5C" + ".nc" 
    new_filename_3 = 'WindS_' + str(year_future) + locationname + "_2C"+ ".nc" 
    WindS_adjust_1.to_netcdf(path = 'BiasCorrected/' + locationname + '/' + new_filename_1)
    WindS_adjust_15.to_netcdf(path = 'BiasCorrected/' + locationname + '/' + new_filename_2)
    WindS_adjust_2.to_netcdf(path = 'BiasCorrected/' + locationname + '/' + new_filename_3)
    
    counter = counter + 1
    
print(mean)
print(std)

# Save Data of the mean and std in a table
np.savetxt('./BiasCorrected/' + locationname + '/' + 'mean'+ locationname + '.csv', mean, header = 'Hist, Hist (BC), 1.5C, 1.5C (BC), 2C, 2C (BC)', delimiter=',', fmt=['%0.3f','%0.3f','%0.3f', '%0.3f', '%0.3f', '%0.3f'], comments='' )
np.savetxt('./BiasCorrected/' + locationname + '/' + 'std'+ locationname + '.csv', std, header = 'Hist, Hist (BC), 1.5C, 1.5C (BC), 2C, 2C (BC)', delimiter=',', fmt=['%0.3f', '%0.3f', '%0.3f', '%0.3f', '%0.3f', '%0.3f'], comments='' )