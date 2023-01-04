import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from datetime import datetime, timedelta 

import cartopy.crs as ccrs 
import cartopy.feature 

import cmasher as cmr

# helper functions

def _generate_area_map(figsize=[8, 8], ax=None, rows_cols=[1, 1], 
                      extent=[-80, -20, -70, -40], fz=12, combo_figure=False):
    
    """
    Helper function to generate cartopy map with polar orthographic projection
    
    Input arguments: 
    figsize (list): two-element list specifying figure dimensions 
    rows_cols (list): two-element list specifying number or rows and columns for subplot 
    extent (list): list specifying longitude and latitude extents for cartopy map [lon_min, lon_max, lat_min, lat_max]
    fz (float): float or int specifying font size for map labels
    
    Output:
    
    returns axes handle
    """

    
    central_lon, central_lat = extent[0]+(extent[1]-extent[0])/2, extent[2]+(extent[3]-extent[2])/2
    
    fig, axes = plt.subplots(*rows_cols, figsize=figsize, 
                             subplot_kw={'projection': ccrs.Orthographic(central_lon, central_lat)})
    
    if np.max(rows_cols)==1:
        axes = [axes]
        
    for ax in axes:
        ax.set_extent(extent)
        #ax.gridlines()
        ax.coastlines() #resolution='50m'
        ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')

        glines = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                              linewidth=2, color='gray', alpha=0.5, linestyle='--')

        glines.xlabels_left = False
        glines.xlabels_right = False
        glines.right_labels = False
        glines.top_labels = False

        glines.xlabel_style = {'size': fz}
        glines.ylabel_style = {'size': fz}

    return axes

    
def _generate_area_map_combo_figure(figsize=[8, 8], extent=[-80, -20, -70, -40], fz=12):
    
    """
    Like generate_area_map() but generates combo plot for paper figure
    
    Output:
    
    returns axes handle
    """

    
    central_lon, central_lat = extent[0]+(extent[1]-extent[0])/2, extent[2]+(extent[3]-extent[2])/2
    
    fig = plt.figure(figsize=figsize)
    
    gs = fig.add_gridspec(2, 6, wspace=1.2, hspace=0.3)
    ax = fig.add_subplot(gs[0, :4], projection=ccrs.Orthographic(central_lon, central_lat))
    ax.set_extent([-180, 180, -90, 90])
        
    ax.set_extent(extent)
    #ax.gridlines()
    ax.coastlines() #resolution='50m'
    ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')

    glines = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=2, color='gray', alpha=0.5, linestyle='--')

    glines.xlabels_left = False
    glines.xlabels_right = False
    glines.right_labels = False
    glines.top_labels = False

    glines.xlabel_style = {'size': fz}
    glines.ylabel_style = {'size': fz}
    
    # generate other subplots
    ax2 = fig.add_subplot(gs[0, 4:])
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax5 = fig.add_subplot(gs[1, 4:])
    
    # add labels 
    y_top = 0.85
    y_bottom = 0.44
    text_locs = [[0.09, y_top], [0.63, y_top], [0.06, y_bottom], [0.36, y_bottom], [0.64, y_bottom]]
    plot_labels = 'abcde'
    
    for ii, locs in enumerate(text_locs):
        plt.text(*locs, plot_labels[ii].upper(), fontsize=fz*3, transform=fig.transFigure, weight="bold")
    
    axes_list = [ax, ax2, ax3, ax4, ax5]

    return axes_list, gs, fig
    
    
def _plot_soccom_float_data(soccom_ds, ax, tr, fz=10):
    
    """
    Function to plot soccom float data in combo-figure (panel B). Called in generate_panel_map()
    """
    
    vnames = ['POT_TEMP_ADJUSTED', 'PSAL_ADJUSTED',  'CHLA_ADJUSTED'] #'DOXY_ADJUSTED',
    vnames_dict = {'POT_TEMP_ADJUSTED': 'Potential temperature', 'PSAL_ADJUSTED': 'Salinity', 'CHLA_ADJUSTED': 'Chlorophyll-a'}
    units_dict = {'POT_TEMP_ADJUSTED': '$^{\circ}$C', 'PSAL_ADJUSTED': 'PSU', 'CHLA_ADJUSTED': 'mg/m$^3$'}
    ms=2

    float_num = soccom_ds['PLATFORM_NUMBER'].values[0].decode("utf-8").strip()
    #print(float_num)
    float_dtimes = pd.to_datetime(soccom_ds['JULD'].values)
    tsel = np.logical_and(float_dtimes>=tr[0], float_dtimes<=tr[1])

    new_pres = np.linspace(10, 1000, 200)
    vname = 'CHLA_ADJUSTED' #'DOXY_ADJUSTED',
    ms=2


    plt.sca(ax)
    vble_sub = soccom_ds[vname][tsel, :]
    pres_sub = soccom_ds['PRES_ADJUSTED'][tsel, :]
    nprof = pres_sub.shape[0]
    dtimes_sub = float_dtimes[tsel]
    cols = cmr.get_sub_cmap('rainbow', 0.1, 0.99, N=nprof)


    for jj in range(nprof):
        time_str = dtimes_sub[jj].strftime("%b %d, %Y")
        no_val = np.isnan(vble_sub[jj, :])
        vble_jj = vble_sub[jj, :].dropna("N_LEVELS")
        ax.plot(vble_jj, pres_sub[jj,:][no_val==False], '-o', ms=ms, label=time_str, color=cols(jj))

    plt.ylabel("Depth (m)", fontsize=fz)
    plt.xlabel("%s (%s)" %(vnames_dict[vname], units_dict[vname]), fontsize=fz)
    plt.ylim(400, 0)
    ax.tick_params(labelsize=fz) 

    plt.legend(ncol=2, fontsize=fz-4)


def _plot_box_avg_data(axes, data_dict, focus_box, tr, fz=10, focus_years=[2021, 2022]):
    
    """
    Function to plot area-averged SST, SIC, and Chlorophylly data in combo-figure (panels C-E). 
    Called in generate_panel_map()
    """
    
    # add sst, chl, sea ice subplots
    ship_coords_mean = np.array(data_dict['ship_coords']).mean(axis=0)
    minlat, maxlat = ship_coords_mean[0]-2, ship_coords_mean[0]+2
    minlon, maxlon = ship_coords_mean[1]-4, ship_coords_mean[1]+4

    focus_box_lons360 = focus_box[:, 0]%360
    years = np.arange(1982, 2023) 
    mons = np.arange(1, 13)

    
    focus_cols = cmr.get_sub_cmap('Reds', 0.5, 0.95, N=len(focus_years))

    vnames = ['sst', 'sic', 'chl']

    #fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #dtimes = pd.to_datetime(boxmean.time.values)
    for ii, ax in enumerate(axes):

        # get variable
        vble = data_dict[vnames[ii]]
        vble_clim = data_dict['%s_clim'%vnames[ii]]

        # compute box mean
        latr = np.logical_and(vble.lat>=focus_box[:, 1].min(), vble.lat<=focus_box[:, 1].max())
        if np.max(vble.lon)<=180:
            lonr = np.logical_and(vble.lon>=focus_box[:, 0].min(), vble.lon<=focus_box[:, 0].max())
        else:
            lonr = np.logical_and(vble.lon>=focus_box_lons360.min(), vble.lon<=focus_box_lons360.max())


        box_sel = np.logical_and(latr, lonr)
        boxmean = vble.where(box_sel).mean(dim=['lat', 'lon'])
        boxmean_clim = vble_clim.where(box_sel).mean(dim=['lat', 'lon'])
        
        # limit to end of time period
        #print(boxmean)
        dtimes = pd.to_datetime(boxmean.time.values)
        boxmean = boxmean.where(dtimes<=tr[-1]+timedelta(days=30))
        

        # plot climatology
        ax.plot(mons, boxmean_clim, lw=2, color='k', label='climatology', zorder=100)
        ax.set_ylabel(data_dict['%s_label'%vnames[ii]], fontsize=fz)

        # plot yearly ann cycle
        years_ii = np.array([dt.year for dt in data_dict["%s_dtimes"%vnames[ii]]])
        years_unique = np.unique(years_ii)

        kk = 0

        for jj, yr in enumerate(years_unique):

            tsel = years_ii==yr
            boxmean_yr = boxmean.isel(time=tsel)

            if ii==0 and jj==0:
                print(boxmean_yr)

            if yr in focus_years:
                col=focus_cols(kk)
                lw=2
                kk +=1
                ax.plot(mons[:len(boxmean_yr)], boxmean_yr, '-s', lw=lw, color=col, label=yr, ms=5)

            else:
                col= 'lightblue' #'0.3'
                lw=1
                alpha=0.8

                if jj==0:
                    ax.plot(mons[:len(boxmean_yr)], boxmean_yr, lw=lw, color=col,
                                 alpha=alpha, label='%s$-$%s' %(years_ii[0], focus_years[0]-1))
                else:
                    ax.plot(mons[:len(boxmean_yr)], boxmean_yr, lw=lw, color=col, alpha=alpha)


        # apply some formatting
        plt.sca(ax)
        plt.legend(loc=0, fontsize=fz-4, ncol=2)
        ax.tick_params(labelsize=fz) 
        plt.xticks(mons)
        plt.xlim(1, 12)
        plt.xlabel('month', fontsize=fz)

        
def convert_dms_dec(deg, mins=0, secs=0): 
    
    """
    Function to convert map coordinates from degree, minute, seconds to decimal
    
    Input:
    
    deg (float): location in degrees
    mins (float): location in minutes
    secs (float): location in seconds
    
    """
    
    dec = (deg + mins/60 + secs/(60*60))
    
    return dec 


def generate_panel_map(data_dict, ax=None, add_sic=True, focus_box_size=[5, 10], save_plot=False, time_avg=False, 
                       line_col_update={}, map_vble='sst_anom',  fz=8, leg_fz=6, cmap='', leg_loc=0, 
                       fig_size=(15, 4), fig_fmt='.png', dpi=300, sie_pct=20,
                       tr=np.array([datetime(2021, 11, 1), datetime(2022, 1, 15)]), combo_fig=False, plot_dir=''):
    
    """
    function to generate panel map
    Note: this more complex than what is needed for this paper
    
    Input:
    
    data_dict (dict): dictionary-like object containing variables to be plotted (see notebook for example)
    
    """

    # unpack data_dict
    map_dtimes = data_dict['%s_dtimes'%map_vble]
    clvls = data_dict['%s_clvls'%map_vble]
    map_data = data_dict[map_vble]
        
    # limit to focus time range
    ti = np.logical_and(map_dtimes>=tr.min(), map_dtimes<=tr.max())
    dtimes_sub = map_dtimes[ti]
    months = [dt.month for dt in dtimes_sub]
    #print(months)

    # update default line colors if necessary
    line_cols = {'sic_clim': 'k', 'sic_mon': 'magenta', 'ship': 'green', 
                 'soccom_float': 'cyan', 'argo_float': 'teal', 'region_box': 'k'}
    
    for key in line_col_update:
        if key in line_cols:
            line_cols[key] = line_col_update[key]
        else:
            print('%s is not valid. See color entries below:' %key)
            print(line_cols)

                
    # specify colormap for background data 
    if cmap=='':
        cmap = data_dict['%s_cmap'%map_vble]
            

    # generate regional map(s)
    if time_avg:
        
        map_data_mean = map_data.isel(time=ti).mean(dim='time')
        
        if combo_fig:
            axes_list, gs, fig = _generate_area_map_combo_figure(figsize=fig_size, extent=[-70, -30, -70, -50], fz=fz) 
            axes = [axes_list[0]]
        else:
            axes = _generate_area_map(figsize=fig_size, rows_cols=[1, 1], 
                                     extent=[-70, -30, -70, -50], fz=fz)
    else:    
        # create panel for each month
        axes = generate_area_map(figsize=fig_size, rows_cols=[1, len(dtimes_sub)], 
                                 extent=[-70, -30, -70, -50], fz=fz)

    
    # create box for focus region centered around ship coordinates
    # this box is used for area-averaging in a separate step
    ship_coords = data_dict['ship_coords']
    ship_coords_mean = np.array(ship_coords).mean(axis=0)
    minlat, maxlat = ship_coords_mean[0]-focus_box_size[0]/2, ship_coords_mean[0]+focus_box_size[0]/2
    minlon, maxlon = ship_coords_mean[1]-focus_box_size[1]/2, ship_coords_mean[1]+focus_box_size[1]/2

    verts = np.array([[minlon, maxlat], [minlon, minlat], 
                      [maxlon, minlat], [maxlon, maxlat], 
                      [minlon, maxlat]])

    # interpolate longitudinal segments of box to higher resolution (lons are curved in polar projection)
    vert_lats = verts[:, 1]
    vert_lons = verts[:, 0]
    vert_lats_hires = np.array([])
    vert_lons_hires = np.array([])
    for kk in range(len(vert_lats)-1):
        vert_lats_hires = np.append(vert_lats_hires, np.linspace(vert_lats[kk], vert_lats[kk+1], 50))
        vert_lons_hires = np.append(vert_lons_hires, np.linspace(vert_lons[kk], vert_lons[kk+1], 50))

    focus_box_coords = np.vstack([vert_lons_hires, vert_lats_hires]).T
    
    
    # define misc variables
    XX, YY = np.meshgrid(map_data.lon, map_data.lat)
    sie_str = ''
    #ms = 3
    mean_ship_coords = ship_coords.mean(axis=0)

    leg_lbl_list = []
    # generate plot for each axis
    for ii, ax in enumerate(axes):

        plt.sca(axes[ii])

        # add background surface data 
        # Two options: plot average for time range or make map for each month
        if time_avg:
            map_tt = map_data_mean
            time_str = "%s-%s"%(tr[0].strftime("%b-%Y"), tr[-1].strftime("%b-%Y"))                   
        else:
            map_tt = map_data.sel(time=dtimes_sub[ii], method='nearest').values
        
        im = ax.contourf(XX, YY, map_tt, clvls, vmin=clvls.min(), vmax=clvls.max(), 
                         cmap=cmap, extend='both', transform=ccrs.PlateCarree())
        
        # add ship location
        ship_col = line_cols['ship']
        
        ln1, = ax.plot(ship_coords[:, 1].mean(), ship_coords[:, 0].mean(), '*', color=ship_col, ms=10,
                       transform=ccrs.PlateCarree(), label='ship location', zorder=10)
        
        #CS_list.append(ln1[0])
        leg_lbl_list.append('ship track')
        
        # add soccom float track  
        float_ds = data_dict['soccom_ds']
        float_num = float_ds['PLATFORM_NUMBER'].values[0].decode("utf-8").strip()
        float_col = line_cols['soccom_float']
        float_dtimes = pd.to_datetime(float_ds['JULD'].values)

        float_tsel = np.logical_and(float_dtimes>=tr[0], float_dtimes<=tr[1])

        if len(float_ds['LONGITUDE'][float_tsel])==0:
            print("Warning: no data was found for float %s." %float_num)
            
        ln2, = ax.plot(float_ds['LONGITUDE'][float_tsel], float_ds['LATITUDE'][float_tsel], '-', 
                marker='o', color=float_col, transform=ccrs.PlateCarree(), 
                label='SOCCOM float %s'%float_num, ms=4, lw=1 )

        leg_lbl_list.append('SOCCOM float %s'%float_num)
        
        # add sea ice extent clim
        if add_sic:
            
            sie_str = '_sie' # used for generating figure filename
            
            if time_avg:
                #idx = -1 # plot the last month (Note: we should probably make this a time average to be consistent)
                sic_clim_map = data_dict['sic_clim'].sel(month=months).mean(dim='month')
                sic_mon_map = data_dict['sic'].sel(time=dtimes_sub, method='nearest').mean(dim='time')
                
                mon_str1 = '%s$-$%s' %(dtimes_sub[0].strftime('%b'), dtimes_sub[-1].strftime('%b'))
                mon_str2 = '%s$-$%s' %(dtimes_sub[0].strftime('%b %Y'), dtimes_sub[-1].strftime('%b %Y'))
                
                leg_lbl_list.append("SIE (%s climatology)" %mon_str1)
                leg_lbl_list.append("SIE (%s)" %mon_str2)
                
            else:
                #idx = ii
                
                sic_clim_map = data_dict['sic_clim'].sel(month=dtimes_sub[ii].month)
                sic_mon_map = data_dict['sic'].sel(time=dtimes_sub[ii], method='nearest')
                
                leg_lbl_list.append(dtimes_sub[ii].strftime("SIE (%b climatology)"))
                leg_lbl_list.append(dtimes_sub[ii].strftime("SIE (%b-%Y)"))
                
            sic_lw = 3
            sic = data_dict['sic']
            
            # clean up SIC artifacts near scotia islands
            lat_mask = np.logical_and(data_dict['sic'].lat>-57, data_dict['sic'].lon<340)==False 
            
            # plot sea ice extent contours
            CS1 = ax.contour(sic.lon, sic.lat, sic_clim_map.where(lat_mask), [sie_pct], 
                             colors=line_cols['sic_clim'], linestyles='--',transform=ccrs.PlateCarree(), 
                             linewidths=sic_lw)
            
            for c in CS1.collections:
                c.set_dashes([(0, (1.5, 0.5))])

            CS2 = ax.contour(data_dict['sic'].lon, data_dict['sic'].lat, 
                             sic_mon_map.where(lat_mask), [sie_pct], colors=line_cols['sic_mon'], 
                             transform=ccrs.PlateCarree(), linewidths=sic_lw)

            
            


        # add focus box
        ax.plot(focus_box_coords[:, 0], focus_box_coords[:, 1], ':', color=line_cols['region_box'], lw=2, 
                transform=ccrs.Geodetic())
        
        # add legend
        h_list = [ln1, ln2]
        h_list.append(CS1.legend_elements()[0][0]) 
        h_list.append(CS2.legend_elements()[0][0]) 
            
        leg = plt.legend(h_list, leg_lbl_list, loc=leg_loc, fontsize=leg_fz, 
                         framealpha=1, ncol=2, handlelength=3) 
        leg.get_frame().set_facecolor('0.2')
        plt.setp(leg.get_texts(), color='w')

        

    # add figure colorbar
    if combo_fig:
        cbar = plt.colorbar(im, extend='max')
    else:
        cb_ax = plt.gcf().add_axes([0.93, 0.23, 0.02, 0.55])
        cbar = plt.colorbar(im, cax=cb_ax, extend='max')
    
    cbar.set_ticks(clvls[::4])
    cbar.ax.set_ylabel(data_dict['%s_label'%map_vble], fontsize=fz)
    cbar.ax.tick_params(labelsize=fz) 
    cbar.solids.set_edgecolor("face")
    
    
    if combo_fig:
        # add other subplots
        _plot_soccom_float_data(data_dict['soccom_ds'], axes_list[1], fz=fz, tr=tr)
        
        _plot_box_avg_data(axes_list[2:], data_dict, focus_box=focus_box_coords, fz=fz, tr=tr, focus_years=[2021, 2022])
        
        if save_plot:
            plt.savefig(os.path.join(plot_dir, 'figure2_%s_map%s'%(map_vble, fig_fmt)), dpi=dpi, bbox_inches='tight')
        
    else:
        if save_plot:
            time_str = "%s_%s"%(tr[0].strftime("%b-%Y"), tr[-1].strftime("%b-%Y") )
            plt.savefig(os.path.join(plot_dir, '%s_%s%s%s'%(map_vble, time_str, sie_str, fig_fmt)),
                        dpi=dpi, bbox_inches='tight')
     
    if combo_fig:
        return focus_box_coords, axes_list, fig
    else:
        return focus_box_coords