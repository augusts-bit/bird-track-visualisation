import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as PathEffects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
from IPython.display import HTML
import  matplotlib.pyplot as plt
from cartopy.io.img_tiles import GoogleTiles
import matplotlib.image as mpimg
import pandas as pd
import geopandas as gpd
from IPython.display import HTML
import random
import tqdm
import sys

# Function to subset data by ringnumber and sort
def subset_ringnr(data, ringnr):
    gdf = data[(data["ringnr"] == ringnr)]
    gdf = gdf[(gdf['lat'].notna())]
    gdf = gdf[(gdf['lon'].notna())]
    gdf = gdf[pd.notnull(gdf['date_time'])]
    gdf = gdf.to_crs(3857) # to pseudo mercator

    # Sort by time
    gdf['date_time'] = pd.to_datetime(gdf['date_time'], format='mixed')
    gdf = gdf.sort_values(by='date_time')

    return gdf

# Function to reduce points by significant movement
def reduce_by_movement(gdf, significance=0.5):
    # Make new column that contains both lat and lon
    gdf['latpluslon'] = gdf['lat'] + gdf['lon']
    
    # Only take the rows with significant lat lon difference
    indexes_tokeep = []
    compare_number = None
    
    for i in range(len(gdf)):
        current_value = gdf.iloc[i]['latpluslon']
        if i == 0:
            indexes_tokeep.append(i)
            compare_number = current_value
        else:
            prev_value = gdf.iloc[i-1]['latpluslon']
            if abs(current_value - prev_value) > significance or abs(current_value - compare_number) > significance:
                indexes_tokeep.append(i)
                compare_number = current_value
    
    # Subset based on the indexes to keep
    subset_gdf = gdf.iloc[indexes_tokeep]
    return subset_gdf

# Function to reduce points by time
def reduce_by_time(gdf, resample='24h'):
    ringnr = gdf['ringnr'].iloc[0]
    soort_nl = gdf['species_nl'].iloc[0]
    gdf = gdf.set_index('date_time')
    gdf_resampled = gdf.resample(resample).mean(numeric_only=True)
    gdf_resampled['lat'] = gdf_resampled['lat'].interpolate()
    gdf_resampled['lon'] = gdf_resampled['lon'].interpolate()
    gdf = gpd.GeoDataFrame(gdf_resampled, geometry=gpd.points_from_xy(gdf_resampled.lon, gdf_resampled.lat))
    gdf = gdf.reset_index()

    # Make sure it still has a necessary columns
    gdf['ringnr'] = ringnr
    gdf['soort'] = soort_nl

    return gdf

# Function to animate single logger
def plot_animation_single(gdf, projection='ortho', tail_length = 100, frames_between_points = 10, interval=5):

    # Map extent
    fig = plt.figure(figsize=(12, 12))

    # Initialize Basemap
    m = Basemap(projection=projection, resolution=None, lat_0=0, lon_0=0)
    m.shadedrelief(scale=1)
    # m.bluemarble(scale=1)

    # Loop through each pair of points and create points in between
    lats_list = []
    lons_list = []
    dates_list = []
    for i in range(len(gdf)):
        start_lat = gdf.iloc[i]['lat']
        start_lon = gdf.iloc[i]['lon']
        start_date = gdf.iloc[i]['date_time']

        if i != len(gdf)-1:
            end_lat = gdf.iloc[i+1]['lat']
            end_lon = gdf.iloc[i+1]['lon']
            end_date = gdf.iloc[i+1]['date_time']
        else:
            continue

        # Generate points between start and end
        lats = np.linspace(start_lat, end_lat, frames_between_points)
        lons = np.linspace(start_lon, end_lon, frames_between_points)
        dates = pd.date_range(start=start_date, end=end_date, periods=frames_between_points)
        lats_list.append(lats)
        lons_list.append(lons)
        dates_list.append(dates)

    # Concatenate points
    lats = np.concatenate(lats_list)
    lons = np.concatenate(lons_list)
    dates = np.concatenate(dates_list)

    # Create a point with tail object on the Basemap
    point, = m.plot([], [], 'mo', markersize=10)
    tail, = m.plot([], [], 'm-', linewidth=2)
    # date_text = plt.text(0.05, 0.95, '', transform=plt.gca().transAxes, fontsize=15, verticalalignment='top', color='green')
    date_text = plt.text(0.05, 0.95, '', transform=plt.gca().transAxes, fontsize=18, verticalalignment='top', color='white',
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])

    # Initialize the point
    def init():
        point.set_data([], [])
        date_text.set_text('')
        return point, date_text

    # Update the point position
    def update(frame):
        x, y = m(lons[frame], lats[frame])
        point.set_data(x, y)

        # tail_x, tail_y = m(lons[:frame+1], lats[:frame+1])
        # tail.set_data(tail_x, tail_y)

        if frame < tail_length:
            tail_x, tail_y = m(lons[:frame + 1], lats[:frame + 1])
            tail.set_data(tail_x, tail_y)
        else:
            tail_x, tail_y = m(lons[frame-tail_length:frame + 1], lats[frame-tail_length:frame + 1])
            tail.set_data(tail_x, tail_y)

        # date_text.set_text({dates[frame]})
        current_date = pd.to_datetime(dates[frame])
        date_text.set_text(f'{current_date.strftime("%Y-%m-%d")}') # %H:00")}')

        return point, tail, date_text

    # Create the animation
    ani = FuncAnimation(fig, update, frames=tqdm.tqdm(range(len(lats)), file=sys.stdout), init_func=init, blit=True, interval=interval) # frames = len(lats_list),
    # HTML(ani.to_html5_video())

    ani.save("single_example.mp4")

# Function to animate multiple/all loggers
def plot_animation_multiple(gdf, projection='ortho', tail_length = 100, frames_between_points=10, interval=5):
    # Map extent
    fig = plt.figure(figsize=(12, 12))

    # Initialize Basemap
    m = Basemap(projection=projection, resolution=None, lat_0=0, lon_0=0)
    m.shadedrelief(scale=1)
    # m.bluemarble(scale=1)

    # Loop through ringnrs
    ringnumbers = gdf['ringnr'].unique().tolist()
    all_lats_list = []
    all_lons_list = []
    all_dates_list = []
    for ringnr in ringnumbers:
        ringnr_gdf = gdf.loc[gdf['ringnr'] == ringnr]

        # Loop through each pair of points and create points in between
        lats_list = []
        lons_list = []
        dates_list = []
        for i in range(len(ringnr_gdf)):
            start_lat = ringnr_gdf.iloc[i]['lat']
            start_lon = ringnr_gdf.iloc[i]['lon']
            # start_date = ringnr_gdf.index[i]
            start_date = ringnr_gdf.iloc[i]['date_time']

            if i != len(ringnr_gdf) - 1:
                end_lat = ringnr_gdf.iloc[i + 1]['lat']
                end_lon = ringnr_gdf.iloc[i + 1]['lon']
                # end_date = ringnr_gdf.index[i + 1]
                end_date = ringnr_gdf.iloc[i + 1]['date_time']
            else:
                continue

            # Generate points between start and end
            lats = np.linspace(start_lat, end_lat, frames_between_points)
            lons = np.linspace(start_lon, end_lon, frames_between_points)
            dates = pd.date_range(start=start_date, end=end_date, periods=frames_between_points)
            lats_list.append(lats)
            lons_list.append(lons)
            dates_list.append(dates)

        # Concatenate points
        if len(lats_list) > 1: 
            lats = np.concatenate(lats_list)
            lons = np.concatenate(lons_list)
            dates = np.concatenate(dates_list)
            all_lats_list.append(lats)
            all_lons_list.append(lons)
            all_dates_list.append(dates)

    # Flatten all dates and sort them to get the chronological order
    all_dates_flat = np.concatenate(all_dates_list)
    unique_sorted_dates = np.unique(all_dates_flat)

    # Create lists for point and tail objects with random colours
    def generate_random_color():
        return plt.cm.tab10(random.random())

    points = []
    tails = []
    for _ in ringnumbers:
        color = generate_random_color()
        point, = m.plot([], [], 'o', color=color, markersize=10)
        tail, = m.plot([], [], '-', color=color, linewidth=2)
        points.append(point)
        tails.append(tail)

    # Add text object for the date
    date_text = plt.text(0.05, 0.95, '', transform=plt.gca().transAxes, fontsize=18, verticalalignment='top', color='black',
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='white')])

    # Initialize the points
    def init():
        for point, tail in zip(points, tails):
            point.set_data([], [])
            tail.set_data([], [])
        date_text.set_text('')
        return points + tails + [date_text]

    # Update the point position
    def update(frame):
        current_time = unique_sorted_dates[frame]
        for lats, lons, dates, point, tail in zip(all_lats_list, all_lons_list, all_dates_list, points, tails):
            idx = np.searchsorted(dates, current_time)
            if idx < len(lats):
                x, y = m(lons[idx], lats[idx])
                point.set_data(x, y)

                # tail_x, tail_y = m(lons[:idx + 1], lats[:idx + 1])
                # tail.set_data(tail_x, tail_y)

                if idx < tail_length:
                    tail_x, tail_y = m(lons[:idx + 1], lats[:idx + 1])
                    tail.set_data(tail_x, tail_y)
                else:
                    tail_x, tail_y = m(lons[idx-tail_length:idx + 1], lats[idx-tail_length:idx + 1])
                    tail.set_data(tail_x, tail_y)

        current_time = pd.to_datetime(current_time)
        date_text.set_text(f'{current_time.strftime("%Y-%m-%d %H:00")}')
        return points + tails + [date_text]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=tqdm.tqdm(range(len(unique_sorted_dates)), file=sys.stdout), init_func=init, blit=True, interval=interval)
    ani.save("multiple_example.mp4")

# Function to animate with moving basemaps
def plot_animation_moving(gdf, zoom_level = 10, extent_margin = 0.1, tail_length = 50, frames_between_points = 10, interval=5, plot_windfarms = False, windfarms = None):

    tiler = GoogleTiles(style="street")
    mercator = tiler.crs

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': mercator}) # ccrs.Orthographic()})

    # Initial extent
    initial_extent = [gdf.iloc[0]['lon']-extent_margin, gdf.iloc[0]['lon']+extent_margin, gdf.iloc[0]['lat']-extent_margin, gdf.iloc[0]['lat']+extent_margin]
    ax.set_extent(initial_extent)

    # Add background and features
    ax.add_image(tiler, zoom_level)
    # ax.stock_img()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.coastlines(resolution='10m')
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)

    if plot_windfarms == True:
        windfarms.plot(ax=ax)

    # Loop through each pair of points and create points in between
    lats_list = []
    lons_list = []
    dates_list = []
    for i in range(len(gdf)):
        start_lat = gdf.iloc[i]['lat']
        start_lon = gdf.iloc[i]['lon']
        start_date = gdf.iloc[i]['date_time']

        if i != len(gdf)-1:
            end_lat = gdf.iloc[i+1]['lat']
            end_lon = gdf.iloc[i+1]['lon']
            end_date = gdf.iloc[i+1]['date_time']
        else:
            continue

        # Generate points between start and end
        lats = np.linspace(start_lat, end_lat, frames_between_points)
        lons = np.linspace(start_lon, end_lon, frames_between_points)
        dates = pd.date_range(start=start_date, end=end_date, periods=frames_between_points)
        lats_list.append(lats)
        lons_list.append(lons)
        dates_list.append(dates)

    # Concatenate points
    lats = np.concatenate(lats_list)
    lons = np.concatenate(lons_list)
    dates = np.concatenate(dates_list)

    # Create a point with tail object on the Basemap
    # line, = ax.plot(lons[0], lats[0], linewidth=1, color='m', transform=ccrs.PlateCarree())
    point, = ax.plot(lons[0:1], lats[0:1], 'mo', markersize=10, transform=ccrs.PlateCarree())
    tail, = ax.plot(lons[:1], lats[:1], 'm-', linewidth=2, transform=ccrs.PlateCarree())
    date_text = plt.text(0.05, 0.95, '', transform=plt.gca().transAxes, fontsize=18, verticalalignment='top', color='white',
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])

    # Initialize 
    # def init():
    #     point.set_data([], [])
    #     tail.set_data([], [])
    #     date_text.set_text('')
    #     return point, tail, date_text

    def add_image(factory, *factory_args, **factory_kwargs):
        img, extent, origin = factory.image_for_domain(
            ax._get_extent_geom(factory.crs),
            factory_args[0],
        )
        ax.imshow(
            img,
            extent=extent,
            origin=origin,
            transform=factory.crs,
            *factory_args[1:],
            **factory_kwargs
        )

    # Update the point position
    def update(frame):

        # line.set_data(lons[:frame], lats[:frame])
        point.set_data([lons[frame]], [lats[frame]])
        if frame < tail_length:
            tail.set_data(lons[:frame+1], lats[:frame+1])
        else:
            tail.set_data(lons[frame-tail_length:frame+1], lats[frame-tail_length:frame+1])

        # date_text.set_text({dates[frame]})
        current_date = pd.to_datetime(dates[frame])
        date_text.set_text(f'{current_date.strftime("%Y-%m-%d %H:00")}')

        # Update the map extent to follow the line
        ax.set_extent([lons[frame] - extent_margin, lons[frame] + extent_margin, lats[frame] - extent_margin, lats[frame] + extent_margin])

        if frame % 1000:
            for artist in ax.get_images():  # Remove all image layers
                artist.remove()
            fig.canvas.draw_idle()
            add_image(tiler, zoom_level)
        
        return point, tail, date_text
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=tqdm.tqdm(range(len(lats)), file=sys.stdout), blit=False, interval=interval) # frames = len(lats_list),
    # HTML(ani.to_html5_video())

    ani.save("moving_example.mp4")

# Function to animate multiple/all loggers with stationary basemap
def plot_animation_multiple_stationary(gdf, projection='ortho', zoom_level = 10, extent_margin = 0.8, lon_centre = 4.4, lat_centre = 53.2, with_tail = False,
                                       tail_length = 750, frames_between_points=10, interval=2, plot_windfarms = False, windfarms1 = None, windfarms2 = None):
    
    # Tiles
    class ESRImap(GoogleTiles):
        def _image_url(self, tile):
            x, y, z = tile
            url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
                'World_Topo_Map/MapServer/tile/{z}/{y}/{x}.jpg').format(
                z=z, y=y, x=x)
            return url
    tiler = GoogleTiles(style="satellite")
    # tiler = ESRImap()
    mercator = tiler.crs

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': mercator}) # ccrs.Orthographic()})
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    # Initial extent
    initial_extent = [lon_centre-(extent_margin*1.65), lon_centre+(extent_margin*1.65), lat_centre-extent_margin, lat_centre+extent_margin]
    ax.set_extent(initial_extent)

    # Add background and features
    ax.add_image(tiler, zoom_level)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.add_feature(cfeature.LAND)
    # ax.coastlines(resolution='10m')
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)

    # Windfarms
    if plot_windfarms == True:
        windfarms1.plot(ax=ax, color = 'skyblue')
        windfarms2.plot(ax=ax, markersize=1, color='darkslategray')

    # Loop through ringnrs
    ringnumbers = gdf['ringnr'].unique().tolist()
    all_lats_list = []
    all_lons_list = []
    all_dates_list = []
    for ringnr in tqdm.tqdm(ringnumbers):
        ringnr_gdf = gdf.loc[gdf['ringnr'] == ringnr]

        # # Skip if not start in North Holland
        # first_lat = ringnr_gdf.iloc[0]['lat']
        # first_lon = ringnr_gdf.iloc[0]['lon']

        # min_lat = 52.25
        # max_lat = 53.25
        # min_lon = 4.4
        # max_lon = 5.6
        # if not (min_lat <= first_lat <= max_lat and min_lon <= first_lon <= max_lon):
        #     continue  # Skip this ringnr if the first point is outside North Holland

        # Loop through each pair of points and create points in between
        lats_list = []
        lons_list = []
        dates_list = []
        for i in range(len(ringnr_gdf)):
            start_lat = ringnr_gdf.iloc[i]['lat']
            start_lon = ringnr_gdf.iloc[i]['lon']
            # start_date = ringnr_gdf.index[i]
            start_date = ringnr_gdf.iloc[i]['date_time']

            if i != len(ringnr_gdf) - 1:
                end_lat = ringnr_gdf.iloc[i + 1]['lat']
                end_lon = ringnr_gdf.iloc[i + 1]['lon']
                # end_date = ringnr_gdf.index[i + 1]
                end_date = ringnr_gdf.iloc[i + 1]['date_time']
            else:
                continue

            # Generate points between start and end
            lats = np.linspace(start_lat, end_lat, frames_between_points)
            lons = np.linspace(start_lon, end_lon, frames_between_points)
            dates = pd.date_range(start=start_date, end=end_date, periods=frames_between_points)
            lats_list.append(lats)
            lons_list.append(lons)
            dates_list.append(dates)

        # Concatenate points
        if len(lats_list) > 1: 
            lats = np.concatenate(lats_list)
            lons = np.concatenate(lons_list)
            dates = np.concatenate(dates_list)
            all_lats_list.append(lats)
            all_lons_list.append(lons)
            all_dates_list.append(dates)

    # Flatten all dates and sort them to get the chronological order
    all_dates_flat = np.concatenate(all_dates_list)
    unique_sorted_dates = np.unique(all_dates_flat)

    # Create lists for point and tail objects with random colours
    def generate_random_color():
        return plt.cm.tab10(random.random())

    points = []
    tails = []
    for _ in ringnumbers:
        color = generate_random_color()
        point, = ax.plot([], [], 'o', color=color, markersize=8, zorder=10)
        tail, = ax.plot([], [], '-', color=color, linewidth=2, zorder=5)
        points.append(point)
        tails.append(tail)

    # Add text object for the date
    date_text = plt.text(0.05, 0.95, '', transform=plt.gca().transAxes, fontsize=22, verticalalignment='top', color='black',
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='white')])

    # Initialize the points
    def init():
        for point, tail in zip(points, tails):
            point.set_data([], [])
            tail.set_data([], [])
        date_text.set_text('')
        return points + tails + [date_text]

    # Update the point position
    def update(frame):
        current_time = unique_sorted_dates[frame]
        for lats, lons, dates, point, tail in zip(all_lats_list, all_lons_list, all_dates_list, points, tails):
            idx = np.searchsorted(dates, current_time)
            if idx < len(lats):
                # x, y point
                x, y = ax.projection.transform_point(lons[idx], lats[idx], ccrs.PlateCarree())
                point.set_data([x], [y]) # point.set_data([lons[idx]], [lats[idx]])

                # tail
                if with_tail == True:
                    if frame < tail_length:
                        # tail.set_data(lons[:frame+1], lats[:frame+1])
                        transformed_tail = ax.projection.transform_points(ccrs.PlateCarree(), np.array(lons[:idx+1]), np.array(lats[:idx+1]))
                        tail_x, tail_y = transformed_tail[:, 0], transformed_tail[:, 1]
                        tail.set_data([tail_x], [tail_y]) # tail.set_data(lons[:idx+1], lats[:idx+1])
                    else:
                        transformed_tail = ax.projection.transform_points(ccrs.PlateCarree(), np.array(lons[idx-tail_length:idx+1]), np.array(lats[idx-tail_length:idx+1]))
                        tail_x, tail_y = transformed_tail[:, 0], transformed_tail[:, 1]
                        tail.set_data([tail_x], [tail_y]) # tail.set_data(lons[:idx+1], lats[:idx+1])
                else:
                    transformed_tail = ax.projection.transform_points(ccrs.PlateCarree(), np.array(lons[:idx+1]), np.array(lats[:idx+1]))
                    tail_x, tail_y = transformed_tail[:, 0], transformed_tail[:, 1]
                    tail.set_data([tail_x], [tail_y]) # tail.set_data(lons[:idx+1], lats[:idx+1])

        current_time = pd.to_datetime(current_time)
        date_text.set_text(f'{current_time.strftime("%Y-%m-%d")}') # %H:00")}')
        return points + tails + [date_text]

    # Create the animation
    ani = FuncAnimation(fig, update, frames=tqdm.tqdm(range(len(unique_sorted_dates)), file=sys.stdout), init_func=init, blit=True, interval=interval)
    ani.save("multiple_example_stationary.mp4")

# Function to animate with moving basemaps on a globe 
def plot_animation_moving_globe(gdf, zoom_level = 10, extent_margin = 0.1, tail_length = 50, frames_between_points = 10, interval=5, plot_windfarms = False, windfarms = None):

    # Tiles
    class ESRImap(GoogleTiles):
        def _image_url(self, tile):
            x, y, z = tile
            url = ('https://server.arcgisonline.com/ArcGIS/rest/services/' \
                'World_Physical_Map/MapServer/tile/{z}/{y}/{x}.jpg').format(
                z=z, y=y, x=x)
            return url
    # tiler = GoogleTiles(style="satellite")
    # tiler = GoogleTiles(style="terrain")
    tiler = GoogleTiles(style="street")
    # tiler = ESRImap()
    mercator = tiler.crs

    # Set up the figure and axis
    proj = ccrs.Orthographic() # ccrs.Orthographic()
    proj.threshold /= 10 # zo dat er geen tiles missen bij de polen https://github.com/SciTools/cartopy/issues/1907
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': proj})

    # Initial extent
    initial_extent = [gdf.iloc[0]['lon']-extent_margin, gdf.iloc[0]['lon']+extent_margin, gdf.iloc[0]['lat']-extent_margin, gdf.iloc[0]['lat']+extent_margin]
    ax.set_extent(initial_extent)

    # Add background and features
    ax.add_image(tiler, zoom_level)
    # ax.stock_img()
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    # ax.coastlines(resolution='10m')
    # ax.add_feature(cfeature.LAND)
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.LAKES, alpha=0.5)
    # ax.add_feature(cfeature.RIVERS)

    if plot_windfarms == True:
        windfarms.plot(ax=ax)

    # Loop through each pair of points and create points in between
    lats_list = []
    lons_list = []
    dates_list = []
    for i in range(len(gdf)):
        start_lat = gdf.iloc[i]['lat']
        start_lon = gdf.iloc[i]['lon']
        start_date = gdf.iloc[i]['date_time']

        if i != len(gdf)-1:
            end_lat = gdf.iloc[i+1]['lat']
            end_lon = gdf.iloc[i+1]['lon']
            end_date = gdf.iloc[i+1]['date_time']
        else:
            continue

        # Generate points between start and end
        lats = np.linspace(start_lat, end_lat, frames_between_points)
        lons = np.linspace(start_lon, end_lon, frames_between_points)
        dates = pd.date_range(start=start_date, end=end_date, periods=frames_between_points)
        lats_list.append(lats)
        lons_list.append(lons)
        dates_list.append(dates)

    # Concatenate points
    lats = np.concatenate(lats_list)
    lons = np.concatenate(lons_list)
    dates = np.concatenate(dates_list)

    # Create a point with tail object on the Basemap
    # line, = ax.plot(lons[0], lats[0], linewidth=1, color='m', transform=ccrs.PlateCarree())
    point, = ax.plot(lons[0:1], lats[0:1], 'mo', markersize=10, transform=ccrs.PlateCarree())
    tail, = ax.plot(lons[:1], lats[:1], 'm-', linewidth=2, transform=ccrs.PlateCarree())
    date_text = plt.text(0.05, 0.95, '', transform=plt.gca().transAxes, fontsize=25, verticalalignment='top', color='white',
                         path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])

    # Initialize 
    # def init():
    #     point.set_data([], [])
    #     tail.set_data([], [])
    #     date_text.set_text('')
    #     return point, tail, date_text

    def add_image(factory, *factory_args, **factory_kwargs):
        img, extent, origin = factory.image_for_domain(
            ax._get_extent_geom(factory.crs),
            factory_args[0],
        )
        ax.imshow(
            img,
            extent=extent,
            origin=origin,
            transform=factory.crs,
            *factory_args[1:],
            **factory_kwargs
        )

    # Update the point position
    def update(frame):

        # line.set_data(lons[:frame], lats[:frame])
        point.set_data([lons[frame]], [lats[frame]])
        if frame < tail_length:
            tail.set_data(lons[:frame+1], lats[:frame+1])
        else:
            tail.set_data(lons[frame-tail_length:frame+1], lats[frame-tail_length:frame+1])

        # date_text.set_text({dates[frame]})
        current_date = pd.to_datetime(dates[frame])
        date_text.set_text(f'{current_date.strftime("%Y-%m-%d")}') # %H:00")}')

        # Update the map extent to follow the line
        ax.set_extent([lons[frame] - extent_margin, lons[frame] + extent_margin, lats[frame] - extent_margin, lats[frame] + extent_margin])

        if frame % 1000:
            for artist in ax.get_images():  # Remove all image layers
                artist.remove()
            fig.canvas.draw_idle()
            add_image(tiler, zoom_level)
        
        return point, tail, date_text
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=tqdm.tqdm(range(len(lats)), file=sys.stdout), blit=False, interval=interval) # frames = len(lats_list),
    # HTML(ani.to_html5_video())

    ani.save("moving_globe_example.mp4")