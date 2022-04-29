#########
# Code for Figure 1 (Map)
#########

library(tidyverse)
library(ggmap)
library(maps)
library(mapplots)
library(rgdal)
library(sf)
library(ncdf4)
library(raster)



SO_MPAs <- st_read("mpa-shapefile-EPSG102020.shp")



# Maps for Ryan et al. fin whale paper----
scotia_bbox_MPAs <- st_bbox(c(xmin = -70, 
                              xmax = -30,
                              ymin = -65, 
                              ymax = -50),
                            crs = 4326) %>% 
  st_as_sfc() %>% 
  st_transform(st_crs(SO_MPAs))

SO_MPAs_cropped <- SO_MPAs %>% 
  st_crop(scotia_bbox_MPAs)


scotia_bbox_CCAMLR <- st_bbox(c(xmin = -75, 
                                xmax = -20,
                                ymin = -75, 
                                ymax = -50),
                              crs = 4326) %>% 
  st_as_sfc() %>% 
  st_transform(st_crs(CCAMLR_mgmt_areas))

CCAMLR_areas_cropped <- CCAMLR_mgmt_areas %>% 
  st_crop(scotia_bbox_CCAMLR)


NGshiptrack <- c(left=-46.07917,bottom=-60.35,right=-45.8975,top=-60.35)
x <- c(NGshiptrack["left"], NGshiptrack["left"], 
       NGshiptrack["right"], NGshiptrack["right"])
y <- c(NGshiptrack["bottom"], NGshiptrack["top"], 
       NGshiptrack["top"], NGshiptrack["bottom"])
NG_df <- data.frame(x, y)


Bp_supergroup <- basemap(limits = c(-65, -30, -65, -53), shapefiles = "Antarctic",
                         #bathymetry = TRUE, 
                         lon.interval = 10, rotate = TRUE) +
  geom_sf(data = SO_MPAs_cropped, fill = NA, color = "darkred") +
  #annotation_north_arrow(location = "tr", which_north = "true") +
  # geom_spatial_point(aes(x = -46, y = -60.2), 
  #                    shape = 23, fill = "goldenrod3",
  #                    color = "black", size = 2.5) +
  geom_spatial_segment(aes(x = -46.07917, y = -60.41222,
                           xend = -45.8975, yend = -60.30806), 
                       color = "goldenrod3") +
  geom_sf_label(data = CCAMLR_areas_cropped,
                aes(label = Name)) +
  annotation_scale(location = "br") 
# coord_sf(xlim = c(-70,-30), ylim = c(-65,-50),
#          crs = 4326)

Bp_supergroup


Bp_supergroup + coord_sf(xlim = c(-65,-30), ylim = c(-65,-53),
                         crs = 4326)



dev.copy2pdf(file="Bp_supergroup_sighting.pdf", 
             width=7, height=5)



interest_region <- c(left=-65,bottom=-65,right=-35,top=-53)
x <- c(interest_region["left"], interest_region["left"], 
       interest_region["right"], interest_region["right"])
y <- c(interest_region["bottom"], interest_region["top"], 
       interest_region["top"], interest_region["bottom"])
df <- data.frame(x, y)



Bp_supergroup_zoomout <- basemap(limits = c(-90, -30, -80, 40), 
                                 lon.interval = 20, lat.interval = 20) +
  annotation_north_arrow(location = "tr", which_north = "true") +
  geom_polygon(aes(x=x, y=y), data=df, fill = NA, color = "black") +
  labs(x = NULL, y = NULL) 

Bp_supergroup_zoomout


dev.copy2pdf(file="Bp_supergroup_zoomout.pdf", 
             width=2.5, height=6)
