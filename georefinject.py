import os

import numpy as np
import tifffile
import torch
import tqdm
from osgeo import gdal, osr

from utils import convert_to_color

def geo_inject(inference_dataset, model, out_path):
    """Inject georeference information. 
    Feeds test samples to model and generates geo-referenced prediction outputs.
    """
    
    for i in tqdm.tqdm(range(len(inference_dataset))):

        x_tensor = inference_dataset[i][0]
        tensor = torch.unsqueeze(torch.tensor(x_tensor), axis = 0)
        tensor = tensor.float() 
        tensor = torch.tensor(tensor, device = 'cuda')

        pr_mask = model.predict(tensor).detach().cpu().numpy()
        pr_mask = pr_mask.squeeze()
        pr_max  = np.argmax(pr_mask.squeeze(), 0)
        pr_max11 = convert_to_color(pr_max)
    
        image_dir = inference_dataset[i][-1]
        ds = gdal.Open(image_dir)
        
        im = tifffile.imread(image_dir)
        width = im.shape[1]
        height = im.shape[0]

        tfw = ds.GetGeoTransform()
        prj = ds.GetProjection()
        srs = osr.SpatialReference(wkt=prj)
        driver = gdal.GetDriverByName("GTiff")
        out_p = os.path.join(out_path, os.path.split(image_dir)[-1])
        outdata = driver.Create(out_p, height, width, 3, gdal.GDT_Byte)
        outdata.SetGeoTransform(tfw)##sets same geotransform as input
        outdata.SetProjection(ds.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(pr_max11[:,:,0])
        outdata.GetRasterBand(2).WriteArray(pr_max11[:,:,1])
        outdata.GetRasterBand(3).WriteArray(pr_max11[:,:,2])

        outdata.FlushCache()
        outdata = None
        ds=None
