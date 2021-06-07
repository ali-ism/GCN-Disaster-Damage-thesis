import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
from typing import Iterable

class CmapString:
    def __init__(self, palette: str, domain: Iterable[str]) -> None:
        self.domain = domain
        domain_unique = np.unique(domain)
        self.hash_table = {key: i_str for i_str, key in enumerate(domain_unique)}
        self.mpl_cmap = matplotlib.cm.get_cmap(palette, lut=len(domain_unique))

    def color(self, x: str, **kwargs):
        return self.mpl_cmap(self.hash_table[x], **kwargs)
    
    def color_list(self, **kwargs):
        return [self.mpl_cmap(self.hash_table[x], **kwargs) for x in self.domain]


def plot_on_image(labels: pd.DataFrame, subset: str, zone: str):
    plt.figure(figsize=(12,8))
    subset_marker = subset[subset.find('/')+len('/'):subset.rfind('_')]
    img = plt.imread(f'datasets/xbd/{subset_marker}/images/{zone}_post_disaster.png')
    plt.imshow(img)
    cmap = {'no-damage': 'blue', 'minor-damage': 'orange', 'major-damage': 'red', 'destroyed': 'purple', 'un-classified': 'white'}
    for _, row in labels[labels['zone']==zone].iterrows():
        plt.scatter(row['xcoords'], row['ycoords'], label=row['zone'], color=cmap[row['class']])
    plt.axis('off')
    plt.show()


def plot_on_map(labels: pd.DataFrame, mapbox=True):
    cmap = CmapString(palette='viridis', domain=labels['zone'].values)
    if mapbox:
        fig = px.scatter_mapbox(data_frame=labels, lat='lat', lon='long',
                                color=cmap.color_list(), mapbox_style='open-street-map',
                                hover_name='class', zoom=10)
        fig.layout.update(showlegend=False)
        fig.show()
    else:
        plt.figure(figsize=(12,8))
        for _, row in labels.iterrows():
            plt.scatter(row['long'], row['lat'], label=row['zone'], color=cmap.color(row['zone']))
        plt.axis('off')
        plt.show()