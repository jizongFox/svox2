# this is to use nerfstudio dataloader to load data and convert it into opencv convention.
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig

from .dataset_base import DatasetBase


class NerfStudioDataset(DatasetBase):

    def __init__(self):
        super().__init__()
        self.dataset = VanillaDataManagerConfig(dataparser=NerfstudioDataParserConfig(), ).setup()

    def shuffle_rays(self):
        super().shuffle_rays()

    def gen_rays(self, factor=1):
        super().gen_rays(factor)

    def opengl2opencv(self, rays):
        ...
