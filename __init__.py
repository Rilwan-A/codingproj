
from diffusion import DiffusionAlvarez, DiffusionCSDI

from datasets.datasets import DsetYahooStocks, Dset_V2, DsetYahooStocksMaxMin

from  models import MAP_MNAME_MODEL


MAP_NAME_DIFFUSION = {
    'alvarez':DiffusionAlvarez,
    'csdi': DiffusionCSDI
}

MAP_NAME_DSET = {
    "electricity":Dset_V2,
    "ettml_1056":Dset_V2,
    "mujocco":Dset_V2,
    # "PEMS-BAY":GenericDataset,
    "ptbxl_248":Dset_V2,
    "ptbxl_1000":Dset_V2,
    'stocks':DsetYahooStocks,
    'stocks_maxmin':DsetYahooStocksMaxMin

    # "Solar":GenericDataset
    }

