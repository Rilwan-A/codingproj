
from models.SSSDS4Imputer import SSSDS4Imputer
from models.SSSDSAImputer import SSSDSAImputer
from models.CSDIS4 import CSDIS4

MAP_MNAME_MODEL = {
    'CSDIS4':CSDIS4,
    'SSSDS4':SSSDS4Imputer,
    'SSSDSA':SSSDSAImputer 
    }