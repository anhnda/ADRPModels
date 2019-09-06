import  os , yaml
import logging.config
import config
C_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_CONF = None
try:
    LOG_CONF  = config.LOG_CON
except:
    pass
if LOG_CONF == None:
    config.LOG_CONF = "%s/logger.yaml"%C_DIR
with open(config.LOG_CONF) as f:
    D = yaml.load(f)
    #print D
    D.setdefault('version', 1)
    logging.config.dictConfig(D)


# create logger
allLogger = logging.getLogger('allLogger')
fileLogger = logging.getLogger('fileLogger')
consoleLogger = logging.getLogger('consoleLogger')

def infoAll(msg):
    allLogger.info(msg)
def infoFile(msg):
    fileLogger.info(msg)
