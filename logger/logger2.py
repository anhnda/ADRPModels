import  os , yaml
import logging.config
import const
C_DIR = os.path.abspath(os.path.dirname(__file__))
LOG_CONF = None
try:
    LOG_CONF = const.LOG_CON
except:
    pass
if LOG_CONF == None:
    const.LOG_CONF = "%s/logger.yaml" % C_DIR

class MyLogger():
    def __init__(self,logPath = None):

        with open(const.LOG_CONF) as f:
            D = yaml.load(f)
            #print D
            D.setdefault('version', 1)
        if logPath != None:
            D['handlers']['file']['filename'] = logPath

        logging.config.dictConfig(D)

        # create logger
        self.allLogger = logging.getLogger('allLogger')
        self.fileLogger = logging.getLogger('fileLogger')
        self.consoleLogger = logging.getLogger('consoleLogger')

    def infoAll(self,msg):
        self.allLogger.info(msg)

    def infoFile(self,msg):
        self.fileLogger.info(msg)
