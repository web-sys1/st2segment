from stream2segment.process import yaml_load
from stream2segment.process.gui.main import show_gui
import os

thisdir = os.path.dirname(__file__)

dburl = yaml_load(os.path.join(thisdir, 'download.yaml'))['dburl']
c = '%s/paramtable.yaml' % thisdir
p = '%s/paramtable.py' % thisdir

show_gui(dburl, p, c, port=8000)