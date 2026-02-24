#!.venv/bin/python3

from pyspedas import mms, download

rnge = ['2019-08-11', '2019-08-18']

mms.scm(trange=rnge, time_clip=True, datatype='scb', data_rate='brst', latest_version=True, no_update=False)
#mms.edp(trange=rnge, time_clip=True, datatype='dce', data_rate='brst', latest_version=True, no_update=False)
