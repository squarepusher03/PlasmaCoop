#!.venv/bin/python3

from pyspedas import mms, download

rnge = ['2020-09-02/06:30', '2020-09-02/07:00']

mms.scm(trange=rnge, time_clip=True, datatype='schb', data_rate='brst', latest_version=True, no_update=False)
mms.edp(trange=rnge, time_clip=True, datatype='dce', data_rate='brst', latest_version=True, no_update=False)
