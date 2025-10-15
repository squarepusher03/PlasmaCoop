#!.venv/bin/python3

from pyspedas import themis

rnge = ['2014-07-01', '2014-12-31']

for p in ['a']:
    themis.state(trange=rnge, probe=p, downloadonly=True)
    themis.sst(trange=rnge, probe=p, downloadonly=True)
    themis.fgm(trange=rnge, probe=p, downloadonly=True)
