import os
import sys
import datetime

# EDP

# MMS1
# 7AM, 10AM, 11AM, 8-9PM
# 1AM - 5AM, 1 Ts around 2PM
# 12AM, 2AM for 2 TS, 9AM for 2, 10 for 2, 11AM, and one for 2PM



# SCM
# one TS around 3AM, then same as EDP
#

dates = [
	datetime.datetime(2021, 2, 23),
	datetime.datetime(2021, 3, 2),
	#datetime.datetime(2017, 1, 25)
]

probes = [1,]

if __name__ == '__main__':
	for date in dates:
		for field in ['B', 'E', 'FGM']:
			for probe in probes:
				os.system(f'{sys.executable} -u download.py {probe} {date.strftime('%Y-%m-%d')} {(date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')} {field}')
