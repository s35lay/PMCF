from xmlrpc.client import MININT
import numpy as np
import json
import pickle
from TimeLogger import log
from scipy.sparse import csr_matrix
import time

def ok(year, month):
	# return True
	if year >= 2018 and year <= 2018 and month >=1 and month <= 6:
		return True
minn = 2022
maxx = 0
def transTime(date):
	timeArr = time.strptime(date, '%Y-%m-%d %H:%M:%S')
	year = timeArr.tm_year
	month = timeArr.tm_mon
	global minn
	global maxx
	minn = min(minn, year)
	maxx = max(maxx, year)
	if ok(year, month):
		return time.mktime(timeArr)
	return None

def mapping(infile):
	usrId = dict()
	itmId = dict()
	usrid, itmid = [0, 0]
	interaction = list()
	with open(infile, 'r', encoding='utf-8') as fs:
		for line in fs:
			data = json.loads(line.strip())
			row = data['user_id']
			col = data['business_id']
			timeStamp = transTime(data['date'])
			if timeStamp is None:
				continue
			if row not in usrId:
				usrId[row] = usrid
				interaction.append(dict())
				usrid += 1
			if col not in itmId:
				itmId[col] = itmid
				itmid += 1
			usr = usrId[row]
			itm = itmId[col]
			interaction[usr][itm] = timeStamp
	print('minimum and maximum year', minn, maxx)
	return interaction, usrid, itmid

def checkFunc1(cnt):
	return cnt >= 3
def checkFunc2(cnt):
	return cnt >= 3
def checkFunc3(cnt):
	return cnt >= 3

def filter(interaction, usrnum, itmnum, ucheckFunc, icheckFunc, filterItem=True):
	# get keep set
	usrKeep = set()
	itmKeep = set()
	itmCnt = [0] * itmnum
	for usr in range(usrnum):
		data = interaction[usr]
		usrCnt = 0
		for col in data:
			itmCnt[col] += 1
			usrCnt += 1
		if ucheckFunc(usrCnt):
			usrKeep.add(usr)
	for itm in range(itmnum):
		if not filterItem or icheckFunc(itmCnt[itm]):
			itmKeep.add(itm)

	# filter data
	retint = list()
	usrid = 0
	itmid = 0
	itmId = dict()
	for row in range(usrnum):
		if row not in usrKeep:
			continue
		usr = usrid
		usrid += 1
		retint.append(dict())
		data = interaction[row]
		for col in data:
			if col not in itmKeep:
				continue
			if col not in itmId:
				itmId[col] = itmid
				itmid += 1
			itm = itmId[col]
			retint[usr][itm] = data[col]
	return retint, usrid, itmid

def split(interaction, usrnum, itmnum):
	pickNum = 10000
	# random pick
	usrPerm = np.random.permutation(usrnum)
	pickUsr = usrPerm[:pickNum]

	tstInt = [None] * usrnum
	exception = 0
	for usr in pickUsr:
		temp = list()
		data = interaction[usr]
		for itm in data:
			temp.append((itm, data[itm]))
		if len(temp) == 0:
			exception += 1
			continue
		temp.sort(key=lambda x: x[1])
		tstInt[usr] = temp[-1][0]
		interaction[usr][tstInt[usr]] = None
	print('Exception:', exception, np.sum(np.array(tstInt)!=None))
	return interaction, tstInt

def trans(interaction, usrnum, itmnum):
	r, c, d = [list(), list(), list()]
	for usr in range(usrnum):
		if interaction[usr] == None:
			continue
		data = interaction[usr]
		for col in data:
			if data[col] != None:
				r.append(usr)
				c.append(col)
				d.append(data[col])
	intMat = csr_matrix((d, (r, c)), shape=(usrnum, itmnum))
	return intMat

prefix = 'sparse_yelp/'
log('Start')
interaction, usrnum, itmnum = mapping(prefix + 'review')
log('Id Mapped, usr %d, itm %d' % (usrnum, itmnum))

checkFuncs = [checkFunc1, checkFunc2, checkFunc3]
for i in range(3):
	filterItem = True# if i < 2 else False
	interaction, usrnum, itmnum = filter(interaction, usrnum, itmnum, checkFuncs[i], checkFuncs[i], filterItem)
	print('Filter', i, 'times:', usrnum, itmnum)
log('Sparse Samples Filtered, usr %d, itm %d' % (usrnum, itmnum))

trnInt, tstInt = split(interaction, usrnum, itmnum)
log('Datasets Splited')
trnMat = trans(trnInt, usrnum, itmnum)
log('Train Mat Done')
with open(prefix+'trn_mat', 'wb') as fs:
	pickle.dump(trnMat, fs)
with open(prefix+'tst_int', 'wb') as fs:
	pickle.dump(tstInt, fs)
log('Interaction Data Saved')