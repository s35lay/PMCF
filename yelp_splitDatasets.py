import pickle
import numpy as np

file = 'sparse_yelp/yelp_dataset'

class buffWriter:
	def __init__(self, filename):
		self.buffSize = 100000
		self.buff = [None] * self.buffSize
		self.cur = 0
		self.filename = 'sparse_yelp/' + filename

	def write(self, s):
		if self.cur < self.buffSize:
			self.buff[self.cur] = s
			self.cur += 1
		else:
			self.writeBuff()

	def writeBuff(self):
		with open(self.filename, 'a', encoding='latin1') as fs:
			for i in range(self.cur):
				fs.write(self.buff[i])
		self.cur = 0

reviewBuff = buffWriter('review')
tipBuff = buffWriter('tip')
checkinBuff = buffWriter('checkin')
photoBuff = buffWriter('photo')
userBuff = buffWriter('user')
bussBuff = buffWriter('bussiness')
trashBuff = buffWriter('trash')
buffs = [reviewBuff, tipBuff, checkinBuff, photoBuff, userBuff, bussBuff, trashBuff]
with open(file, 'r', encoding='latin1') as fs:
	for line in fs:
		if not line.startswith('{'):
			line = line[line.find('{'):]

		if '\"review_id\"' in line:
			reviewBuff.write(line)
		elif '\"friends\"' in line:
			userBuff.write(line)
		elif '\"photo_id\"' in line:
			photoBuff.write(line)
		elif '\"longitude\"' in line and '\"latitude\"' in line:
			bussBuff.write(line)
		elif '\"compliment_count\"' in line:
			tipBuff.write(line)
		elif '\"business_id\"' in line and '\"date\"' in line and '\"review_id\"' not in line:
			checkinBuff.write(line)
		else:
			trashBuff.write(line)
			print(line)
for buff in buffs:
	buff.writeBuff()