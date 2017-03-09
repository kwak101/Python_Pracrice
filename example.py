# -*- coding: utf-8 -*-
# "Non-Ascii character를 막기 위함
#
# this is python programming exercise
#
# summary:
# string = 'str' + 'str1' + 'str2' * how_many_to_repeat_the_string
# print(), int(), str() function
# try:
# except:
# list = [el,...]
# dictionary = {k:v,...}
# tuple = (invariant, ...)
# function list() vs function tuple()
# if ():
#   sentence
# for valel in container:
#   sentence
# def function (param,...):
#   sentence

contacts={
    'john': {
        'tel'   :   '555-2049',
        'email' :   'john@email.com'
    },
    'sue' : {
        'tel'   : '444-5555',
        'email' : 'sue@naver.com'
    },
    '은호' : {
        'tel'   : '222-4444',
        'email' : '은호@이메일.컴'
    }
}

for contact in contacts:
    print("{0}의 연락처 정보:".format(contact))
    print(contacts[contact]['tel'])
    print(contacts[contact]['email'])

def display_facts(facts):
    for fact in facts:
        print("{0}의 팩트는 \'{1}\'이다.".format(fact, facts[fact]))

facts_one={
    '은호':'공부를 잘해요',
    '소영':'요리를 잘해요',
    '용재':'일을 잘해요'
}

facts_two={
    'eun-ho':'studies well',
    'so-young':'cooks well',
    'yong-jae':'works well'
}

for facts in [facts_one,facts_two]:
    display_facts(facts)

days_of_the_week=("일요일","월요일","화요일","수요일","목요일","금요일","토요일")
(day_1, day_2, day_3, day_4, day_5, day_6, day_7) = days_of_the_week
sunday=days_of_the_week[0]
print(sunday)
print(day_2)
print()

days_of_the_week_list = list(days_of_the_week)
print("type of this object is {0}".format(type(days_of_the_week)))
print("type of this object is {0}".format(type(days_of_the_week_list)))

print ("print contacts_as_list_of_tuple\n")
contacts_as_list_of_tuple = (('은호','010-3167-7994'),('소영','010-6271-7994'),('용재','010-6281-7994'))
for element in contacts_as_list_of_tuple:
    print (element)

for (name,phone) in contacts_as_list_of_tuple:
    print ("{0} - {1}".format(name,phone))

#hosts=open('/etc/hosts')
#hosts_file_contents = hosts.read()
#print(hosts_file_contents)
#hosts.close()

with open('/etc/hosts') as hosts:
    print ('file closed? {0}'.format(hosts.closed))
    print (hosts.read())
print('file read end')
print('file closed? {0}'.format(hosts.closed))

with open('/etc/hosts') as file:
    for line in file:
        print(line.rstrip())

with open('/Users/kwak101/MyWorkingLab/SRE.txt.mp3','rb') as mp3:
    print (mp3.mode)

try:
    with open('./contacts2.txt','w') as contacts:
        contacts.write(contacts_as_list_of_tuple.__str__()+'\n')
except:
    print("file open error")

lineno=1
try:
    file = open('./example.py')
    for line in file:
        print("{0:>8}: {1}".format(lineno,line.rstrip()))
        lineno +=1
except:
    print('file open error')

import sys
list=[]
try:
    file = open ('./data.txt')
    for line in file:
        list.append(line.rstrip())
    file.close()
    list.sort()
    file2 = open ('./data_sorted.txt', 'w')
    for item in list:
        file2.write(item+'\n')
    file2.close()
except:
    print ('file access error')
    sys.exit(1)

import time
print(time.asctime())
print(time.timezone)


import cat_say

def main():
    cat_say.cat_say("huhuhuhuhu")
    cat_say.cat_say("hahahahahaha")


if __name__ == '__main__':
    main()

