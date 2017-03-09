#!/usr/bin/python
# -*- coding: utf-8 -*-

def cat_say(text):
    """말하는 고양이 그림을 만든다"""
    textlen = str(text).__len__()
    print ('        {0}'.format(' ' * textlen))
    print ('       <{0}>'.format(text))
    print ('        {0}'.format('-' * textlen))
    print ('        /')

def main():
    text = raw_input ('what can cat say?')
    cat_say(text)

if __name__ == '__main__':
    main()
