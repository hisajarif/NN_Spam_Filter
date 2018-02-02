# import all dependecies
import os
from mailparser import MailParser
import re
from tqdm import tqdm


def textExtraction(f):
    parser = MailParser()
    raw_mail=parser.parse_from_file(f)
    body = parser.body
    return body

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    text2=cleantext.split("\n")
    fcc=" "
    for i in range(len(text2)):
        if text2[i]!='\n':
            fcc+=text2[i]
    return fcc

def Mail2txt(path):
	DEBUG =False
	directory=path
	l = []
	for file in os.listdir(directory):
		img = directory + file
		if DEBUG : print (img)
		l.append(img)
	l=sorted(l)	
	holdtext = []
	for i in tqdm(range(len(l))):
		holdtext.append(cleanhtml(textExtraction(l[i])))
	return holdtext

