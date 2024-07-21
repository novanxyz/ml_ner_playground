import sys
import fileinput

from lang_util.spell import SpellCorrector
spell = SpellCorrector()

file = sys.argv[1]
print(file)
for line in fileinput.input(file,inplace=True):
    try:
        word, tags = line.split("\t")        
        word = spell.validate(word)
        print("%s\t%s" % (word, tags) )
    except:
        print(line)
    