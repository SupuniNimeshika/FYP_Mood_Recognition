with open('mood.tsv') as f:
    contents = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
file=open('new.tsv','w')
contents = [x.strip() for x in contents]
for content in contents:
    mood =content.split(' ',1)[0]
    comment = (content.split(' ',1)[1])
    sen = mood+'|'+comment
    print(sen)
    file.write(sen+'\n')
file.close()