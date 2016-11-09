import csv
import numpy as np
t = "./"
back = ".csv"
filename = "./influenza.fna"
aaa = [0, 1, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 2, 3, 198, 198, 198, 198, 198, 4, 198, 198, 5, 6, 198, 198, 198, 7, 198, 8, 198, 198, 198, 198, 9, 198, 198, 198, 10, 198, 11, 198, 198, 198, 12, 13, 14, 198, 198, 15, 198, 16, 198, 198, 198, 17, 18, 198, 198, 19, 20, 198, 21, 198, 198, 198, 22, 23, 24, 198, 198, 198, 25, 198, 26, 198, 198, 198, 198, 198, 27, 198, 198, 198, 198, 198, 198, 198, 198, 28, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 29, 198, 198, 198, 30, 31, 198, 198, 198, 198, 32, 198, 198, 198, 198, 198, 198, 33, 198, 198, 198, 198, 198, 198, 34, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 35, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 36, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198, 198]


def getlabel(h,n):
    label = 11*h+n-12
    l = aaa[label]
    if l == 198:
        w = False
    else:
        w = True
    return l,w
filelist = list()
writers = list()
num = 20
segementL = 1000
#row = ["%d" % i for i in range(average+1)]
for i in range(num):
    name = t+str(i)+back
    csvfile = file(name,'wb')
    filelist.append(csvfile)
    writer = csv.writer(csvfile)
    #writer.writerow(row)
    writers.append(writer)

with open(filename,"r") as f:
    guide = f.readline()
    test = 0
    size = 0
    while(1):
        if(test%5000==0):
            print test
        jump = 0
        index = guide.find("virus")
        if(index == -1):
            jump = 1
        
        type = guide[index - 2]
        if(cmp("A",type) != 0):
            jump = 1
            
        first = guide.find("(")
        second = guide.find("(",first+1)
        if second == -1:
            jump = 1
        
        close = guide.find(")")
        h = guide.find("H",second,close)
        n = guide.find("N",second,close)
        if(h>=0 and n >= 0):
            hn = guide[h+1:n]
            nn = guide[n+1:close]
            if(hn.isdigit() and nn.isdigit()):
                label,w = getlabel(int(hn),int(nn))
        else:
            jump = 1
        comma = guide.find(",")
        close = guide.rfind(")")
        #protein = guide[close+2:comma]
        
        data = ""
        line = f.readline()
        while(line.find(">")==-1):
            if jump == 0:
                l = len(line)
                line = line[:l-1]
                line = line.replace('A',"1")
                line = line.replace('T',"2")
                line = line.replace('G',"3")
                line = line.replace('C',"4")
                data+=line
            line = f.readline()
            if(line == ''):
                break
        if jump == 0:
            l = len(data)
            if w and data.isdigit():
                data = np.array(list(data))
                A = np.piecewise(data,[data=='1',data!='1'],[1,0]).tostring()
                T = np.piecewise(data,[data=='2',data!='2'],[1,0]).tostring()
                G = np.piecewise(data,[data=='3',data!='3'],[1,0]).tostring()
                C = np.piecewise(data,[data=='4',data!='4'],[1,0]).tostring()
                while l >segementL:
                    dA = A[:segementL]
                    dT = T[:segementL]
                    dG = G[:segementL]
                    dC = C[:segementL]
                    data = dA+dT+dG+dC
                    writer = writers[size%num]
                    writer.writerow([label,data])
                    A = A[segementL:]
                    T = T[segementL:]
                    G = G[segementL:]
                    C = C[segementL:]
                    size +=1
                    l -= segementL
                A = A.ljust(segementL,'0')
                T = T.ljust(segementL,'0')
                G = G.ljust(segementL,'0')
                C = C.ljust(segementL,'0')
                data = A+T+G+C
                writer = writers[size%num]
                writer.writerow([label,data])
                size+=1
                test+=1
        guide = line
        if(guide == ''):
            break

print test
print size
for i in range(num):
    filelist[i].close()