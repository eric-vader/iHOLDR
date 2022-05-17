# https://stackoverflow.com/questions/63626389/how-to-sort-points-along-a-hilbert-curve-without-using-hilbert-indices

N=9 # 9 points
n=2 # 2 dimension 
m=3 # order of Hilbert curve
def BitTest(x,od):
    result = x & (1 << od)
    return int(bool(result))

def BitFlip(b,pos):
    b ^= 1 << pos
    return b
def partition(A,st,en,od,ax,di):
    i = st
    j = en
    while True:
        while i < j and BitTest(A[i][ax],od) == di:
            i = i + 1
        while i < j and BitTest(A[j][ax],od) != di:
            j = j - 1
            
        if j <= i:
            return i
        
        A[i], A[j] = A[j], A[i]

def HSort(A,st,en,od,c,e,d,di,cnt):
    if en<=st: 
        return
    p = partition(A,st,en,od,(d+c)%n,BitTest(e,(d+c)%n))

    if c==n-1:
        if od==0:
            return
        
        d2= (d+n+n-(2 if di else cnt + 2)) % n
        e=BitFlip(e,d2)
        e=BitFlip(e,(d+c)%n)
        HSort(A,st,p-1,od-1,0,e,d2,False,0)
        
        e=BitFlip(e,(d+c)%n)
        e=BitFlip(e,d2)
        d2= (d+n+n-(cnt + 2 if di else 2))%n
        HSort(A,p,en,od-1,0,e,d2,False,0)
    else:
        HSort(A,st,p-1,od,c+1,e,d,False,(1 if di else cnt+1))
        e=BitFlip(e,(d+c)%n)
        e=BitFlip(e,(d+c+1)%n)
        HSort(A,p,en,od,c+1,e,d,True,(cnt+1 if di else 1))
        e=BitFlip(e,(d+c+1)%n)
        e=BitFlip(e,(d+c)%n)
        
array = [[2,2],[2,4],[3,4],[2,5],[3,5],[1,6],[3,6],[5,6],[3,7]]
HSort(array,st=0,en=N-1,od=m-1,c=0,e=0,d=0,di=False,cnt=0)
print(array)