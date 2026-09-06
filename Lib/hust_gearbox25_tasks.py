def shift_distance(s,t): return .5*(abs(s//5-t//5)/4+abs(s%5-t%5)/4)
def task_set():
    pairs=[(s,t) for s in range(25) for t in range(25) if s!=t]
    groups={'small':[p for p in pairs if shift_distance(*p)<=.25], 'medium':[p for p in pairs if .25<shift_distance(*p)<=.75], 'large':[p for p in pairs if shift_distance(*p)>.75]}
    out=[]
    for level,items in groups.items():
        items=sorted(items,key=lambda p:(shift_distance(*p),p),reverse=level=='large')
        idx=[round(i*(len(items)-1)/19) for i in range(20)]
        out += [(level,*items[j]) for j in idx]
    return out
