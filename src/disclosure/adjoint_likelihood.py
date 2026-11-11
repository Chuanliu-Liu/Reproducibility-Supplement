"""Likelihood-side sum-product of the unchanged original disclosure prefixes.

Fixed safe Minkowski boxes; complete bundle factors; full joint likelihood;
two mutable boxes, immutable base, explicit forward class-weight rows.
Float mode is diagnostic. Enclosure mode uses Arb base endpoints and the
separately validated packed directed kernel. No sampler is supplied here.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as F
import ctypes
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
from .coordinates import UnimodularCertificate, contribution_box
from .compiled import NumericalConditioningError, StateBudgetExceeded
from .oracle import ImpossibleObservation
from .estimate import PathBudgetExceeded


def fraction(value):
    if isinstance(value,F):return value
    if isinstance(value, np.longdouble):
        n,d=value.as_integer_ratio();return F(n,d)
    return F(float(value))


def directed_dyadic(m, e, dtype, upper):
    """Exact integer quantization before a validated exact dyadic conversion."""
    m,e=int(m),int(e)
    if m < 0: raise ValueError('Nonnegative endpoint required')
    if not m:return dtype(0)
    info=np.finfo(dtype);bits=info.nmant+1
    cut=max(0,m.bit_length()-bits,info.minexp-bits+1-e)
    k=m >> cut
    if upper and m != k << cut:k+=1
    value=np.ldexp(dtype(str(k)),e+cut)
    exact=F(m)*F(2)**e;represented=fraction(value)
    if not np.isfinite(value) or (represented<exact if upper else represented>exact):
        raise NumericalConditioningError('Directed conversion invariant failed')
    return value


def posterior_risk_bounds(lower, upper, labels, loss):
    lower=list(map(fraction,lower));upper=list(map(fraction,upper))
    if not sum(upper):raise ImpossibleObservation('Exact zero mass')
    if sum(lower)<=0:raise NumericalConditioningError('Uncertified normalization lower bound')
    bounds=[]
    for i,(lo,hi) in enumerate(zip(lower,upper)):
        dl=lo+sum(upper[:i])+sum(upper[i+1:])
        du=hi+sum(lower[:i])+sum(lower[i+1:])
        bounds.append((lo/dl if lo else F(0),hi/du if hi else F(0)))
    groups={}
    for label,(lo,hi) in zip(labels,bounds):
        a,b=groups.get(label,(F(0),F(0)));groups[label]=(a+lo,b+hi)
    if loss=='zero_one':risk=(max(F(0),1-max(b for a,b in groups.values())),min(F(1),1-max(a for a,b in groups.values())))
    elif loss=='brier':risk=(max(F(0),(1-sum(b*b for a,b in groups.values()))/2),min(F(1),(1-sum(a*a for a,b in groups.values()))/2))
    else:raise ValueError(loss)
    return bounds,risk


@dataclass
class ALMPath:
    increments: object
    risks: object
    target_posteriors: object
    enclosures: object = None
    increment_bounds: object = None


class AdjointLikelihood:
    def __init__(self, model, release, *, players=None, matrix=None, dtype='float64',
                 certified=False, precision=128, directed_library=None, tile_cells=16384,
                 max_bytes=768*1024**2, admission=None):
        begin=time.perf_counter();self.model,self.release=model,release
        self.q,self.h=model.contributions[0].shape[1],len(model.mixture)
        if self.q not in (1,2) or len(release.observed)!=self.q:raise ValueError('q1/q2 full likelihood required')
        self.dtype=np.dtype(dtype).type;self.certified=certified;self.precision=precision
        if self.dtype not in (np.float64,np.longdouble):raise ValueError('binary64 or host longdouble required')
        self.tile_cells,self.max_bytes,self.admission=tile_cells,max_bytes,admission
        self.players=tuple(i for i in range(len(model.probs)) if i!=model.target) if players is None else tuple(players)
        if len(set(self.players))!=len(self.players) or model.target in self.players or any(i<0 or i>=len(model.probs) for i in self.players):raise ValueError('Original players required')
        self.fixed=tuple(i for i in range(len(model.probs)) if i!=model.target and i not in self.players)
        identity=tuple(tuple(int(i==j) for j in range(self.q)) for i in range(self.q))
        self.coord=UnimodularCertificate.create(identity if matrix is None else matrix,self.q)
        self.c=[np.array([self.coord.encode(row) for row in owner],dtype=np.int64) for owner in model.contributions]
        self.p=[np.asarray(p,dtype=self.dtype) for p in model.probs]
        self.identity=(model.identity,release.identity,model.target)
        self.counters=dict(paths=0,resets=0,base_likelihood_cells=0,stencil_products=0,lookup_cells=0,allocated_cells=0,peak_mutable_bytes=0,zero_class_revivals=0)
        self.phase_seconds=dict(base=0.,reset=0.,prefix=0.,arithmetic=0.,lookup=0.)
        self.lib=None
        if certified:
            self.lib=ctypes.CDLL(str(directed_library))
            self.kernel=self.lib.alm_shift64 if self.dtype==np.float64 else self.lib.alm_shiftld
            ptr=np.ctypeslib.ndpointer(dtype=self.dtype,flags='C_CONTIGUOUS')
            self.kernel.argtypes=[ptr,ptr,*[ctypes.c_size_t]*5,ctypes.c_double if self.dtype==np.float64 else ctypes.c_longdouble,ctypes.c_int]
            self.kernel.restype=ctypes.c_int
            if self.lib.alm_round()!=0 or self.lib.alm_mantissa(int(self.dtype!=np.float64))!=np.finfo(self.dtype).nmant+1:raise NumericalConditioningError('Unvalidated rounding/mantissa')
        box=contribution_box(self.c);self.base_low=np.array(box['low'],dtype=np.int64)
        self.base=self._likelihood_table(box)
        self.base.setflags(write=False)
        self.compile_seconds=time.perf_counter()-begin

    def _admit(self, needed):
        if needed>self.max_bytes:raise StateBudgetExceeded('ALM retained/base/two-buffer byte cap: '+str(needed))
        if self.admission:self.admission(needed)
        self.counters['peak_mutable_bytes']=max(self.counters['peak_mutable_bytes'],needed)

    @property
    def memory_bytes(self):
        return self.base.nbytes+sum(p.nbytes+c.nbytes for p,c in zip(self.p,self.c))

    def _likelihood_table(self, box):
        begin=time.perf_counter();shape=tuple(box['widths']);endpoints=2 if self.certified else 1
        self._admit(endpoints*math.prod(shape)*np.dtype(self.dtype).itemsize+8*self.tile_cells*64)
        base=np.zeros((endpoints,1,*shape),dtype=self.dtype)
        inv=np.asarray(self.coord.inverse,dtype=np.int64)
        if self.certified and self.release.kind=='laplace':
            from flint import arb,ctx
            old=ctx.prec;ctx.prec=self.precision
            try:
                # Build all original-coordinate likelihood factors with Arb,
                # then multiply them on the FULL q-dimensional argument box.
                # Dropped 1/prod(2b) is one state/class-independent constant.
                per=[]
                for d in range(self.q):
                    lo=sum(int(inv[d,j])*(box['low'][j] if inv[d,j]>=0 else box['high'][j]) for j in range(self.q))
                    hi=sum(int(inv[d,j])*(box['high'][j] if inv[d,j]>=0 else box['low'][j]) for j in range(self.q))
                    values=[]
                    for s in range(lo,hi+1):
                        y=F(float(self.release.observed[d]));b=F(float(self.release.scales[d]));e=abs(F(s)-y)/b
                        a=(-(arb(e.numerator)/arb(e.denominator))).exp()
                        values.append([directed_dyadic(*a.lower().man_exp(),self.dtype,False),directed_dyadic(*a.upper().man_exp(),self.dtype,True)])
                    per.append((lo,np.array(values,dtype=self.dtype)))
            finally:ctx.prec=old
        else:per=None
        flat=base.reshape(endpoints,1,-1)
        for start in range(0,math.prod(shape),self.tile_cells):
            ids=np.arange(start,min(start+self.tile_cells,math.prod(shape)))
            coords=np.stack(np.unravel_index(ids,shape),axis=1)+self.base_low
            original=coords@inv.T
            if self.release.kind=='exact':
                values=np.all(original==np.asarray(self.release.observed),axis=1).astype(self.dtype)
                flat[:,0,start:start+len(ids)]=values
            elif self.certified:
                if per is None:raise ValueError('Certified adapter requires exact/rational small reference or Laplace')
                values=np.ones((2,len(ids)),dtype=self.dtype)
                for d,(offset,array) in enumerate(per):values=self._multiply(values,array[original[:,d]-offset].T)
                flat[:,0,start:start+len(ids)]=values
            else:
                # Includes the complete joint release adapter exactly once.
                logs=np.asarray(self.release.log_likelihood(original),dtype=self.dtype)
                values=np.exp(logs)
                if np.any(np.isfinite(logs)&(values==0)):raise NumericalConditioningError('Positive base likelihood underflow')
                flat[0,0,start:start+len(ids)]=values
        if not np.isfinite(base).all():raise NumericalConditioningError('Nonfinite base')
        self.counters['base_likelihood_cells']+=math.prod(shape)
        self.phase_seconds['base']+=time.perf_counter()-begin
        return base

    def _multiply(self,a,b):
        v=a*b
        if not self.certified:
            if np.any((a>0)&(b>0)&(v==0)):raise NumericalConditioningError('Positive factor underflow')
            return v
        lo=np.nextafter(v[0],self.dtype(-np.inf));hi=np.nextafter(v[1],self.dtype(np.inf))
        lo=np.maximum(lo,0);hi=np.where((a[1]==0)|(b[1]==0),0,hi)
        return np.array([lo,hi],dtype=self.dtype)

    def _update(self,table,low,i):
        begin=time.perf_counter();cl=self.c[i].min(axis=0);ch=self.c[i].max(axis=0)
        dims=tuple(int(d-(b-a)) for d,a,b in zip(table.shape[2:],cl,ch))
        shape=(table.shape[0],self.h,*dims);needed=table.nbytes+math.prod(shape)*np.dtype(self.dtype).itemsize+self.base.nbytes+self.tile_cells*64
        self._admit(needed);out=np.zeros(shape,dtype=self.dtype)
        for h in range(self.h):
            sourceh=min(h,table.shape[1]-1)
            for v,f in enumerate(self.c[i]):
                p=self.p[i][h,v]
                if p==0:continue
                shift=f-cl
                self.counters['stencil_products']+=math.prod(dims)*(2 if self.certified else 1)
                if self.certified:
                    for e in (0,1):
                        # Kernel is contiguous and streams rows: no object grid.
                        rows,cols=(1,dims[0]) if self.q==1 else dims
                        oldcols=table.shape[-1];dr,dc=(0,int(shift[0])) if self.q==1 else map(int,shift)
                        if self.kernel(out[e,h],table[e,sourceh],rows,cols,oldcols,dr,dc,float(p),e):raise NumericalConditioningError('Directed kernel failed')
                else:
                    width=dims[-1];rows=1 if self.q==1 else dims[0];tile_rows=max(1,self.tile_cells//width)
                    for r in range(0,rows,tile_rows):
                        for col in range(0,width,self.tile_cells):
                            if self.q==1:
                                src=table[0,sourceh,col+shift[0]:min(col+self.tile_cells,width)+shift[0]];dest=out[0,h,col:col+self.tile_cells]
                            else:
                                src=table[0,sourceh,r+shift[0]:min(r+tile_rows,rows)+shift[0],col+shift[1]:min(col+self.tile_cells,width)+shift[1]];dest=out[0,h,r:r+tile_rows,col:col+self.tile_cells]
                            prod=src*p
                            if np.any((src>0)&(prod==0)):raise NumericalConditioningError('Positive shift product underflow')
                            dest+=prod
        self.counters['allocated_cells']+=out.size
        self.phase_seconds['arithmetic']+=time.perf_counter()-begin
        return out,low-cl

    def evaluate_path(self,world,order,loss='zero_one',*,posteriors=True,deadline=None,stop_prefix=0):
        if self.identity!=(self.model.identity,self.release.identity,self.model.target):raise ValueError('Stale model/release cache')
        order=tuple(map(int,order));n=len(order)
        if len(world)!=len(self.p) or len(order)!=len(self.players) or set(order)!=set(self.players):raise ValueError('Original permutation required')
        begin=time.perf_counter();weights=[np.repeat(np.asarray(self.model.mixture,dtype=self.dtype)[None,:],2 if self.certified else 1,axis=0)]
        totals=[np.zeros(self.q,dtype=np.int64)]
        for i in order:
            value=int(world[i])
            if value<0 or value>=len(self.c[i]):raise ValueError('Invalid complete bundle value')
            a=self._multiply(weights[-1],np.repeat(self.p[i][:,value][None,:],2 if self.certified else 1,axis=0))
            scale=a[-1].max()
            if scale==0:raise ImpossibleObservation('Impossible disclosed world')
            scaled=a/scale
            if self.certified:
                scaled[0]=np.maximum(0,np.nextafter(scaled[0],self.dtype(-np.inf)));scaled[1]=np.where(a[1]==0,0,np.nextafter(scaled[1],self.dtype(np.inf)))
            weights.append(scaled);totals.append(totals[-1]+self.c[i][value])
        self.phase_seconds['prefix']+=time.perf_counter()-begin
        begin=time.perf_counter();table=self.base;low=self.base_low.copy()
        for i in self.fixed:table,low=self._update(table,low,i)
        self.counters['resets']+=1;self.phase_seconds['reset']+=time.perf_counter()-begin
        risks=[None]*(n+1);states=[None]*(n+1);bounds=[None]*(n+1)
        for k in range(n,-1,-1):
            if deadline is not None and time.perf_counter()>=deadline:raise PathBudgetExceeded(dict(next_prefix=k,completed_prefixes=n-k,risks=[None if v is None else float(v) for v in risks]))
            begin=time.perf_counter();mass=np.zeros((table.shape[0],self.model.domains[self.model.target]),dtype=self.dtype)
            for a,f in enumerate(self.c[self.model.target]):
                index=totals[k]+f-low
                if any(v<0 or v>=d for v,d in zip(index,table.shape[2:])):raise ValueError('Shifted-domain invariant')
                values=table[(slice(None),slice(None),*map(int,index))]
                for h in range(self.h):
                    p=self.p[self.model.target][h,a];s=self._multiply(weights[k][:,h],np.repeat(p,table.shape[0]))
                    term=self._multiply(s,values[:,min(h,values.shape[1]-1)])
                    v=mass[:,a]+term
                    if self.certified:
                        v[0]=max(0,np.nextafter(v[0],self.dtype(-np.inf)));v[1]=0 if mass[1,a]==0 and term[1]==0 else np.nextafter(v[1],self.dtype(np.inf))
                    mass[:,a]=v
                    self.counters['lookup_cells']+=1
            if self.certified:
                pb,rb=posterior_risk_bounds(mass[0],mass[1],self.model.target_labels,loss)
                bounds[k]=dict(posterior_bounds=pb,risk_bounds=rb,target_mass_bounds=([fraction(v) for v in mass[0]],[fraction(v) for v in mass[1]]))
                posterior=np.array([float((a+b)/2) for a,b in pb]);risk=(rb[0]+rb[1])/2
            else:
                if mass.sum()<=0:raise ImpossibleObservation('Zero normalization')
                posterior=mass[0]/mass[0].sum();groups={}
                for label,p in zip(self.model.target_labels,posterior):groups[label]=groups.get(label,self.dtype(0))+p
                if loss=='zero_one':risk=1-max(groups.values())
                elif loss=='brier':risk=(1-sum(p*p for p in groups.values()))/2
                else:raise ValueError(loss)
            states[k]=posterior;risks[k]=risk;self.phase_seconds['lookup']+=time.perf_counter()-begin
            if k==stop_prefix:break
            table,low=self._update(table,low,order[k-1])
            self.counters['zero_class_revivals']+=int(((weights[k][-1]==0)&(weights[k-1][-1]>0)).sum())
        inc=np.zeros(len(world));ib={}
        for k,i in enumerate(order,1):
            if risks[k-1] is not None and risks[k] is not None:
                inc[i]=float(risks[k-1]-risks[k])
                if self.certified:
                    a,b=bounds[k-1]['risk_bounds'],bounds[k]['risk_bounds'];ib[i]=(a[0]-b[1],a[1]-b[0])
        self.counters['paths']+=1
        return ALMPath(inc,risks,states,bounds if self.certified else None,ib if self.certified else None)

    def change_release(self,release):
        return type(self)(self.model,release,players=self.players,matrix=self.coord.matrix,dtype=self.dtype,certified=self.certified,precision=self.precision,directed_library=self.lib._name if self.lib else None,tile_cells=self.tile_cells,max_bytes=self.max_bytes,admission=self.admission)

    def retarget(self,target,labels=None):
        model=replace(self.model,target=target,target_labels=tuple(range(self.model.domains[target])) if labels is None else tuple(labels))
        full_game=set(self.players)==set(range(len(self.model.probs)))-{self.model.target}
        players=None if full_game else tuple(i for i in self.players if i!=target)
        return type(self)(model,self.release,players=players,matrix=self.coord.matrix,dtype=self.dtype,certified=self.certified,precision=self.precision,directed_library=self.lib._name if self.lib else None,tile_cells=self.tile_cells,max_bytes=self.max_bytes,admission=self.admission)
