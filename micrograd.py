#!/usr/bin/env python
# coding: utf-8

# # BUILDING MICROGRAD

# ![ac39e557f27d4cd9c2c32ab79f3983fc.png](attachment:ac39e557f27d4cd9c2c32ab79f3983fc.png)

# A tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

# # SOME BASIC IMPORTS

# In[2]:


import numpy as np
import math
import matplotlib.pyplot as plt
import random


# # Let's Define f(x)

# In[4]:


def f(x):
    return 3*x**2 -4*x +5


# In[5]:


f(3.0)


# # Let's PLOT

# In[6]:


xs = np.arange(-5,5,0.25)
ys = f(xs)
print("\n---------------xs-------------------")
print(xs)
print("\n---------------ys-------------------\n")
print(ys)
plt.plot(xs,ys)


# # THE DERIVATIVE

# In[7]:


h = 0.001
x = 3.0
(f(x+h)-f(x))/h


# # A BIT BIG

# In[8]:


a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)


# In[9]:


h = 0.0001
a = 2.0
b = -3.0
c = 10.0
d1 = a*b + c
print("Initially : ",d1)
a = a+h
d2 = a*b + c
print("After a nudge : ",d2)
print("Slope : ",(d2-d1)/h)


# # The Value Object

# In[10]:


class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0
        self._backward = lambda: None
        
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self,other):
        return Value(self.data+other.data,(self,other),"+")
    def __mul__(self,other):
        return Value(self.data*other.data,(self,other),"*")
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')
        return out

a = Value(2.00,label='a')
b = Value(-3.00,label='b')
c = Value(10.00,label='c')
e = a*b
e.label = 'e'
d = e+c
d.label = 'd'
f = Value(-2.0,label='f')
L = d*f
L.label = 'L'
L


# In[10]:


L._prev


# # FORWARD PASS OF ARITHMETIC OPERATION

# ![Screenshot%20from%202023-03-24%2014-00-43.png](attachment:Screenshot%20from%202023-03-24%2014-00-43.png)

# In[ ]:





# # The Back-Propagation

# #### In Back-Propagtion, we will start from the end and calculate the derivative of the final value with respect to it's leaf nodes ( The Preceding Nodes )

# ![Screenshot%20from%202023-03-24%2014-44-41.png](attachment:Screenshot%20from%202023-03-24%2014-44-41.png)

# #### If we have computed the value of L after a forward-pass, we have to compute the derivative of L w.r.to L , w.r.to f, w.r.to d, w.r.to e, w.r.to c, w.r.to a and w.r.to b

# In[11]:


from graphviz import Digraph
 
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR'):
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ %s | data %.4f | grad = %.4f }" % (n.label,n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op), 
            dot.edge(str(id(n)) + n._op, str(id(n)))
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot


# In[12]:


L.grad = 1
draw_dot(L)


# # THE GRADIENT

# #### The Gradient of L can be computed as the following.
#     L =  f.d
#     d(L)/df = d = 4.00
#     d(L)/dd = f = -2.00
#      
#     L = f.(c+e) 
#     d(L)/dc = f = -2.00
#     d(L)/de = f = -2.00
#     
#     L = f.(c+a*b)
#     d(L)/da = f.b = 6
#     d(L)/db = f.a = -4
#     

# In[13]:


f.grad = 4.00
d.grad = -2.00
c.grad = -2.00
e.grad = -2.00
a.grad = 6.00
b.grad = -4.00


# # THE NEURAL NETWORK 

# ![cool_neural_network.jpg](attachment:cool_neural_network.jpg)

# In[14]:


# Inputs
x1 = Value(2.0,label='x1')
x2 = Value(0.0,label='x2')

# Weights
w1 = Value(-3.0,label='w1')
w2 = Value(1.0,label='w2')
 
# Bias
b = Value(6.88137358,label='b') 

# The Weighted Sum
x1w1 = x1*w1
x1w1.label = 'x1w1'
x2w2 = x2*w2
x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b
n.label = 'n'
out = n.tanh()
out.label='out'
draw_dot(out)


# # THE GRADIENT AGAIN ( Manually )

# ![Screenshot%20from%202023-03-24%2016-04-08.png](attachment:Screenshot%20from%202023-03-24%2016-04-08.png)

#      d(out)/dout = 1
#      d(out)/dn = 1-tanh²n = 1-out*out = 0.4999999
#       
#      n  = b + (x1w1+x2w2)
#      out = tanh(b + (x1w1+x2w2)) = tanh(n)
#      d(out)/db = 0.4999999
#      d(out)/d(x1w1+x2w2) = 0.4999999
#      d(out)/d(x1w1) = 0.4999999
#      d(out)/d(x2w2) = 0.4999999
#      
#      ON APPLYING CHAIN RULE
#      -----------------------------------------------------
#      x1.grad = w1.data*x1w1.grad
#      w1.grad = x1.data*x1w1.grad
#      x2.grad = w2.data*x2w2.grad
#      w2.grad = x2.data*x2w2.grad

# In[15]:


out.grad = 1
n.grad = 0.5
b.grad = 0.5
x1w1x2w2.grad = 0.5
x1w1.grad = 0.5
x2w2.grad = 0.5
 
x1.grad = w1.data*x1w1.grad
w1.grad = x1.data*x1w1.grad

x2.grad = w2.data*x2w2.grad
w2.grad = x2.data*x2w2.grad


# In[16]:


draw_dot(out)


# # FUNCTIONS FOR BACK-PROPAGATION

# In[16]:


class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0
        self._backward = lambda:None
        
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other),"+")
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self,other):
       return self*other
        
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),"*")
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad = out.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad = (1-t**2) * out.grad
        out._backward = _backward
        return out

a = Value(2.00,label='a')
b = Value(-3.00,label='b')
c = Value(10.00,label='c')
e = a*b
e.label = 'e'
d = e+c
d.label = 'd'
f = Value(-2.0,label='f')
L = d*f
L.label = 'L'


# In[17]:


out.grad = 1
out._backward()
n._backward()
b._backward()
x1w1x2w2._backward()
x1w1._backward()
x2w2._backward()

draw_dot(out)


# # Topological Order for BackPropagation

# ### We are calling the _backward() mehod manually for each nodes. Let's go with The Topo.

# In[18]:


out.grad = 1
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
build_topo(out)
for node in reversed(topo):
    node._backward()
draw_dot(out)


# In[55]:


class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0
        self._backward = lambda:None
        
    def __repr__(self):
        return f"Value(data={self.data})"
    def __radd__(self,other):
        return self+other
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other),"+")
        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __sub__(self,other):
        return self+(-other)
    
    def __rmul__(self,other):
       return self*other
        
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),"*")
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out
    
    def __truediv__(self,other):
        return self*other**-1
    
    def __pow__(self,other):
        assert isinstance(other,(int,float)), "Only supports int/float powers for now"
        out = Value(self.data**other,(self,),f'**{other}')
        def _backward():
            self.grad = other*self.grad**(other-1)*out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad = out.data * out.grad
        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,(self,),'tanh')
        def _backward():
            self.grad = (1-t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()

a = Value(2.00,label='a')
b = Value(-3.00,label='b')
c = Value(10.00,label='c')
e = a*b
e.label = 'e'
d = e+c
d.label = 'd'
f = Value(-2.0,label='f')
L = d*f
L.label = 'L'


# In[21]:


# Inputs
x1 = Value(2.0,label='x1')
x2 = Value(0.0,label='x2')

# Weights
w1 = Value(-3.0,label='w1')
w2 = Value(1.0,label='w2')
 
# Bias
b = Value(6.88137358,label='b') 

# The Weighted Sum
x1w1 = x1*w1
x1w1.label = 'x1w1'
x2w2 = x2*w2
x2w2.label = 'x2w2'
x1w1x2w2 = x1w1 + x2w2
x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b
n.label = 'n'
out = n.tanh()
out.label='out'
out.backward()
draw_dot(out)


# # SAME WITH PYTORCH

# ![ab6765630000ba8a2b9320406e01747de0853e7d.jpg](attachment:ab6765630000ba8a2b9320406e01747de0853e7d.jpg)

# In[22]:


import torch
x1 = torch.Tensor([2.0]).double(); x1.requires_grad = True
x2 = torch.Tensor([0.0]).double(); x2.requires_grad = True
w1 = torch.Tensor([-3.0]).double(); w1.requires_grad = True
w2 = torch.Tensor([1.0]).double(); w2.requires_grad = True
b = torch.Tensor([6.88137358]).double(); b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = torch.tanh(n)
print(o.data.item())
o.backward()
print(x1.grad.item())
print(x2.grad.item())
print(w1.grad.item())
print(w2.grad.item())


# In[72]:


torch.Tensor([2.0]).dtype


# # THE NEURAL NETWORK

# In[70]:


class Neuron:
    def __init__(self,nin):
        self.w = [ Value(random.uniform(-1,1)) for _ in range(nin) ]
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self,x):
        wsum = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out = wsum.tanh()
        return out

class Layer:
    def __init__(self,nin,nout):
        self.neurons = [ Neuron(nin) for _ in range(nout)  ]
    
    def __call__(self,x):
        outs = [ n(x) for n in self.neurons ]
        return outs
    
class MLP:
    def __init__(self,nin,nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
        
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

x = [ 2.0, 3.0 ]
n = Neuron(2) # CREATES TWO NEURONS
print(n(x)) # COMPUTES THE ACTIVATION AFTER FINDING THE WEIGHTED SUM
draw_dot(n(x))
l = Layer(2,3)  # COMPUTES THE ACTIVATIONS AFTER FINDING THE WEIGHTED SUM
print(l(x))
mlp = MLP(3,[4,4,1]) # THIS IS THE SO-CALLED MLP

xs = [
     [2.0,3.0,-1.0],
     [3.0,-1.0,0.5],
     [0.5,1.0,1.0],
     [1.0,1.0,-1.0]
]
ys = [1.0,-1.0,-1.0,1.0]
ypred = [n(x) for x in xs]
loss = sum([(yout-ygt)**2 for ygt,yout in zip(ys,ypred)])
loss.backward()
mlp.layers[0].neurons[0].w[0]
draw_dot(loss)


# In[71]:


mlp.layers[0].neurons[0].w[1].grad

