#命令式编程
def add(a,b):
    return a+b

def fancy_func(a,b,c,d):
    e=add(a,b)
    f=add(c,d)
    g=add(e,f)
    return g

fancy_func(1,2,3,4) #10

#符号式编程
#通常，符号式编程的程序需要下面3个步骤
#1.定义计算流程
#2.把计算流程编译成可执行的程序
#3.给定输入，调用编译好的的程序执行
def add_str():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3, 4))
'''

prog = evoke_str()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
#命令式编程更方便。符号式编程更高效并更容易移植。另一方面，符号式编程可以将程序变成一个与Python无关的格式，从而可以使程序在非Python环境下运行，以避开Python解释器的性能问题。