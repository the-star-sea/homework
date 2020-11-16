#CS205 C/ C++ Programming – Matrix
Name：张通

SID: 11911611
##Part 1 - Analysis
这次作业是写矩阵乘法，最简单的算法是使用三重循环直接计算

类的声明：column,row,data

优化：

1. 矩阵的转置

   从一级缓存中提取中间结果的速度远大于低级的缓存,而转置后，j不必一段一段的跳着读，cache命中率提高
   1000*1000维矩阵乘法中，转置为3813ms,未转置则为4615ms


2. avx

在1.的基础上，我们进一步使用了avx,一次可以计算8组乘法或加法
 1000*1000维矩阵乘法中，未使用avx为3813ms,使用avx为584ms

3. omp

omp即多线程的运算，与avx不同的是，omp基于cpu的多核心，而avx是基于寄存器，oomp的使用后文有说明，我的电脑是12 核心
在2.基础上，1000*1000维矩阵乘法中，未使用omp为584ms,使用omp为96ms

20000*20000矩阵运算时核心利用率
![image.png](https://i.loli.net/2020/11/15/pF4vOGSC8L2fNrQ.png)

4. blockcache

与1.原理相似，cache不变，矩阵越大，越难装进cache中，高速缓存缺失将缺失更多，
使用分块矩阵能充分利用高速缓存的局部性将矩阵乘法分块实现，优化性能
但在较小矩阵乘法中，分块矩阵会拖慢速度，在我的电脑上较优的blocksize为1024（从32 开始调试）
在3.基础上，1000*1000维矩阵乘法中，未使用block为96ms,使用block 为573ms
在3.基础上，5000*5000维矩阵乘法中，未使用block为14282ms,使用block为 5416ms

5. 编译器优化

我选择了O3，"unroll-loops", "Ofast"和inline关键字
内嵌函数没有了调用的开销，unroll-loops减少判断循环次数
O3针对循环有很多优化，但很危险，很多优化省略了语法检查并使线程出bug可能大大提高
在4.基础上，5000*5000维矩阵乘法中，未使用编译器优化为5416ms,使用编译器优化为 3635ms

6. 变量的传输

尽量使用指针传变量，不使用return，频繁使用的如matrix,直接全局变量声明

总结
  |     |   转置 | avx     | omp    |  blockcache| 编译器优化
  :----: | :----: |  :----: | :----: | :----: | :----:
  优化比|1.210 | 6.529|6.083 |2.637 | 1.490

![image.png](https://i.loli.net/2020/11/15/FLEO1IMJA9BnzNe.png)





##Part 2 - Code
https://github.com/the-star-sea/homework
请使用提供的所有文件
##Part 3 - Result & Verification
1.计算
![image.png](https://i.loli.net/2020/11/14/sD1gybc6EojxzRA.png)
2.测试速度
![image.png](https://i.loli.net/2020/11/14/qSEmpAn7cjLTB8J.png)
![image.png](https://i.loli.net/2020/11/14/E6JMmTP7Rj4eN3H.png)
3.输入错误
![image.png](https://i.loli.net/2020/11/15/MPsf8XqlT5uCbag.png)
##Part 4 - Difficulties & Solutions
1. omp的使用

一开始我是使用thread,不过发现omp比thread优化性能更好，因为能够动态优化，不过
omp并非我最开始所想象的加一个#pragma omp parallel for 就完事，omp中变量有shared
和private之分，使用不当，不同线程间变量彼此影响，且很难debug（时对时错），出于安全的考虑，我最
终选择将复杂的部分写入另一方法中，并私有了一些关键的变量，解决了问题

2. 编译器优化的使用

编译器的优化不能一股脑写上去，而是要逐条检查，如"unroll-loops"，观察是否会冲突
，一些编译选项会拖慢速度甚至出错

3. block caching

block caching 实在让我头疼。开始我只对最外两层用了block,blocksize 为32发现
速度反而慢了20多倍.后来经过调试，发现在我的笔记本上，blocksize=1024时对于大的
数据效果极好，与openblas极为接近

4. avx

avx的编写毫无疑问是照葫芦画瓢，结果我照的葫芦是个错葫芦。百思不得其解后询问了老师
下定决心自己写一个，最后在stackoverflow的帮助下终于写出正确的代码
