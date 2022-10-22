import time
from tqdm import tqdm

a=[1,2,3]
pbar = tqdm(enumerate(a))
for batch_idx,i in pbar:
#用法：tqdm(可迭代对象）=会显示进度条的可迭代对象
#所以仍然是可迭代对象，可以使用诸如for i in 【可迭代对象】等遍历形式。
    time.sleep(1)#程序休息一秒钟
    print(i)
