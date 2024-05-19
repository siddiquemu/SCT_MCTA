from multiprocessing import Process, Queue
import torch

def f(q,cam,i):
    q.put([42, None, 'hello'])
    print('cam {} gpu {}'.format(cam,i))

if __name__ == '__main__':
    gpus = torch.cuda.device_count()
    cam=[2,9]
    for i in range(gpus):
        q = Queue()
        p = Process(target=f, args=(q,cam[i],i))
        p.start()
        #print(q.get())    # prints "[42, None, 'hello']"
        p.join()
    for i in range(gpus):
        print(q.get())