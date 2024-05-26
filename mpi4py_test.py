from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD       #必要的一行
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 1:                                                          
    data = np.arange(10, dtype='i')                                    
    comm.Send([data, MPI.INT], dest=0, tag=11)                         
    print("process {} Send buffer-like array {}".format(rank, data))
else:                                                                  
    data = np.empty(10, dtype='i')                                     
    comm.Recv([data, MPI.INT], source=1, tag=11)                       
    print("process {} recv buffer-like array {}".format(rank, data))


###一樣要注意順序問題！可以改（1)rank == 1 (2)dest=0 (3)source=1 看看結果！