
# learning 0.5
CUDA_VISIBLE_DEVICES=0,1 python3 /root/DL-project/1.py --lr 0.5 --name CNN_0.5

# learning 0.1
CUDA_VISIBLE_DEVICES=0,1 python3 /root/DL-project/1.py --lr 0.1 --name CNN_0.1

# learning 0.05
CUDA_VISIBLE_DEVICES=0,1 python3 /root/DL-project/1.py --lr 0.05 --name CNN_0.05

# learning 0.01
CUDA_VISIBLE_DEVICES=0,1 python3 /root/DL-project/1.py --lr 0.01 --name CNN_0.01

# learning 0.005
CUDA_VISIBLE_DEVICES=0,1 python3 /root/DL-project/1.py --lr 0.005 --name CNN_0.005

# learning 0.001
CUDA_VISIBLE_DEVICES=0,1 python3 /root/DL-project/1.py --lr 0.0001 --name CNN_0.0001
