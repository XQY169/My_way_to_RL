import torch
import torch.nn.functional as F
from torch import optim

a = torch.tensor([1.0], requires_grad=True,dtype=torch.float)
b = torch.tensor([1.0], requires_grad=True,dtype=torch.float)
c = torch.tensor([1.0], requires_grad=True,dtype=torch.float)

y_target1 = torch.tensor([1368.9,1363.6,1370.7,1387.5,1379.5,1374.7,1378.3,1424.8,1370.7,1390.4])
queue1 = torch.tensor([10.99,11.22,11,11.09,10.98,11.14,11.18,10.77,11.22,10.99])
delay1 = torch.tensor([12.02,12,12,11.52,11.78,11.81,11.7,10.86,11.84,11.49])
wait1 = torch.tensor([9.23,9.25,9.17,8.85,9.03,9.05,8.97,8.32,9.11,8.85])

y_target2 = torch.tensor([1998.2,1948.6,1967.1,1905.3,1942.4,1962.6,1945.9,1925.3,1981.4,1926.2])
queue2 = torch.tensor([4.05,4.46,4.15,4.63,4.43,4.37,4.27,4.49,4.24,4.53])
delay2 = torch.tensor([5.47,5.85,5.69,6.22,5.92,5.73,5.91,6.07,5.55,6.05])
wait2 = torch.tensor([4.05,4.29,4.35,4.69,4.38,4.2,4.47,4.54,4.11,4.5])

y_target3 = torch.tensor([1464.9,1470.1,1456.2,1454,1409.6,1425.5,1477.7,1358,1449.9,1466,1438.5,1394.3,1410.7,1437.7,1424.7])
queue3 = torch.tensor([9.59,9.02,9.3,9.72,9.6,9.98,9.41,9.27,9.59,9.89,9.82,9.45,9.19,9.52,9.67])
delay3 = torch.tensor([10.13,10.39,10.46,40.48,11.28,10.78,10.14,12.85,10.66,10.05,10.59,11.92,11.58,10.79,11.03])
wait3 = torch.tensor([8.53,8.66,8.82,8.5,9.56,9.02,8.25,10.73,8.61,8.3,8.86,9.78,9.69,9,9.13])

y_target4 = torch.tensor([1298,1273.5,1290.3,1312.3,1299.9,1354.1,1367.8])
queue4 = torch.tensor([9.36,10.07,8.86,10.38,9.95,8.32,9.02])
delay4 = torch.tensor([14.87,15.02,15.59,14,14.44,14.27,13.27])
wait4 = torch.tensor([11.78,11.97,12.3,10.59,11.29,10.98,10.27])

y_target = torch.cat((y_target1,y_target2,y_target3,y_target4))
queue = torch.cat((queue1,queue2,queue3,queue4))
delay = torch.cat((delay1,delay2,delay3,delay4))
wait = torch.cat((wait1,wait2,wait3,wait4))

optimizer = optim.Adam([a,b,c],lr=0.0001)

# batch_size = 16

for i in range(30000):
    print(i)
    # indices = torch.randperm(y_target.size(0))[0:batch_size]
    y = (1 / (1 + a * delay) + 1 / (1 + b * queue) + 1 / (1 + c * wait))
    loss = F.mse_loss(y, y_target/1000)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(1/a,1/b,1/c)