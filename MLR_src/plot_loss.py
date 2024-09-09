import matplotlib.pyplot as plt
from torch import load, save
import sys

if len(sys.argv[1:]) != 0:
    d = int(sys.argv[1:][0])
else:
    d=1

# plot loss data from training
loss_data = load(f'mvae_loss_data_recurr{d}.pt')
'''x = len(loss_data['retinal_train'])
loss_data['retinal_train']= loss_data['retinal_train'][61:x-45]
loss_data['retinal_test']= loss_data['retinal_test'][61:x-45]
loss_data['cropped_train'] = loss_data['cropped_train'][:x-45]
loss_data['cropped_test'] = loss_data['cropped_test'][:x-45]'''

plt.plot(loss_data['retinal_train'][101:], label='Retinal Training Error')
plt.plot(loss_data['retinal_test'][101:], label='Retinal Test Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Training')
plt.legend()
plt.title('Retinal Loss Over Epochs of Training')
plt.savefig('plot_loss_retinal.png') # plt.show()#

plt.close()
plt.plot(loss_data['cropped_train'], label='Cropped Training Error')
plt.plot(loss_data['cropped_test'], label='Cropped Test Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Training')
plt.legend()
plt.title('Cropped Loss Over Epochs of Training')
plt.savefig('plot_loss_cropped.png')

#loss_data = save(loss_data,'mvae_loss_data_recurr.pt')
