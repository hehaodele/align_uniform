from socket import gethostname

hostname = gethostname()

data_folder = '/afs/csail.mit.edu/u/h/hehaodele/radar/Hao/datasets'


if hostname == 'MadBoy':
    data_folder = '/media/hehaodele/AngryBoy/datasets/STL10'

print(hostname)