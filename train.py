import data_processor as dp
import nets

dataset_path = 'cache/datasets_processed/data_fma_small_mfcc'
num_epochs = 200
batch_size = 64
num_partitions = 6
avg_over_num_partitions = 6

partitions = dp.load_created_partitions(dataset_path)
if len(partitions) == 0:
    raise Exception('No partitions found')

ae = nets.AutoEncoder(None, (20, 90))
ae.train_with_generator(num_epochs, batch_size, )