import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import time
from datamodule import DataModuleClass
from modelmodule import model  
import sys

try:
    path = sys.argv[1]
    gru = int(sys.argv[2])

except:
    raise SystemExit(f"Usage: {sys.argv[0]} <path_to_fasta> <gru>")

kfold = 5
batch_size = 32
max_epoch = 100
lr = 0.001
p = 3

if __name__ == "__main__":
    s = time.process_time()
    data = DataModuleClass(directory=path, batch_size=batch_size, kfold=kfold)
    f = time.process_time()
    d = f - s
    for k in range(kfold):
        start = time.process_time() 
        clf = model(hidden_size=32, learning_rate=lr, gru=gru)
        data.set_current_id(k)
        checkpoint_callback = ModelCheckpoint(filename='{epoch:02d}-{Precision:.2f}-{Recall:.2f}-{MCC:.2f}', monitor='MCC', save_top_k=1, mode='max') 
        earlyStopping_callback = EarlyStopping(monitor='val_loss', patience=p, mode='min')   
        trainer = pl.Trainer(callbacks=[earlyStopping_callback, checkpoint_callback], max_epochs=max_epoch, num_nodes=1, gpus=1) 
        trainer.fit(clf, data)
        stop = time.process_time()
        diff = stop - start
        version = trainer.logger.version
        with open('Training Times.txt', 'a') as f:
            print(f'Training time for version {version} and fold {data.id} in seconds was:', "{:.0f}".format(diff+d), file=f, sep='\n')