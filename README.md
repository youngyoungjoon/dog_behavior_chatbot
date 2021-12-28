# dog_behavior_chatbot
A chatbot that teaches you about dog behavior questions.


## For dataset, follow the below way

### Folder
```
labeling_tally
    ㄴblepharitis
          ㄴblepharitis_images
          ㄴblepharitis_labelme
          ㄴblepharitis_labelimg
    ㄴcataract
    ㄴCherry_eye
```

## usage

```
train_torch.py

python train_torch.py --gpus 2 --accelerator ddp --train --max_epochs 100  

if your gpu is 2 --gpus 2

'path' = You have to enter the top-level folder.
```
