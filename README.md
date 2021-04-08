# thesis

To read experiment settings:
```
with open('exp_setting.json', 'r') as JSON:
    settings_dict = json.load(JSON)
```

To initialize GPU and random engines
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

# TODO
- test benson resnet 50 weights
- generate disaster_dict if it doesn't exist [Done]
- build y, use label encoding [Done]
- discard unclassified [Done]
- still not moving anything to(device), keep that in mind
- continue working on iid_dataset.py