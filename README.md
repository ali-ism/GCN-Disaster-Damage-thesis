# thesis

To read experiment settings:
```
with open('exp_setting.json', 'r') as JSON:
    settings_dict = json.load(JSON)
```

# TODO
- try downloading Benson ResNet50 weights
- resnet weights are now provided more directly, but need to implement a script to download them if not present.
- still not moving anything to(device), keep that in mind
- continue working on iid_dataset.py