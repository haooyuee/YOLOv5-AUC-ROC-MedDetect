# YOLOv5-AUROC-MedDetect
 Implement auroc metric on yolov5

```
YOLOv5-AUROC-MedDetect
├─ LICENSE
├─ README.md
└─ yolov5-auc
   ├─ .dockerignore
   ├─ .pre-commit-config.yaml
   ├─ benchmarks.py
   ├─ CITATION.cff
   ├─ classify
   │  ├─ predict.py
   │  ├─ train.py
   │  ├─ tutorial.ipynb
   │  └─ val.py
   ├─ CONTRIBUTING.md
   ├─ data
   │  ├─ Argoverse.yaml
   │  ├─ coco.yaml
   │  ├─ coco128-seg.yaml
   │  ├─ coco128.yaml
   │  ├─ GlobalWheat2020.yaml
   │  ├─ hyps
   │  │  ├─ hyp.no-augmentation.yaml
   │  │  ├─ hyp.Objects365.yaml
   │  │  ├─ hyp.scratch-high.yaml
   │  │  ├─ hyp.scratch-low.yaml
   │  │  ├─ hyp.scratch-med.yaml
   │  │  └─ hyp.VOC.yaml
   │  ├─ ImageNet.yaml
   │  ├─ images
   │  │  ├─ bus.jpg
   │  │  └─ zidane.jpg
   │  ├─ Objects365.yaml
   │  ├─ scripts
   │  │  ├─ download_weights.sh
   │  │  ├─ get_coco.sh
   │  │  ├─ get_coco128.sh
   │  │  └─ get_imagenet.sh
   │  ├─ SKU-110K.yaml
   │  ├─ VisDrone.yaml
   │  ├─ VOC.yaml
   │  └─ xView.yaml
   ├─ detect.py
   ├─ export.py
   ├─ hubconf.py
   ├─ LICENSE
   ├─ models
   │  ├─ common.py
   │  ├─ experimental.py
   │  ├─ hub
   │  │  ├─ anchors.yaml
   │  │  ├─ yolov3-spp.yaml
   │  │  ├─ yolov3-tiny.yaml
   │  │  ├─ yolov3.yaml
   │  │  ├─ yolov5-bifpn.yaml
   │  │  ├─ yolov5-fpn.yaml
   │  │  ├─ yolov5-p2.yaml
   │  │  ├─ yolov5-p34.yaml
   │  │  ├─ yolov5-p6.yaml
   │  │  ├─ yolov5-p7.yaml
   │  │  ├─ yolov5-panet.yaml
   │  │  ├─ yolov5l6.yaml
   │  │  ├─ yolov5m6.yaml
   │  │  ├─ yolov5n6.yaml
   │  │  ├─ yolov5s-ghost.yaml
   │  │  ├─ yolov5s-LeakyReLU.yaml
   │  │  ├─ yolov5s-transformer.yaml
   │  │  ├─ yolov5s6.yaml
   │  │  └─ yolov5x6.yaml
   │  ├─ segment
   │  │  ├─ yolov5l-seg.yaml
   │  │  ├─ yolov5m-seg.yaml
   │  │  ├─ yolov5n-seg.yaml
   │  │  ├─ yolov5s-seg.yaml
   │  │  └─ yolov5x-seg.yaml
   │  ├─ tf.py
   │  ├─ yolo.py
   │  ├─ yolov5l.yaml
   │  ├─ yolov5m.yaml
   │  ├─ yolov5n.yaml
   │  ├─ yolov5s.yaml
   │  ├─ yolov5x.yaml
   │  └─ __init__.py
   ├─ README.md
   ├─ README.zh-CN.md
   ├─ requirements.txt
   ├─ segment
   │  ├─ predict.py
   │  ├─ train.py
   │  ├─ tutorial.ipynb
   │  └─ val.py
   ├─ setup.cfg
   ├─ train.py
   ├─ tutorial.ipynb
   ├─ utils
   │  ├─ activations.py
   │  ├─ augmentations.py
   │  ├─ autoanchor.py
   │  ├─ autobatch.py
   │  ├─ aws
   │  │  ├─ mime.sh
   │  │  ├─ resume.py
   │  │  ├─ userdata.sh
   │  │  └─ __init__.py
   │  ├─ callbacks.py
   │  ├─ dataloaders.py
   │  ├─ docker
   │  │  ├─ Dockerfile
   │  │  ├─ Dockerfile-arm64
   │  │  └─ Dockerfile-cpu
   │  ├─ downloads.py
   │  ├─ flask_rest_api
   │  │  ├─ example_request.py
   │  │  ├─ README.md
   │  │  └─ restapi.py
   │  ├─ general.py
   │  ├─ google_app_engine
   │  │  ├─ additional_requirements.txt
   │  │  ├─ app.yaml
   │  │  └─ Dockerfile
   │  ├─ loggers
   │  │  ├─ clearml
   │  │  │  ├─ clearml_utils.py
   │  │  │  ├─ hpo.py
   │  │  │  ├─ README.md
   │  │  │  └─ __init__.py
   │  │  ├─ comet
   │  │  │  ├─ comet_utils.py
   │  │  │  ├─ hpo.py
   │  │  │  ├─ optimizer_config.json
   │  │  │  ├─ README.md
   │  │  │  └─ __init__.py
   │  │  ├─ wandb
   │  │  │  ├─ wandb_utils.py
   │  │  │  └─ __init__.py
   │  │  └─ __init__.py
   │  ├─ loss.py
   │  ├─ metrics.py
   │  ├─ plots.py
   │  ├─ segment
   │  │  ├─ augmentations.py
   │  │  ├─ dataloaders.py
   │  │  ├─ general.py
   │  │  ├─ loss.py
   │  │  ├─ metrics.py
   │  │  ├─ plots.py
   │  │  └─ __init__.py
   │  ├─ torch_utils.py
   │  ├─ triton.py
   │  └─ __init__.py
   └─ val.py

```