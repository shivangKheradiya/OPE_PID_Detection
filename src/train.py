from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

register_coco_instances("pid_train", {}, "datasets/pid_symbols/train.json", "datasets/pid_symbols/train")
register_coco_instances("pid_val", {}, "datasets/pid_symbols/val.json", "datasets/pid_symbols/val")

cfg = get_cfg()
cfg.merge_from_file("configs/pid_config.yaml")
cfg.DATASETS.TRAIN = ("pid_train",)
cfg.DATASETS.TEST = ("pid_val",)
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5   # number of symbol types

trainer = DefaultTrainer(cfg)
trainer.train()