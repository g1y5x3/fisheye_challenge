"""
For distributed training, run the following example command. 
Making sure --nproc_per_node and -devices has the same param.

python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32

"""
import copy, torch, json, wandb, argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.nn.tasks import DetectionModel
from utils import get_image_id

# libraries that are monkey patched
import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as torch_utils
from ultralytics.data.augment import Albumentations
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.engine.validator import BaseValidator
from yolov8_monkey_patches import albumentation_init, load_model_custom, get_flops_pass, parse_dcn_model  
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images

def update_metrics_new(self, preds, batch):
    """Metrics."""
    for si, pred in enumerate(preds):
        self.seen += 1
        npr = len(pred)
        stat = dict(
            conf=torch.zeros(0, device=self.device),
            pred_cls=torch.zeros(0, device=self.device),
            tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
        )
        pbatch = self._prepare_batch(si, batch)
        cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
        nl = len(cls)
        stat["target_cls"] = cls
        if npr == 0:
            if nl:
                for k in self.stats.keys():
                    self.stats[k].append(stat[k])
                if self.args.plots:
                    self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
            continue

        # Predictions
        if self.args.single_cls:
            pred[:, 5] = 0
        predn = self._prepare_pred(pred, pbatch)
        stat["conf"] = predn[:, 4]
        stat["pred_cls"] = predn[:, 5]

        print("FROM MONKEY PATCH")
        print("predn")
        print(predn.shape)
        print("bbox")
        print(bbox.shape)
        print("cls")
        print(cls.shape)
        print("FROM MONKEY PATCH")

        # Evaluate
        if nl:
            stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
        for k in self.stats.keys():
            self.stats[k].append(stat[k])

        # Save
        if self.args.save_json:
            self.pred_to_json(predn, batch["im_file"][si])
        if self.args.save_txt:
            file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
            self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

def init_metrics_new(self, model):
    """Initialize evaluation metrics for YOLO."""
    val = self.data.get(self.args.split, "")  # validation path
    self.is_coco = isinstance(val, str) and "coco" in val and val.endswith(f"{os.sep}val2017.txt")  # is COCO
    self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
    self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
    self.names = model.names
    self.nc = len(model.names)
    self.metrics.names = self.names
    self.metrics.plot = self.args.plots

    # add more metrics tracking
    self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
    self.confusion_matrix_center = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
    self.confusion_matrix_edge = ConfusionMatrix(nc=self.nc, conf=self.args.conf)

    self.seen = 0
    self.jdict = []

    # add more metrics tracking
    self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
    self.stats_center = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
    self.stats_edge = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

def new_call(self, trainer=None, model=None):
    """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
    gets priority).
    """
    self.training = trainer is not None
    augment = self.args.augment and (not self.training)
    if self.training:
        self.device = trainer.device
        self.data = trainer.data
        self.args.half = self.device.type != "cpu"  # force FP16 val during training
        model = trainer.ema.ema or trainer.model
        model = model.half() if self.args.half else model.float()
        # self.model = model
        print("adding loss")
        self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.loss_center = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.loss_edge = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
        model.eval()
    else:
        callbacks.add_integration_callbacks(self)
        model = AutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, self.args.batch),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
        )
        # self.model = model
        self.device = model.device  # update device
        self.args.half = model.fp16  # update half
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        if engine:
            self.args.batch = model.batch_size
        elif not pt and not jit:
            self.args.batch = 1  # export.py models default to batch-size 1
            LOGGER.info(f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

        if str(self.args.data).split(".")[-1] in ("yaml", "yml"):
            self.data = check_det_dataset(self.args.data)
        elif self.args.task == "classify":
            self.data = check_cls_dataset(self.args.data, split=self.args.split)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

        if self.device.type in ("cpu", "mps"):
            self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
        if not pt:
            self.args.rect = False
        self.stride = model.stride  # used in get_dataloader() for padding
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        model.eval()
        model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

    self.run_callbacks("on_val_start")
    dt = (
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
        Profile(device=self.device),
    )
    bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val

    for batch_i, batch in enumerate(bar):
        self.run_callbacks("on_val_batch_start")
        self.batch_i = batch_i
        # Preprocess
        with dt[0]:
            batch = self.preprocess(batch)

        # Inference
        with dt[1]:
            preds = model(batch["img"], augment=augment)

        # Loss
        with dt[2]:
            if self.training:
                self.loss += model.loss(batch, preds)[1]

        # Postprocess
        with dt[3]:
            preds = self.postprocess(preds)

        #print("preds")
        #print(len(preds))
        #print(preds[0].shape)
        #print("batch")
        #print(len(batch))
        #print(batch["bboxes"].shape)

        self.update_metrics(preds, batch)

        if self.args.plots and batch_i < 3:
            self.plot_val_samples(batch, batch_i)
            self.plot_predictions(batch, preds, batch_i)

        self.run_callbacks("on_val_batch_end")
    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
    self.finalize_metrics()
    self.print_results()
    self.run_callbacks("on_val_end")
    if self.training:
        model.float()
        results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
        return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
    else:
        LOGGER.info(
            "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
            % tuple(self.speed.values())
        )
        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / "predictions.json"), "w") as f:
                LOGGER.info(f"Saving {f.name}...")
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats

if __name__ == "__main__":

  # monkey patches
  BaseValidator.__call__ = new_call 
  DetectionValidator.init_metrics = init_metrics_new
  DetectionValidator.update_metrics = update_metrics_new
  Albumentations.__init__ = albumentation_init
  DetectionTrainer.get_model = load_model_custom
  tasks.parse_model = parse_dcn_model
  torch_utils.get_flops = get_flops_pass

  parser = argparse.ArgumentParser(description="yolov8x fisheye experiment")
  parser.add_argument('-devices', type=int, default=1, help="batch size")
  parser.add_argument('-model', type=str, default="yolov8x_dcn.yaml", help="model file name")
  parser.add_argument('-imgsz', type=int, default=1280, help="image size used for training")
  parser.add_argument('-frac', type=float, default=1.0, help="fraction of the data being used")
  parser.add_argument('-epoch', type=int, default=1, help="number of epoch")
  parser.add_argument('-bs', type=int, default=16, help="number of batches")
  parser.add_argument('-wd', type=float, default=0.0005, help="weight decay")
  parser.add_argument('-conf', type=float, default=0.001, help="confidence threshold")
  parser.add_argument('-iou', type=float, default=0.5, help="intersection of union")
  parser.add_argument('-project', type=str, default="fisheye-challenge", help="project name")
  parser.add_argument('-name', type=str, default="yolov8x", help="run name")
  args = parser.parse_args()
  
  device = 0 if args.devices == 1 else [i for i in range(args.devices)]

  train_args = dict(project=args.project, name=args.name, model=args.model, data="fisheye.yaml",
                    device=device, epochs=args.epoch, batch=args.bs, imgsz=args.imgsz, fraction=args.frac,
                    exist_ok=True,
                    conf=args.conf, iou=args.iou,
                    lr0=2e-5, warmup_bias_lr=2e-5/3, weight_decay=args.wd,
                    optimizer="AdamW", seed=0,
                    box=7.5, cls=0.5, dfl=1.5,
                    close_mosaic=10,
                    degrees=0, translate=0.1, scale=0.0, shear=0.0,
                    perspective=0.0, flipud=0.0, fliplr=0.5, 
                    mosaic=1.0, mixup=0.0,
                    deterministic=True, verbose=True,
                    pretrained=True)

  trainer = DetectionTrainer(overrides=train_args)
  #trainer.add_callback("on_val_end", save_eval_json_with_id)
  #trainer.add_callback("on_train_epoch_end", check_dcn_weights_offsets)
  trainer.train()

