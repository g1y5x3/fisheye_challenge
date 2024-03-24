"""
For distributed training, run the following example command. 
Making sure --nproc_per_node and -devices has the same param.

python -m torch.distributed.run --nproc_per_node 2 yolov8x_train_fisheye.py -devices 2 -epoch 1 -bs 32

"""
import time, copy, torch, json, wandb, warnings, argparse
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.nn.tasks import DetectionModel
from utils import get_image_id

# libraries that are monkey patched
import ultralytics.nn.tasks as tasks
import ultralytics.utils.torch_utils as torch_utils
from ultralytics.engine.trainer import BaseTrainer
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
from ultralytics.utils import LOGGER, ops, RANK
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

        # Evaluate
        if nl:

            #print("predn")
            #print(predn.shape)
            #print(predn)
            #predn_dis = (predn[:,0]-0.5)**2 + (predn[:,1]-0.5)**2
            #print(predn_dis)
            #print(f"edge {torch.sum(predn_dis>0.125)}")
            #print(f"center {torch.sum(predn_dis<0.125)}")
            #print("bbox")
            #print(bbox.shape)
            #print("cls")
            #print(cls.shape)

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

        print(f"loss {self.loss_center}")

        self.loss_morning = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.loss_afternoon = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.loss_evening = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.loss_night = torch.zeros_like(trainer.loss_items, device=trainer.device)

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

        # print("\n")
        # print("pred")
        # print(len(preds))
        # print(preds[0].shape)
        # print(len(preds[1]))
        # print(preds[1][0].shape)
        # print(preds[1][1].shape)
        # print(preds[1][2].shape)

        with dt[2]:
            if self.training:
                self.loss += model.loss(batch, preds)[1]

        # Postprocess
        with dt[3]:
            preds = self.postprocess(preds)

        #print("===== after postprocess  =====")
        #print("pred")
        #print(len(preds))
        #print(preds[0].shape)
        #print(len(preds[1]))
        #print(preds[1][0].shape)
        #print(preds[1][1].shape)
        #print(preds[1][2].shape)

        #print("batch")
        #print(len(batch["im_file"]))
        #print(batch["bboxes"].shape)
        #dis = (batch["bboxes"][:,0]-0.5)**2 + (batch["bboxes"][:,1]-0.5)**2
        #print(dis.shape)
        #print(f"edge {torch.sum(dis>0.125)}")
        #print(f"center {torch.sum(dis<0.125)}")

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

def _do_train_new(self, world_size=1):
    """Train completed, evaluate and plot if specified by arguments."""
    if world_size > 1:
        self._setup_ddp(world_size)
    self._setup_train(world_size)

    nb = len(self.train_loader)  # number of batches
    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
    last_opt_step = -1
    self.epoch_time = None
    self.epoch_time_start = time.time()
    self.train_time_start = time.time()
    self.run_callbacks("on_train_start")
    LOGGER.info(
        f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
        f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
        f"Logging results to {colorstr('bold', self.save_dir)}\n"
        f'Starting training for ' + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
    )
    if self.args.close_mosaic:
        base_idx = (self.epochs - self.args.close_mosaic) * nb
        self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    epoch = self.start_epoch
    while True:
        self.epoch = epoch
        self.run_callbacks("on_train_epoch_start")
        self.model.train()
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        # Update dataloader attributes (optional)
        if epoch == (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()
            self.train_loader.reset()

        if RANK in (-1, 0):
            LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(self.train_loader), total=nb)
        self.tloss = None
        self.optimizer.zero_grad()
        for i, batch in pbar:
            self.run_callbacks("on_train_batch_start")
            # Warmup
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]  # x interp
                self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                for j, x in enumerate(self.optimizer.param_groups):
                    # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(
                        ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                    )
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

            # Forward
            with torch.cuda.amp.autocast(self.amp):

                dis = (batch["bboxes"][:,0]-0.5)**2 + (batch["bboxes"][:,1]-0.5)**2

                batch = self.preprocess_batch(batch)
                self.loss, self.loss_items = self.model(batch)
                if RANK != -1:
                    self.loss *= world_size
                self.tloss = (
                    (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                )

                weight = (torch.sum(dis>0.125)*4 + torch.sum(dis<0.125))/dis.shape[0]
                self.loss *= weight

            # Backward
            self.scaler.scale(self.loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= self.accumulate:
                self.optimizer_step()
                last_opt_step = ni

                # Timed stopping
                if self.args.time:
                    self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                    if RANK != -1:  # if DDP training
                        broadcast_list = [self.stop if RANK == 0 else None]
                        dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                        self.stop = broadcast_list[0]
                    if self.stop:  # training time exceeded
                        break

            # Log
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
            loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
            losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
            if RANK in (-1, 0):
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                    % (f"{epoch + 1}/{self.epochs}", mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1])
                )
                self.run_callbacks("on_batch_end")
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)

            self.run_callbacks("on_train_batch_end")

        self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
        self.run_callbacks("on_train_epoch_end")
        if RANK in (-1, 0):
            final_epoch = epoch + 1 == self.epochs
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self.metrics, self.fitness = self.validate()
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            if self.args.time:
                self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

            # Save model
            if self.args.save or final_epoch:
                self.save_model()
                self.run_callbacks("on_model_save")

        # Scheduler
        t = time.time()
        self.epoch_time = t - self.epoch_time_start
        self.epoch_time_start = t
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.scheduler.step()
        self.run_callbacks("on_fit_epoch_end")
        torch.cuda.empty_cache()  # clear GPU memory at end of epoch, may help reduce CUDA out of memory errors

        # Early Stopping
        if RANK != -1:  # if DDP training
            broadcast_list = [self.stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            self.stop = broadcast_list[0]
        if self.stop:
            break  # must break all DDP ranks
        epoch += 1

    if RANK in (-1, 0):
        # Do final val with best.pt
        LOGGER.info(
            f"\n{epoch - self.start_epoch + 1} epochs completed in "
            f"{(time.time() - self.train_time_start) / 3600:.3f} hours."
        )
        self.final_eval()
        if self.args.plots:
            self.plot_metrics()
        self.run_callbacks("on_train_end")
    torch.cuda.empty_cache()
    self.run_callbacks("teardown")


if __name__ == "__main__":

  # monkey patches
  BaseTrainer._do_train = _do_train_new
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

