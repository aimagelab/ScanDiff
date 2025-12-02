import os
import numpy as np
from tqdm import tqdm
from src.utils.pylogger import RankedLogger
import torch
import torch.nn as nn
from PIL import Image
import pickle
from src.utils.train_utils import get_device
from typing import List
from src.gazetools.metrics.run_metrics import run_metrics, run_density
from src.gazetools.display import save_image_scanpaths
from pathlib import Path
from src.utils.test_utils import process_data_for_metrics_computation
from src.data.unified_datamodule import UnifiedDataModule
import json
from pathlib import Path

log = RankedLogger(__name__, rank_zero_only=True)

class Evaluator:
    def __init__(
        self,
        eval_root_path: str,
        metrics_to_compute: List[str],
        data_to_extract: List[str],
        datamodule: UnifiedDataModule,
        limit_test_batches: float = 1.0, # 1 means using the 100% of the test set
    ) -> None:
        self.eval_root_path = eval_root_path
        self.datamodule = datamodule
        self.metrics_to_compute = metrics_to_compute
        self.data_to_extract = data_to_extract
        self.device = get_device()
        self.limit_test_batches = limit_test_batches

    def test(self, model, diffusion, epoch, is_validation=False):
        preds, target = None, None
        metrics = None
        
        if 'preds' in self.data_to_extract:
            preds, target = self.extract_predictions(model, diffusion, epoch, is_validation)
            
        if 'qualitatives' in self.data_to_extract:
            self.save_qualitatives(epoch, preds)
            
        if 'metrics' in self.data_to_extract:
            metrics = self.compute_metrics(epoch, preds, target, is_validation)
        
        return metrics

    def extract_predictions(self, model, diffusion, epoch, is_validation=False):
        log.info(f"Starting evaluation at epoch {epoch}...")

        model.to(self.device)
        model.eval()

        with torch.no_grad():
            self.datamodule.setup()
            
            if is_validation:
                data_loader = self.datamodule.val_dataloader()
            else:
                data_loader = self.datamodule.test_dataloader()

            target = {}
            preds = {}

            for curr_step, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                if curr_step >= len(data_loader) * self.limit_test_batches: #limit the number of loops during debugging
                    break
                
                img_filename = batch["img_filename"][0]
                original_img_size = batch["original_img_size"][0]
                gt_scanpaths = batch["scanpath"].to(self.device)
                dataset_name = batch["dataset"][0]
                
                key = None
                question_id = ""
                if dataset_name == "aird":
                    question_id = batch["question_id"][0]
                    key = (img_filename, question_id)
                elif dataset_name in ['cocosearch18_tp', 'cocosearch18_ta']:
                    key = (img_filename, batch['task'][0])
                else: # freeviewing dataset (e.g osie)
                    key = img_filename
                

                if key not in target:
                    target[key] = {}
                    target[key]["size"] = original_img_size
                    target[key]['dataset'] = dataset_name
                    target[key]["scanpaths"] = gt_scanpaths
                    target[key]["question_id"] = question_id
                else:
                    target[key]["scanpaths"] = torch.cat(
                        (target[key]["scanpaths"], gt_scanpaths), dim=0
                    )
                    target[key]["size"] = original_img_size
                    target[key]['dataset'] = dataset_name
                    target[key]["question_id"] = question_id

                if key in preds:
                    continue

                num_viewers = batch["num_viewers"][0]
                img_condition = batch["img"].repeat(num_viewers, 1, 1).to(self.device)
                
                if model.task in ["visual_search", "unified_model"]:
                    task_embedding = batch['task_embedding'].to(self.device)
                    task_embedding = task_embedding.repeat(num_viewers, 1)
                else:
                    task_embedding = None

                # this generates "num_viewers" (e.g. 10) different initial noise samples
                max_len = gt_scanpaths.shape[1]

                initial_noise = torch.randn(num_viewers, max_len, model.scanpath_emb_size).to(self.device)
                
                y = img_condition
                model_kwargs = dict(y=y, task_embedding=task_embedding)  # img conditioning

                samples = diffusion.p_sample_loop(
                    model,
                    initial_noise.shape,
                    initial_noise,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=self.device,
                )
                
                pred_scanpath = model.get_coords_and_time(samples)
                
                # predict padding ad different length for scanpaths
                token_validity_preds = model.token_validity_predictor(samples)
                token_validity_preds = nn.Softmax(dim=-1)(token_validity_preds)
                token_validity_preds = token_validity_preds.argmax(
                    dim=-1
                )  # NB: 1 means that the fixation is valid, 0 otherwise

                scanpath_lengths = torch.cumprod(token_validity_preds, dim=-1).sum(-1)
                mask = (
                    (
                        torch.arange(samples.shape[1]).to("cuda").unsqueeze(0)
                        >= scanpath_lengths.unsqueeze(1)
                    )
                    .unsqueeze(2)
                    .repeat(1, 1, 3)
                )  # 3 is x,y,t

                # pad scanpaths after the predicted length
                padded_scanpaths = pred_scanpath
                padded_scanpaths[mask] = self.datamodule.test_collators[0].PAD[
                    0
                ]  # padding value

                if key not in preds:
                    preds[key] = {}

                preds[key]["size"] = original_img_size
                preds[key]["scanpaths"] = padded_scanpaths
                preds[key]["question_id"] = question_id
                preds[key]['dataset'] = dataset_name
            

            self.save_predictions_and_gt_scanpaths(epoch, preds, target, is_validation)
                
            return preds, target

    def save_predictions_and_gt_scanpaths(self, epoch, preds, target, is_validation=False):
        log.info(f"Saving predictions in {self.eval_root_path}/generations...")

        os.makedirs(Path(self.eval_root_path, f'generations_epoch_{epoch}'), exist_ok=True)
        
        if is_validation:
            dataset_name = self.datamodule.val_datasets[0].__class__.__name__ + '_validation'
        else:
            dataset_name = self.datamodule.test_datasets[0].__class__.__name__ + '_test'
            
        with open(
            Path(
                self.eval_root_path, f'generations_epoch_{epoch}',
                f"generations_{dataset_name}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(preds, f)

        with open(
            Path(
                self.eval_root_path, f'generations_epoch_{epoch}',
                f"original_{dataset_name}.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(target, f)
        # NB: PREDS AND TARGETS SHOULD HAVE THE SAME KEYS AND THE SAME LENGTH OF EACH KEY.

    def save_qualitatives(self, epoch=None, preds=None, is_validation=False):        
        if is_validation:
            dataset_name = self.datamodule.val_datasets[0].__class__.__name__ + '_validation'
        else:
            dataset_name = self.datamodule.test_datasets[0].__class__.__name__ + '_test'
        
        log.info(
            f"Saving qualitatives in {self.eval_root_path}/qualitatives for the dataset {dataset_name}..."
        )

        if preds is None:
            try:
                with open(
                    Path(
                        self.eval_root_path, f'generations_epoch_{epoch}',
                        f"generations_{dataset_name}.pkl",
                    ),
                    "rb",
                ) as f:
                    preds = pickle.load(f)
            except FileNotFoundError:
                log.error("No predictions found.")
                return

        root_path = Path(
            self.eval_root_path, f'qualitatives',
            dataset_name,
        )
        os.makedirs(root_path, exist_ok=True)
        for img_filename, data in preds.items():
            if isinstance(img_filename, tuple):
                img_name = img_filename[0].rsplit(".", 1)[0]
                task = img_filename[1]
                dir_name = img_name + '_' + task
            else:
                dir_name = img_filename.rsplit(".", 1)[0]
            os.makedirs(Path(root_path, dir_name), exist_ok=True)

            for idx, s in enumerate(data["scanpaths"]):
                
                if data['dataset'] == 'cocosearch18_tp':
                    suffix = '_tp'
                elif data['dataset'] == 'cocosearch18_ta':
                    suffix = '_ta'
                else:
                    suffix = ''
                    
                img_root_path = Path(
                    self.datamodule.test_datasets[0].root_path, "images" + suffix
                )
                
                if isinstance(img_filename, tuple):
                    pil_img = Image.open(Path(img_root_path, task, img_filename[0]))
                else:
                    pil_img = Image.open(Path(img_root_path, img_filename))

                # convert coords from 512,320 to original size
                original_width, original_height = pil_img.size

                s_length = torch.ones(s.shape[0])
                s_length[s[:, 0] == self.datamodule.test_collators[0].PAD[0]]  = 0
                s_length = int(s_length.sum().item())

                s = s.cpu().numpy()
                x = s[:, 0] * original_width
                y = s[:, 1] * original_height
                
                t = [] # this competitor does not predict the time (e.g. dg3)
                
                if s.shape[-1] > 2:
                    # qualitatives and metrics are computed considering milliseconds
                    if self.datamodule.time_in_ms: # time is already in ms
                        t = s[:, 2]
                    else: #time is in seconds, then put it in milliseconds
                        t = s[:, 2] * 1000
                    
                    t=t[:s_length]
                
                save_image_scanpaths(
                    pil_img,
                    x=x[:s_length],
                    y=y[:s_length],
                    t=t,
                    save_path=Path(
                        root_path, dir_name, f"subject_{idx}.jpg"
                    ),
                )

    def compute_metrics(self, epoch, pred_scanpaths=None, gt_scanpaths=None, is_validation=False):
        log.info("Running scanpath metrics on ...")
        
        print('****** the epoch is********')
        print(epoch)
        print(self.eval_root_path)
        
        if is_validation:
            dataset_name = self.datamodule.val_datasets[0].__class__.__name__ + '_validation'
        else:
            dataset_name = self.datamodule.test_datasets[0].__class__.__name__ + '_test'
        
        
        if pred_scanpaths is None or gt_scanpaths is None:
            log.error("No predictions or ground truth scanpaths provided.")
            log.info(
                "Checking if predictions and ground truth scanpaths have been previously saved..."
            )
            if not Path(self.eval_root_path, f'generations_epoch_{epoch}').exists():
                log.error("No saved predictions found.")
                exit(1)
            else:
                with open(
                    Path(
                        self.eval_root_path, f'generations_epoch_{epoch}',
                        f"generations_{dataset_name}.pkl",
                    ),
                    "rb",
                ) as f:
                    pred_scanpaths = pickle.load(f)

                with open(
                    Path(
                        self.eval_root_path, f'generations_epoch_{epoch}',
                        f"original_{dataset_name}.pkl",
                    ),
                    "rb",
                ) as f:
                    gt_scanpaths = pickle.load(f)
                    
        gt_scanpaths = process_data_for_metrics_computation(
           self.datamodule, gt_scanpaths, is_original=True,
        )
        pred_scanpaths = process_data_for_metrics_computation(
           self.datamodule, pred_scanpaths, is_original=False,
        )

        if 'sequence_score' in self.metrics_to_compute or 'sequence_score_time' in self.metrics_to_compute or 'diversity_sequence_score' in self.metrics_to_compute:
            #load cluster data for the specific dataset
            
            if is_validation:
                dataset_name_no_suffix = self.datamodule.val_datasets[0].name
                root_dataset_location = self.datamodule.val_datasets[0].root_path
            else:
                dataset_name_no_suffix = self.datamodule.test_datasets[0].name
                root_dataset_location = self.datamodule.test_datasets[0].root_path
            ss_clusters = np.load(Path(root_dataset_location, f'clusters_{dataset_name_no_suffix}_512_384.npy'), allow_pickle=True).item()
        else:
            ss_clusters = None
        
        metrics, DSS_per_img, DSS_per_img_no_dur, covered_human_scanpaths_no_dur, covered_human_scanpaths, all_human_vs_human, all_human_vs_model, all_model_vs_model = run_metrics(gt_scanpaths, pred_scanpaths, self.datamodule, self.metrics_to_compute, ss_clusters, is_validation=is_validation)
        
        if 'distributions' in self.metrics_to_compute:
            log.info('Plotting distributions...')
            os.makedirs(Path(self.eval_root_path, 'distrib_plots'), exist_ok=True)
            
            real = {}
            pred = {}
            for key in gt_scanpaths.keys():
                real[key] = gt_scanpaths[key]['scanpaths']
                pred[key] = pred_scanpaths[key]['scanpaths']
                
            run_density(real, pred, Path(self.eval_root_path, 'distrib_plots'))
        
        log.info("Saving metrics to file...")
        metrics_dict = metrics.to_dict()
        
        os.makedirs(Path(self.eval_root_path, f'metrics_epoch_{epoch}'), exist_ok=True)

        if DSS_per_img:
            np.save(
                Path(self.eval_root_path, f"DSS_{dataset_name}.npy"),
                DSS_per_img,
            )
        
        if DSS_per_img_no_dur:
            np.save(
                Path(self.eval_root_path, f"DSS_no_dur_{dataset_name}.npy"),
                DSS_per_img_no_dur,
            )
        
        if covered_human_scanpaths:
            with open(Path(self.eval_root_path, f"covered_human_scanpaths_{dataset_name}.pkl"), "wb") as f:
                pickle.dump(covered_human_scanpaths, f)
        
        if covered_human_scanpaths_no_dur:
            with open(Path(self.eval_root_path, f"covered_human_scanpaths_no_dur_{dataset_name}.pkl"), "wb") as f:
                pickle.dump(covered_human_scanpaths_no_dur, f)
        
        if all_human_vs_human:
            np.save(
                Path(self.eval_root_path, f"all_human_vs_human_{dataset_name}.npy"),
                all_human_vs_human,
            )
        
        if all_human_vs_model:
            np.save(
                Path(self.eval_root_path, f"all_human_vs_model_{dataset_name}.npy"),
                all_human_vs_model,
            )
        
        if all_model_vs_model:
            np.save(
                Path(self.eval_root_path, f"all_model_vs_model_{dataset_name}.npy"),
                all_model_vs_model,
            )
        
        with open(
            Path(
                self.eval_root_path, f'metrics_epoch_{epoch}',
                f"metrics_{dataset_name}.json",
            ),
            "w",
        ) as f:
            json.dump(metrics_dict, f, indent=4)
            
        # adjust metrics dict for logging
        appendix = 'test' if not is_validation else 'validation'
        
        logging_dict = {}
        for key, value in metrics_dict.items():
            logging_dict[f'{appendix}/{key}_Model'] = metrics_dict[key]['Model']
            if 'KLD' in metrics_dict[key]:
                logging_dict[f'{appendix}/{key}_KLD'] = metrics_dict[key]['KLD']
            
        return logging_dict
