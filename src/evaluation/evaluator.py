# src/evaluation/evaluator.py
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
import tempfile
import traceback
import uuid

from ..utils.logger import get_logger
from .metrics import calculate_iou, calculate_mask_iou


class Evaluator:
    """Evaluator for Mask R-CNN model."""
    
    def __init__(self, device: torch.device):
        """Initialize the evaluator.
        
        Args:
            device: Device to use for evaluation
        """
        self.device = device
        self.logger = get_logger(name="evaluator")
    
    def evaluate(self, predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
        """Evaluate model predictions against ground truth.
        
        Args:
            predictions: List of model predictions
            targets: List of ground truth annotations
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if len(predictions) == 0:
                self.logger.warning("No predictions to evaluate")
                return self._get_empty_metrics()
            
            # Convert predictions and targets to COCO format
            coco_gt = self._convert_to_coco_format(targets, is_gt=True)
            coco_pred = self._convert_to_coco_format(predictions, is_gt=False)
            
            # Create temporary file paths more reliably (Windows-compatible)
            temp_dir = tempfile.gettempdir()
            gt_file_path = os.path.join(temp_dir, f"gt_{uuid.uuid4().hex}.json")
            pred_file_path = os.path.join(temp_dir, f"pred_{uuid.uuid4().hex}.json")
            
            try:
                # Write to temporary files
                with open(gt_file_path, 'w') as f:
                    json.dump(coco_gt, f)
                with open(pred_file_path, 'w') as f:
                    json.dump(coco_pred, f)
                
                # Load COCO API objects
                try:
                    coco_gt_api = COCO(gt_file_path)
                    
                    # Try loading predictions directly as list first
                    try:
                        if len(coco_pred["annotations"]) > 0:
                            coco_pred_api = coco_gt_api.loadRes(coco_pred["annotations"])
                        else:
                            self.logger.warning("No predictions to evaluate")
                            return self._get_empty_metrics()
                    except Exception as e1:
                        self.logger.warning(f"Error loading predictions as list: {e1}")
                        # Try loading from file instead
                        try:
                            coco_pred_api = coco_gt_api.loadRes(pred_file_path)
                        except Exception as e2:
                            self.logger.error(f"Error loading predictions from file: {e2}")
                            return self._get_empty_metrics()
                    
                    # Evaluate bounding boxes
                    coco_eval = COCOeval(coco_gt_api, coco_pred_api, 'bbox')
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    
                    bbox_metrics = {
                        'mAP': coco_eval.stats[0],
                        'mAP_50': coco_eval.stats[1],
                        'mAP_75': coco_eval.stats[2],
                        'mAP_small': coco_eval.stats[3],
                        'mAP_medium': coco_eval.stats[4],
                        'mAP_large': coco_eval.stats[5],
                        'AR_max_1': coco_eval.stats[6],
                        'AR_max_10': coco_eval.stats[7],
                        'AR_max_100': coco_eval.stats[8],
                        'AR_small': coco_eval.stats[9],
                        'AR_medium': coco_eval.stats[10],
                        'AR_large': coco_eval.stats[11],
                    }
                    
                    # Try to evaluate masks if available
                    try:
                        coco_eval = COCOeval(coco_gt_api, coco_pred_api, 'segm')
                        coco_eval.evaluate()
                        coco_eval.accumulate()
                        coco_eval.summarize()
                        
                        # Get segmentation metrics
                        segm_metrics = {
                            'mask_mAP': coco_eval.stats[0],
                            'mask_mAP_50': coco_eval.stats[1],
                            'mask_mAP_75': coco_eval.stats[2],
                            'mask_mAP_small': coco_eval.stats[3],
                            'mask_mAP_medium': coco_eval.stats[4],
                            'mask_mAP_large': coco_eval.stats[5],
                            'mask_AR_max_1': coco_eval.stats[6],
                            'mask_AR_max_10': coco_eval.stats[7],
                            'mask_AR_max_100': coco_eval.stats[8],
                            'mask_AR_small': coco_eval.stats[9],
                            'mask_AR_medium': coco_eval.stats[10],
                            'mask_AR_large': coco_eval.stats[11],
                        }
                        
                        # Combine metrics
                        metrics = {**bbox_metrics, **segm_metrics}
                        return metrics
                    except Exception as e:
                        # If mask evaluation fails, return only bbox metrics
                        self.logger.warning(f"Mask evaluation failed: {e}")
                        return bbox_metrics
                
                except Exception as e:
                    self.logger.error(f"Error in COCO evaluation: {e}")
                    self.logger.error(traceback.format_exc())
                    return self._get_empty_metrics()
                
            finally:
                # Clean up temporary files
                try:
                    if os.path.exists(gt_file_path):
                        os.remove(gt_file_path)
                    if os.path.exists(pred_file_path):
                        os.remove(pred_file_path)
                except Exception as cleanup_error:
                    self.logger.warning(f"Error cleaning up temporary files: {cleanup_error}")
        
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {e}")
            self.logger.error(traceback.format_exc())
            return self._get_empty_metrics()
    
    def _get_empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'mAP': 0.0,
            'mAP_50': 0.0,
            'mAP_75': 0.0,
            'mAP_small': 0.0,
            'mAP_medium': 0.0,
            'mAP_large': 0.0,
            'AR_max_1': 0.0,
            'AR_max_10': 0.0,
            'AR_max_100': 0.0,
            'AR_small': 0.0,
            'AR_medium': 0.0,
            'AR_large': 0.0
        }
    
    def _convert_to_coco_format(self, data: List[Dict], is_gt: bool) -> Dict:
        """Convert data to COCO format.
        
        Args:
            data: List of annotations or predictions
            is_gt: Whether data is ground truth annotations
            
        Returns:
            Dictionary in COCO format
        """
        images = []
        annotations = []
        
        # Categories (GN, GL, GS)
        categories = [
            {"id": 1, "name": "Normal", "supercategory": "glomeruli"},
            {"id": 2, "name": "Partially_sclerotic", "supercategory": "glomeruli"},
            {"id": 3, "name": "Sclerotic", "supercategory": "glomeruli"},
            {"id": 4, "name": "Uncertain", "supercategory": "glomeruli"}
        ]
        
        # Create a unique annotation ID
        ann_id = 1
        
        for idx, item in enumerate(data):
            try:
                # Create image entry
                image_id = idx + 1
                
                # Get image dimensions, handling both tensor and non-tensor cases
                if "orig_size" in item:
                    orig_size = item["orig_size"]
                    if isinstance(orig_size, torch.Tensor):
                        height = int(orig_size[0].item())
                        width = int(orig_size[1].item())
                    else:
                        height, width = int(orig_size[0]), int(orig_size[1])
                else:
                    height, width = 1000, 1000  # Default size
                
                images.append({
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": item.get("image_id", f"image_{image_id}.jpg")
                })
                
                # Handle boxes, labels, scores, masks
                boxes = item.get("boxes", [])
                labels = item.get("labels", [])
                scores = item.get("scores", []) if not is_gt else []
                masks = item.get("masks", [])
                
                # Skip if no boxes
                if len(boxes) == 0:
                    continue
                
                # Convert tensors to lists
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy().tolist()
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy().tolist()
                if isinstance(scores, torch.Tensor) and len(scores) > 0:
                    scores = scores.cpu().numpy().tolist()
                
                # Process each box
                for i in range(len(boxes)):
                    try:
                        # Get box
                        box = boxes[i]
                        
                        # Get label (ensure it's an integer)
                        if i < len(labels):
                            label = int(labels[i])
                        else:
                            label = 1  # Default to class 1
                        
                        # Get score for predictions
                        score = None
                        if not is_gt and i < len(scores):
                            score = float(scores[i])
                        
                        # Convert box from [x1, y1, x2, y2] to [x, y, width, height]
                        x1, y1, x2, y2 = box
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        # Create annotation entry
                        ann = {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": label,
                            "bbox": [float(x1), float(y1), float(bbox_width), float(bbox_height)],
                            "area": float(bbox_width * bbox_height),
                            "iscrowd": 0
                        }
                        
                        # Add score for predictions
                        if not is_gt and score is not None:
                            ann["score"] = score
                        
                        # Create a polygon segmentation from bbox (COCO format requires segmentation)
                        ann["segmentation"] = [[
                            float(x1), float(y1),
                            float(x1+bbox_width), float(y1),
                            float(x1+bbox_width), float(y1+bbox_height),
                            float(x1), float(y1+bbox_height)
                        ]]
                        
                        # Try to add mask segmentation if available
                        if isinstance(masks, torch.Tensor) and i < masks.shape[0]:
                            try:
                                from pycocotools import mask as mask_util
                                mask = masks[i].cpu().numpy()
                                if len(mask.shape) > 2:
                                    mask = mask[0]  # Take first channel if multi-channel
                                
                                # Ensure binary mask
                                binary_mask = (mask > 0.5).astype(np.uint8)
                                
                                # Only use mask if it has any foreground pixels
                                if binary_mask.sum() > 0:
                                    rle = mask_util.encode(np.asfortranarray(binary_mask))
                                    if isinstance(rle["counts"], bytes):
                                        rle["counts"] = rle["counts"].decode("utf-8")
                                    ann["segmentation"] = rle
                            except Exception as e:
                                self.logger.warning(f"Error processing mask: {e}")
                                # Keep polygon segmentation as fallback
                        
                        annotations.append(ann)
                        ann_id += 1
                    except Exception as e:
                        self.logger.warning(f"Error processing box {i}: {e}")
                        continue
            
            except Exception as e:
                self.logger.warning(f"Error processing item {idx}: {e}")
                continue
        
        return {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
    
    def evaluate_classifications(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """Evaluate classification performance per glomeruli type."""
        class_names = ['Normal', 'Partially_sclerotic', 'Sclerotic', 'Uncertain']
        metrics_per_class = {}
        
        for class_idx, class_name in enumerate(class_names):
            # Class index is +1 because 0 is background
            class_idx_in_data = class_idx + 1
            
            # Count TP, FP, FN for this class
            tp = 0
            fp = 0
            fn = 0
            
            for pred, target in zip(predictions, targets):
                pred_boxes = pred.get("boxes", [])
                pred_labels = pred.get("labels", [])
                target_boxes = target.get("boxes", [])
                target_labels = target.get("labels", [])
                
                # Process tensors efficiently
                device = pred_boxes.device if isinstance(pred_boxes, torch.Tensor) else torch.device('cpu')
                
                # Convert to tensors if needed, staying on the same device
                if not isinstance(pred_boxes, torch.Tensor):
                    pred_boxes = torch.tensor(pred_boxes, device=device)
                if not isinstance(pred_labels, torch.Tensor):
                    pred_labels = torch.tensor(pred_labels, device=device)
                if not isinstance(target_boxes, torch.Tensor):
                    target_boxes = torch.tensor(target_boxes, device=device)
                if not isinstance(target_labels, torch.Tensor):
                    target_labels = torch.tensor(target_labels, device=device)
                
                # Skip if empty
                if len(pred_boxes) == 0 or len(target_boxes) == 0:
                    continue
                    
                # Prefilter boxes by class for efficiency
                pred_class_mask = pred_labels == class_idx_in_data
                target_class_mask = target_labels == class_idx_in_data
                
                pred_class_boxes = pred_boxes[pred_class_mask]
                target_class_boxes = target_boxes[target_class_mask]
                
                # Count target instances of this class
                target_instances = target_class_mask.sum().item()
                
                # Count matched instances
                matched_target_indices = set()
                
                # Skip computation if either is empty
                if len(pred_class_boxes) == 0 or len(target_class_boxes) == 0:
                    fp += len(pred_class_boxes)
                    fn += target_instances
                    continue
                    
                # Compute IoU matrix efficiently
                # Computing matrix of ious between all pred and target boxes
                iou_matrix = torch.zeros((len(pred_class_boxes), len(target_class_boxes)), device=device)
                
                for p_idx, p_box in enumerate(pred_class_boxes):
                    for t_idx, t_box in enumerate(target_class_boxes):
                        iou_matrix[p_idx, t_idx] = calculate_iou(p_box, t_box)
                
                # For each prediction, find best matching target
                best_target_indices = iou_matrix.max(dim=1)[1]
                best_target_ious = iou_matrix.max(dim=1)[0]
                above_threshold = best_target_ious > 0.5
                
                # Count true positives and false positives
                for p_idx, (above_thresh, best_t_idx) in enumerate(zip(above_threshold, best_target_indices)):
                    if above_thresh and best_t_idx.item() not in matched_target_indices:
                        tp += 1
                        matched_target_indices.add(best_t_idx.item())
                    else:
                        fp += 1
                
                # Remaining unmatched targets are false negatives
                fn += target_instances - len(matched_target_indices)
                
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_per_class[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
        
        return metrics_per_class