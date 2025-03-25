# src/evaluation/evaluator.py
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
import tempfile

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
        if len(predictions) == 0:
            self.logger.warning("No predictions to evaluate")
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
        
        # Convert predictions and targets to COCO format
        coco_gt = self._convert_to_coco_format(targets, is_gt=True)
        coco_pred = self._convert_to_coco_format(predictions, is_gt=False)
        
        # Save to temporary files
        with tempfile.NamedTemporaryFile(suffix='.json') as gt_file:
            with tempfile.NamedTemporaryFile(suffix='.json') as pred_file:
                with open(gt_file.name, 'w') as f:
                    json.dump(coco_gt, f)
                with open(pred_file.name, 'w') as f:
                    json.dump(coco_pred, f)
                
                # Load COCO API objects
                coco_gt_api = COCO(gt_file.name)
                coco_pred_api = coco_gt_api.loadRes(pred_file.name)
                
                # Run COCO evaluation
                coco_eval = COCOeval(coco_gt_api, coco_pred_api, 'bbox')
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()
                
                # Get bbox metrics
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
                
                # Run COCO evaluation for masks
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
            # Create image entry
            image_id = idx + 1
            
            # Get image dimensions, handling both tensor and non-tensor cases
            if "orig_size" in item:
                orig_size = item["orig_size"]
                if isinstance(orig_size, torch.Tensor):
                    height = orig_size[0].item()
                    width = orig_size[1].item()
                else:
                    height, width = orig_size
            else:
                height, width = 1000, 1000  # Default size
            
            images.append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": item.get("image_id", f"image_{image_id}.jpg")
            })
            
            if is_gt:
                # Handle ground truth data
                boxes = item.get("boxes", [])
                labels = item.get("labels", [])
                masks = item.get("masks", [])
                
                # Convert to tensor if not already
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                if not isinstance(masks, torch.Tensor):
                    masks = torch.tensor(masks)
                
                for box_idx in range(len(boxes)):
                    box = boxes[box_idx].cpu().tolist()
                    label = labels[box_idx].item() if isinstance(labels[box_idx], torch.Tensor) else labels[box_idx]
                    mask = masks[box_idx].cpu().numpy() if isinstance(masks, torch.Tensor) and masks.size(0) > 0 else None
                    
                    # Convert box from [x1, y1, x2, y2] to [x, y, width, height]
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Create annotation entry
                    ann = {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    }
                    
                    # Add segmentation mask if available
                    if mask is not None:
                        from pycocotools import mask as mask_util
                        rle = mask_util.encode(np.asfortranarray(mask))
                        rle["counts"] = rle["counts"].decode("utf-8")
                        ann["segmentation"] = rle
                    
                    annotations.append(ann)
                    ann_id += 1
            else:
                # Handle predictions
                boxes = item.get("boxes", [])
                labels = item.get("labels", [])
                scores = item.get("scores", [])
                masks = item.get("masks", [])
                
                # Convert to tensor if not already
                if not isinstance(boxes, torch.Tensor):
                    boxes = torch.tensor(boxes)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                if not isinstance(scores, torch.Tensor):
                    scores = torch.tensor(scores)
                if not isinstance(masks, torch.Tensor):
                    masks = torch.tensor(masks)
                
                for box_idx in range(len(boxes)):
                    box = boxes[box_idx].cpu().tolist()
                    label = labels[box_idx].item() if isinstance(labels[box_idx], torch.Tensor) else labels[box_idx]
                    score = scores[box_idx].item() if isinstance(scores[box_idx], torch.Tensor) else scores[box_idx]
                    mask = masks[box_idx].cpu().numpy() if isinstance(masks, torch.Tensor) and len(masks) > 0 else None
                    
                    # Convert box from [x1, y1, x2, y2] to [x, y, width, height]
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Create annotation entry
                    ann = {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": label,
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "score": score,
                        "iscrowd": 0
                    }
                    
                    # Add segmentation mask if available
                    if mask is not None:
                        from pycocotools import mask as mask_util
                        rle = mask_util.encode(np.asfortranarray(mask))
                        rle["counts"] = rle["counts"].decode("utf-8")
                        ann["segmentation"] = rle
                    
                    annotations.append(ann)
                    ann_id += 1
        
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
        
    def evaluate_by_stain(self, predictions: List[Dict], targets: List[Dict], stain_types: List[str]) -> Dict:
        """Evaluate performance by stain type.
        
        Args:
            predictions: List of predictions
            targets: List of ground truth annotations
            stain_types: List of stain types for each image
            
        Returns:
            Dictionary with metrics per stain type
        """
        stain_names = set(stain_types)
        metrics_per_stain = {}
        
        for stain_name in stain_names:
            # Get indices for this stain type
            indices = [i for i, s in enumerate(stain_types) if s == stain_name]
            
            # Skip if no images of this stain type
            if not indices:
                continue
            
            # Get predictions and targets for this stain type
            stain_predictions = [predictions[i] for i in indices]
            stain_targets = [targets[i] for i in indices]
            
            # Evaluate this stain type
            try:
                stain_metrics = self.evaluate(stain_predictions, stain_targets)
                metrics_per_stain[stain_name] = stain_metrics
            except Exception as e:
                self.logger.error(f"Error evaluating stain type {stain_name}: {e}")
                metrics_per_stain[stain_name] = {"error": str(e)}
        
        return metrics_per_stain
    
    def analyze_errors(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """Analyze common error patterns.
        
        Args:
            predictions: List of predictions
            targets: List of ground truth annotations
            
        Returns:
            Dictionary with error analysis
        """
        class_names = ['background', 'Normal', 'Partially_sclerotic', 'Sclerotic', 'Uncertain']
        
        # Error analysis struct
        error_analysis = {
            "missed_detections": [],  # Instances not detected
            "false_detections": [],   # False positive detections
            "misclassifications": [], # Correctly detected but wrong class
            "size_based_errors": {    # Errors by size
                "small": {"missed": 0, "false": 0, "misclassified": 0, "total": 0},
                "medium": {"missed": 0, "false": 0, "misclassified": 0, "total": 0},
                "large": {"missed": 0, "false": 0, "misclassified": 0, "total": 0}
            },
            "confusion_matrix": np.zeros((5, 5), dtype=int)  # Including background
        }
        
        for idx, (pred, target) in enumerate(zip(predictions, targets)):
            # Get image dimensions for relative size categorization
            img_height, img_width = target.get("orig_size", (1000, 1000))
            img_area = img_height * img_width
            
            small_threshold = 0.001 * img_area  # 0.1% of image area
            medium_threshold = 0.01 * img_area  # 1% of image area

            # Convert to numpy if needed
            if isinstance(pred_boxes, torch.Tensor):
                pred_boxes = pred_boxes.cpu().numpy()
            if isinstance(pred_labels, torch.Tensor):
                pred_labels = pred_labels.cpu().numpy()
            if isinstance(pred_scores, torch.Tensor):
                pred_scores = pred_scores.cpu().numpy()
            if isinstance(target_boxes, torch.Tensor):
                target_boxes = target_boxes.cpu().numpy()
            if isinstance(target_labels, torch.Tensor):
                target_labels = target_labels.cpu().numpy()
            
            # Match predictions to targets
            matched_targets = [-1] * len(target_boxes)
            matched_preds = [-1] * len(pred_boxes)
            
            # First pass: match by IoU
            for t_idx, t_box in enumerate(target_boxes):
                best_iou = 0.5  # IoU threshold
                best_p_idx = -1
                
                for p_idx, p_box in enumerate(pred_boxes):
                    if matched_preds[p_idx] >= 0:
                        continue
                    
                    iou = calculate_iou(
                        torch.tensor(t_box),
                        torch.tensor(p_box)
                    )
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_p_idx = p_idx
                
                if best_p_idx >= 0:
                    matched_targets[t_idx] = best_p_idx
                    matched_preds[best_p_idx] = t_idx
                    
                    # Update confusion matrix
                    t_label = target_labels[t_idx]
                    p_label = pred_labels[best_p_idx]
                    error_analysis["confusion_matrix"][t_label, p_label] += 1
                    
                    # Check for misclassification
                    if t_label != p_label:
                        error_analysis["misclassifications"].append({
                            "image_idx": idx,
                            "true_class": class_names[t_label],
                            "pred_class": class_names[p_label],
                            "iou": best_iou,
                            "score": pred_scores[best_p_idx] if len(pred_scores) > best_p_idx else None
                        })
                        
                        # Update size-based errors
                        box_area = (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])
                        if box_area < small_threshold:
                            error_analysis["size_based_errors"]["small"]["missed"] += 1
                            error_analysis["size_based_errors"]["small"]["total"] += 1
                        elif box_area < medium_threshold:
                            error_analysis["size_based_errors"]["medium"]["missed"] += 1
                            error_analysis["size_based_errors"]["medium"]["total"] += 1
                        else:
                            error_analysis["size_based_errors"]["large"]["missed"] += 1
                            error_analysis["size_based_errors"]["large"]["total"] += 1
                
                else:
                    # Missed detection
                    error_analysis["missed_detections"].append({
                        "image_idx": idx,
                        "true_class": class_names[target_labels[t_idx]],
                        "box": target_boxes[t_idx].tolist() if isinstance(target_boxes[t_idx], np.ndarray) else target_boxes[t_idx]
                    })
                    
                    # Update size-based errors
                    box_area = (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])
                    if box_area < 32*32:
                        error_analysis["size_based_errors"]["small"]["missed"] += 1
                        error_analysis["size_based_errors"]["small"]["total"] += 1
                    elif box_area < 96*96:
                        error_analysis["size_based_errors"]["medium"]["missed"] += 1
                        error_analysis["size_based_errors"]["medium"]["total"] += 1
                    else:
                        error_analysis["size_based_errors"]["large"]["missed"] += 1
                        error_analysis["size_based_errors"]["large"]["total"] += 1
            
            # Find false detections
            for p_idx, matched in enumerate(matched_preds):
                if matched < 0:
                    # False detection
                    error_analysis["false_detections"].append({
                        "image_idx": idx,
                        "pred_class": class_names[pred_labels[p_idx]],
                        "score": pred_scores[p_idx] if len(pred_scores) > p_idx else None,
                        "box": pred_boxes[p_idx].tolist() if isinstance(pred_boxes[p_idx], np.ndarray) else pred_boxes[p_idx]
                    })
                    
                    # Update size-based errors
                    p_box = pred_boxes[p_idx]
                    box_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                    if box_area < 32*32:
                        error_analysis["size_based_errors"]["small"]["false"] += 1
                    elif box_area < 96*96:
                        error_analysis["size_based_errors"]["medium"]["false"] += 1
                    else:
                        error_analysis["size_based_errors"]["large"]["false"] += 1
        
        return error_analysis