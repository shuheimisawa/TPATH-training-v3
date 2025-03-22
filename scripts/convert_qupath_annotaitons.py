# scripts/convert_qupath_annotations.py
import os
import argparse
import json
import csv
import glob
import numpy as np
import shapely.geometry
from shapely.affinity import affine_transform
from tqdm import tqdm
import xml.etree.ElementTree as ET

from src.utils.logger import get_logger
from src.utils.io import save_json
from src.utils.slide_io import SlideReader


def parse_args():
    parser = argparse.ArgumentParser(description='Convert QuPath annotations to project format')
    
    parser.add_argument('--annotations-dir', type=str, required=True,
                        help='Path to directory with QuPath annotation exports')
    parser.add_argument('--slides-dir', type=str, required=True,
                        help='Path to directory with slide files')
    parser.add_argument('--output-dir', type=str, default='data/annotations',
                        help='Path to output directory')
    parser.add_argument('--format', type=str, choices=['geojson', 'csv', 'xml'], default='geojson',
                        help='Format of QuPath annotations')
    parser.add_argument('--class-map', type=str, default=None,
                        help='Path to JSON file mapping QuPath classes to project classes')
    
    return parser.parse_args()


def load_class_map(class_map_path):
    """Load class mapping from JSON file."""
    if class_map_path and os.path.exists(class_map_path):
        with open(class_map_path, 'r') as f:
            return json.load(f)
    else:
        # Default class mapping
        return {
            "Normal": "Normal",
            "Sclerotic": "Sclerotic",
            "Partially sclerotic": "Partially_sclerotic",
            "Uncertain": "Uncertain",
            # Alternative names that might be used
            "GN": "Normal",
            "GS": "Sclerotic",
            "GL": "Partially_sclerotic",
            "Normal Glomerulus": "Normal",
            "Sclerosed Glomerulus": "Sclerotic",
            "Partially Sclerosed": "Partially_sclerotic",
            "Partially Sclerotic": "Partially_sclerotic",
            "Partial Sclerosis": "Partially_sclerotic"
        }


def convert_geojson(geojson_path, class_map, slide_reader=None):
    """Convert GeoJSON export from QuPath to project format."""
    logger = get_logger(name="convert_qupath")
    
    try:
        # Load GeoJSON file
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        
        annotations = []
        
        # Extract features
        for feature in geojson_data.get('features', []):
            # Get properties
            properties = feature.get('properties', {})
            classification = properties.get('classification', {})
            class_name = classification.get('name')
            
            # Check if this is a valid glomerulus annotation
            if not class_name or class_name not in class_map:
                continue
            
            # Map to project class
            project_class = class_map[class_name]
            
            # Get geometry
            geometry = feature.get('geometry', {})
            geometry_type = geometry.get('type')
            coordinates = geometry.get('coordinates')
            
            if not coordinates:
                continue
            
            # Handle different geometry types
            if geometry_type == 'Polygon':
                # Simplify by only using outer ring
                polygon_coords = coordinates[0]
                
                # Create annotation
                x_coords = [coord[0] for coord in polygon_coords]
                y_coords = [coord[1] for coord in polygon_coords]
                
                if not x_coords or not y_coords:
                    continue
                
                # Calculate bounding box
                x_min, y_min = min(x_coords), min(y_coords)
                x_max, y_max = max(x_coords), max(y_coords)
                width, height = x_max - x_min, y_max - y_min
                
                # Convert polygon to project format
                segmentation = []
                for x, y in zip(x_coords, y_coords):
                    segmentation.extend([x, y])
                
                annotation = {
                    'bbox': [x_min, y_min, width, height],
                    'category': project_class,
                    'segmentation': [segmentation]
                }
                
                annotations.append(annotation)
                
            elif geometry_type == 'Point':
                if slide_reader:
                    # Points need to be converted to rectangular regions
                    # Use a fixed size based on typical glomerulus size
                    # Adjust as needed based on your data
                    pixel_size = 30  # In pixels at level 0
                    
                    x, y = coordinates
                    
                    # Create a square box around the point
                    x_min, y_min = x - pixel_size, y - pixel_size
                    width, height = pixel_size * 2, pixel_size * 2
                    
                    # Create annotation
                    annotation = {
                        'bbox': [x_min, y_min, width, height],
                        'category': project_class,
                        'segmentation': [[x_min, y_min, x_min + width, y_min, 
                                          x_min + width, y_min + height, x_min, y_min + height]]
                    }
                    
                    annotations.append(annotation)
        
        return annotations
    
    except Exception as e:
        logger.error(f"Error converting GeoJSON {geojson_path}: {e}")
        return []


def convert_csv(csv_path, class_map, slide_reader=None):
    """Convert CSV export from QuPath to project format."""
    logger = get_logger(name="convert_qupath")
    
    try:
        annotations = []
        
        # Load CSV file
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Check if this is a valid glomerulus annotation
                class_name = row.get('Class')
                
                if not class_name or class_name not in class_map:
                    continue
                
                # Map to project class
                project_class = class_map[class_name]
                
                # Get coordinates - CSV format may vary, adjust field names as needed
                try:
                    # For ROI annotations
                    if 'ROI (X,Y)' in row:
                        # Format: "X1,Y1, X2,Y2, X3,Y3, ..."
                        roi_str = row['ROI (X,Y)']
                        coords = [float(coord) for coord in roi_str.replace(' ', '').split(',')]
                        
                        # Group into (x,y) pairs
                        polygon_coords = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                        
                        # Calculate bounding box
                        x_coords = [coord[0] for coord in polygon_coords]
                        y_coords = [coord[1] for coord in polygon_coords]
                        
                        x_min, y_min = min(x_coords), min(y_coords)
                        x_max, y_max = max(x_coords), max(y_coords)
                        width, height = x_max - x_min, y_max - y_min
                        
                        # Convert polygon to project format
                        segmentation = []
                        for x, y in polygon_coords:
                            segmentation.extend([x, y])
                        
                        annotation = {
                            'bbox': [x_min, y_min, width, height],
                            'category': project_class,
                            'segmentation': [segmentation]
                        }
                        
                        annotations.append(annotation)
                    
                    # For bounding box annotations
                    elif all(field in row for field in ['Centroid X µm', 'Centroid Y µm', 'Width µm', 'Height µm']):
                        # Convert from µm to pixels if slide_reader is provided
                        if slide_reader and 'pixel_size_x' in dir(slide_reader) and 'pixel_size_y' in dir(slide_reader):
                            # Get pixel size in µm
                            pixel_size_x = slide_reader.pixel_size_x
                            pixel_size_y = slide_reader.pixel_size_y
                            
                            # Convert to pixels
                            x_center = float(row['Centroid X µm']) / pixel_size_x
                            y_center = float(row['Centroid Y µm']) / pixel_size_y
                            width = float(row['Width µm']) / pixel_size_x
                            height = float(row['Height µm']) / pixel_size_y
                            
                            # Calculate bounding box
                            x_min = x_center - width / 2
                            y_min = y_center - height / 2
                            
                            # Create annotation
                            annotation = {
                                'bbox': [x_min, y_min, width, height],
                                'category': project_class,
                                'segmentation': [[x_min, y_min, x_min + width, y_min, 
                                                  x_min + width, y_min + height, x_min, y_min + height]]
                            }
                            
                            annotations.append(annotation)
                
                except Exception as row_error:
                    logger.warning(f"Error processing row in CSV: {row_error}")
                    continue
        
        return annotations
    
    except Exception as e:
        logger.error(f"Error converting CSV {csv_path}: {e}")
        return []


def convert_xml(xml_path, class_map, slide_reader=None):
    """Convert XML export from QuPath to project format."""
    logger = get_logger(name="convert_qupath")
    
    try:
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        
        # Find annotations in the XML
        for annotation_elem in root.findall('.//Annotation'):
            # Get attributes
            annotation_id = annotation_elem.get('Id')
            annotation_type = annotation_elem.get('Type')
            class_name = None
            
            # Try to find the class name
            for attr in annotation_elem.findall('.//Attribute'):
                if attr.get('Name') == 'Class':
                    class_name = attr.get('Value')
                    break
            
            # Check if this is a valid glomerulus annotation
            if not class_name or class_name not in class_map:
                continue
            
            # Map to project class
            project_class = class_map[class_name]
            
            # Get coordinates
            coords = []
            for region in annotation_elem.findall('.//Region'):
                for vertex in region.findall('.//Vertex'):
                    x = float(vertex.get('X'))
                    y = float(vertex.get('Y'))
                    coords.append((x, y))
            
            if not coords:
                continue
            
            # Calculate bounding box
            x_coords = [coord[0] for coord in coords]
            y_coords = [coord[1] for coord in coords]
            
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            width, height = x_max - x_min, y_max - y_min
            
            # Convert polygon to project format
            segmentation = []
            for x, y in coords:
                segmentation.extend([x, y])
            
            annotation = {
                'bbox': [x_min, y_min, width, height],
                'category': project_class,
                'segmentation': [segmentation]
            }
            
            annotations.append(annotation)
        
        return annotations
    
    except Exception as e:
        logger.error(f"Error converting XML {xml_path}: {e}")
        return []


def main():
    # Parse arguments
    args = parse_args()
    
    # Create logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(
        name="convert_qupath",
        log_file=os.path.join(args.output_dir, "convert_qupath.log")
    )
    
    # Load class mapping
    class_map = load_class_map(args.class_map)
    logger.info(f"Using class mapping: {class_map}")
    
    # Find annotation files
    if args.format == 'geojson':
        annotation_files = glob.glob(os.path.join(args.annotations_dir, "*.geojson"))
    elif args.format == 'csv':
        annotation_files = glob.glob(os.path.join(args.annotations_dir, "*.csv"))
    else:  # xml
        annotation_files = glob.glob(os.path.join(args.annotations_dir, "*.xml"))
    
    logger.info(f"Found {len(annotation_files)} annotation files")
    
    # Process each annotation file
    for annotation_file in tqdm(annotation_files, desc="Converting annotations"):
        # Extract slide name from annotation filename
        slide_name = os.path.splitext(os.path.basename(annotation_file))[0]
        
        # Find corresponding slide file
        slide_extensions = ['.svs', '.ndpi', '.tif', '.tiff']
        slide_path = None
        
        for ext in slide_extensions:
            potential_path = os.path.join(args.slides_dir, f"{slide_name}{ext}")
            if os.path.exists(potential_path):
                slide_path = potential_path
                break
        
        slide_reader = None
        if slide_path:
            try:
                # Open slide for reference
                slide_reader = SlideReader(slide_path)
            except Exception as e:
                logger.warning(f"Could not open slide {slide_path}: {e}")
        else:
            logger.warning(f"Could not find slide for {slide_name}, proceeding without slide reference")
        
        # Convert annotations based on format
        if args.format == 'geojson':
            annotations = convert_geojson(annotation_file, class_map, slide_reader)
        elif args.format == 'csv':
            annotations = convert_csv(annotation_file, class_map, slide_reader)
        else:  # xml
            annotations = convert_xml(annotation_file, class_map, slide_reader)
        
        # Close slide reader if opened
        if slide_reader:
            slide_reader.close()
        
        # Save converted annotations
        if annotations:
            output_file = os.path.join(args.output_dir, f"{slide_name}_annotations.json")
            
            # Create the final annotations structure
            slide_annotations = {
                slide_name: {
                    'file_path': f"{slide_name}.svs" if slide_path is None else os.path.basename(slide_path),
                    'annotations': annotations
                }
            }
            
            # Save to JSON
            save_json(slide_annotations, output_file)
            logger.info(f"Saved {len(annotations)} annotations to {output_file}")
        else:
            logger.warning(f"No valid annotations found in {annotation_file}")
    
    logger.info("Conversion completed")


if __name__ == '__main__':
    main()