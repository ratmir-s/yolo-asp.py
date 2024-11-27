import os
import xml.etree.ElementTree as ET

def convert_to_yolo(size, box):
    """
    Converts bounding box coordinates to YOLO format.
    size: tuple (width, height)
    box: tuple (xmin, ymin, xmax, ymax)
    Returns: (x_center, y_center, width, height) normalized
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    return x_center * dw, y_center * dh, width * dw, height * dh

def parse_xml_to_yolo(xml_file, output_dir, class_mapping):
    """
    Parses an XML file and creates a corresponding YOLO .txt file.
    xml_file: Path to the XML file
    output_dir: Directory to save the .txt file
    class_mapping: Dictionary mapping class names to YOLO class IDs
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image size
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)
    
    # Get image filename
    filename = root.find("filename").text
    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_filepath = os.path.join(output_dir, txt_filename)
    
    with open(txt_filepath, "w") as txt_file:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_mapping:
                print(f"Warning: Class {class_name} not in class mapping. Skipping.")
                continue
            class_id = class_mapping[class_name]
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            # Convert to YOLO format
            x_center, y_center, width, height = convert_to_yolo((img_width, img_height), (xmin, ymin, xmax, ymax))
            # Write to txt file
            txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def convert_dataset(xml_dir, output_dir, class_mapping):
    """
    Converts all XML annotations in a directory to YOLO format.
    xml_dir: Directory containing XML files
    output_dir: Directory to save YOLO .txt files
    class_mapping: Dictionary mapping class names to YOLO class IDs
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_dir, xml_file)
            parse_xml_to_yolo(xml_path, output_dir, class_mapping)

# Example usage
xml_directory = r"normal_1\Annotations"
output_directory = r"data\labels\train"
class_mapping = {"red": 0,"yellow":1,"green":2,"off":3}  # Map your classes to integer IDs

convert_dataset(xml_directory, output_directory, class_mapping)
