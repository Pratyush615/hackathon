import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time

def create_mapping_from_meta(meta_file_path='meta.json', output_file='class_mapping.txt'):
    """Create a class mapping text file from meta.json"""
    try:
        class_mapping = load_class_mapping_from_meta(meta_file_path)
        if not class_mapping:
            print("Failed to load class mapping from meta.json")
            return False
            
        with open(output_file, 'w') as f:
            for idx in sorted(class_mapping.keys()):
                f.write(f"{class_mapping[idx]}\n")
                
        print(f"Successfully created {output_file} with {len(class_mapping)} classes")
        return True
        
    except Exception as e:
        print(f"Error creating mapping file from meta.json: {e}")
        return False

def load_class_mapping_from_meta(meta_file_path='meta.json', num_classes=None):
    """Load class mapping from meta.json file used during training"""
    try:
        import json
        
        with open(meta_file_path, 'r') as f:
            meta_data = json.load(f)
        
        if 'classes' not in meta_data:
            print(f"Warning: No 'classes' field found in {meta_file_path}")
            return None
            
        # Create mapping from class index to name
        # Ensure 'background' is class 0
        class_mapping = {0: "background"}
        
        # Add each class from the meta file
        for i, class_info in enumerate(meta_data['classes'], start=1):
            if 'title' in class_info:
                class_mapping[i] = class_info['title']
        
        print(f"Loaded {len(class_mapping)-1} ingredient classes from {meta_file_path}")
        
        # If num_classes is specified, ensure we have mapping up to that number
        if num_classes and num_classes > len(class_mapping):
            for i in range(len(class_mapping), num_classes):
                class_mapping[i] = f"ingredient_{i}"
            print(f"Extended mapping to cover all {num_classes} classes")
            
        return class_mapping
        
    except Exception as e:
        print(f"Error loading class mapping from meta.json: {e}")
        return None

def get_class_mapping(num_classes):
    """Create a mapping from class index to ingredient name"""
    # First try to load from meta.json (training dataset metadata)
    meta_file_paths = ["meta.json", "dataset_meta.json", "classes_meta.json"]
    
    for meta_path in meta_file_paths:
        if os.path.exists(meta_path):
            class_mapping = load_class_mapping_from_meta(meta_path, num_classes)
            if class_mapping:
                return class_mapping
    
    # Next try to load from a text mapping file
    mapping_file_paths = ["class_mapping.txt", "ingredient_classes.txt", "class_names.txt"]
    
    for file_path in mapping_file_paths:
        if os.path.exists(file_path):
            print(f"Loading class mapping from {file_path}")
            class_mapping = {}
            try:
                with open(file_path, 'r') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if ',' in line:
                                idx, name = line.split(',', 1)
                                try:
                                    idx = int(idx.strip())
                                    class_mapping[idx] = name.strip()
                                except ValueError:
                                    class_mapping[i] = line
                            else:
                                class_mapping[i] = line
                
                # Ensure background class exists
                if 0 not in class_mapping:
                    class_mapping[0] = "background"
                
                print(f"Loaded {len(class_mapping)} class names")
                return class_mapping
            except Exception as e:
                print(f"Error loading class mapping from file: {e}")
    
    # Create a default mapping file with common food ingredients
    try:
        print("\nNo class mapping file found. Creating default mapping with common food ingredients.")
        default_ingredients = [
            "background", "apple", "banana", "orange", "carrot", "potato", "onion", "tomato", 
            "lettuce", "cucumber", "pepper", "chicken", "beef", "pork", "fish", "rice", 
            "pasta", "bread", "cheese", "egg", "milk", "butter", "oil", "garlic", "ginger", 
            "mushroom", "broccoli", "spinach", "corn", "peas", "beans", "lemon", "lime", 
            "avocado", "strawberry", "blueberry", "grape", "watermelon", "kiwi", "pineapple", 
            "peach", "pear", "plum", "cherry", "mango", "cabbage", "cauliflower", "celery", 
            "asparagus", "eggplant", "squash", "pumpkin", "sweet potato", "zucchini", "radish", 
            "shrimp", "crab", "lobster", "clam", "mussel", "yogurt", "cream", "flour", "sugar", 
            "salt", "pepper", "cinnamon", "basil", "parsley", "cilantro", "thyme", "oregano", 
            "mint", "rosemary", "dill", "bay leaf", "cumin", "paprika", "turmeric", "chili", 
            "mustard", "honey", "maple syrup", "soy sauce", "vinegar", "wine", "beer", "tequila", 
            "vodka", "rum", "whiskey", "gin", "chocolate", "vanilla", "almond", "walnut", 
            "pecan", "cashew", "peanut", "pistachio", "pine nut", "hazelnut", "coconut", "raisin"
        ]
        
        # Create mapping dictionary
        class_mapping = {}
        for i, ingredient in enumerate(default_ingredients[:num_classes]):
            class_mapping[i] = ingredient
        
        # Fill any remaining classes
        for i in range(len(default_ingredients), num_classes):
            class_mapping[i] = f"ingredient_{i}"
        
        # Save the mapping to a file for future use
        with open("default_class_mapping.txt", 'w') as f:
            for i in range(num_classes):
                f.write(f"{i},{class_mapping[i]}\n")
                
        print(f"Created default mapping with {len(class_mapping)} classes")
        print("Saved to 'default_class_mapping.txt'")
        
        return class_mapping
        
    except Exception as e:
        print(f"Error creating default mapping: {e}")
        
        # Final fallback: simple numeric mapping
        print("Using simple generic mapping")
        class_mapping = {0: "background"}
        for i in range(1, num_classes):
            class_mapping[i] = f"ingredient_{i}"
        
        return class_mapping

def load_model(model_path, device=None):
    """Load the trained segmentation model with correct architecture"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading checkpoint to determine model structure...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            print("Trying to load with additional parameters...")
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    # Determine number of classes from the checkpoint
    classifier_weight_key = None
    num_classes = None
    
    # Look for the classifier weights to determine number of classes
    possible_keys = [
        'classifier.4.weight', 
        'model.classifier.4.weight',
        'module.classifier.4.weight'
    ]
    
    for key in possible_keys:
        if key in checkpoint:
            classifier_weight_key = key
            num_classes = checkpoint[key].shape[0]
            print(f"Found classifier weight with {num_classes} output classes")
            break
    
    if num_classes is None:
        # Try to infer from state_dict keys
        for key in checkpoint.keys():
            if 'classifier' in key and 'weight' in key:
                parts = key.split('.')
                if len(parts) >= 2 and parts[-1] == 'weight':
                    try:
                        num_classes = checkpoint[key].shape[0]
                        classifier_weight_key = key
                        print(f"Inferred {num_classes} classes from {key}")
                        break
                    except:
                        continue
    
    if num_classes is None:
        print("Could not determine number of classes, defaulting to 103 based on error message")
        num_classes = 103
    
    # Check if model has aux_classifier
    has_aux = any('aux_classifier' in key for key in checkpoint.keys())
    print(f"Model {'has' if has_aux else 'does not have'} auxiliary classifier")
    
    # Initialize the same model architecture
    try:
        # Try to initialize with the same architecture as in training
        if has_aux:
            print("Using DeepLabV3+ with ResNet50 backbone")
            model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
        else:
            print("Using DeepLabV3+ with ResNet50 backbone without aux_loss")
            model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=False)
    except Exception as e:
        print(f"Error initializing ResNet50 model: {e}")
        print("Falling back to MobileNetV3 architecture")
        if has_aux:
            model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
        else:
            model = models.segmentation.deeplabv3_mobilenet_v3_large(weights=None, aux_loss=False)
    
    # Set up output layer for correct number of classes
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    if has_aux:
        aux_in_channels = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in_channels, num_classes, kernel_size=1)
    
    print(f"Initialized model with {num_classes} output classes")
    
    # Try to load the state dict - with some handling for different model versions
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print("Loading from 'model_state_dict' key")
        state_dict = checkpoint['model_state_dict']
    else:
        print("Loading directly from checkpoint")
        state_dict = checkpoint
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully with non-strict loading")
    except Exception as e:
        print(f"Error loading state dict directly: {e}")
        print("Trying to load with key mapping...")
        
        # Create a new state dict with correct keys
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if it exists (from DataParallel)
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("Model loaded successfully after key mapping")
        except Exception as e:
            print(f"Warning: Could not load all weights: {e}")
            print("Continuing with partially loaded model")
    
    model = model.to(device)
    model.eval()
    
    return model, device, num_classes

def preprocess_frame(frame, device):
    """Preprocess a video frame for the model"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # Keep original frame for display
        original_frame = frame.copy()
        
        # Ensure frame is in RGB format for PIL
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Check if it's BGR (from OpenCV) and convert to RGB
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
        else:
            raise ValueError("Invalid frame format")
            
        # Apply preprocessing
        input_tensor = transform(frame).unsqueeze(0).to(device)
        return input_tensor, original_frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        # Create a blank test image
        print("Creating blank test image instead")
        blank = np.zeros((512, 512, 3), dtype=np.uint8)
        blank.fill(128)  # Medium gray
        original_frame = blank.copy()
        input_tensor = transform(blank).unsqueeze(0).to(device)
        return input_tensor, original_frame

def predict_segmentation_with_confidence(model, input_tensor, original_frame, num_classes, confidence_threshold=0.5, class_mapping=None):
    """Run segmentation prediction with confidence thresholding"""
    with torch.no_grad():
        try:
            output = model(input_tensor)['out']
            
            # Add debug info
            print(f"Raw output shape: {output.shape}")
            print(f"Output min/max values: {output.min().item()}, {output.max().item()}")
            
            # Apply softmax to get probabilities
            output_probs = F.softmax(output, dim=1)
            
            # More debug info
            print(f"After softmax - max probability: {output_probs.max().item()}")
            
            # Get the predicted class and confidence for each pixel
            max_probs, output_predictions = output_probs.max(1)
            
            # Debug the distribution of class predictions
            unique_pred_classes, pred_counts = torch.unique(output_predictions, return_counts=True)
            print(f"Predicted classes: {unique_pred_classes.cpu().numpy()}")
            print(f"Prediction counts: {pred_counts.cpu().numpy()}")
            
            # Convert to numpy
            output_predictions = output_predictions.squeeze(0).cpu().numpy()
            confidence_map = max_probs.squeeze(0).cpu().numpy()
            
            # Check average confidence
            print(f"Average confidence: {confidence_map.mean():.4f}")
            
            # Create a mask for low confidence areas (below threshold)
            low_confidence_mask = confidence_map < confidence_threshold
            print(f"Percentage of low confidence pixels: {np.mean(low_confidence_mask) * 100:.2f}%")
            
            # Apply the mask (set low confidence areas to background class 0)
            thresholded_predictions = output_predictions.copy()
            thresholded_predictions[low_confidence_mask] = 0
            
            # Debug thresholded predictions
            unique_thresh_classes, thresh_counts = np.unique(thresholded_predictions, return_counts=True)
            print(f"Classes after thresholding: {unique_thresh_classes}")
            print(f"Counts after thresholding: {thresh_counts}")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return None, original_frame, None, None, None, []
    
    # Find unique classes in the predictions (excluding background class 0)
    unique_classes = np.unique(thresholded_predictions)
    detected_ingredients = []
    
    for class_id in unique_classes:
        if class_id > 0:  # Skip background class
            # Calculate percentage of pixels for this class
            class_pixels = np.sum(thresholded_predictions == class_id)
            total_pixels = thresholded_predictions.size
            percentage = (class_pixels / total_pixels) * 100
            
            # Only consider classes that occupy a significant portion of the image
            if percentage > 0.5:  # Lower threshold to catch smaller ingredients
                if class_mapping and class_id in class_mapping:
                    ingredient_name = class_mapping[class_id]
                else:
                    ingredient_name = f"Ingredient {class_id}"
                detected_ingredients.append((class_id, ingredient_name, percentage))
    
    # Sort ingredients by percentage (descending)
    detected_ingredients.sort(key=lambda x: x[2], reverse=True)
    
    # Create a colormap for visualization
    colormap = plt.cm.get_cmap('viridis', num_classes)
    colored_mask = colormap(thresholded_predictions)
    
    # Convert from RGBA to RGB
    colored_mask = (colored_mask[:, :, :3] * 255).astype(np.uint8)
    
    # Resize to match original frame size
    h, w = original_frame.shape[:2]
    colored_mask_resized = cv2.resize(colored_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create confidence visualization (brighter = higher confidence)
    confidence_viz = (confidence_map * 255).astype(np.uint8)
    confidence_viz_resized = cv2.resize(confidence_viz, (w, h), interpolation=cv2.INTER_LINEAR)
    confidence_viz_colored = cv2.applyColorMap(confidence_viz_resized, cv2.COLORMAP_JET)
    
    # Create overlay (original image with segmentation mask)
    overlay = cv2.addWeighted(original_frame, 0.6, colored_mask_resized, 0.4, 0)
    
    # Add label overlays for each detected ingredient
    labeled_overlay = overlay.copy()
    for class_id, ingredient_name, percentage in detected_ingredients:
        # Create binary mask for this class
        class_mask = (thresholded_predictions == class_id).astype(np.uint8)
        class_mask_resized = cv2.resize(class_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(class_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # For larger regions, add label
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Only label larger regions, reduced threshold
                # Find center of mass of the contour
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Create label text with confidence percentage
                    label_text = f"{ingredient_name}: {percentage:.1f}%"
                    
                    # Add a background rectangle for the text
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(labeled_overlay, 
                                  (cx - 5, cy - text_size[1] - 5), 
                                  (cx + text_size[0] + 5, cy + 5), 
                                  (0, 0, 0), -1)
                    
                    # Add text label
                    cv2.putText(labeled_overlay, label_text, (cx, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return thresholded_predictions, original_frame, colored_mask_resized, labeled_overlay, confidence_viz_colored, detected_ingredients

def setup_display_window():
    """Set up display window with trackbar for confidence threshold"""
    cv2.namedWindow('Segmentation Results', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Segmentation Results', 1200, 800)
    
    # Create trackbar for confidence threshold
    cv2.createTrackbar('Confidence %', 'Segmentation Results', 50, 100, lambda x: None)
    
    return

def run_live_segmentation(model_path, camera_id=0):
    """Run segmentation on live camera feed"""
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model, device, num_classes = load_model(model_path)
        print(f"Model loaded successfully. Using device: {device}")
        
        # Create class mapping for ingredient names
        class_mapping = get_class_mapping(num_classes)
        print(f"Created class mapping for {num_classes} classes")
        
    except Exception as e:
        print(f"Critical error loading model: {e}")
        return
    
    # Set up camera
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        print("Trying default camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any camera")
            return
    
    # Set up display window
    setup_display_window()
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Create a separate window for detected ingredients
    cv2.namedWindow('Detected Ingredients', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detected Ingredients', 400, 600)
    
    print("Press 'q' to quit, 's' to save current frame")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Get current confidence threshold from trackbar
            confidence_threshold = cv2.getTrackbarPos('Confidence %', 'Segmentation Results') / 100.0
            
            # Convert BGR to RGB (OpenCV uses BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preprocess frame
            input_tensor, original_frame = preprocess_frame(frame_rgb, device)
            
            # Run prediction with confidence threshold
            predictions, original_frame, segmentation_mask, labeled_overlay, confidence_viz, detected_ingredients = predict_segmentation_with_confidence(
                model, input_tensor, frame, num_classes, confidence_threshold, class_mapping
            )
            
            if predictions is None:
                print("Prediction failed, trying again...")
                continue
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Update FPS every second
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Add FPS and confidence threshold info to the display
            info_text = f"FPS: {fps:.1f} | Confidence Threshold: {confidence_threshold:.2f}"
            cv2.putText(labeled_overlay, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Create a 2x2 grid display
            h, w = labeled_overlay.shape[:2]
            display = np.zeros((h*2, w*2, 3), dtype=np.uint8)
            
            # Place the images in the grid
            display[:h, :w] = original_frame  # Top-left: Original
            display[:h, w:w*2] = segmentation_mask  # Top-right: Segmentation
            display[h:h*2, :w] = labeled_overlay  # Bottom-left: Labeled Overlay
            display[h:h*2, w:w*2] = confidence_viz  # Bottom-right: Confidence
            
            # Add labels
            cv2.putText(display, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Segmentation", (w+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Labeled Overlay", (10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Confidence", (w+10, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the combined display
            cv2.imshow('Segmentation Results', display)
            
            # Create and show the detected ingredients list
            ingredients_display = np.zeros((600, 400, 3), dtype=np.uint8)
            cv2.putText(ingredients_display, "DETECTED INGREDIENTS:", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if detected_ingredients:
                for i, (class_id, name, percentage) in enumerate(detected_ingredients):
                    # Get a color for this ingredient from the colormap
                    color_array = plt.cm.get_cmap('viridis', num_classes)(class_id)[:3]
                    color = tuple(int(c * 255) for c in color_array)
                    # OpenCV uses BGR
                    color = (color[2], color[1], color[0])
                    
                    text = f"{name}: {percentage:.1f}%"
                    y_pos = 80 + (i * 40)
                    
                    # Add ingredient name and percentage
                    cv2.putText(ingredients_display, text, (20, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Print ingredients to console as well
                ingredient_names = [f"{name} ({percentage:.1f}%)" for _, name, percentage in detected_ingredients]
                print(f"Detected ingredients: {', '.join(ingredient_names)}")
            else:
                cv2.putText(ingredients_display, "No ingredients detected", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
                print("No ingredients detected")
            
            cv2.imshow('Detected Ingredients', ingredients_display)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save current frame and results
                save_path = f"segmentation_result_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(save_path, display)
                
                # Also save ingredients list
                ingredients_path = f"ingredients_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(ingredients_path, ingredients_display)
                
                print(f"Saved results to {save_path} and {ingredients_path}")
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in live segmentation: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Camera resources released")

def process_single_image(model_path, image_path, confidence_threshold=0.5, output_path=None):
    """Process a single image file for debugging or testing"""
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        model, device, num_classes = load_model(model_path)
        print(f"Model loaded successfully. Using device: {device}")
        
        # Create class mapping for ingredient names
        class_mapping = get_class_mapping(num_classes)
        print(f"Created class mapping with {len(class_mapping)} ingredients")
        
    except Exception as e:
        print(f"Critical error loading model: {e}")
        return
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Warning: Image file {image_path} not found!")
        # Try alternative filenames
        alternatives = ["apples.jpeg", "apple.jpg", "apples.jpg", "food.jpg", "ingredients.jpg"]
        for alt in alternatives:
            if os.path.exists(alt):
                image_path = alt
                print(f"Found alternative image: {image_path}")
                break
        else:
            print("No valid image found!")
            return
    
    # Load and preprocess image
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
            
        # Convert BGR to RGB (OpenCV loads as BGR, but we need RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        input_tensor, original_image = preprocess_frame(image_rgb, device)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return
    
    # Run prediction
    print("Running segmentation...")
    predictions, original_image, segmentation_mask, labeled_overlay, confidence_viz, detected_ingredients = predict_segmentation_with_confidence(
        model, input_tensor, image, num_classes, confidence_threshold, class_mapping
    )
    
    if predictions is None:
        print("Prediction failed")
        return
    
    # Print detected ingredients
    if detected_ingredients:
        print("\nDetected ingredients:")
        for class_id, name, percentage in detected_ingredients:
            print(f"  - {name}: {percentage:.1f}%")
    else:
        print("No ingredients detected")
    
    # Create a 2x2 grid display
    h, w = labeled_overlay.shape[:2]
    display = np.zeros((h*2, w*2, 3), dtype=np.uint8)
    
    # Place the images in the grid
    display[:h, :w] = original_image  # Top-left: Original
    display[:h, w:w*2] = segmentation_mask  # Top-right: