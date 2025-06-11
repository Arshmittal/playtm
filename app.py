import os
import base64
import cv2
import numpy as np
import tempfile
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app, supports_credentials=True)  # Enable credentials for cross-origin requests

# Directories for storing images and training data
UPLOAD_DIR = "/tmp/game_photos"
TRAINING_DIR = "/tmp/training_data"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)

# File to store user stats
STATS_FILE = os.path.join(TRAINING_DIR, "user_stats.json")
# Load or create user stats
def load_user_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_stats(stats):
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f)

user_stats = load_user_stats()

# Generate a unique session ID for anonymous users
def get_session_id():
    if 'user_id' not in session:
        session['user_id'] = f"anonymous_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}"
    return session['user_id']



def process_image(image_data, return_image=False, detect_lines=False):
    """Process image to detect real 3D choley objects (not drawings)"""
    try:
        # Handle both full base64 string and raw base64 data
        if isinstance(image_data, str) and ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Create a copy for visualization
        vis_img = img.copy()
        original_img = img.copy()
        
        # Convert to different color spaces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        # Method 1: Shadow Detection (key for 3D objects)
        # Real choley cast shadows on white background
        # Detect shadows by finding darker regions
        shadow_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        shadow_mask = cv2.bitwise_not(shadow_thresh)  # Invert to get dark regions
        
        # Method 2: Gradient/Edge detection for 3D objects
        # Real objects have gradual color transitions, not sharp edges
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)
        
        # Create gradient mask for moderate gradients (not sharp pen lines)
        gradient_mask = cv2.inRange(gradient_magnitude, 20, 100)  # Moderate gradients
        
        # Method 3: Color variation detection
        # Real choley have color variations, not uniform colors like drawings
        
        # Split BGR channels
        b, g, r = cv2.split(img)
        
        # Calculate color variation (standard deviation in local regions)
        kernel = np.ones((9, 9), np.float32) / 81
        b_var = cv2.filter2D(b.astype(np.float32), -1, kernel)
        g_var = cv2.filter2D(g.astype(np.float32), -1, kernel)
        r_var = cv2.filter2D(r.astype(np.float32), -1, kernel)
        
        # Areas with color variation (real objects) vs uniform areas (drawings/background)
        color_std = np.std(np.stack([b, g, r], axis=-1), axis=-1)
        color_variation_mask = (color_std > 5).astype(np.uint8) * 255
        
        # Method 4: Texture analysis using Local Binary Pattern concept
        # Real choley have texture, drawings are smooth
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        texture_diff = cv2.absdiff(gray, blurred)
        texture_mask = cv2.threshold(texture_diff, 3, 255, cv2.THRESH_BINARY)[1]
        
        # Method 5: Brightness variation (3D objects have highlights and shadows)
        # Calculate local brightness statistics
        kernel_large = np.ones((15, 15), np.float32) / 225
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel_large)
        brightness_diff = cv2.absdiff(gray.astype(np.float32), local_mean)
        brightness_mask = (brightness_diff > 8).astype(np.uint8) * 255
        
        # Combine all methods with weights
        # Focus on shadow detection and color variation (strongest indicators of 3D objects)
        combined_mask = np.zeros_like(gray)
        
        # Add shadow information (most important)
        combined_mask = cv2.add(combined_mask, shadow_mask * 0.4)
        
        # Add color variation (second most important)
        combined_mask = cv2.add(combined_mask, color_variation_mask * 0.3)
        
        # Add brightness variation
        combined_mask = cv2.add(combined_mask, brightness_mask * 0.2)
        
        # Add texture information
        combined_mask = cv2.add(combined_mask, texture_mask * 0.1)
        
        # Convert to binary
        combined_mask = (combined_mask > 30).astype(np.uint8) * 255
        
        # Morphological operations to clean up and connect nearby regions
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remove small noise
        cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        # Connect nearby regions
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        # Fill holes
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # Distance transform to find centers of objects
        dist_transform = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)
        
        # Find local maxima (centers of choley)
        local_maxima = cv2.dilate(dist_transform, kernel_small)
        local_maxima = (dist_transform == local_maxima) & (dist_transform > 5)
        
        # Find contours from the cleaned mask
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Parameters for real choley (more lenient for 3D objects)
        min_area = 80      # Minimum area for choley
        max_area = 3000    # Maximum area for choley
        min_circularity = 0.2  # Very flexible for 3D objects
        max_circularity = 1.0
        
        # Filter contours
        seed_count = 0
        valid_contours = []
        centers = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Calculate basic shape properties
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                    
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Calculate compactness
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Check if this region has the characteristics of a 3D object
                # Extract the region for analysis
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [contour], 255)
                region_mean = cv2.mean(gray, mask=mask)[0]
                
                # Real choley should have:
                # 1. Reasonable shape properties
                # 2. Not be too dark (like pen marks) or too bright (like paper)
                # 3. Have some color/brightness variation
                
                if (circularity > min_circularity and
                    0.3 < aspect_ratio < 3.0 and  # Not too elongated
                    solidity > 0.6 and  # Reasonably solid shape
                    80 < region_mean < 220):  # Not pure black (pen) or pure white (paper)
                    
                    seed_count += 1
                    valid_contours.append(contour)
                    
                    # Calculate center
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = x + w // 2, y + h // 2
                    
                    centers.append((cX, cY))
        
        # Initialize line detection results
        lines_detected = []
        line_counts = []
        
        # Detect lines if requested and we have enough objects
        if detect_lines and len(centers) >= 3:
            lines_detected, line_counts = detect_seed_lines(centers, vis_img)
        
        # Draw detection results on visualization image
        for i, contour in enumerate(valid_contours):
            # Draw contour
            cv2.drawContours(vis_img, [contour], 0, (0, 255, 0), 2)
            
            # Get center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(contour)
                cX, cY = x + w // 2, y + h // 2
            
            # Draw center point
            cv2.circle(vis_img, (cX, cY), 4, (255, 0, 0), -1)
            
            # Draw number label
            cv2.putText(
                vis_img, 
                str(i + 1),
                (cX - 10, cY - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )
        
        # Add count to image
        cv2.putText(vis_img, f"Choley detected: {seed_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Debug: Show what the algorithm is detecting
        if seed_count == 0:
            cv2.putText(vis_img, "Try: Better lighting, cleaner background", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        if return_image:
            # Convert the visualization image to base64
            _, buffer = cv2.imencode('.jpg', vis_img)
            vis_img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            if detect_lines:
                return seed_count, vis_img_b64, lines_detected, line_counts
            return seed_count, vis_img_b64
        
        if detect_lines:
            return seed_count, lines_detected, line_counts
        return seed_count
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        if return_image:
            if detect_lines:
                return 0, None, [], []
            return 0, None
        if detect_lines:
            return 0, [], []
        return 0


def detect_seed_lines(centers, vis_img=None):
    """
    Detect lines of seeds from their center points
    """
    if len(centers) < 3:
        return [], []
    
    # Convert centers to numpy array for easier processing
    points = np.array(centers)
    
    # Parameters for line detection
    max_distance = 80  # Maximum distance between choley
    min_points = 3     # Minimum number of points to form a line
    line_threshold = 0.15  # Threshold for line detection
    
    # Store detected lines
    lines_detected = []
    line_counts = []
    
    # Keep track of points that have been assigned to lines
    assigned_points = set()
    
    # Try to find lines starting from each point
    for i, point in enumerate(points):
        if i in assigned_points:
            continue
        
        # Find all points within max_distance of this point
        distances = np.sqrt(np.sum((points - point)**2, axis=1))
        nearby_indices = np.where(distances < max_distance)[0]
        
        if len(nearby_indices) < min_points:
            continue
        
        # Try to fit a line to these points
        nearby_points = points[nearby_indices]
        
        # Use RANSAC to find the best line
        best_line = None
        best_inliers = []
        best_score = 0
        
        # Try multiple random pairs of points to find the best line
        for _ in range(15):  # More iterations for better results
            if len(nearby_indices) < 2:
                break
                
            # Select two random points to define a line
            idx1, idx2 = np.random.choice(len(nearby_indices), 2, replace=False)
            p1 = nearby_points[idx1]
            p2 = nearby_points[idx2]
            
            # Skip if points are too close
            if np.sqrt(np.sum((p1 - p2)**2)) < 15:
                continue
            
            # Calculate line parameters (ax + by + c = 0)
            a = p2[1] - p1[1]
            b = p1[0] - p2[0]
            c = p2[0]*p1[1] - p1[0]*p2[1]
            norm = np.sqrt(a*a + b*b)
            
            if norm < 1e-10:  # Avoid division by zero
                continue
                
            # Normalize
            a, b, c = a/norm, b/norm, c/norm
            
            # Calculate distances from all nearby points to this line
            distances_to_line = np.abs(a*nearby_points[:,0] + b*nearby_points[:,1] + c)
            
            # Count inliers (points close to the line)
            inliers = np.where(distances_to_line < line_threshold * max_distance)[0]
            
            if len(inliers) > best_score:
                best_score = len(inliers)
                best_inliers = inliers
                best_line = (a, b, c)
        
        # If we found a good line with enough points
        if best_line is not None and len(best_inliers) >= min_points:
            # Get the actual points in the line
            line_points = [nearby_indices[i] for i in best_inliers]
            
            # Add these points to assigned_points
            assigned_points.update(line_points)
            
            # Sort points along the line
            a, b, c = best_line
            if abs(b) > abs(a):  # Line is more horizontal
                sorted_indices = np.argsort([points[i][0] for i in line_points])
            else:  # Line is more vertical
                sorted_indices = np.argsort([points[i][1] for i in line_points])
            
            sorted_line_points = [line_points[i] for i in sorted_indices]
            line_centers = [centers[i] for i in sorted_line_points]
            
            # Add to detected lines
            lines_detected.append(line_centers)
            line_counts.append(len(line_centers))
            
            # Draw the line on the visualization image if provided
            if vis_img is not None:
                # Draw line connecting the points
                for j in range(len(line_centers) - 1):
                    pt1 = line_centers[j]
                    pt2 = line_centers[j + 1]
                    cv2.line(vis_img, pt1, pt2, (0, 0, 255), 3)
                
                # Draw a label for the line
                if len(line_centers) > 0:
                    mid_point = line_centers[len(line_centers) // 2]
                    cv2.putText(
                        vis_img,
                        f"Line {len(lines_detected)}: {len(line_centers)} choley",
                        (mid_point[0] - 50, mid_point[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        2
                    )
    
    return lines_detected, line_counts

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    """API status endpoint"""
    return jsonify({
        "status": "online",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    })

# Training endpoint removed

@app.route("/api/user-stats", methods=["GET"])
def get_user_stats():
    """Get user's training statistics"""
    user_id = get_session_id()
    stats = user_stats.get(user_id, {})
    return jsonify(stats)

@app.route("/api/start-game", methods=["POST"])
def start_game():
    """Handle start game photo submission"""
    data = request.json
    image_data = data.get("image")
    
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_id = get_session_id()
    filename = f"{user_id}_start_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Save start game image
        if isinstance(image_data, str) and ',' in image_data:
            img_data = base64.b64decode(image_data.split(',')[1])
        else:
            img_data = base64.b64decode(image_data)
            
        with open(filepath, "wb") as f:
            f.write(img_data)
        
        session['game_start_photo'] = filepath
        session['game_start_time'] = datetime.now().timestamp()

        # Process the image to get the starting almond count and annotated image
        start_count, annotated_image = process_image(image_data, return_image=True)
        session['start_count'] = start_count

        return jsonify({
            "message": "Game started",
            "start_time": session['game_start_time'],
            "start_count": start_count,
            "annotated_image": annotated_image
        })
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

@app.route("/api/end-game", methods=["POST"])
def end_game():
    """Handle end game photo submission and calculate score"""
    data = request.json
    image_data = data.get("image")
    
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    
    if 'game_start_photo' not in session:
        return jsonify({"error": "No start game photo found"}), 400
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_id = get_session_id()
    filename = f"{user_id}_end_{timestamp}.jpg"
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    try:
        # Save end game image
        if isinstance(image_data, str) and ',' in image_data:
            img_data = base64.b64decode(image_data.split(',')[1])
        else:
            img_data = base64.b64decode(image_data)
            
        with open(filepath, "wb") as f:
            f.write(img_data)
        
        # Get the start count from session or reprocess the start image
        start_count = session.get('start_count')
        start_annotated_image = None
        
        if start_count is None:
            with open(session['game_start_photo'], 'rb') as f:
                start_image = base64.b64encode(f.read()).decode('utf-8')
            start_count, start_annotated_image = process_image(start_image, return_image=True)
        else:
            # Get annotated start image
            with open(session['game_start_photo'], 'rb') as f:
                start_image = base64.b64encode(f.read()).decode('utf-8')
            _, start_annotated_image = process_image(start_image, return_image=True)
        
        # Process end image to count almonds, detect lines, and get annotated image
        end_count, end_annotated_image, lines_detected, line_counts = process_image(
            image_data, return_image=True, detect_lines=True
        )
        
        # Count lines with exactly 10 choley (perfect lines)
        perfect_lines = sum(1 for count in line_counts if count == 10)
        
        # Count lines with at least 3 choley (minimum for a line)
        total_lines = len(line_counts)
        
        # Calculate straightness penalty for each line
        # This is based on the maximum deviation of any point from the line
        line_straightness = []
        straightness_penalty = 0
        
        if lines_detected:
            for line_points in lines_detected:
                if len(line_points) >= 3:
                    # Calculate line parameters using first and last point
                    x1, y1 = line_points[0]
                    x2, y2 = line_points[-1]
                    
                    # Avoid division by zero
                    if x2 - x1 == 0:
                        # Vertical line
                        max_deviation = max(abs(x - x1) for x, y in line_points)
                    else:
                        # Calculate line equation: y = mx + b
                        m = (y2 - y1) / (x2 - x1)
                        b = y1 - m * x1
                        
                        # Calculate maximum deviation from line
                        max_deviation = max(abs(y - (m * x + b)) for x, y in line_points)
                    
                    # Normalize deviation by line length
                    line_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
                    normalized_deviation = max_deviation / max(line_length, 1)
                    
                    # Add penalty for non-straight lines
                    if normalized_deviation > 0.05:  # Threshold for "straight enough"
                        straightness_penalty += int(normalized_deviation * 100)
                    
                    line_straightness.append(normalized_deviation)
        
        # Calculate score based on:
        # 1. Number of perfect lines (10 choley) - 15 points each (increased reward)
        # 2. Number of other valid lines - 3 points each (decreased reward)
        # 3. Total choley count - 1 point each
        # 4. Penalty for non-straight lines
        
        perfect_line_score = perfect_lines * 15
        other_line_score = (total_lines - perfect_lines) * 3
        seed_score = end_count
        
        # Apply straightness penalty
        straightness_penalty = min(straightness_penalty, perfect_line_score + other_line_score)  # Cap penalty
        
        total_score = perfect_line_score + other_line_score + seed_score - straightness_penalty
        
        # Store game results
        user_stats.setdefault(user_id, {}).setdefault('game_history', []).append({
            'timestamp': datetime.now().isoformat(),
            'start_count': start_count,
            'end_count': end_count,
            'perfect_lines': perfect_lines,
            'total_lines': total_lines,
            'straightness_penalty': straightness_penalty,
            'score': total_score
        })
        save_user_stats(user_stats)
        
        return jsonify({
            "message": "Game completed",
            "start_count": start_count,
            "end_count": end_count,
            "perfect_lines": perfect_lines,
            "total_lines": total_lines,
            "line_counts": line_counts,
            "straightness_penalty": straightness_penalty,
            "score": total_score,
            "start_annotated_image": start_annotated_image,
            "end_annotated_image": end_annotated_image
        })
    except Exception as e:
        print(f"Error in end_game: {str(e)}")
        return jsonify({"error": f"Failed to process game: {str(e)}"}), 500

@app.route("/api/detect", methods=["POST"])
def detect_seeds():
    """Simple endpoint to detect seeds in an image"""
    data = request.json
    image_data = data.get("image")
    
    if not image_data:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        # Process the image to count choley and get annotated image
        choley_count, annotated_image = process_image(image_data, return_image=True)
        
        return jsonify({
            "count": choley_count,
            "annotated_image": annotated_image
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
