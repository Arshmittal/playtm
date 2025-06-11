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



import cv2
import numpy as np
import base64

def process_image(image_data, return_image=False, detect_lines=False):
    """Simplified and more reliable choley detection"""
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
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Simple thresholding with multiple approaches
        height, width = gray.shape
        
        # Calculate background color (assume corners are background)
        corners = [
            gray[0:20, 0:20],           # Top-left
            gray[0:20, width-20:width], # Top-right
            gray[height-20:height, 0:20], # Bottom-left
            gray[height-20:height, width-20:width] # Bottom-right
        ]
        background_values = [np.mean(corner) for corner in corners]
        background_mean = np.mean(background_values)
        
        print(f"Background mean: {background_mean}")
        
        # Approach 1: Simple threshold based on background
        # Choley are typically darker than white background
        if background_mean > 200:  # White background
            threshold_value = background_mean - 30  # Detect anything darker than background
            _, thresh1 = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        else:  # Darker background
            threshold_value = background_mean + 20
            _, thresh1 = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        
        # Approach 2: Adaptive thresholding
        thresh2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 10
        )
        
        # Approach 3: Otsu's thresholding
        _, thresh3 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine thresholding methods
        combined_thresh = cv2.bitwise_or(thresh1, thresh2)
        combined_thresh = cv2.bitwise_or(combined_thresh, thresh3)
        
        # Apply light smoothing to reduce noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_OPEN, kernel_small)
        
        # Apply closing to fill small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_thresh = cv2.morphologyEx(combined_thresh, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(combined_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"Found {len(contours)} contours")
        
        # More lenient parameters for choley detection
        min_area = 30       # Smaller minimum area
        max_area = 5000     # Larger maximum area
        min_width = 8       # Minimum width
        max_width = 100     # Maximum width
        min_height = 8      # Minimum height
        max_height = 100    # Maximum height
        
        # Filter contours by size and basic shape
        seed_count = 0
        valid_contours = []
        centers = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < min_area or area > max_area:
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check size constraints
            if (w < min_width or w > max_width or 
                h < min_height or h > max_height):
                continue
            
            # Check aspect ratio (not too elongated)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5.0:
                continue
            
            # Calculate some basic shape properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Calculate circularity (4π×Area)/(perimeter²)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Very lenient circularity check
            if circularity < 0.1:  # Very low threshold
                continue
            
            # Check if the contour is not just noise
            # Look at the intensity values in the region
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            
            # Get mean intensity of the region
            region_mean = cv2.mean(gray, mask=mask)[0]
            
            # Skip if region is too similar to background (likely noise)
            intensity_diff = abs(region_mean - background_mean)
            if intensity_diff < 5:  # Very small difference from background
                continue
            
            # This looks like a valid choley
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
            
            print(f"Choley {seed_count}: area={area:.1f}, circularity={circularity:.3f}, intensity_diff={intensity_diff:.1f}")
        
        # Initialize line detection results
        lines_detected = []
        line_counts = []
        
        # Detect lines if requested and we have enough objects
        if detect_lines and len(centers) >= 3:
            lines_detected, line_counts = detect_seed_lines(centers, vis_img)
        
        # Draw detection results
        for i, contour in enumerate(valid_contours):
            # Draw contour in green
            cv2.drawContours(vis_img, [contour], 0, (0, 255, 0), 2)
            
            # Get center and bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            cX, cY = x + w // 2, y + h // 2
            
            # Draw center point in blue
            cv2.circle(vis_img, (cX, cY), 3, (255, 0, 0), -1)
            
            # Draw number label
            cv2.putText(
                vis_img, 
                str(i + 1),
                (cX - 8, cY + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )
            
            # Draw bounding rectangle for debugging
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 255), 1)
        
        # Add detection info to image
        cv2.putText(vis_img, f"Choley detected: {seed_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.putText(vis_img, f"Background: {background_mean:.1f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Debug information
        if seed_count == 0:
            cv2.putText(vis_img, "No choley detected. Try:", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(vis_img, "- Better lighting", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(vis_img, "- Cleaner background", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(vis_img, "- Spread choley apart", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        print(f"Final detection: {seed_count} choley found")
        
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
        import traceback
        traceback.print_exc()
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
