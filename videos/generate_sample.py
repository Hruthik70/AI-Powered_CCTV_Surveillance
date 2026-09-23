import cv2
import numpy as np
import os
import random
import math

def generate_synthetic_cctv_video(output_path="videos/crowd.mp4", duration_sec=20, fps=25, width=960, height=540):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    num_pedestrians = 16
    
    # Initialize pedestrian states
    pedestrians = []
    for i in range(num_pedestrians):
        pedestrians.append({
            "id": i + 1,
            "x": random.uniform(50, width - 100),
            "y": random.uniform(100, height - 120),
            "vx": random.uniform(-2.5, 2.5),
            "vy": random.uniform(-1.5, 1.5),
            "color": (random.randint(180, 240), random.randint(180, 240), random.randint(180, 240)),
            "shirt": (random.randint(40, 200), random.randint(40, 200), random.randint(40, 200)),
            "size_scale": random.uniform(0.85, 1.15)
        })
        
    for frame_idx in range(total_frames):
        # Create CCTV background (indoor lobby floor tiles with lighting)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (45, 45, 48)  # Dark surveillance background
        
        # Draw floor tiles grid
        for x in range(0, width, 60):
            cv2.line(frame, (x, 0), (x, height), (55, 55, 60), 1)
        for y in range(0, height, 60):
            cv2.line(frame, (0, y), (width, y), (55, 55, 60), 1)
            
        # Draw architectural pillars / walls
        cv2.rectangle(frame, (0, 0), (width, 50), (30, 30, 32), -1)
        cv2.putText(frame, "BUILDING A - LOBBY ENTRANCE [CCTV CAM-01]", (20, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # In frame range 100-300, make pedestrians cluster in center to trigger Overcrowding
        is_crowd_phase = (100 <= frame_idx <= 350)
        
        for p in pedestrians:
            if is_crowd_phase:
                target_x, target_y = width / 2.0, height / 2.0
                p["vx"] += (target_x - p["x"]) * 0.003
                p["vy"] += (target_y - p["y"]) * 0.003
                # Max speed limit
                speed = math.hypot(p["vx"], p["vy"])
                if speed > 2.0:
                    p["vx"] = (p["vx"] / speed) * 2.0
                    p["vy"] = (p["vy"] / speed) * 2.0
            else:
                if random.random() < 0.04:
                    p["vx"] += random.uniform(-0.5, 0.5)
                    p["vy"] += random.uniform(-0.5, 0.5)
                    
            # Move towards restricted zone (top-right) for pedestrian 3 and 7
            if p["id"] in (3, 7) and frame_idx > 150:
                p["vx"] = 2.0
                p["vy"] = -1.0
                
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            
            # Boundary bounce
            if p["x"] < 50:
                p["x"] = 50
                p["vx"] *= -1
            elif p["x"] > width - 60:
                p["x"] = width - 60
                p["vx"] *= -1
                
            if p["y"] < 70:
                p["y"] = 70
                p["vy"] *= -1
            elif p["y"] > height - 70:
                p["y"] = height - 70
                p["vy"] *= -1
                
            # Draw synthetic person (head, body, limbs)
            px, py = int(p["x"]), int(p["y"])
            s = p["size_scale"]
            
            # Shadow
            cv2.ellipse(frame, (px, py + int(45 * s)), (int(16 * s), int(6 * s)), 0, 0, 360, (20, 20, 20), -1)
            # Legs
            cv2.line(frame, (px - 4, py + int(20 * s)), (px - 6, py + int(45 * s)), (30, 30, 30), int(3 * s))
            cv2.line(frame, (px + 4, py + int(20 * s)), (px + 6, py + int(45 * s)), (30, 30, 30), int(3 * s))
            # Torso
            cv2.rectangle(frame, (px - int(10 * s), py - int(10 * s)), (px + int(10 * s), py + int(25 * s)), p["shirt"], -1)
            # Head
            cv2.circle(frame, (px, py - int(20 * s)), int(9 * s), (220, 190, 170), -1)
            # Hair/Cap
            cv2.circle(frame, (px, py - int(23 * s)), int(8 * s), (40, 30, 20), -1)
            
        # Add realistic CCTV scanlines and noise
        noise = np.random.randint(0, 15, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Add CCTV timestamp overlay
        timestamp_str = f"REC ● 2026-09-22 17:42:{frame_idx//fps:02d}:{int((frame_idx%fps)*3.3):02d}"
        cv2.putText(frame, timestamp_str, (width - 320, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        
        out.write(frame)
        
    out.release()
    print(f"Synthesized realistic CCTV crowd footage at: {output_path}")

if __name__ == "__main__":
    generate_synthetic_cctv_video()
