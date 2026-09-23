import math
import numpy as np
from typing import Dict, Any, Tuple, List
from detection.tracker import TrackedObject

class CrowdEngine:
    """
    Phase 3 & 4: Crowd Counting, Density & Spatial Clustering Engine
    Categorizes crowd levels and identifies physical clusters / hotspots on the camera screen.
    """
    def __init__(self, crowd_threshold: int = 12, warning_threshold: int = 6, cluster_distance: float = 140.0):
        self.crowd_threshold = crowd_threshold
        self.warning_threshold = warning_threshold
        self.cluster_distance = cluster_distance

    def evaluate(self, person_count: int) -> Tuple[str, str, float]:
        """
        Global crowd status categorization.
        """
        if person_count >= self.crowd_threshold:
            ratio = min(1.0, person_count / float(self.crowd_threshold))
            return "OVERCROWDED", "HIGH", round(ratio, 2)
        elif person_count >= self.warning_threshold:
            return "MODERATE", "MEDIUM", 0.75
        else:
            return "NORMAL", "INFO", 0.50

    def find_crowd_clusters(self, tracks: List[TrackedObject], min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """
        Identifies spatial clusters/hotspots of people congregating together.
        Returns bounding boxes and track IDs for each dense group.
        """
        if len(tracks) < min_cluster_size:
            return []

        points = [t.current_position for t in tracks]
        n = len(points)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            
            # Breadth-first search / cluster expansion
            cluster_indices = [i]
            visited[i] = True
            queue = [i]

            while queue:
                curr = queue.pop(0)
                cx, cy = points[curr]

                for j in range(n):
                    if not visited[j]:
                        jx, jy = points[j]
                        dist = math.hypot(cx - jx, cy - jy)
                        if dist <= self.cluster_distance:
                            visited[j] = True
                            cluster_indices.append(j)
                            queue.append(j)

            if len(cluster_indices) >= min_cluster_size:
                # Calculate cluster bounding box with padding
                cluster_tracks = [tracks[idx] for idx in cluster_indices]
                min_x = min(t.bbox[0] for t in cluster_tracks) - 15
                min_y = min(t.bbox[1] for t in cluster_tracks) - 15
                max_x = max(t.bbox[2] for t in cluster_tracks) + 15
                max_y = max(t.bbox[3] for t in cluster_tracks) + 15

                clusters.append({
                    "bbox": [max(0, min_x), max(0, min_y), max_x, max_y],
                    "count": len(cluster_tracks),
                    "track_ids": [t.track_id for t in cluster_tracks],
                    "density_ratio": round(len(cluster_tracks) / float(self.crowd_threshold), 2)
                })

        return clusters
