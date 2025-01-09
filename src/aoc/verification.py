import math
import json

from collections import defaultdict

def calculate_centroid_and_radius(data):
    results = []
    
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        
        x_min, y_min = points[0]
        x_max, y_max = points[1]
        centroid_x = (x_min + x_max) / 2
        centroid_y = (y_min + y_max) / 2
        
        diagonal = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
        radius = diagonal / (2 * math.sqrt(2))
        
        results.append({
            'label': label,
            'centroid': (centroid_x, centroid_y),
            'radius': radius
        })
    
    return results

def find_shots_in_radius(data, shots):
    centroids_with_radii = calculate_centroid_and_radius(data)
    results = defaultdict(list)

    for entry in centroids_with_radii:
        label = entry['label']
        centroid_x, centroid_y = entry['centroid']
        radius = entry['radius']
        
        shots_in_radius = []
        for shot in shots:
            shot_label = shot['label']
            shot_x, shot_y = shot['coordinates']
            
            distance = math.sqrt((shot_x - centroid_x) ** 2 + (shot_y - centroid_y) ** 2)
            print(f"{distance=}, {radius=}")
            if distance <= radius and shot_label == label:
                shots_in_radius.append(shot)
        
        results[label].append(shots_in_radius)

    return results

with open('', 'r') as f:
    data = json.load(f)

# shots = [
#     {'label': "200", 'coordinates': (3375.0, 5140.0)}
# ]

results = find_shots_in_radius(data, shots)

for label, shots in results.items():
    print(f"Label: {label}, Shots in radius: {shots}")

