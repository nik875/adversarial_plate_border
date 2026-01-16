import Levenshtein
from fast_alpr import ALPR

# 1. Initialize the ALPR engine
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model"
)


def find_target_in_image(image_path, target_plate="VRJ7774"):
    # 2. Get all predictions from the image
    results = alpr.predict(image_path)

    best_match = None
    min_dist = float('inf')

    # 3. Iterate through detected plates to find the closest match
    for res in results:
        detected_text = res.text.upper().strip()

        # Calculate Levenshtein distance
        distance = Levenshtein.distance(detected_text, target_plate)

        # Update if this is the closest match so far
        if distance < min_dist:
            min_dist = distance
            best_match = res

        # Stop early if we find an exact match
        if distance == 0:
            break

    return best_match, min_dist


# Example Execution
image = "car_entry.jpg"
match, dist = find_target_in_image(image)

if match:
    print(f"Target: VRJ7774")
    print(f"Found: {match.text} | Distance: {dist}")
    if dist == 0:
        print("Status: EXACT MATCH FOUND")
    elif dist <= 2:
        print("Status: LIKELY MATCH (Fuzzy)")
    else:
        print("Status: NO CLOSE MATCH")
