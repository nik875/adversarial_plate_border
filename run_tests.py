import csv
import re
from pathlib import Path
from plate_detector import LicensePlateDetector   # <-- rename if needed

# ================= CONFIG =================

DATASET_ROOT = Path("physical_world_test/full test/organized")
OUTPUT_CSV = "full_results_largedet.csv"

# Regex to extract x and y from filename
FILENAME_REGEX = re.compile(r"x_([+-]\d+)_y_(\d+)", re.IGNORECASE)

# =========================================
MAX_IMAGES = None   # set to None to process everything
PLATE_LABEL_MAP = {
    "control": "control",
    "plate1": "impersonation",  # VJJ7744 patch
    "plate2": "disruption",      # SHX8459 patch
}


def run_batch():
    detector = LicensePlateDetector()
    detector.initialize_alpr()

    rows = []
    processed = 0

    for time_dir in DATASET_ROOT.iterdir():
        if not time_dir.is_dir():
            continue
        time_of_day = time_dir.name

        for plate_dir in time_dir.iterdir():
            if not plate_dir.is_dir():
                continue
            plate_dir_name = plate_dir.name
            plate = PLATE_LABEL_MAP.get(plate_dir_name, plate_dir_name)

            for y_dir in plate_dir.iterdir():
                if not y_dir.is_dir():
                    continue

                y = int(y_dir.name.split("_")[1])

                for img_path in y_dir.iterdir():
                    if not img_path.is_file():
                        continue

                    if MAX_IMAGES is not None and processed >= MAX_IMAGES:
                        write_csv(rows)
                        print(f"\n🛑 Stopped early after {processed} images")
                        print(f"✅ Partial CSV written to {OUTPUT_CSV}")
                        return

                    match = FILENAME_REGEX.search(img_path.name)
                    if not match:
                        continue

                    x = int(match.group(1))

                    print(f"🔍 [{processed+1}] {time_of_day}/{plate}/{img_path.name}")

                    try:
                        all_plates, target_plates = detector.detect_target_plates(str(img_path))

                        if all_plates:
                            # pick highest-confidence plate overall
                            best_plate = max(all_plates, key=lambda p: p["confidence"])
                            detected_text = best_plate["text"]
                            detected_conf = best_plate["confidence"]
                        else:
                            detected_text = None
                            detected_conf = None

                        is_correct = detected_text == detector.target_plate

                        rows.append({
                            "time_of_day": time_of_day,
                            "condition": plate,                     # control / disruption / impersonation
                            "x": x,
                            "y": y,
                            "filename": img_path.name,
                            "any_plate_detected": len(all_plates) > 0,
                            "detected_plate_text": detected_text,
                            "detected_plate_confidence": detected_conf,
                            "is_correct_plate": is_correct,
                        })

                    except Exception as e:
                        print(f"❌ Error on {img_path.name}: {e}")
                        rows.append({
                            "time_of_day": time_of_day,
                            "plate": plate,
                            "x": x,
                            "y": y,
                            "filename": img_path.name,
                            "any_plate_detected": False,
                            "target_detected": False,
                            "target_confidence": None,
                        })

                    processed += 1

    write_csv(rows)
    print(f"\n✅ Finished full run ({processed} images). CSV saved.")


def write_csv(rows):
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time_of_day",
                "condition",
                "x",
                "y",
                "filename",
                "any_plate_detected",
                "detected_plate_text",
                "detected_plate_confidence",
                "is_correct_plate",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_batch()
