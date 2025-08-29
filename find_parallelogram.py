import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    with open("edges.pkl", "rb") as f:
        edges = pickle.load(f)
    edges = np.max(edges, axis=2)
    print(edges.shape)

    # Get coordinates where edge was detected
    binary = edges != 0
    inds = np.transpose(np.indices(binary.shape), (1, 2, 0))
    inds = inds[binary]

    # Normalize coordinates
    x = inds[..., 1]
    y = -inds[..., 0]

    # Plot original edge points
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, marker='.', color='blue', alpha=0.6, s=1, label='Edge points')

    # Hough line detection
    # Convert binary edge image to uint8 for OpenCV
    edge_img = (binary * 255).astype(np.uint8)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(edge_img, 1, np.pi / 180, threshold=200)

    if lines is not None:
        print(f"Detected {len(lines)} lines")

        # Function to select 4 lines that best form a parallelogram
        def select_parallelogram_lines(lines, angle_tolerance=5):
            lines_data = []

            # Extract and normalize angles for grouping
            for line in lines:
                rho, theta = line[0]
                # Normalize angle to [0, π) range
                angle_deg = np.degrees(theta) % 180
                lines_data.append((rho, theta, angle_deg))

            # Group lines by similar angles
            from collections import defaultdict
            angle_groups = defaultdict(list)

            for i, (rho, theta, angle_deg) in enumerate(lines_data):
                # Find existing group or create new one
                assigned = False
                for group_angle in angle_groups.keys():
                    if abs(angle_deg - group_angle) < angle_tolerance or abs(angle_deg -
                                                                             group_angle) > (180 - angle_tolerance):
                        angle_groups[group_angle].append((i, rho, theta, angle_deg))
                        assigned = True
                        break

                if not assigned:
                    angle_groups[angle_deg].append((i, rho, theta, angle_deg))

            # Find the two largest groups (should be our parallel line pairs)
            sorted_groups = sorted(angle_groups.items(), key=lambda x: len(x[1]), reverse=True)

            if len(sorted_groups) < 2:
                print("Warning: Could not find two groups of parallel lines")
                return lines[:4] if len(lines) >= 4 else lines

            # Select best two lines from each of the top two groups
            selected_lines = []
            for _, group in sorted_groups[:2]:
                if len(group) >= 2:
                    # Sort by rho to get lines furthest apart
                    group_sorted = sorted(group, key=lambda x: x[1])  # Sort by rho
                    # Take the two most extreme rho values (furthest apart)
                    selected_lines.append(lines[group_sorted[0][0]])
                    selected_lines.append(lines[group_sorted[-1][0]])
                else:
                    selected_lines.append(lines[group[0][0]])

            print(f"Selected {len(selected_lines)} lines for parallelogram")
            return selected_lines

        # Select best 4 lines for parallelogram
        selected_lines = select_parallelogram_lines(lines)

        # Function to make parallel pairs truly parallel
        def make_truly_parallel(selected_lines):
            if len(selected_lines) != 4:
                return selected_lines

            lines_data = []
            for line in selected_lines:
                rho, theta = line[0]
                angle_deg = np.degrees(theta) % 180
                lines_data.append([rho, theta, angle_deg])

            # Group into two parallel pairs based on angle similarity
            lines_data = np.array(lines_data)
            angles = lines_data[:, 2]

            # Find two main angle groups
            angle_diffs = []
            for i in range(len(angles)):
                for j in range(i + 1, len(angles)):
                    diff = min(abs(angles[i] - angles[j]), 180 - abs(angles[i] - angles[j]))
                    angle_diffs.append((diff, i, j))

            # Sort by similarity and group
            angle_diffs.sort()

            # Find the best pairing that creates two groups of 2
            used = set()
            pairs = []
            for diff, i, j in angle_diffs:
                if i not in used and j not in used and len(pairs) < 2:
                    pairs.append([i, j])
                    used.add(i)
                    used.add(j)
                    if len(used) == 4:
                        break

            # If we couldn't pair all 4, just take first 4
            if len(pairs) != 2:
                pairs = [[0, 1], [2, 3]]

            corrected_lines = []

            for pair in pairs:
                i, j = pair
                # Get average angle for this parallel pair
                theta1, theta2 = lines_data[i, 1], lines_data[j, 1]

                # Handle angle wraparound at π
                if abs(theta1 - theta2) > np.pi / 2:
                    if theta1 < theta2:
                        theta1 += np.pi
                    else:
                        theta2 += np.pi

                avg_theta = (theta1 + theta2) / 2

                # Apply small equal and opposite corrections
                correction = 0.001  # Small angle correction in radians
                theta1_corrected = avg_theta + correction
                theta2_corrected = avg_theta - correction

                # Keep original rho values, update only theta
                corrected_lines.append([[lines_data[i, 0], theta1_corrected]])
                corrected_lines.append([[lines_data[j, 0], theta2_corrected]])

            print(f"Applied parallel corrections to {len(corrected_lines)} lines")
            return corrected_lines

        # Apply parallel corrections
        corrected_lines = make_truly_parallel(selected_lines)

        # Get image dimensions for line endpoint calculation
        h, w = edges.shape

        # Plot all detected lines in light red
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)

            if abs(b) > abs(a):
                x1, x2 = 0, w - 1
                y1 = (rho - a * x1) / b
                y2 = (rho - a * x2) / b
            else:
                y1, y2 = 0, h - 1
                x1 = (rho - b * y1) / a
                x2 = (rho - b * y2) / a

            y1_norm = -y1
            y2_norm = -y2
            plt.plot([x1, x2], [y1_norm, y2_norm], 'pink', linewidth=1, alpha=0.3)

        # Plot corrected parallelogram lines in bright red
        for line in corrected_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)

            if abs(b) > abs(a):
                x1, x2 = 0, w - 1
                y1 = (rho - a * x1) / b
                y2 = (rho - a * x2) / b
            else:
                y1, y2 = 0, h - 1
                x1 = (rho - b * y1) / a
                x2 = (rho - b * y2) / a

            y1_norm = -y1
            y2_norm = -y2
            plt.plot([x1, x2], [y1_norm, y2_norm], 'red', linewidth=3, alpha=0.9)
    else:
        print("No lines detected")

    plt.legend()
    plt.title('Edge Points with Hough Lines')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig("img.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
