import os
import io
import threading
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pillow_heif import register_heif_opener
from flask import Flask, render_template_string, request, jsonify, redirect, url_for, send_file

register_heif_opener()

# Configuration
BASE_PATH = "physical_world_test/full test/organized"
X_COORD = 0   # Straight on
Y_COORD = 5   # Closest available distance
LIGHTING_CONDITIONS = ['full sun', 'dusk', 'dark no flash', 'dark flash']
PORT = 8090

app = Flask(__name__)

# Global state
images = {}
image_paths = {}
corners_map = {}
done_event = threading.Event()


def lighting_to_key(l):
    return l.replace(' ', '_')

def key_to_lighting(k):
    return k.replace('_', ' ')

def find_image_for_condition(lighting, x, y, condition='plate2'):
    possible_names = [
        f"x_{x:+03d}_y_{y:02d}.heic",
        f"x_{x:+03d}_y_{y:02d}.jpg",
        f"x_{x:+03d}_y_{y:02d}.png",
    ]
    base_dir = os.path.join(BASE_PATH, lighting, condition, f"y_{y:02d}")
    for name in possible_names:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    print(f"Warning: Could not find image for {lighting}, x={x}, y={y}")
    return None

def load_and_convert_image(image_path):
    try:
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def blur_region(image, corners):
    if len(corners) != 4:
        return image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    pts = np.array(corners, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    blurred = cv2.GaussianBlur(image, (199, 199), 60)
    result = image.copy()
    result[mask == 255] = blurred[mask == 255]
    return result

def save_grid(imgs_dict, filename):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes = axes.flatten()
    for idx, lighting in enumerate(LIGHTING_CONDITIONS):
        ax = axes[idx]
        if lighting in imgs_dict:
            ax.imshow(imgs_dict[lighting])
            ax.set_title(lighting.title(), fontsize=16, fontweight='bold')
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'{lighting.title()}\n(Not available)',
                    ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
    plt.suptitle('Sample Images from Each Lighting Condition', fontsize=24, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filename}")


# ---------- Flask routes ----------

IMAGE_PAGE = """<!DOCTYPE html>
<html>
<head>
  <title>Corner Selection ({{ current }}/{{ total }})</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif;
           display: flex; flex-direction: column; align-items: center; padding: 16px; }
    h2 { margin: 8px 0 4px; }
    p  { margin: 4px 0 10px; color: #aaa; font-size: 14px; }
    #status { font-size: 16px; margin-bottom: 8px; }
    canvas { cursor: crosshair; border: 2px solid #444; max-width: 90vw; max-height: 70vh; display: block; }
    .btn { padding: 10px 24px; margin: 8px 6px 0; font-size: 15px; cursor: pointer;
           border: none; border-radius: 4px; }
    #resetBtn  { background: #555; color: #eee; }
    #submitBtn { background: #2a7; color: #fff; }
    #submitBtn:disabled { background: #444; color: #888; cursor: default; }
  </style>
</head>
<body>
  <h2>{{ lighting.title() }} &mdash; image {{ current }} of {{ total }}</h2>
  <p>Click the 4 corners of the license plate: <strong>top-left → top-right → bottom-right → bottom-left</strong></p>
  <div id="status">Corners marked: 0 / 4</div>
  <canvas id="canvas"></canvas>
  <div>
    <button class="btn" id="resetBtn"  onclick="resetCorners()">Reset</button>
    <button class="btn" id="submitBtn" onclick="submitCorners()" disabled>Next →</button>
  </div>

  <script>
    const canvas = document.getElementById('canvas');
    const ctx    = canvas.getContext('2d');
    const img    = new Image();
    let corners  = [];
    let scaleX = 1, scaleY = 1;

    img.onload = function() {
      const maxW = window.innerWidth  * 0.88;
      const maxH = window.innerHeight * 0.68;
      const s    = Math.min(maxW / img.width, maxH / img.height, 1);
      canvas.width  = img.width  * s;
      canvas.height = img.height * s;
      scaleX = s; scaleY = s;
      redraw();
    };
    img.src = '/img/{{ key }}';

    function redraw() {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      corners.forEach(([x, y], i) => {
        const cx = x * scaleX, cy = y * scaleY;
        ctx.beginPath();
        ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
        ctx.fillStyle   = 'red';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth   = 2;
        ctx.stroke();
        ctx.fillStyle   = 'white';
        ctx.font        = 'bold 14px Arial';
        ctx.fillText(i + 1, cx + 8, cy - 6);
      });
      if (corners.length === 4) {
        ctx.beginPath();
        ctx.moveTo(corners[0][0]*scaleX, corners[0][1]*scaleY);
        for (let i = 1; i < 4; i++) ctx.lineTo(corners[i][0]*scaleX, corners[i][1]*scaleY);
        ctx.closePath();
        ctx.strokeStyle = 'red';
        ctx.lineWidth   = 2;
        ctx.stroke();
      }
    }

    canvas.addEventListener('click', function(e) {
      if (corners.length >= 4) return;
      const rect = canvas.getBoundingClientRect();
      const x = Math.round((e.clientX - rect.left)  / scaleX);
      const y = Math.round((e.clientY - rect.top)   / scaleY);
      corners.push([x, y]);
      document.getElementById('status').textContent = `Corners marked: ${corners.length} / 4`;
      redraw();
      if (corners.length === 4) document.getElementById('submitBtn').disabled = false;
    });

    function resetCorners() {
      corners = [];
      document.getElementById('status').textContent = 'Corners marked: 0 / 4';
      document.getElementById('submitBtn').disabled = true;
      redraw();
    }

    function submitCorners() {
      fetch('/corners/{{ key }}', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({corners: corners})
      }).then(r => r.json()).then(data => {
        if (data.next) window.location.href = '/image/' + data.next;
        else           window.location.href = '/done';
      });
    }
  </script>
</body>
</html>"""

DONE_PAGE = """<!DOCTYPE html>
<html>
<head><title>Done</title>
<style>body{background:#111;color:#eee;font-family:Arial;display:flex;
  flex-direction:column;align-items:center;justify-content:center;height:100vh;margin:0;}
h2{color:#2a7;}</style>
</head>
<body>
  <h2>✅ All corners submitted!</h2>
  <p>The server is now applying blurs and saving the final grid.</p>
  <p>You can close this tab.</p>
</body>
</html>"""


@app.route('/')
def index():
    for l in LIGHTING_CONDITIONS:
        if l in images and l not in corners_map:
            return redirect(url_for('show_image', key=lighting_to_key(l)))
    return redirect(url_for('done'))

@app.route('/image/<key>')
def show_image(key):
    lighting = key_to_lighting(key)
    if lighting not in images:
        return f"Image not available for '{lighting}'", 404
    idx = LIGHTING_CONDITIONS.index(lighting)
    available = [l for l in LIGHTING_CONDITIONS if l in images]
    return render_template_string(IMAGE_PAGE,
                                  lighting=lighting,
                                  key=key,
                                  current=available.index(lighting) + 1,
                                  total=len(available))

@app.route('/img/<key>')
def get_image(key):
    lighting = key_to_lighting(key)
    if lighting not in images:
        return "Not found", 404
    pil_img = Image.fromarray(images[lighting])
    buf = io.BytesIO()
    pil_img.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')

@app.route('/corners/<key>', methods=['POST'])
def submit_corners(key):
    lighting = key_to_lighting(key)
    data = request.get_json()
    corners_map[lighting] = [tuple(c) for c in data['corners']]
    print(f"✓ Corners received for {lighting}: {corners_map[lighting]}")
    for l in LIGHTING_CONDITIONS:
        if l in images and l not in corners_map:
            return jsonify({'next': lighting_to_key(l)})
    return jsonify({'next': None})

@app.route('/done')
def done():
    done_event.set()
    return render_template_string(DONE_PAGE)


# ---------- Main ----------

if __name__ == '__main__':
    print("Loading images from each lighting condition...")
    for lighting in LIGHTING_CONDITIONS:
        img_path = find_image_for_condition(lighting, X_COORD, Y_COORD)
        if img_path:
            img = load_and_convert_image(img_path)
            if img is not None:
                h = img.shape[0]
                img = img[int(h * 0.2):int(h * 0.8), :]
                images[lighting] = img
                image_paths[lighting] = img_path
                print(f"✓ Loaded: {lighting}")

    if not images:
        print("Error: No images could be loaded!")
        exit(1)

    save_grid(images, 'sample_images_grid_unblurred.png')

    print(f"\n{'='*60}")
    print(f"Web UI ready — open in your browser:")
    print(f"  http://localhost:{PORT}")
    print(f"If SSH tunnelling:  ssh -L {PORT}:localhost:{PORT} <your-server>")
    print(f"{'='*60}\n")

    t = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False), daemon=True)
    t.start()

    done_event.wait()
    print("\nAll corners received — applying blurs...")

    blurred_images = {}
    for lighting, img in images.items():
        if lighting in corners_map:
            blurred_images[lighting] = blur_region(img.copy(), corners_map[lighting])
            print(f"✓ Blurred: {lighting}")
        else:
            print(f"⚠ Skipped blur for {lighting} (no corners submitted)")
            blurred_images[lighting] = img

    save_grid(blurred_images, 'sample_images_grid.png')
    print("Done!")
