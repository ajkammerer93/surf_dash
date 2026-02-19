"""Generate an Instagram profile picture (1080x1080) from the wave favicon using pure Pillow.
Renders at 2x then downscales with LANCZOS for smooth anti-aliased lines."""
from PIL import Image, ImageDraw

SIZE = 1080
SUPERSAMPLE = 2  # render at 2x for AA
SS = SIZE * SUPERSAMPLE
VIEWBOX = 32
PADDING = 140 * SUPERSAMPLE
SCALE = (SS - 2 * PADDING) / VIEWBOX


def quadratic_bezier(p0, p1, p2, steps=80):
    """Sample a quadratic bezier curve into line segments."""
    points = []
    for i in range(steps + 1):
        t = i / steps
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        points.append((x, y))
    return points


def svg_to_canvas(points):
    return [(PADDING + x * SCALE, PADDING + y * SCALE) for x, y in points]


# Wave path 1: M2,20 Q8,10 14,18 Q20,26 26,16 Q29,11 30,12
wave1_pts = svg_to_canvas(
    quadratic_bezier((2, 20), (8, 10), (14, 18)) +
    quadratic_bezier((14, 18), (20, 26), (26, 16)) +
    quadratic_bezier((26, 16), (29, 11), (30, 12))
)

# Wave path 2: M2,24 Q8,16 14,22 Q20,28 26,20 Q29,16 30,17
wave2_pts = svg_to_canvas(
    quadratic_bezier((2, 24), (8, 16), (14, 22)) +
    quadratic_bezier((14, 22), (20, 28), (26, 20)) +
    quadratic_bezier((26, 20), (29, 16), (30, 17))
)

# Draw at supersampled size
img = Image.new("RGB", (SS, SS), (0, 0, 0))
draw = ImageDraw.Draw(img)

w1 = round(3 * SCALE)
w2 = round(2.5 * SCALE)

# Draw thick lines by stamping circles along each path (no joint artifacts)
for pts, color, w in [(wave2_pts, (51, 136, 255), w2), (wave1_pts, (68, 255, 136), w1)]:
    r = w / 2
    for x, y in pts:
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

# Downscale with LANCZOS for smooth anti-aliasing
img = img.resize((SIZE, SIZE), Image.LANCZOS)
img.save("wave_profile_pic.png")
print(f"Saved wave_profile_pic.png ({SIZE}x{SIZE})")
