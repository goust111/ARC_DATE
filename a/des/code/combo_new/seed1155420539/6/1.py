import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5

PANEL_SIZE_RANGE = (4, 8)
SPRITE_COUNT_CHOICES = (3, 4)
MAX_EXTRA_CANVAS = 6
PLACEMENT_RETRIES = 300
MAX_GEN_ATTEMPTS = 400
NON_SPECIAL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": len(coords),
                }
            )
    return comps


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _find_template(grid: np.ndarray, bg: int) -> dict:
    for comp in sorted(_monochrome_components(grid, bg), key=lambda d: (-d["area"], d["bbox"])):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        th, tw = patch.shape
        if th < 7 or tw < 7:
            continue
        if not (
            np.all(patch[0, :] == color)
            and np.all(patch[-1, :] == color)
            and np.all(patch[:, 0] == color)
            and np.all(patch[:, -1] == color)
        ):
            continue
        inner = patch[1:-1, 1:-1]
        row_hits = [i for i in range(inner.shape[0]) if np.all(inner[i, :] == color)]
        col_hits = [j for j in range(inner.shape[1]) if np.all(inner[:, j] == color)]
        if len(row_hits) != 1 or len(col_hits) != 1:
            continue
        mid_r = row_hits[0]
        mid_c = col_hits[0]
        if mid_r == 0 or mid_r == inner.shape[0] - 1 or mid_c == 0 or mid_c == inner.shape[1] - 1:
            continue
        return {
            "color": color,
            "bbox": (r0, c0, r1, c1),
            "mid_r": mid_r,
            "mid_c": mid_c,
        }
    raise AssertionError("template frame not found")


def _hole_corner(mask: np.ndarray) -> tuple[str, tuple[int, int, int, int]]:
    comps = _mask_components(mask)
    assert len(comps) == 1, "each panel must contain at most one rectangular gap"
    coords = comps[0]
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(sub), "gap must be a filled rectangle"
    assert int(mask.sum()) == sub.size, "gap must be rectangular"
    h, w = mask.shape
    if r0 == 0 and c0 == 0:
        return "tl", (r0, c0, r1, c1)
    if r0 == 0 and c1 == w - 1:
        return "tr", (r0, c0, r1, c1)
    if r1 == h - 1 and c0 == 0:
        return "bl", (r0, c0, r1, c1)
    if r1 == h - 1 and c1 == w - 1:
        return "br", (r0, c0, r1, c1)
    raise AssertionError("gap must touch exactly one panel corner")


def _extract_holes(inner: np.ndarray, bg: int, mid_r: int, mid_c: int) -> list[dict]:
    quadrants = [
        (0, mid_r, 0, mid_c),
        (0, mid_r, mid_c + 1, inner.shape[1]),
        (mid_r + 1, inner.shape[0], 0, mid_c),
        (mid_r + 1, inner.shape[0], mid_c + 1, inner.shape[1]),
    ]
    rotation_by_corner = {"br": 0, "bl": 1, "tl": 2, "tr": 3}
    holes = []
    for qr0, qr1, qc0, qc1 in quadrants:
        panel = inner[qr0:qr1, qc0:qc1]
        gap_mask = panel == bg
        if not np.any(gap_mask):
            continue
        corner, (r0, c0, r1, c1) = _hole_corner(gap_mask)
        hole_h = r1 - r0 + 1
        hole_w = c1 - c0 + 1
        if corner in {"br", "tl"}:
            source_shape = (hole_h, hole_w)
        else:
            source_shape = (hole_w, hole_h)
        holes.append(
            {
                "inner_pos": (qr0 + r0, qc0 + c0),
                "hole_shape": (hole_h, hole_w),
                "source_shape": source_shape,
                "rot_k": rotation_by_corner[corner],
            }
        )
    assert holes, "template must contain at least one gap"
    return holes


def _collect_outside_sprites(grid: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = frame_bbox
    outside = np.array(grid, copy=True)
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    sprites = []
    for comp in _monochrome_components(outside, bg):
        sr0, sc0, sr1, sc1 = comp["bbox"]
        patch = outside[sr0 : sr1 + 1, sc0 : sc1 + 1]
        color = int(comp["color"])
        if int(comp["area"]) != patch.size or not np.all(patch == color):
            continue
        sprites.append({"color": color, "shape": patch.shape, "patch": patch.copy()})
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the black rectangular gap inside the quartered template using the uniquely matching outside rectangle.

    Input structure:
    - The background is black, with several small monochrome rectangular sprites scattered outside a larger template.
    - The template is a grey rectangular frame on the right, and a grey cross splits its interior into four equal panels.
    - Each panel is almost fully filled with one color, but one panel contains a black rectangular gap anchored at a corner near the central cross.

    Rule:
    - Find the grey frame with the internal cross and crop to its interior, excluding only the outer grey border.
    - Detect the panel that contains the black rectangular gap and read the gap size and which corner of its panel it occupies.
    - Among the outside sprites, choose the unique rectangle whose original height/width match the required pre-rotation shape for that corner, rotate it accordingly, and paste its color into the gap.

    Key insight:
    - Size identifies the candidate rectangle, but the gap's corner relative to the template center determines whether the source rectangle must keep or swap its orientation before being inserted.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frame = _find_template(g, bg)
    r0, c0, r1, c1 = frame["bbox"]
    inner = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    holes = _extract_holes(inner, bg, int(frame["mid_r"]), int(frame["mid_c"]))
    sprites = _collect_outside_sprites(g, bg, frame["bbox"])

    used = set()
    for hole in holes:
        matches = [(i, s) for i, s in enumerate(sprites) if i not in used and tuple(map(int, s["shape"])) == hole["source_shape"]]
        assert len(matches) == 1, "gap must match exactly one outside rectangle"
        idx, sprite = matches[0]
        used.add(idx)
        patch = np.rot90(sprite["patch"], int(hole["rot_k"]))
        hh, ww = hole["hole_shape"]
        assert patch.shape == (hh, ww)
        rr, cc = hole["inner_pos"]
        inner[rr : rr + hh, cc : cc + ww] = patch
    return inner


# --------------- generator ---------------


def _sample_hole_size(panel_h: int, panel_w: int) -> tuple[int, int]:
    for _ in range(100):
        hole_h = random.randint(1, panel_h - 2)
        hole_w = random.randint(1, panel_w - 2)
        if hole_h != hole_w:
            return hole_h, hole_w
    raise RuntimeError("failed to sample non-square hole size")


def _build_template(
    panel_h: int,
    panel_w: int,
    panel_colors: list[int],
    hole_quadrant: int,
    hole_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    hole_h, hole_w = hole_shape
    frame_h = 2 * panel_h + 3
    frame_w = 2 * panel_w + 3
    template = np.full((frame_h, frame_w), GREY, dtype=int)

    template[1 : 1 + panel_h, 1 : 1 + panel_w] = panel_colors[0]
    template[1 : 1 + panel_h, panel_w + 2 : frame_w - 1] = panel_colors[1]
    template[panel_h + 2 : frame_h - 1, 1 : 1 + panel_w] = panel_colors[2]
    template[panel_h + 2 : frame_h - 1, panel_w + 2 : frame_w - 1] = panel_colors[3]

    inner_starts = {
        0: (0, 0),
        1: (0, panel_w + 1),
        2: (panel_h + 1, 0),
        3: (panel_h + 1, panel_w + 1),
    }
    hole_starts = {
        0: (panel_h - hole_h, panel_w - hole_w),
        1: (panel_h - hole_h, 0),
        2: (0, panel_w - hole_w),
        3: (0, 0),
    }

    inner_r0, inner_c0 = inner_starts[hole_quadrant]
    rel_r0, rel_c0 = hole_starts[hole_quadrant]
    abs_r0 = 1 + inner_r0 + rel_r0
    abs_c0 = 1 + inner_c0 + rel_c0
    template[abs_r0 : abs_r0 + hole_h, abs_c0 : abs_c0 + hole_w] = BLACK
    return template, (inner_r0 + rel_r0, inner_c0 + rel_c0), hole_starts[hole_quadrant]


def _target_source_shape(hole_quadrant: int, hole_shape: tuple[int, int]) -> tuple[int, int]:
    hole_h, hole_w = hole_shape
    if hole_quadrant in (0, 3):
        return hole_h, hole_w
    return hole_w, hole_h


def _fits(occ: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occ.shape[0], top + h + pad)
    c1 = min(occ.shape[1], left + w + pad)
    return not np.any(occ[r0:r1, c0:c1])


def _occupy(occ: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occ.shape[0], top + h + pad)
    c1 = min(occ.shape[1], left + w + pad)
    occ[r0:r1, c0:c1] = True


def _random_size(exclude: set[tuple[int, int]], max_h: int, max_w: int) -> tuple[int, int]:
    for _ in range(200):
        h = random.randint(1, max_h)
        w = random.randint(1, max_w)
        shape = (h, w)
        if shape not in exclude:
            return shape
    raise RuntimeError("failed to sample distractor size")


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_SIZE_RANGE)
    panel_w = random.randint(*PANEL_SIZE_RANGE)
    hole_shape = _sample_hole_size(panel_h, panel_w)
    hole_quadrant = random.randrange(4)
    sprite_count = random.choice(SPRITE_COUNT_CHOICES)

    colors = random.sample(NON_SPECIAL_COLORS, 4 + sprite_count)
    panel_colors = colors[:4]
    sprite_colors = colors[4:]

    template, hole_inner_pos, _ = _build_template(panel_h, panel_w, panel_colors, hole_quadrant, hole_shape)
    frame_h, frame_w = template.shape

    target_shape = _target_source_shape(hole_quadrant, hole_shape)
    opposite_shape = hole_shape if target_shape == hole_shape[::-1] else hole_shape[::-1]
    sprite_specs = [
        {"shape": target_shape, "color": sprite_colors[0], "target": True},
        {"shape": opposite_shape, "color": sprite_colors[1], "target": False},
    ]

    used_shapes = {target_shape, opposite_shape}
    max_rand_h = max(2, panel_h - 1)
    max_rand_w = max(2, panel_w - 1)
    for i in range(2, sprite_count):
        shape = _random_size(used_shapes, max_rand_h, max_rand_w)
        used_shapes.add(shape)
        sprite_specs.append({"shape": shape, "color": sprite_colors[i], "target": False})
    random.shuffle(sprite_specs)

    max_sprite_w = max(spec["shape"][1] for spec in sprite_specs)
    min_left_w = max(max_sprite_w + 2, 8)
    gap = random.randint(1, 2)
    max_left_w = 30 - frame_w - gap
    if min_left_w > max_left_w:
        return None
    left_w = random.randint(min_left_w, max_left_w)
    canvas_w = left_w + gap + frame_w

    min_canvas_h = max(frame_h + 2, 16)
    max_canvas_h = min(30, min_canvas_h + MAX_EXTRA_CANVAS)
    if min_canvas_h > max_canvas_h:
        return None
    canvas_h = random.randint(min_canvas_h, max_canvas_h)

    frame_top = random.randint(1, canvas_h - frame_h - 1)
    frame_left = left_w + gap
    canvas = np.full((canvas_h, canvas_w), BLACK, dtype=int)
    canvas[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = template

    occ = np.zeros((canvas_h, left_w), dtype=bool)
    ordered_specs = sorted(sprite_specs, key=lambda d: (-(d["shape"][0] * d["shape"][1]), d["shape"]))
    target_color = None
    for spec in ordered_specs:
        h, w = spec["shape"]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, canvas_h - h)
            left = random.randint(0, left_w - w)
            if not _fits(occ, top, left, h, w):
                continue
            canvas[top : top + h, left : left + w] = spec["color"]
            _occupy(occ, top, left, h, w)
            if spec["target"]:
                target_color = spec["color"]
            placed = True
            break
        if not placed:
            return None

    assert target_color is not None
    expected = template[1:-1, 1:-1].copy()
    hr, hc = hole_inner_pos
    hh, ww = hole_shape
    expected[hr : hr + hh, hc : hc + ww] = target_color

    try:
        produced = main(canvas)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
