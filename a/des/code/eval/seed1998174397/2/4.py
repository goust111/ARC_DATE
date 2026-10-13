import random
from typing import Optional

import numpy as np


BG = 0
PATTERN_COLOR = 4
ARROW_COLOR = 7
FRAME_COLORS = (1, 2, 3, 5, 6, 8, 9)
ROT_STEPS = {"up": 0, "right": 3, "down": 2, "left": 1}
SCRATCH_SHAPE = (80, 80)
FRAME_COUNT_RANGE = (2, 4)
PATTERN_SIZE_RANGE = (3, 5)
INNER_PAD_RANGE = (2, 5)
OUTER_MARGIN_RANGE = (1, 7)
FINAL_MARGIN_RANGE = (1, 4)
MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr in range(max(0, cr - 1), min(h, cr + 2)):
                    for nc in range(max(0, cc - 1), min(w, cc + 2)):
                        if nr == cr and nc == cc:
                            continue
                        if seen[nr, nc] or int(grid[nr, nc]) != color:
                            continue
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "size": int(len(pts)),
                }
            )
    return comps


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(comp["size"]) != 2 * (h + w) - 4:
        return False
    pts = {tuple(map(int, p)) for p in comp["coords"]}
    for r, c in pts:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    return True


def _arrow_direction(coords: np.ndarray) -> Optional[str]:
    if len(coords) != 3:
        return None
    rr = coords[:, 0] - int(coords[:, 0].min())
    cc = coords[:, 1] - int(coords[:, 1].min())
    pts = frozenset((int(r), int(c)) for r, c in zip(rr, cc))
    patterns = {
        frozenset({(0, 0), (0, 2), (1, 1)}): "up",
        frozenset({(0, 1), (1, 0), (1, 2)}): "down",
        frozenset({(0, 0), (2, 0), (1, 1)}): "left",
        frozenset({(1, 0), (0, 1), (2, 1)}): "right",
    }
    return patterns.get(pts)


def _arrow_shaft(coords: np.ndarray, direction: str) -> tuple[int, int]:
    if direction == "up":
        idx = int(np.argmax(coords[:, 0]))
    elif direction == "down":
        idx = int(np.argmin(coords[:, 0]))
    elif direction == "left":
        idx = int(np.argmax(coords[:, 1]))
    else:
        idx = int(np.argmin(coords[:, 1]))
    return int(coords[idx, 0]), int(coords[idx, 1])


def _matches_frame_arrow(arrow: dict, frame_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = frame_bbox
    sr, sc = arrow["shaft"]
    direction = arrow["direction"]
    if direction == "up":
        return sr == r0 - 1 and c0 < sc < c1
    if direction == "down":
        return sr == r1 + 1 and c0 < sc < c1
    if direction == "left":
        return sc == c0 - 1 and r0 < sr < r1
    return sc == c1 + 1 and r0 < sr < r1


def _crop_mask(grid: np.ndarray, mask: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(mask)
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    arr = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    arr[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return arr


def _center_top_left(frame_bbox: tuple[int, int, int, int], item_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    item_h, item_w = map(int, item_shape)
    assert inner_h >= item_h and inner_w >= item_w
    top = int(r0 + 1 + (inner_h - item_h) // 2)
    left = int(c0 + 1 + (inner_w - item_w) // 2)
    return top, left


def _paste_non_bg(dest: np.ndarray, src: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = src.shape
    mask = src != bg
    window = dest[top : top + h, left : left + w]
    assert np.all(window[mask] == bg)
    window[mask] = src[mask]


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _arrow_points(frame_bbox: tuple[int, int, int, int], direction: str, rng: random.Random) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    if direction == "up":
        col = rng.randint(c0 + 1, c1 - 1)
        return [(r0 - 1, col), (r0 - 2, col - 1), (r0 - 2, col + 1)]
    if direction == "down":
        col = rng.randint(c0 + 1, c1 - 1)
        return [(r1 + 1, col), (r1 + 2, col - 1), (r1 + 2, col + 1)]
    if direction == "left":
        row = rng.randint(r0 + 1, r1 - 1)
        return [(row, c0 - 1), (row - 1, c0 - 2), (row + 1, c0 - 2)]
    row = rng.randint(r0 + 1, r1 - 1)
    return [(row, c1 + 1), (row - 1, c1 + 2), (row + 1, c1 + 2)]


def _bbox_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    return _bbox_from_coords(np.argwhere(mask))


def _paste_full(dest: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = src.shape
    dest[top : top + h, left : left + w] = src


def _build_structure(grid: np.ndarray) -> tuple[list[dict], dict[int, str], np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = _color_components(g, bg)
    frames = [comp for comp in comps if _is_frame_component(comp)]
    assert len(frames) >= 2, "expected nested hollow frames"
    frames.sort(key=lambda comp: _bbox_area(comp["bbox"]))
    for inner, outer in zip(frames, frames[1:]):
        assert _contains(outer["bbox"], inner["bbox"]), "frames must be strictly nested"

    frame_mask = np.zeros(g.shape, dtype=bool)
    for comp in frames:
        frame_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True

    arrows = []
    arrow_mask = np.zeros(g.shape, dtype=bool)
    for comp in comps:
        if comp in frames:
            continue
        direction = _arrow_direction(comp["coords"])
        if direction is None:
            continue
        shaft = _arrow_shaft(comp["coords"], direction)
        arrow = {"direction": direction, "shaft": shaft, "coords": comp["coords"]}
        arrows.append(arrow)
        arrow_mask[comp["coords"][:, 0], comp["coords"][:, 1]] = True

    assert len(arrows) == len(frames) - 1, "expected one arrow per non-innermost frame"
    frame_dirs: dict[int, str] = {}
    used = set()
    for idx, frame in enumerate(frames[1:], start=1):
        matches = [j for j, arrow in enumerate(arrows) if _matches_frame_arrow(arrow, frame["bbox"])]
        assert len(matches) == 1, "each outer frame must have exactly one arrow"
        used.add(matches[0])
        frame_dirs[idx] = arrows[matches[0]]["direction"]
    assert len(used) == len(arrows)

    inner = frames[0]["bbox"]
    inside_inner = np.zeros(g.shape, dtype=bool)
    inside_inner[inner[0] + 1 : inner[2], inner[1] + 1 : inner[3]] = True
    motif_mask = (g != bg) & (~frame_mask) & (~arrow_mask)
    assert motif_mask.any(), "expected a seed motif inside the innermost frame"
    assert np.all(motif_mask <= inside_inner), "unexpected non-frame residue outside innermost interior"
    motif = _crop_mask(g, motif_mask, bg)
    return frames, frame_dirs, motif, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the innermost motif outward across nested frames, rotating once per layer
    according to the arrow attached to each destination frame.
    """
    g = np.asarray(grid, dtype=int)
    frames, frame_dirs, motif, bg = _build_structure(g)

    out = g.copy()
    current = motif.copy()
    for idx, frame in enumerate(frames[1:], start=1):
        current = np.rot90(current, k=ROT_STEPS[frame_dirs[idx]])
        top, left = _center_top_left(frame["bbox"], current.shape)
        _paste_non_bg(out, current, top, left, bg)
    return out


def _normalize_points(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    norm = sorted((r - min_r, c - min_c) for r, c in points)
    return [(int(r), int(c)) for r, c in norm]


def _shape_array(points: list[tuple[int, int]]) -> np.ndarray:
    h = max(r for r, _ in points) + 1
    w = max(c for _, c in points) + 1
    arr = np.zeros((h, w), dtype=int)
    for r, c in points:
        arr[r, c] = 1
    return arr


def _random_pattern(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        target = rng.randint(*PATTERN_SIZE_RANGE)
        points = {(0, 0)}
        while len(points) < target:
            frontier = set()
            for r, c in points:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if -2 <= nr <= 2 and -2 <= nc <= 2 and (nr, nc) not in points:
                        frontier.add((nr, nc))
            if not frontier:
                break
            points.add(rng.choice(sorted(frontier)))
        pts = _normalize_points(points)
        arr = _shape_array(pts)
        if arr.shape[0] < 2 or arr.shape[1] < 2:
            continue
        if any(np.array_equal(arr, np.rot90(arr, k)) for k in (1, 2, 3)):
            continue
        out = np.full(arr.shape, BG, dtype=int)
        out[arr == 1] = PATTERN_COLOR
        return out
    raise RuntimeError("failed to generate asymmetric pattern")


def _can_place_pattern(grid: np.ndarray, pattern: np.ndarray, top: int, left: int) -> bool:
    h, w = pattern.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    mask = pattern != BG
    return bool(np.all(grid[top : top + h, left : left + w][mask] == BG))


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    input_grid = np.full(SCRATCH_SHAPE, BG, dtype=int)
    output_grid = np.full(SCRATCH_SHAPE, BG, dtype=int)

    frame_count = rng.randint(*FRAME_COUNT_RANGE)
    frame_colors = rng.sample(FRAME_COLORS, frame_count)
    pattern = _random_pattern(rng)
    current_pattern = pattern.copy()

    inner_h = int(pattern.shape[0] + rng.randint(*INNER_PAD_RANGE))
    inner_w = int(pattern.shape[1] + rng.randint(*INNER_PAD_RANGE))
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    sr = SCRATCH_SHAPE[0] // 2 - frame_h // 2
    sc = SCRATCH_SHAPE[1] // 2 - frame_w // 2
    frame_bbox = (sr, sc, sr + frame_h - 1, sc + frame_w - 1)

    _draw_frame(input_grid, frame_bbox, frame_colors[0])
    _draw_frame(output_grid, frame_bbox, frame_colors[0])
    top, left = _center_top_left(frame_bbox, pattern.shape)
    _paste_non_bg(input_grid, pattern, top, left, BG)
    _paste_non_bg(output_grid, pattern, top, left, BG)

    for layer in range(1, frame_count):
        direction = rng.choice(["up", "right", "down", "left"])
        next_pattern = np.rot90(current_pattern, k=ROT_STEPS[direction])

        placed = False
        for _ in range(250):
            occ_bbox = _bbox_mask(output_grid != BG)
            mt = rng.randint(*OUTER_MARGIN_RANGE)
            mb = rng.randint(*OUTER_MARGIN_RANGE)
            ml = rng.randint(*OUTER_MARGIN_RANGE)
            mr = rng.randint(*OUTER_MARGIN_RANGE)
            interior = (occ_bbox[0] - mt, occ_bbox[1] - ml, occ_bbox[2] + mb, occ_bbox[3] + mr)
            frame_bbox = (interior[0] - 1, interior[1] - 1, interior[2] + 1, interior[3] + 1)
            if (
                frame_bbox[0] < 2
                or frame_bbox[1] < 2
                or frame_bbox[2] >= SCRATCH_SHAPE[0] - 2
                or frame_bbox[3] >= SCRATCH_SHAPE[1] - 2
            ):
                continue

            top, left = _center_top_left(frame_bbox, next_pattern.shape)
            if not _can_place_pattern(output_grid, next_pattern, top, left):
                continue

            arrow_pts = _arrow_points(frame_bbox, direction, rng)
            if any(
                r < 0 or c < 0 or r >= SCRATCH_SHAPE[0] or c >= SCRATCH_SHAPE[1]
                for r, c in arrow_pts
            ):
                continue
            if any(int(output_grid[r, c]) != BG for r, c in arrow_pts):
                continue

            _draw_frame(input_grid, frame_bbox, frame_colors[layer])
            _draw_frame(output_grid, frame_bbox, frame_colors[layer])
            for r, c in arrow_pts:
                input_grid[r, c] = ARROW_COLOR
                output_grid[r, c] = ARROW_COLOR
            _paste_non_bg(output_grid, next_pattern, top, left, BG)
            current_pattern = next_pattern
            placed = True
            break
        if not placed:
            return None

    bbox = _bbox_mask(input_grid != BG)
    cropped_in = input_grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]
    cropped_out = output_grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1]

    pad_t = rng.randint(*FINAL_MARGIN_RANGE)
    pad_b = rng.randint(*FINAL_MARGIN_RANGE)
    pad_l = rng.randint(*FINAL_MARGIN_RANGE)
    pad_r = rng.randint(*FINAL_MARGIN_RANGE)
    final_h = int(cropped_in.shape[0] + pad_t + pad_b)
    final_w = int(cropped_in.shape[1] + pad_l + pad_r)
    if final_h > MAX_SIDE or final_w > MAX_SIDE:
        return None

    final_in = np.full((final_h, final_w), BG, dtype=int)
    final_out = np.full((final_h, final_w), BG, dtype=int)
    _paste_full(final_in, cropped_in, pad_t, pad_l)
    _paste_full(final_out, cropped_out, pad_t, pad_l)

    try:
        produced = main(final_in)
    except AssertionError:
        return None
    if not np.array_equal(produced, final_out):
        return None
    return final_in


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame rotation puzzle")
