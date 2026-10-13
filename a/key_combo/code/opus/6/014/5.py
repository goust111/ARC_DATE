from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg


BG = 0
BAR = 5
CONTROL = 9
TARGET_COLORS = (1, 2, 3, 4, 7)

GRID_H_RANGE = (15, 20)
GRID_W_RANGE = (15, 20)
MAX_GEN_ATTEMPTS = 500

_RAW_SHAPES = [
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
]


def canonical_shape_key(mask: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    """Canonicalize a boolean mask up to rotation and horizontal reflection."""
    canon = np.asarray(mask, dtype=bool)
    best = (canon.tobytes(), tuple(map(int, canon.shape)))
    cur = canon
    for _ in range(4):
        key = (cur.tobytes(), tuple(map(int, cur.shape)))
        if key < best:
            best = key
        flipped = np.fliplr(cur)
        key = (flipped.tobytes(), tuple(map(int, flipped.shape)))
        if key < best:
            best = key
        cur = np.rot90(cur)
    return best


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    cur = np.asarray(mask, dtype=bool)
    for _ in range(4):
        for candidate in (cur, np.fliplr(cur)):
            key = candidate.tobytes()
            if key not in seen:
                seen.add(key)
                out.append(candidate.copy())
        cur = np.rot90(cur)
    return out


def _rotates_to_self(mask: np.ndarray) -> bool:
    return np.array_equal(mask, np.rot90(mask, -1))


def _build_shape_library() -> tuple[dict[int, list[np.ndarray]], dict[int, list[np.ndarray]]]:
    by_area: dict[int, list[np.ndarray]] = {}
    asym_by_area: dict[int, list[np.ndarray]] = {}
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for mask in _RAW_SHAPES:
        key = canonical_shape_key(mask)
        if key in seen:
            continue
        seen.add(key)
        area = int(mask.sum())
        by_area.setdefault(area, []).append(mask.copy())
        if not _rotates_to_self(mask):
            asym_by_area.setdefault(area, []).append(mask.copy())
    return by_area, asym_by_area


SHAPES_BY_AREA, ASYM_SHAPES_BY_AREA = _build_shape_library()
AREA_CHOICES = tuple(sorted(area for area, masks in SHAPES_BY_AREA.items() if len(masks) >= 2 and ASYM_SHAPES_BY_AREA.get(area)))


def _coords_from_mask(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        coords.append((int(top + r), int(left + c)))
    return coords


def _mask_from_coords(coords: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in coords:
        out[int(r), int(c)] = True
    return out


def _find_marker(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    bar = np.argwhere(grid == BAR)
    control = np.argwhere(grid == CONTROL)
    assert bar.shape == (3, 2), "expected one 3-pixel marker bar"
    assert control.shape == (1, 2), "expected one control pixel"

    rows = sorted(int(r) for r in bar[:, 0])
    cols = sorted(int(c) for c in bar[:, 1])
    ctrl = (int(control[0, 0]), int(control[0, 1]))

    if len(set(rows)) == 1:
        center = (rows[0], cols[1])
        assert cols == [center[1] - 1, center[1], center[1] + 1]
        assert abs(ctrl[0] - center[0]) == 1 and ctrl[1] == center[1]
    else:
        assert len(set(cols)) == 1
        center = (rows[1], cols[0])
        assert rows == [center[0] - 1, center[0], center[0] + 1]
        assert abs(ctrl[1] - center[1]) == 1 and ctrl[0] == center[0]

    step = (center[0] - ctrl[0], center[1] - ctrl[1])
    assert abs(step[0]) + abs(step[1]) == 1
    return center, step


def _ray_cells(center: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r, c = center[0] + step[0], center[1] + step[1]
    out: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        out.append((int(r), int(c)))
        r += step[0]
        c += step[1]
    return out


def _component_mask(obj: dict, grid_shape: tuple[int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, obj["bbox"])
    rel = [(int(r) - r0, int(c) - c0) for r, c in obj["coords"]]
    return _mask_from_coords(rel, (r1 - r0 + 1, c1 - c0 + 1))


def _component_shape_key(obj: dict) -> tuple[bytes, tuple[int, int]]:
    return canonical_shape_key(_component_mask(obj, (0, 0)))


def _rotate_coords_cw(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    out: list[tuple[int, int]] = []
    for r, c in coords:
        rr = cr + (float(c) - cc)
        cc2 = cc - (float(r) - cr)
        nr = int(round(rr))
        nc = int(round(cc2))
        assert abs(rr - nr) < 1e-6 and abs(cc2 - nc) < 1e-6, "rotation must stay on lattice"
        out.append((nr, nc))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast the marker ray, classify only the hit objects by local shape frequency, then erase duplicates and rotate local singletons.

    Input structure:
    - The grid has black background, one 3-pixel grey marker bar, and one maroon control pixel adjacent to the bar center.
    - The control pixel and the bar center define one horizontal or vertical ray pointing away from the control.
    - The remaining foreground objects are monochromatic 4-connected target components.

    Rule:
    - Extend the ray from the bar center to the grid edge in the direction away from the control pixel.
    - Every target object touched by any ray cell enters the hit set; untouched objects are ignored.
    - Compute each hit object's canonical binary shape up to rotation and flip, and count frequencies only inside the hit set.
    - Hit objects whose canonical shape occurs two or more times are erased to background.
    - Hit objects whose canonical shape occurs exactly once are rotated 90 degrees clockwise in place around their own bounding-box center.
    - The marker, the control pixel, and every untouched object remain unchanged.

    Key insight:
    - The ray does not directly choose the final action; it only defines a local candidate pool, and shape frequency is evaluated inside that pool rather than globally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    center, step = _find_marker(g)
    ray = set(_ray_cells(center, step, g.shape))

    target_colors = sorted(set(map(int, g.ravel())) - {bg, BAR, CONTROL})
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=target_colors, return_format="dict")
    assert objects, "expected target objects"

    hits = [obj for obj in objects if any((int(r), int(c)) in ray for r, c in obj["coords"])]
    assert hits, "expected at least one ray hit"

    keys = [_component_shape_key(obj) for obj in hits]
    freq = Counter(keys)

    out = g.copy()
    for obj in hits:
        coords = obj["coords"].astype(int)
        out[coords[:, 0], coords[:, 1]] = bg

    for obj, key in zip(hits, keys):
        if freq[key] != 1:
            continue
        color = int(obj["color"])
        rotated = _rotate_coords_cw(obj["coords"], obj["bbox"])
        for r, c in rotated:
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1], "rotated object left the grid"
            assert out[r, c] == bg, "rotated object overlaps surviving content"
        for r, c in rotated:
            out[r, c] = color

    return out


def _variant(mask: np.ndarray) -> np.ndarray:
    return random.choice(_shape_variants(mask))


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _direction_spec(h: int, w: int) -> tuple[str, tuple[int, int], tuple[int, int], list[int]]:
    direction = random.choice(("right", "left", "down", "up"))
    if direction == "right":
        ray_row = random.randint(2, h - 3)
        center = (ray_row, random.randint(1, 3))
        control = (ray_row, center[1] - 1)
        line_positions = list(range(center[1] + 3, w - 1))
    elif direction == "left":
        ray_row = random.randint(2, h - 3)
        center = (ray_row, random.randint(w - 4, w - 2))
        control = (ray_row, center[1] + 1)
        line_positions = list(range(1, center[1] - 2))
    elif direction == "down":
        ray_col = random.randint(2, w - 3)
        center = (random.randint(1, 3), ray_col)
        control = (center[0] - 1, ray_col)
        line_positions = list(range(center[0] + 3, h - 1))
    else:
        ray_col = random.randint(2, w - 3)
        center = (random.randint(h - 4, h - 2), ray_col)
        control = (center[0] + 1, ray_col)
        line_positions = list(range(1, center[0] - 2))
    return direction, center, control, line_positions


def _choose_spaced_positions(candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    if len(candidates) < count:
        return None
    for _ in range(200):
        picks = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _place_marker(grid: np.ndarray, blocked: np.ndarray, direction: str, center: tuple[int, int], control: tuple[int, int]) -> None:
    r, c = center
    if direction in ("right", "left"):
        bar = [(r - 1, c), (r, c), (r + 1, c)]
    else:
        bar = [(r, c - 1), (r, c), (r, c + 1)]
    _paint(grid, bar, BAR)
    grid[control] = CONTROL
    _mark_blocked(blocked, bar + [control])


def _hit_top_left(direction: str, line_coord: int, position: int) -> tuple[int, int]:
    if direction in ("right", "left"):
        return line_coord - 1, position - 1
    return position - 1, line_coord - 1


def _bbox_crosses_ray(direction: str, top: int, left: int, line_coord: int) -> bool:
    if direction in ("right", "left"):
        return top <= line_coord <= top + 2
    return left <= line_coord <= left + 2


def _can_place(mask: np.ndarray, top: int, left: int, blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    coords = _coords_from_mask(mask, top, left)
    return all(0 <= r < h and 0 <= c < w and not blocked[r, c] for r, c in coords)


def _place_object(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> list[tuple[int, int]]:
    coords = _coords_from_mask(mask, top, left)
    assert _can_place(mask, top, left, blocked)
    _paint(grid, coords, color)
    _mark_blocked(blocked, coords)
    return coords


def _random_off_ray_location(
    blocked: np.ndarray,
    direction: str,
    line_coord: int,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    candidates: list[tuple[int, int]] = []
    for top in range(h - 2):
        for left in range(w - 2):
            if _bbox_crosses_ray(direction, top, left, line_coord):
                continue
            candidates.append((top, left))
    random.shuffle(candidates)
    for top, left in candidates:
        if not blocked[top : top + 3, left : left + 3].any():
            return top, left
    return None


def _extract_target_objects(grid: np.ndarray) -> list[dict]:
    return find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=list(TARGET_COLORS),
        return_format="dict",
    )


def _semantic_signature(grid: np.ndarray) -> tuple[list[dict], list[dict], Counter]:
    center, step = _find_marker(grid)
    ray = set(_ray_cells(center, step, grid.shape))
    objects = _extract_target_objects(grid)
    hits = [obj for obj in objects if any((int(r), int(c)) in ray for r, c in obj["coords"])]
    freq = Counter(_component_shape_key(obj) for obj in hits)
    return objects, hits, freq


def _semantic_ok(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    objects, hits, freq = _semantic_signature(inp)
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if len(objects) != 5 or len(hits) != 3:
        return False
    if sorted(freq.values()) != [1, 2]:
        return False

    hit_areas = {int(obj["area"]) for obj in hits}
    if len(hit_areas) != 1:
        return False

    all_freq = Counter(_component_shape_key(obj) for obj in objects)
    singleton_hits = [obj for obj in hits if freq[_component_shape_key(obj)] == 1]
    if len(singleton_hits) != 1:
        return False
    unique_hit = singleton_hits[0]
    unique_key = _component_shape_key(unique_hit)
    if all_freq[unique_key] != 2:
        return False

    rotated = sorted(_rotate_coords_cw(unique_hit["coords"], unique_hit["bbox"]))
    original = sorted((int(r), int(c)) for r, c in unique_hit["coords"])
    if rotated == original:
        return False

    center, step = _find_marker(inp)
    ray = set(_ray_cells(center, step, inp.shape))
    if any((r, c) in ray for r, c in rotated if (r, c) not in original):
        return False

    try:
        recomputed = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, recomputed):
        return False
    if np.array_equal(out, inp):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    direction, center, control, line_positions = _direction_spec(h, w)
    hit_positions = _choose_spaced_positions(line_positions, count=3, min_gap=4)
    if hit_positions is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    _place_marker(grid, blocked, direction, center, control)

    area = random.choice(AREA_CHOICES)
    shared_shape, unique_shape = random.sample(SHAPES_BY_AREA[area], 2)
    unique_candidates = ASYM_SHAPES_BY_AREA[area]
    if canonical_shape_key(unique_shape) not in {canonical_shape_key(m) for m in unique_candidates}:
        unique_shape = random.choice(unique_candidates)
        shared_options = [m for m in SHAPES_BY_AREA[area] if canonical_shape_key(m) != canonical_shape_key(unique_shape)]
        if not shared_options:
            return None
        shared_shape = random.choice(shared_options)

    c_options = [
        mask
        for masks in SHAPES_BY_AREA.values()
        for mask in masks
        if canonical_shape_key(mask) not in {canonical_shape_key(shared_shape), canonical_shape_key(unique_shape)}
    ]
    if not c_options:
        return None
    distractor_shape = random.choice(c_options)

    masks = {
        "A1": _variant(shared_shape),
        "A2": _variant(shared_shape),
        "B1": _variant(unique_shape),
        "B2": _variant(unique_shape),
        "C": _variant(distractor_shape),
    }

    colors = random.sample(TARGET_COLORS, 5)
    color_map = dict(zip(("A1", "A2", "B1", "B2", "C"), colors))

    if direction in ("right", "left"):
        line_coord = center[0]
    else:
        line_coord = center[1]

    hit_names = ["A1", "A2", "B1"]
    random.shuffle(hit_names)
    for name, pos in zip(hit_names, hit_positions):
        top, left = _hit_top_left(direction, line_coord, pos)
        if not _can_place(masks[name], top, left, blocked):
            return None
        _place_object(grid, blocked, masks[name], top, left, color_map[name])

    for name in ("B2", "C"):
        loc = _random_off_ray_location(blocked, direction, line_coord)
        if loc is None:
            return None
        top, left = loc
        if not _can_place(masks[name], top, left, blocked):
            return None
        _place_object(grid, blocked, masks[name], top, left, color_map[name])

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 5")
