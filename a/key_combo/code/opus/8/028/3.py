from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, find_components, get_bg


BG = Color.BLACK
OBSTACLE = Color.BLUE
BLOCK = Color.RED
GEN_ATTEMPTS = 400


def _rows_to_shape(rows: list[str]) -> np.ndarray:
    return np.array([[BLOCK if ch == "1" else BG for ch in row] for row in rows], dtype=int)


SHAPES = {
    "zig5": _rows_to_shape(["110", "011", "010"]),
    "step6": _rows_to_shape(["1100", "0110", "0011"]),
    "hook7": _rows_to_shape(["111", "001", "111"]),
    "ring3": _rows_to_shape(["111", "101", "111"]),
    "ring34": _rows_to_shape(["1111", "1011", "1111"]),
    "ring15": _rows_to_shape(["11111", "10111", "11111"]),
    "double35": _rows_to_shape(["11111", "10101", "11111"]),
    "double44": _rows_to_shape(["1111", "1011", "1101", "1111"]),
    "triple37": _rows_to_shape(["1111111", "1010101", "1111111"]),
    "triple55": _rows_to_shape(["11111", "10101", "11111", "11011", "11111"]),
}


def _count_hole_pixels(crop: np.ndarray) -> int:
    total = 0
    hole_parts = components_mask(crop == BG, connectivity=4)
    for coords in hole_parts:
        r0, c0, r1, c1 = bbox_coords(coords)
        if r0 == 0 or c0 == 0 or r1 == crop.shape[0] - 1 or c1 == crop.shape[1] - 1:
            continue
        total += int(coords.shape[0])
    return total


SHAPE_INFO = {}
for _name, _shape in SHAPES.items():
    SHAPE_INFO[_name] = {
        "shape": _shape,
        "holes": _count_hole_pixels(_shape),
        "area": int(np.count_nonzero(_shape == BLOCK)),
        "height": int(_shape.shape[0]),
        "width": int(_shape.shape[1]),
    }


HOLE_GROUPS = {
    0: ("zig5", "step6", "hook7"),
    1: ("ring3", "ring34", "ring15"),
    2: ("double35", "double44"),
    3: ("triple37", "triple55"),
}


def _linear_relation(xs: list[int], ys: list[int]) -> bool:
    if len(xs) != len(ys) or len(xs) < 2:
        return False
    x0, y0 = int(xs[0]), int(ys[0])
    idx = 1
    while idx < len(xs) and int(xs[idx]) == x0 and int(ys[idx]) == y0:
        idx += 1
    if idx == len(xs):
        return True
    x1, y1 = int(xs[idx]), int(ys[idx])
    if x1 == x0:
        return all(int(y) == y0 for y in ys)
    dx = x1 - x0
    dy = y1 - y0
    return all((int(y) - y0) * dx == dy * (int(x) - x0) for x, y in zip(xs, ys))


def _inspect_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(np.unique(g)).issubset({BG, OBSTACLE, BLOCK})

    blocks = sorted(
        find_components(g, bg=BG, colors=[BLOCK], connectivity=4, monochromatic=True),
        key=lambda obj: (int(obj["bbox"][0]), int(obj["bbox"][1])),
    )
    barriers = {tuple(map(int, coord)) for coord in np.argwhere(g == OBSTACLE)}
    assert len(blocks) == len(barriers) and len(blocks) >= 3

    info = []
    unused = set(barriers)
    for obj in blocks:
        r0, c0, r1, c1 = map(int, obj["bbox"])
        candidates = sorted((r, c) for r, c in unused if r0 <= r <= r1 and c > c1)
        assert len(candidates) == 1, "each block must have exactly one stopper pixel in its row band"
        barrier_r, barrier_c = candidates[0]
        unused.remove(candidates[0])

        crop = crop_bbox(g, obj["bbox"])
        holes = _count_hole_pixels(crop)
        max_shift = barrier_c - 1 - c1
        assert max_shift >= 0, "stopper must be to the right of the block"
        shift = min(holes, max_shift)

        info.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": obj["coords"],
                "area": int(obj["area"]),
                "width": int(c1 - c0 + 1),
                "height": int(r1 - r0 + 1),
                "crop": crop,
                "barrier": (barrier_r, barrier_c),
                "holes": holes,
                "max_shift": max_shift,
                "shift": shift,
            }
        )

    assert not unused
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each red object right by its enclosed-hole count, capped by the blue stopper column.

    Input structure:
    - Black is the background.
    - Red cells form several separated 4-connected objects; each object has exactly one blue single-pixel stopper somewhere to its right within the same row band.
    - A red object's tight bounding box may contain enclosed black hole pixels.

    Rule:
    - For each red object independently, count the black pixels inside its bounding box that belong to background components not touching the box border.
    - Treat that count H as the object's intended rightward translation distance.
    - If moving H steps would make any part of the object reach or cross the blue stopper's column, move only far enough that the object's rightmost pixel ends at the column immediately left of the stopper.
    - Clear the old red cells, redraw the shifted red object, and keep every blue stopper unchanged.

    Key insight:
    - The motion distance is not fixed and not "as far as possible"; it comes from each object's own internal hole count and is then clipped by the physical stopper column.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == BLOCK] = BG

    for obj in _inspect_grid(g):
        shift = int(obj["shift"])
        for r, c in obj["coords"]:
            out[int(r), int(c) + shift] = BLOCK
    return out


def _sample_shape_names(rng: random.Random) -> list[str]:
    extras = [1, 2, 3]
    counts = [0, 1, 2, rng.choice(extras)]
    if rng.random() < 0.5:
        counts.append(rng.choice([1, 2, 3]))

    names = []
    used = set()
    for holes in counts:
        candidates = [name for name in HOLE_GROUPS[holes] if name not in used]
        if not candidates:
            return []
        chosen = rng.choice(candidates)
        used.add(chosen)
        names.append(chosen)
    rng.shuffle(names)
    return names


def _choose_gaps(rng: random.Random, hole_counts: list[int]) -> list[int]:
    multi_hole = [idx for idx, holes in enumerate(hole_counts) if holes >= 2]
    positive = [idx for idx, holes in enumerate(hole_counts) if holes > 0]
    zeroes = [idx for idx, holes in enumerate(hole_counts) if holes == 0]
    if not multi_hole or not positive or not zeroes:
        return []

    collision_idx = rng.choice(multi_hole)
    free_idx = rng.choice([idx for idx in positive if idx != collision_idx] or positive)

    gaps = []
    for idx, holes in enumerate(hole_counts):
        if holes == 0:
            gaps.append(rng.randint(1, 4))
        elif idx == collision_idx:
            gaps.append(rng.randint(1, holes - 1))
        elif idx == free_idx:
            gaps.append(holes + rng.randint(1, 4))
        else:
            if rng.random() < 0.55:
                gaps.append(holes + rng.randint(1, 4))
            else:
                gaps.append(rng.randint(0, holes - 1))
    return gaps


def _build_attempt(rng: random.Random) -> tuple[np.ndarray, list[dict]] | None:
    names = _sample_shape_names(rng)
    if not names:
        return None

    specs = []
    for name in names:
        info = SHAPE_INFO[name]
        specs.append(
            {
                "name": name,
                "shape": info["shape"],
                "holes": int(info["holes"]),
                "area": int(info["area"]),
                "height": int(info["height"]),
                "width": int(info["width"]),
            }
        )

    hole_counts = [int(spec["holes"]) for spec in specs]
    gaps = _choose_gaps(rng, hole_counts)
    if not gaps:
        return None
    for spec, gap in zip(specs, gaps):
        spec["gap"] = int(gap)

    top_margin = rng.randint(0, 2)
    bottom_margin = rng.randint(0, 2)
    band_gaps = [rng.randint(1, 2) for _ in range(max(0, len(specs) - 1))]
    height = top_margin + bottom_margin + sum(spec["height"] for spec in specs) + sum(band_gaps)
    if height > 30:
        return None

    row = top_margin
    for idx, spec in enumerate(specs):
        spec["top"] = int(row)
        row += spec["height"]
        if idx + 1 < len(specs):
            row += band_gaps[idx]

    lefts = []
    needed_width = 0
    for spec in specs:
        max_left = 6 if spec["width"] <= 5 else 4
        left = rng.randint(1, max_left)
        lefts.append(left)
        barrier_col = left + spec["width"] + spec["gap"]
        needed_width = max(needed_width, barrier_col + 1)
    width = needed_width + rng.randint(1, 3)
    if width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    plan = []
    for spec, left in zip(specs, lefts):
        shape = spec["shape"]
        top = int(spec["top"])
        h, w = map(int, shape.shape)
        r0, c0 = top, int(left)
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        patch[shape == BLOCK] = BLOCK
        barrier_col = c0 + w + int(spec["gap"])
        barrier_row = r0 + h // 2
        grid[barrier_row, barrier_col] = OBSTACLE

        plan.append(
            {
                "name": spec["name"],
                "holes": int(spec["holes"]),
                "gap": int(spec["gap"]),
                "barrier_col": int(barrier_col),
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
            }
        )

    return grid, plan


def _validate_attempt(grid: np.ndarray, plan: list[dict]) -> bool:
    try:
        info = _inspect_grid(grid)
    except AssertionError:
        return False

    plan_sorted = sorted(plan, key=lambda item: (int(item["bbox"][0]), int(item["bbox"][1])))
    if len(info) != len(plan_sorted):
        return False

    signatures = set()
    saw_zero = False
    saw_collision = False
    saw_free = False
    holes = []
    areas = []
    widths = []
    heights = []

    for observed, expected in zip(info, plan_sorted):
        if observed["bbox"] != expected["bbox"]:
            return False
        if observed["holes"] != expected["holes"]:
            return False
        if observed["barrier"][1] != expected["barrier_col"]:
            return False

        holes.append(int(observed["holes"]))
        areas.append(int(observed["area"]))
        widths.append(int(observed["width"]))
        heights.append(int(observed["height"]))
        signatures.add(observed["crop"].tobytes())

        if observed["holes"] == 0:
            saw_zero = True
        if observed["holes"] > 0 and observed["shift"] == observed["holes"]:
            saw_free = True
        if 0 < observed["shift"] < observed["holes"]:
            saw_collision = True

    if len(signatures) != len(info):
        return False
    if not (saw_zero and saw_collision and saw_free):
        return False
    if _linear_relation(areas, holes):
        return False
    if _linear_relation(widths, holes):
        return False
    if _linear_relation(heights, holes):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.array_equal(out == OBSTACLE, grid == OBSTACLE):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        built = _build_attempt(rng)
        if built is None:
            continue
        grid, plan = built
        if _validate_attempt(grid, plan):
            return grid
    raise RuntimeError("failed to generate a valid hole-count slide puzzle")
