from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import bbox, components_mask, find_components, get_bg


BG = 0
OBJ_COLOR = 7
SIDE_COLORS = {
    "up": 2,
    "right": 3,
    "down": 4,
    "left": 1,
}
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_SEQUENCE = ("up", "right", "down", "left")
ORTHOGONAL_PAIRS = (
    ("left", "up"),
    ("left", "down"),
    ("right", "up"),
    ("right", "down"),
    ("up", "left"),
    ("up", "right"),
    ("down", "left"),
    ("down", "right"),
)
CANONICAL_REQUIRED = {
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
}
GEN_RETRIES = 500


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, xy)) for xy in np.asarray(coords, dtype=int)}


def _shift_coords(coords: np.ndarray, step: tuple[int, int], amount: int = 1) -> np.ndarray:
    dr, dc = step
    delta = np.array([dr * amount, dc * amount], dtype=int)
    return np.asarray(coords, dtype=int) + delta


def _neighbors8_of_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _frame_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    box = bbox(grid != bg)
    assert box is not None, "expected a non-empty framed puzzle"
    r0, c0, r1, c1 = map(int, box)
    assert r1 - r0 >= 6 and c1 - c0 >= 6, "frame must be at least 7x7"
    return r0, c0, r1, c1


def _dominant_non_bg(values: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in values.tolist() if int(v) != bg]
    assert vals, "expected a colored frame edge"
    counts = np.bincount(vals, minlength=10)
    return int(np.argmax(counts))


def _frame_side_colors(grid: np.ndarray, frame: tuple[int, int, int, int], bg: int) -> dict[str, int]:
    r0, c0, r1, c1 = frame
    colors = {
        "up": _dominant_non_bg(grid[r0, c0 + 1 : c1], bg),
        "right": _dominant_non_bg(grid[r0 + 1 : r1, c1], bg),
        "down": _dominant_non_bg(grid[r1, c0 + 1 : c1], bg),
        "left": _dominant_non_bg(grid[r0 + 1 : r1, c0], bg),
    }
    assert len(set(colors.values())) == 4, "frame sides must use four distinct colors"
    return colors


def _canonical_mask_open_up() -> np.ndarray:
    mask = np.ones((5, 5), dtype=bool)
    center = (2, 2)
    cross = [center, (1, 2), (3, 2), (2, 1), (2, 3)]
    for rc in cross:
        mask[rc] = False
    mask[0, 2] = False
    return mask


def _rotate_from_up(mask: np.ndarray, direction: str) -> np.ndarray:
    turns = {"up": 0, "left": 1, "down": 2, "right": 3}[direction]
    return np.rot90(mask, k=turns)


def _mask_connected(mask: np.ndarray) -> bool:
    comps = components_mask(mask.astype(bool), connectivity=4)
    return len(comps) == 1


def _generate_mask(direction: str) -> np.ndarray:
    mask = _canonical_mask_open_up()
    removable = [tuple(map(int, xy)) for xy in np.argwhere(mask)]
    removable = [rc for rc in removable if rc not in CANONICAL_REQUIRED]
    random.shuffle(removable)
    target_removals = random.randint(0, 4)
    removed = 0
    for rr, cc in removable:
        if removed >= target_removals:
            break
        trial = mask.copy()
        trial[rr, cc] = False
        if int(trial.sum()) < 15:
            continue
        if not _mask_connected(trial):
            continue
        mask = trial
        removed += 1
    return _rotate_from_up(mask, direction)


def _detect_hole(
    grid: np.ndarray,
    bg: int,
    coords: np.ndarray,
    bbox_rc: tuple[int, int, int, int],
) -> tuple[tuple[int, int], str]:
    r0, c0, r1, c1 = map(int, bbox_rc)
    comp = _coord_set(coords)
    candidates: list[tuple[tuple[int, int], str]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != bg:
                continue
            if any(int(grid[r + dr, c + dc]) != bg for dr, dc in DIRS.values()):
                continue
            diagonals = ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))
            if any(pos not in comp for pos in diagonals):
                continue
            open_dirs: list[str] = []
            closed_dirs: list[str] = []
            for name, (dr, dc) in DIRS.items():
                rr = r + 2 * dr
                cc = c + 2 * dc
                if (rr, cc) in comp:
                    closed_dirs.append(name)
                elif 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and int(grid[rr, cc]) == bg:
                    open_dirs.append(name)
            if len(open_dirs) == 1 and len(closed_dirs) == 3:
                candidates.append(((r, c), open_dirs[0]))
    assert len(candidates) == 1, "each object must contain exactly one detectable cross-gap"
    return candidates[0]


def _distance_to_wall(coords: np.ndarray, frame: tuple[int, int, int, int], direction: str) -> int:
    r0, c0, r1, c1 = frame
    rows = coords[:, 0]
    cols = coords[:, 1]
    if direction == "up":
        return int(rows.min() - (r0 + 1))
    if direction == "down":
        return int((r1 - 1) - rows.max())
    if direction == "left":
        return int(cols.min() - (c0 + 1))
    return int((c1 - 1) - cols.max())


def _extract_objects(
    grid: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
) -> list[dict]:
    fr0, fc0, fr1, fc1 = frame
    objects = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 <= fr0 or c0 <= fc0 or r1 >= fr1 or c1 >= fc1:
            continue
        center, direction = _detect_hole(grid, bg, comp["coords"], comp["bbox"])
        distance = _distance_to_wall(comp["coords"], frame, direction)
        objects.append(
            {
                "color": int(comp["color"]),
                "coords": np.asarray(comp["coords"], dtype=int),
                "bbox": tuple(map(int, comp["bbox"])),
                "hole_center": tuple(map(int, center)),
                "direction": direction,
                "distance": distance,
            }
        )
    return objects


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict], dict[str, int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame = _frame_bbox(g, bg)
    side_colors = _frame_side_colors(g, frame, bg)
    objects = _extract_objects(g, bg, frame)
    assert objects, "expected at least one inner object"
    assert len({obj["distance"] for obj in objects}) == len(objects), "object-to-wall distances must be unique"

    ordered = sorted(
        objects,
        key=lambda obj: (
            int(obj["distance"]),
            int(obj["hole_center"][0]),
            int(obj["hole_center"][1]),
        ),
    )

    out = g.copy()
    stats: list[dict] = []
    for obj in ordered:
        coords = np.asarray(obj["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = bg
        step = DIRS[obj["direction"]]
        placed = coords.copy()
        hole_r, hole_c = map(int, obj["hole_center"])
        travelled = 0
        while True:
            nxt = _shift_coords(placed, step)
            if np.any(out[nxt[:, 0], nxt[:, 1]] != bg):
                break
            placed = nxt
            hole_r += step[0]
            hole_c += step[1]
            travelled += 1
        out[placed[:, 0], placed[:, 1]] = int(obj["color"])
        fill_color = int(side_colors[obj["direction"]])
        cross = [(hole_r, hole_c)] + [(hole_r + dr, hole_c + dc) for dr, dc in DIRS.values()]
        for rr, cc in cross:
            out[rr, cc] = fill_color
        stats.append(
            {
                "direction": obj["direction"],
                "raw_distance": int(obj["distance"]),
                "travelled": int(travelled),
                "blocked": int(travelled) < int(obj["distance"]),
            }
        )
    return out, stats, side_colors, frame


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each orange object inside the colored frame according to its cross-gap opening, then color the gap from the target wall.

    Input structure:
    - The background is black.
    - One 1-pixel-thick rectangular frame surrounds the scene; its top, right, bottom, and left edges use four distinct colors.
    - Inside the frame are 3 to 5 separate orange 4-connected objects, each containing one 5-cell cross-shaped background gap with exactly one open arm.

    Rule:
    - For each orange object, detect the opening direction of its cross-shaped gap.
    - Sort the objects by their distance to the corresponding frame wall in that direction, nearest first.
    - Remove each object from its source position and translate it step by step in that direction until the next step would hit the frame or an already present object.
    - After it stops, fill the moved cross-gap with the color of the frame side indicated by the gap opening.
    - Leave the frame unchanged.

    Key insight:
    - The cross-gap opening is dual-purpose: it selects both the motion direction and the wall color used to fill the gap after the object settles.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame = _frame_bbox(g, bg)
    parsed = _extract_objects(g, bg, frame)
    assert 3 <= len(parsed) <= 5, "expected 3 to 5 inner objects"
    out, _stats, _side_colors, _frame = _simulate(g)
    return out


def _stamp_frame(grid: np.ndarray, frame: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame
    grid[r0, c0 + 1 : c1] = SIDE_COLORS["up"]
    grid[r1, c0 + 1 : c1] = SIDE_COLORS["down"]
    grid[r0 + 1 : r1, c0] = SIDE_COLORS["left"]
    grid[r0 + 1 : r1, c1] = SIDE_COLORS["right"]
    grid[r0, c0] = SIDE_COLORS[random.choice(("up", "left"))]
    grid[r0, c1] = SIDE_COLORS[random.choice(("up", "right"))]
    grid[r1, c0] = SIDE_COLORS[random.choice(("down", "left"))]
    grid[r1, c1] = SIDE_COLORS[random.choice(("down", "right"))]


def _coords_from_mask(mask: np.ndarray, top_left: tuple[int, int]) -> np.ndarray:
    base = np.argwhere(mask)
    return base + np.array(top_left, dtype=int)


def _touches_existing(coords: np.ndarray, placed_sets: list[set[tuple[int, int]]]) -> bool:
    current = _coord_set(coords)
    halo = _neighbors8_of_coords(current)
    return any(halo & other for other in placed_sets)


def _path_clear_to_wall(
    coords: np.ndarray,
    frame: tuple[int, int, int, int],
    direction: str,
    placed_sets: list[set[tuple[int, int]]],
) -> bool:
    step = DIRS[direction]
    limit = _distance_to_wall(coords, frame, direction)
    if limit <= 0:
        return False
    for amount in range(limit + 1):
        shifted = _coord_set(_shift_coords(coords, step, amount))
        if any(shifted & other for other in placed_sets):
            return False
    return True


def _build_spec(
    mask: np.ndarray,
    top_left: tuple[int, int],
    direction: str,
    frame: tuple[int, int, int, int],
) -> dict:
    coords = _coords_from_mask(mask, top_left)
    hole_center = (int(top_left[0] + 2), int(top_left[1] + 2))
    return {
        "mask": mask.copy(),
        "top_left": tuple(map(int, top_left)),
        "coords": np.asarray(coords, dtype=int),
        "direction": direction,
        "hole_center": hole_center,
        "distance": _distance_to_wall(coords, frame, direction),
    }


def _place_random_objects(
    frame: tuple[int, int, int, int],
    count: int,
    seed_specs: list[dict] | None = None,
) -> list[dict] | None:
    fr0, fc0, fr1, fc1 = frame
    specs = [] if seed_specs is None else [dict(spec) for spec in seed_specs]
    placed_sets = [_coord_set(spec["coords"]) for spec in specs]
    used_distances = {int(spec["distance"]) for spec in specs}

    while len(specs) < count:
        success = False
        for _ in range(180):
            direction = random.choice(DIR_SEQUENCE)
            mask = _generate_mask(direction)
            h, w = mask.shape
            row = random.randint(fr0 + 1, fr1 - h)
            col = random.randint(fc0 + 1, fc1 - w)
            spec = _build_spec(mask, (row, col), direction, frame)
            if int(spec["distance"]) in used_distances:
                continue
            if _touches_existing(spec["coords"], placed_sets):
                continue
            if not _path_clear_to_wall(spec["coords"], frame, direction, placed_sets):
                continue
            specs.append(spec)
            placed_sets.append(_coord_set(spec["coords"]))
            used_distances.add(int(spec["distance"]))
            success = True
            break
        if not success:
            return None
    return specs


def _generate_blocking_pair(frame: tuple[int, int, int, int]) -> list[dict] | None:
    for _ in range(250):
        first_dir, second_dir = random.choice(ORTHOGONAL_PAIRS)
        first_mask = _generate_mask(first_dir)
        second_mask = _generate_mask(second_dir)
        h1, w1 = first_mask.shape
        h2, w2 = second_mask.shape
        fr0, fc0, fr1, fc1 = frame

        row1 = random.randint(fr0 + 1, fr1 - h1)
        col1 = random.randint(fc0 + 1, fc1 - w1)
        row2 = random.randint(fr0 + 1, fr1 - h2)
        col2 = random.randint(fc0 + 1, fc1 - w2)

        first = _build_spec(first_mask, (row1, col1), first_dir, frame)
        second = _build_spec(second_mask, (row2, col2), second_dir, frame)
        if first["distance"] <= 0 or second["distance"] <= 0:
            continue
        if first["distance"] >= second["distance"]:
            continue
        sets = [_coord_set(first["coords"])]
        if _touches_existing(second["coords"], sets):
            continue
        if not _path_clear_to_wall(first["coords"], frame, first_dir, [_coord_set(second["coords"])]):
            continue
        if not _path_clear_to_wall(second["coords"], frame, second_dir, [_coord_set(first["coords"])]):
            continue

        grid = np.full((fr1 + 2, fc1 + 2), BG, dtype=int)
        _stamp_frame(grid, frame)
        for spec in (first, second):
            grid[spec["coords"][:, 0], spec["coords"][:, 1]] = OBJ_COLOR
        out, stats, _side_colors, _frame = _simulate(grid)
        _ = out
        if any(stat["blocked"] for stat in stats):
            return [first, second]
    return None


def _grid_from_specs(shape: tuple[int, int], frame: tuple[int, int, int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    _stamp_frame(grid, frame)
    for spec in specs:
        coords = spec["coords"]
        grid[coords[:, 0], coords[:, 1]] = OBJ_COLOR
    return grid


def _validate_generated_input(grid: np.ndarray, require_blocking: bool) -> bool:
    bg = get_bg(grid)
    frame = _frame_bbox(grid, bg)
    side_colors = _frame_side_colors(grid, frame, bg)
    if side_colors != SIDE_COLORS:
        return False
    objects = _extract_objects(grid, bg, frame)
    if not (3 <= len(objects) <= 5):
        return False
    if len({obj["direction"] for obj in objects}) < 2:
        return False
    distances = [int(obj["distance"]) for obj in objects]
    if len(set(distances)) != len(distances):
        return False
    sets = [_coord_set(obj["coords"]) for obj in objects]
    for i, obj in enumerate(objects):
        others = [sets[j] for j in range(len(objects)) if j != i]
        if not _path_clear_to_wall(obj["coords"], frame, obj["direction"], others):
            return False
    out, stats, _side_colors, _frame = _simulate(grid)
    if np.array_equal(out, grid):
        return False
    if require_blocking and not any(stat["blocked"] for stat in stats):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.randint(13, 18)
    inner_w = random.randint(13, 18)
    margin_top = random.randint(0, 2)
    margin_left = random.randint(0, 2)
    margin_bottom = random.randint(0, 2)
    margin_right = random.randint(0, 2)
    height = inner_h + 2 + margin_top + margin_bottom
    width = inner_w + 2 + margin_left + margin_right
    if height > 30 or width > 30:
        return None
    frame = (margin_top, margin_left, margin_top + inner_h + 1, margin_left + inner_w + 1)

    require_blocking = random.random() < 0.45
    if require_blocking:
        seeds = _generate_blocking_pair(frame)
        if seeds is None:
            return None
        count = random.randint(4, 5)
    else:
        seeds = []
        count = random.randint(3, 5)

    specs = _place_random_objects(frame, count, seed_specs=seeds)
    if specs is None:
        return None

    if len({spec["direction"] for spec in specs}) < 2:
        return None

    grid = _grid_from_specs((height, width), frame, specs)
    if not _validate_generated_input(grid, require_blocking):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-8 input")
