from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS) not in sys.path:
    sys.path.append(str(ARC_UTILS))

from arc_utils import Color, find_components, get_bg


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    h, w = grid.shape
    assert w % 2 == 1, "input must contain two equal-width panels and one divider column"
    mid = w // 2
    divider = grid[:, mid]
    assert np.all(divider == divider[0]), "divider must be a solid column"
    left = grid[:, :mid]
    right = grid[:, mid + 1 :]
    assert left.shape == right.shape
    bg_left = int(get_bg(left))
    bg_right = int(get_bg(right))
    assert bg_left == bg_right, "both panels must share the same background"
    return left, right, bg_left


def _record_objects(panel: np.ndarray, bg: int) -> list[dict]:
    records = []
    for comp in find_components(panel, bg=bg, connectivity=4, monochromatic=True):
        coords = np.asarray(comp["coords"], dtype=int)
        anchor = coords.min(axis=0)
        rel = tuple(sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in coords))
        coord_key = frozenset((int(r), int(c)) for r, c in coords)
        records.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "anchor": (int(anchor[0]), int(anchor[1])),
                "shape_key": rel,
                "coord_key": coord_key,
                "signature": (int(comp["color"]), rel),
            }
        )
    return records


def _match_objects(left_objs: list[dict], right_objs: list[dict]) -> list[dict]:
    left_by_sig: dict[tuple, list[dict]] = defaultdict(list)
    right_by_sig: dict[tuple, list[dict]] = defaultdict(list)
    for obj in left_objs:
        left_by_sig[obj["signature"]].append(obj)
    for obj in right_objs:
        right_by_sig[obj["signature"]].append(obj)

    pairs: list[dict] = []
    for sig in sorted(set(left_by_sig) | set(right_by_sig), key=str):
        left_group = list(left_by_sig.get(sig, []))
        right_group = list(right_by_sig.get(sig, []))
        assert len(left_group) == len(right_group), "object counts by color+shape must match across panels"

        right_exact = {obj["coord_key"]: obj for obj in right_group}
        used_right_ids: set[int] = set()
        remaining_left: list[dict] = []
        for left_obj in left_group:
            right_obj = right_exact.get(left_obj["coord_key"])
            if right_obj is None or id(right_obj) in used_right_ids:
                remaining_left.append(left_obj)
                continue
            used_right_ids.add(id(right_obj))
            pairs.append({"left": left_obj, "right": right_obj, "velocity": (0, 0)})

        remaining_right = [obj for obj in right_group if id(obj) not in used_right_ids]
        if not remaining_left:
            assert not remaining_right
            continue

        assert len(remaining_left) == len(remaining_right) == 1, "ambiguous moving-object correspondence"
        left_obj = remaining_left[0]
        right_obj = remaining_right[0]
        dr = int(right_obj["anchor"][0] - left_obj["anchor"][0])
        dc = int(right_obj["anchor"][1] - left_obj["anchor"][1])
        assert (dr, dc) != (0, 0)
        pairs.append({"left": left_obj, "right": right_obj, "velocity": (dr, dc)})

    assert len(pairs) == len(left_objs) == len(right_objs)
    return pairs


def _offset_path(dr: int, dc: int) -> list[tuple[int, int]]:
    steps = max(abs(int(dr)), abs(int(dc)))
    assert steps > 0
    path: list[tuple[int, int]] = []
    prev = (0, 0)
    for k in range(1, steps + 1):
        off = (int(round(dr * k / steps)), int(round(dc * k / steps)))
        if off != prev:
            path.append(off)
            prev = off
    if path[-1] != (int(dr), int(dc)):
        path.append((int(dr), int(dc)))
    return path


def _try_offset(
    coords: np.ndarray, offset: tuple[int, int], static_mask: np.ndarray
) -> tuple[bool, str | None, np.ndarray]:
    dr, dc = offset
    moved = coords + np.array([dr, dc], dtype=int)
    h, w = static_mask.shape
    if (
        int(moved[:, 0].min()) < 0
        or int(moved[:, 0].max()) >= h
        or int(moved[:, 1].min()) < 0
        or int(moved[:, 1].max()) >= w
    ):
        return False, "boundary", moved
    if bool(static_mask[moved[:, 0], moved[:, 1]].any()):
        return False, "static", moved
    return True, None, moved


def _slide_coords(
    coords: np.ndarray, velocity: tuple[int, int], static_mask: np.ndarray
) -> tuple[np.ndarray, str | None]:
    best = coords.copy()
    fail_reason: str | None = None
    for offset in _offset_path(*velocity):
        ok, reason, moved = _try_offset(coords, offset, static_mask)
        if not ok:
            fail_reason = reason
            break
        best = moved
    return best, fail_reason


def _paint(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate a third animation frame from two aligned panels while stopping movers at static blockers.

    Input structure:
    - A solid divider column splits the grid into equal left and right panels with the same background.
    - Each panel contains several monochrome 4-connected objects; some stay fixed while others are translated copies.
    - Matching objects preserve both color and shape across the two frames.

    Rule:
    - Compare the left and right panels object by object to recover each moving object's translation vector.
    - Treat unchanged objects in the right panel as static obstacles.
    - Starting from the right panel, move every translated object once more by its own vector; if the full move would hit a static object or leave the panel, keep only the longest valid prefix of that motion.

    Key insight:
    - The two panels are consecutive time steps, so the answer is the next frame rather than a visual diff.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    left, right, bg = _split_panels(g)
    left_objs = _record_objects(left, bg)
    right_objs = _record_objects(right, bg)
    pairs = _match_objects(left_objs, right_objs)

    out = np.full_like(right, bg)
    static_mask = np.zeros(right.shape, dtype=bool)
    for pair in pairs:
        if pair["velocity"] != (0, 0):
            continue
        coords = pair["right"]["coords"]
        _paint(out, coords, pair["right"]["color"])
        static_mask[coords[:, 0], coords[:, 1]] = True

    occupied = static_mask.copy()
    moving_pairs = [pair for pair in pairs if pair["velocity"] != (0, 0)]
    moving_pairs.sort(key=lambda pair: (pair["right"]["color"], pair["velocity"]))
    for pair in moving_pairs:
        final_coords, _reason = _slide_coords(pair["right"]["coords"], pair["velocity"], static_mask)
        assert not bool(occupied[final_coords[:, 0], final_coords[:, 1]].any()), "moving objects must remain disjoint"
        _paint(out, final_coords, pair["right"]["color"])
        occupied[final_coords[:, 0], final_coords[:, 1]] = True

    return out


GRID_ROWS = (9, 14)
GRID_COLS = (9, 14)
MOVER_COUNT = (1, 3)
STATIC_COUNT = (1, 3)
AREA_RANGE = (2, 6)
SHAPE_BOX = (2, 4)
SPEED_CHOICES = (1, 2, 3)
STOP_SPEED_CHOICES = (2, 3)
PADDING = 1
GEN_RETRIES = 400

DIRS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    anchor = coords.min(axis=0)
    return tuple(sorted((int(r - anchor[0]), int(c - anchor[1])) for r, c in coords))


def _random_connected_mask() -> np.ndarray:
    box_h = random.randint(*SHAPE_BOX)
    box_w = random.randint(*SHAPE_BOX)
    max_area = min(AREA_RANGE[1], box_h * box_w)
    area = random.randint(AREA_RANGE[0], max_area)

    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    add_frontier(start)
    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        add_frontier(cell)

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _make_unique_masks(count: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    sigs: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(200):
        if len(masks) == count:
            return masks
        mask = _random_connected_mask()
        sig = _mask_signature(mask)
        if sig in sigs:
            continue
        masks.append(mask)
        sigs.add(sig)
    raise RuntimeError("failed to create enough unique shapes")


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask)
    return coords + np.array([int(r0), int(c0)], dtype=int)


def _coords_with_padding(coords: np.ndarray, shape: tuple[int, int], pad: int) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr in range(int(r) - pad, int(r) + pad + 1):
            for nc in range(int(c) - pad, int(c) + pad + 1):
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _fits_with_padding(coords: np.ndarray, occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    if (
        int(coords[:, 0].min()) < 0
        or int(coords[:, 0].max()) >= h
        or int(coords[:, 1].min()) < 0
        or int(coords[:, 1].max()) >= w
    ):
        return False
    return all((int(r), int(c)) not in occupied for r, c in _coords_with_padding(coords, shape, PADDING))


def _coords_overlap(coords: np.ndarray, occupied: np.ndarray) -> bool:
    return bool(occupied[coords[:, 0], coords[:, 1]].any())


def _place_static_objects(
    left: np.ndarray,
    right: np.ndarray,
    masks: list[np.ndarray],
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, int]], set[tuple[int, int]], np.ndarray] | None:
    h, w = left.shape
    left_occ: set[tuple[int, int]] = set()
    right_occ: set[tuple[int, int]] = set()
    static_mask = np.zeros((h, w), dtype=bool)
    placed: list[tuple[np.ndarray, int]] = []

    for mask, color in zip(masks, colors):
        mh, mw = mask.shape
        options = [(r0, c0) for r0 in range(h - mh + 1) for c0 in range(w - mw + 1)]
        random.shuffle(options)
        chosen = None
        for r0, c0 in options:
            coords = _coords_from_mask(mask, r0, c0)
            if not _fits_with_padding(coords, left_occ, (h, w)):
                continue
            chosen = coords
            break
        if chosen is None:
            return None
        left[chosen[:, 0], chosen[:, 1]] = color
        right[chosen[:, 0], chosen[:, 1]] = color
        placed.append((chosen, color))
        for cell in _coords_with_padding(chosen, (h, w), PADDING):
            left_occ.add(cell)
            right_occ.add(cell)
        static_mask[chosen[:, 0], chosen[:, 1]] = True

    return left, right, placed, left_occ, static_mask


def _place_mover(
    left: np.ndarray,
    right: np.ndarray,
    expected: np.ndarray,
    mask: np.ndarray,
    color: int,
    mode: str,
    left_occ: set[tuple[int, int]],
    right_occ: set[tuple[int, int]],
    static_mask: np.ndarray,
    final_occ: np.ndarray,
) -> bool:
    h, w = left.shape
    dirs = random.sample(DIRS, len(DIRS))
    for dr_unit, dc_unit in dirs:
        speeds = STOP_SPEED_CHOICES if mode != "free" else SPEED_CHOICES
        for speed in random.sample(list(speeds), len(speeds)):
            velocity = (dr_unit * speed, dc_unit * speed)
            mh, mw = mask.shape
            positions = [(r0, c0) for r0 in range(h - mh + 1) for c0 in range(w - mw + 1)]
            random.shuffle(positions)
            for r0, c0 in positions:
                right_coords = _coords_from_mask(mask, r0, c0)
                left_coords = right_coords - np.array(velocity, dtype=int)
                if not _fits_with_padding(left_coords, left_occ, (h, w)):
                    continue
                if not _fits_with_padding(right_coords, right_occ, (h, w)):
                    continue

                final_coords, reason = _slide_coords(right_coords, velocity, static_mask)
                full_coords = right_coords + np.array(velocity, dtype=int)
                full_reached = np.array_equal(final_coords, full_coords)
                if mode == "free" and not full_reached:
                    continue
                if mode == "wall" and reason != "boundary":
                    continue
                if mode == "obstacle" and reason != "static":
                    continue
                if not full_reached and np.array_equal(final_coords, right_coords):
                    continue
                if _coords_overlap(final_coords, final_occ):
                    continue

                left[left_coords[:, 0], left_coords[:, 1]] = color
                right[right_coords[:, 0], right_coords[:, 1]] = color
                expected[final_coords[:, 0], final_coords[:, 1]] = color
                for cell in _coords_with_padding(left_coords, (h, w), PADDING):
                    left_occ.add(cell)
                for cell in _coords_with_padding(right_coords, (h, w), PADDING):
                    right_occ.add(cell)
                final_occ[final_coords[:, 0], final_coords[:, 1]] = True
                return True
    return False


def _choose_modes(n_movers: int) -> list[str]:
    modes = []
    stop_modes = ["wall", "obstacle"]
    if random.random() < 0.75:
        modes.append(random.choice(stop_modes))
    while len(modes) < n_movers:
        modes.append(random.choice(["free", "free", "wall", "obstacle"]))
    random.shuffle(modes)
    return modes[:n_movers]


def _build_input(left: np.ndarray, right: np.ndarray, divider_color: int) -> np.ndarray:
    h, w = left.shape
    out = np.full((h, 2 * w + 1), int(divider_color), dtype=int)
    out[:, :w] = left
    out[:, w + 1 :] = right
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    bg = random.choice([Color.BLACK, Color.TEAL])
    divider_color = random.choice([c for c in range(10) if c != bg])
    n_static = random.randint(*STATIC_COUNT)
    n_movers = random.randint(*MOVER_COUNT)
    total_objects = n_static + n_movers

    palette = [c for c in range(10) if c not in {bg, divider_color}]
    if total_objects > len(palette):
        return None
    colors = random.sample(palette, total_objects)
    masks = _make_unique_masks(total_objects)

    left = np.full((h, w), bg, dtype=int)
    right = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)

    static_pack = _place_static_objects(left, right, masks[:n_static], colors[:n_static])
    if static_pack is None:
        return None
    left, right, placed_static, left_occ, static_mask = static_pack
    right_occ = set(left_occ)
    final_occ = static_mask.copy()
    for coords, color in placed_static:
        expected[coords[:, 0], coords[:, 1]] = color

    modes = _choose_modes(n_movers)
    for mask, color, mode in zip(masks[n_static:], colors[n_static:], modes):
        if not _place_mover(
            left,
            right,
            expected,
            mask,
            color,
            mode,
            left_occ,
            right_occ,
            static_mask,
            final_occ,
        ):
            return None

    inp = _build_input(left, right, divider_color)
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-panel extrapolation input")
