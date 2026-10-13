from __future__ import annotations

from collections import Counter
import numpy as np


BG = 0
BODY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_clockwise(pattern: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(pattern, -turns)


_BASE_UP_PATTERNS = {
    "U": np.array(
        [
            [0, 2, 0],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    "L_left": np.array(
        [
            [0, 2, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    "L_right": np.array(
        [
            [0, 2, 0],
            [0, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
}

_DIRECTION_TURNS = {"up": 0, "right": 1, "down": 2, "left": 3}
_PATTERNS_BY_DIRECTION: dict[str, list[np.ndarray]] = {}
for _direction, _turns in _DIRECTION_TURNS.items():
    _PATTERNS_BY_DIRECTION[_direction] = [
        _rotate_clockwise(pattern, _turns) for pattern in _BASE_UP_PATTERNS.values()
    ]


def _paint_object_patch(direction: str, shape_name: str, marker_color: int) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    pattern = _rotate_clockwise(_BASE_UP_PATTERNS[shape_name], _DIRECTION_TURNS[direction])
    patch[pattern == 1] = BODY
    patch[pattern == 2] = int(marker_color)
    return patch


def _classify_arrow(
    coords: list[tuple[int, int]], color_grid: np.ndarray
) -> tuple[str, set[tuple[int, int]]] | None:
    h, w = color_grid.shape
    r0, c0, r1, c1 = _bbox(coords)
    rel = {(r - r0, c - c0) for r, c in coords}
    touches = {
        side
        for side, cond in (
            ("top", r0 == 0),
            ("bottom", r1 == h - 1),
            ("left", c0 == 0),
            ("right", c1 == w - 1),
        )
        if cond
    }
    if len(touches) != 1:
        return None
    side = next(iter(touches))
    if side == "left" and (r1 - r0 + 1, c1 - c0 + 1) == (3, 2):
        if rel == {(0, 0), (1, 0), (2, 0), (1, 1)}:
            return "right", set(coords)
    if side == "right" and (r1 - r0 + 1, c1 - c0 + 1) == (3, 2):
        if rel == {(0, 1), (1, 1), (2, 1), (1, 0)}:
            return "left", set(coords)
    if side == "top" and (r1 - r0 + 1, c1 - c0 + 1) == (2, 3):
        if rel == {(0, 0), (0, 1), (0, 2), (1, 1)}:
            return "down", set(coords)
    if side == "bottom" and (r1 - r0 + 1, c1 - c0 + 1) == (2, 3):
        if rel == {(1, 0), (1, 1), (1, 2), (0, 1)}:
            return "up", set(coords)
    return None


def _extract_scene(grid: np.ndarray) -> tuple[int, set[tuple[int, int]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    all_comps = _components(g != bg, connectivity=8)
    arrow_cells: set[tuple[int, int]] = set()
    arrow_dirs: set[str] = set()
    for comp in all_comps:
        arrow_info = _classify_arrow(comp, g)
        if arrow_info is None:
            continue
        direction, cells = arrow_info
        arrow_dirs.add(direction)
        arrow_cells |= cells

    assert arrow_dirs == {"up", "down", "left", "right"}, "expected one arrow for each direction"

    objects: list[dict[str, object]] = []
    object_mask = (g != bg).copy()
    for r, c in arrow_cells:
        object_mask[r, c] = False

    for comp in _components(object_mask, connectivity=8):
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "objects must occupy a 3x3 box"
        assert 0 < r0 and 0 < c0 and r1 < h - 1 and c1 < w - 1, "objects must start in the interior"

        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        counts = Counter(int(g[r, c]) for r, c in comp)
        body_color, body_count = max(counts.items(), key=lambda kv: kv[1])
        assert body_count >= 4, "body must dominate the object"
        marker_cells = [(r, c) for r, c in comp if int(g[r, c]) != body_color]
        assert len(marker_cells) == 1, "each object must have one colored opening marker"
        marker_r, marker_c = marker_cells[0]
        marker_color = int(g[marker_r, marker_c])
        marker_rel = (marker_r - r0, marker_c - c0)
        rel_patch = np.zeros((3, 3), dtype=int)
        for r, c in comp:
            rel_patch[r - r0, c - c0] = 2 if (r, c) == (marker_r, marker_c) else 1

        marker_to_dir = {
            (0, 1): "up",
            (2, 1): "down",
            (1, 0): "left",
            (1, 2): "right",
        }
        assert marker_rel in marker_to_dir, "marker must sit on the center of the opening side"
        direction = marker_to_dir[marker_rel]
        assert any(np.array_equal(rel_patch, pat) for pat in _PATTERNS_BY_DIRECTION[direction]), "unexpected object shape"
        assert direction in arrow_dirs, "object direction must match an edge arrow"

        rel_cells = [(rr, cc) for rr in range(3) for cc in range(3) if patch[rr, cc] != bg]
        objects.append(
            {
                "patch": patch,
                "direction": direction,
                "marker_color": marker_color,
                "marker_rel": marker_rel,
                "rel_cells": rel_cells,
                "orig_top": r0,
                "orig_left": c0,
                "top": r0,
                "left": c0,
            }
        )

    assert objects, "expected at least one object"
    return bg, arrow_cells, objects


def _cells_for_object(obj: dict[str, object]) -> set[tuple[int, int]]:
    top = int(obj["top"])
    left = int(obj["left"])
    return {(top + rr, left + cc) for rr, cc in obj["rel_cells"]}


def _simulate(objects: list[dict[str, object]], h: int, w: int) -> None:
    for _ in range(h * w):
        current_cells = [_cells_for_object(obj) for obj in objects]
        occupancy: dict[tuple[int, int], int] = {}
        for idx, cells in enumerate(current_cells):
            for cell in cells:
                occupancy[cell] = idx

        proposals: dict[int, set[tuple[int, int]]] = {}
        for idx, obj in enumerate(objects):
            dr, dc = DIR_VECS[str(obj["direction"])]
            shifted = {(r + dr, c + dc) for r, c in current_cells[idx]}
            if any(r < 0 or r >= h or c < 0 or c >= w for r, c in shifted):
                continue
            if any(occupancy.get(cell, idx) != idx for cell in shifted):
                continue
            proposals[idx] = shifted

        if not proposals:
            return

        conflicts: set[int] = set()
        items = list(proposals.items())
        for i in range(len(items)):
            idx_a, cells_a = items[i]
            for j in range(i + 1, len(items)):
                idx_b, cells_b = items[j]
                if cells_a & cells_b:
                    conflicts.add(idx_a)
                    conflicts.add(idx_b)

        movers = [idx for idx in proposals if idx not in conflicts]
        if not movers:
            return

        for idx in movers:
            dr, dc = DIR_VECS[str(objects[idx]["direction"])]
            objects[idx]["top"] = int(objects[idx]["top"]) + dr
            objects[idx]["left"] = int(objects[idx]["left"]) + dc


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg, arrow_cells, objects = _extract_scene(g)
    _simulate(objects, h, w)

    out = g.copy()
    for r, c in arrow_cells:
        out[r, c] = bg
    for obj in objects:
        patch = obj["patch"]
        orig_top = int(obj["orig_top"])
        orig_left = int(obj["orig_left"])
        out[orig_top : orig_top + 3, orig_left : orig_left + 3][patch != bg] = bg

    for obj in objects:
        mr0 = int(obj["orig_top"]) + int(obj["marker_rel"][0])
        mc0 = int(obj["orig_left"]) + int(obj["marker_rel"][1])
        mr1 = int(obj["top"]) + int(obj["marker_rel"][0])
        mc1 = int(obj["left"]) + int(obj["marker_rel"][1])
        color = int(obj["marker_color"])
        if mr0 == mr1:
            c0, c1 = sorted((mc0, mc1))
            out[mr0, c0 : c1 + 1] = color
        else:
            r0, r1 = sorted((mr0, mr1))
            out[r0 : r1 + 1, mc0] = color

    for obj in objects:
        patch = obj["patch"]
        top = int(obj["top"])
        left = int(obj["left"])
        view = out[top : top + 3, left : left + 3]
        mask = patch != bg
        view[mask] = patch[mask]

    return out


def _paint_arrow(grid: np.ndarray, direction: str, color: int, center: int) -> None:
    h, w = grid.shape
    if direction == "right":
        patch = np.array([[1, 0], [1, 1], [1, 0]], dtype=int)
        r0, c0 = center - 1, 0
    elif direction == "left":
        patch = np.array([[0, 1], [1, 1], [0, 1]], dtype=int)
        r0, c0 = center - 1, w - 2
    elif direction == "down":
        patch = np.array([[1, 1, 1], [0, 1, 0]], dtype=int)
        r0, c0 = 0, center - 1
    else:
        patch = np.array([[0, 1, 0], [1, 1, 1]], dtype=int)
        r0, c0 = h - 2, center - 1
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][patch == 1] = int(color)


def _sample_spaced_positions(
    rng: np.random.Generator, low: int, high: int, count: int, gap: int
) -> list[int] | None:
    candidates = np.arange(low, high + 1)
    if len(candidates) < count:
        return None
    for _ in range(200):
        picked = sorted(int(v) for v in rng.choice(candidates, size=count, replace=False))
        if all(picked[i + 1] - picked[i] >= gap for i in range(count - 1)):
            return picked
    return None


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    grid[top : top + patch.shape[0], left : left + patch.shape[1]][patch != BG] = patch[patch != BG]


def _horizontal_lane_positions(
    rng: np.random.Generator, width: int, direction: str, count: int
) -> list[int] | None:
    final_lefts = (
        [width - 3 * count + 3 * i for i in range(count)]
        if direction == "right"
        else [3 * i for i in range(count)]
    )
    for _ in range(200):
        picks = _sample_spaced_positions(rng, 3, width - 6, count, 4)
        if picks is None:
            return None
        if direction == "right":
            if all(final_lefts[i] - picks[i] >= 2 for i in range(count)):
                return picks
        else:
            if all(picks[i] - final_lefts[i] >= 2 for i in range(count)):
                return picks
    return None


def _vertical_lane_positions(
    rng: np.random.Generator, height: int, direction: str, count: int
) -> list[int] | None:
    final_tops = (
        [height - 3 * count + 3 * i for i in range(count)]
        if direction == "down"
        else [3 * i for i in range(count)]
    )
    for _ in range(200):
        picks = _sample_spaced_positions(rng, 3, height - 6, count, 4)
        if picks is None:
            return None
        if direction == "down":
            if all(final_tops[i] - picks[i] >= 2 for i in range(count)):
                return picks
        else:
            if all(picks[i] - final_tops[i] >= 2 for i in range(count)):
                return picks
    return None


def _build_scene(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(17, 23))
    w = int(rng.integers(17, 23))
    axis_mode = "horizontal" if rng.random() < 0.5 else "vertical"
    grid = np.full((h, w), BG, dtype=int)

    if axis_mode == "horizontal":
        lane_count = 2 if rng.random() < 0.5 else 3
        rows = _sample_spaced_positions(rng, 3, h - 6, lane_count, 4)
        if rows is None:
            return None
        lane_dirs = ["left", "right"]
        if lane_count == 3:
            lane_dirs.append("left" if rng.random() < 0.5 else "right")
        rng.shuffle(lane_dirs)
        multi_idx = int(rng.integers(lane_count))
    else:
        lane_count = 2 if rng.random() < 0.5 else 3
        cols = _sample_spaced_positions(rng, 3, w - 6, lane_count, 4)
        if cols is None:
            return None
        lane_dirs = ["up", "down"]
        if lane_count == 3:
            lane_dirs.append("up" if rng.random() < 0.5 else "down")
        rng.shuffle(lane_dirs)
        multi_idx = int(rng.integers(lane_count))

    object_total = lane_count + 1
    colors = list(rng.choice(np.array(PALETTE), size=4 + object_total, replace=False))
    arrow_colors = colors[:4]
    object_colors = colors[4:]
    rng.shuffle(object_colors)

    arrow_centers = {
        "left": int(rng.integers(3, h - 3)),
        "right": int(rng.integers(3, h - 3)),
        "down": int(rng.integers(3, w - 3)),
        "up": int(rng.integers(3, w - 3)),
    }
    for direction, color in zip(("left", "right", "down", "up"), arrow_colors):
        _paint_arrow(grid, direction, color, arrow_centers[direction])

    objects_added = 0
    for lane_idx in range(lane_count):
        count = 2 if lane_idx == multi_idx else 1
        direction = lane_dirs[lane_idx]
        if axis_mode == "horizontal":
            lefts = _horizontal_lane_positions(rng, w, direction, count)
            if lefts is None:
                return None
            top = rows[lane_idx]
            for j in range(count):
                shape = "U" if count == 2 else ("L_left" if rng.random() < 0.5 else "L_right")
                patch = _paint_object_patch(direction, shape, object_colors[objects_added])
                _place_patch(grid, patch, top, lefts[j])
                objects_added += 1
        else:
            tops = _vertical_lane_positions(rng, h, direction, count)
            if tops is None:
                return None
            left = cols[lane_idx]
            for j in range(count):
                shape = "U" if count == 2 else ("L_left" if rng.random() < 0.5 else "L_right")
                patch = _paint_object_patch(direction, shape, object_colors[objects_added])
                _place_patch(grid, patch, tops[j], left)
                objects_added += 1

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _build_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
