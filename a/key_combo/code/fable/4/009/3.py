from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
START_MARKER = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield int(r) + dr, int(c) + dc


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == int(bg) or color == START_MARKER:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "id": len(comps),
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _coords_distance(a: np.ndarray, b: np.ndarray) -> int:
    dr = np.abs(a[:, None, 0] - b[None, :, 0])
    dc = np.abs(a[:, None, 1] - b[None, :, 1])
    return int(np.maximum(dr, dc).min())


def _object_distance(a: dict, b: dict) -> int:
    return _coords_distance(a["coords"], b["coords"])


def _point_object_distance(point: tuple[int, int], obj: dict) -> int:
    pr, pc = point
    delta = np.abs(obj["coords"] - np.asarray([pr, pc], dtype=int))
    return int(delta.max(axis=1).min())


def _center2(obj: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = obj["bbox"]
    return int(r0 + r1), int(c0 + c1)


def _center_distance(a: dict, b: dict) -> int:
    ar, ac = _center2(a)
    br, bc = _center2(b)
    return max(abs(ar - br), abs(ac - bc))


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    marker_pts = np.argwhere(g == START_MARKER)
    assert marker_pts.shape[0] == 1, "expected exactly one gray start marker pixel"
    marker = tuple(map(int, marker_pts[0]))

    objects = _components(g, bg)
    assert len(objects) >= 2, "expected multiple colored objects"
    colors = [obj["color"] for obj in objects]
    assert len(colors) == len(set(colors)), "object colors must be distinct"

    touching = [idx for idx, obj in enumerate(objects) if _point_object_distance(marker, obj) == 1]
    assert len(touching) == 1, "gray marker must touch exactly one object"
    return objects, marker, touching[0]


def _nearest_chain(objects: list[dict], start_idx: int, distance_fn=_object_distance) -> list[int]:
    chain = [int(start_idx)]
    unvisited = set(range(len(objects)))
    unvisited.remove(int(start_idx))

    while unvisited:
        current = chain[-1]
        distances = [(int(distance_fn(objects[current], objects[idx])), idx) for idx in unvisited]
        best = min(d for d, _idx in distances)
        winners = [idx for d, idx in distances if d == best]
        assert len(winners) == 1, "nearest-neighbor step must have no tie"
        chain.append(int(winners[0]))
        unvisited.remove(int(winners[0]))
    return chain


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate object colors by one step along the chain defined by greedy nearest-neighbor jumps.

    Input structure:
    - The dominant background contains several separated solid monochromatic objects, and all object colors are distinct.
    - Exactly one gray color-5 singleton pixel is adjacent to exactly one object; that object is the chain start.
    - Object distance is the minimum pixel-to-pixel Chebyshev distance between two objects.

    Rule:
    - Starting from the marked object, repeatedly append the nearest unvisited object, with the input guaranteed to have no distance ties.
    - Recolor every object as a whole with the previous object's color on that chain; the start object receives the tail object's color.
    - Keep object shapes and positions fixed, and remove the gray start marker back to the background.

    Key insight:
    - The color mapping is not determined by rows, columns, shapes, or color ids; it is the cyclic shift induced by a hidden nearest-neighbor traversal.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects, marker, start_idx = _parse_scene(g)
    chain = _nearest_chain(objects, start_idx)

    out = g.copy()
    out[marker] = bg
    old_colors = [objects[idx]["color"] for idx in chain]
    for pos, idx in enumerate(chain):
        coords = objects[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = old_colors[pos - 1]
    return out


# ---------------- generator ----------------

GRID_SIDE_RANGE = (15, 26)
OBJECT_COUNT_RANGE = (4, 7)
GENERATION_ATTEMPTS = 1200
PLACE_ATTEMPTS = 300

RECT_SIZES = (
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (1, 4),
    (4, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
)

BASE_MOTIF = (
    ((0, 0), (0, 1), (0, 2), (0, 3)),  # start object
    ((2, 5),),  # pixel-nearest from start
    ((3, 1),),  # center-nearest from start, but farther by pixel distance
)


def _motif_variant() -> list[np.ndarray]:
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (r, -c),
        lambda r, c: (-r, c),
        lambda r, c: (-r, -c),
        lambda r, c: (c, r),
        lambda r, c: (c, -r),
        lambda r, c: (-c, r),
        lambda r, c: (-c, -r),
    )
    transform = random.choice(transforms)
    pieces = []
    all_pts = []
    for obj in BASE_MOTIF:
        pts = np.asarray([transform(r, c) for r, c in obj], dtype=int)
        pieces.append(pts)
        all_pts.extend(map(tuple, pts))

    all_arr = np.asarray(all_pts, dtype=int)
    offset = all_arr.min(axis=0)
    return [pts - offset for pts in pieces]


def _paint_object(grid: np.ndarray, occ: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)
    occ[coords[:, 0], coords[:, 1]] = True


def _can_place_coords(occ: np.ndarray, coords: np.ndarray) -> bool:
    h, w = occ.shape
    for r, c in coords:
        if not (0 <= int(r) < h and 0 <= int(c) < w):
            return False
        r0, r1 = max(0, int(r) - 1), min(h, int(r) + 2)
        c0, c1 = max(0, int(c) - 1), min(w, int(c) + 2)
        if bool(occ[r0:r1, c0:c1].any()):
            return False
    return True


def _rect_coords(top: int, left: int, height: int, width: int) -> np.ndarray:
    return np.asarray(
        [(top + r, left + c) for r in range(int(height)) for c in range(int(width))],
        dtype=int,
    )


def _place_random_rect(
    grid: np.ndarray,
    occ: np.ndarray,
    color: int,
    start_coords: np.ndarray,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACE_ATTEMPTS):
        obj_h, obj_w = random.choice(RECT_SIZES)
        if obj_h + 4 > h or obj_w + 4 > w:
            continue
        top = random.randint(1, h - obj_h - 2)
        left = random.randint(1, w - obj_w - 2)
        coords = _rect_coords(top, left, obj_h, obj_w)
        if not _can_place_coords(occ, coords):
            continue
        if _coords_distance(coords, start_coords) <= 3:
            continue
        _paint_object(grid, occ, coords, color)
        return True
    return False


def _add_start_marker(grid: np.ndarray, start_coords: np.ndarray) -> bool:
    h, w = grid.shape
    start_set = {tuple(map(int, rc)) for rc in start_coords}
    candidates: set[tuple[int, int]] = set()

    for r, c in start_set:
        for rr, cc in _neighbors8(r, c):
            if not (0 <= rr < h and 0 <= cc < w) or int(grid[rr, cc]) != BACKGROUND:
                continue
            ok = True
            for nr, nc in _neighbors8(rr, cc):
                if (nr, nc) in start_set:
                    continue
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BACKGROUND:
                    ok = False
                    break
            if ok:
                candidates.add((rr, cc))

    if not candidates:
        return False
    r, c = random.choice(tuple(candidates))
    grid[r, c] = START_MARKER
    return True


def _row_major_order(objects: list[dict]) -> list[int]:
    return sorted(
        range(len(objects)),
        key=lambda idx: (
            objects[idx]["bbox"][0],
            objects[idx]["bbox"][1],
            objects[idx]["bbox"][2],
            objects[idx]["bbox"][3],
            objects[idx]["area"],
        ),
    )


def _distance_from_start_order(objects: list[dict], start_idx: int) -> list[int]:
    rest = [idx for idx in range(len(objects)) if idx != int(start_idx)]
    rest.sort(key=lambda idx: (_object_distance(objects[start_idx], objects[idx]), objects[idx]["bbox"]))
    return [int(start_idx)] + rest


def _is_monotone(values: list[int]) -> bool:
    return all(a <= b for a, b in zip(values, values[1:])) or all(a >= b for a, b in zip(values, values[1:]))


def _has_center_distance_counterexample(objects: list[dict], chain: list[int]) -> bool:
    for pos, current in enumerate(chain[:-1]):
        remaining = chain[pos + 1 :]
        pixel_next = chain[pos + 1]
        center_distances = [(_center_distance(objects[current], objects[idx]), idx) for idx in remaining]
        best = min(d for d, _idx in center_distances)
        winners = [idx for d, idx in center_distances if d == best]
        if len(winners) == 1 and int(winners[0]) != int(pixel_next):
            return True
    return False


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        objects, marker, start_idx = _parse_scene(grid)
        if len(objects) < OBJECT_COUNT_RANGE[0]:
            return False

        chain = _nearest_chain(objects, start_idx)
        out = main(grid)
        if out.shape != grid.shape or np.array_equal(out, grid):
            return False
        if int(out[marker]) != BACKGROUND:
            return False
        if np.count_nonzero(out == START_MARKER):
            return False

        fg_in = (grid != BACKGROUND) & (grid != START_MARKER)
        fg_out = out != BACKGROUND
        if not np.array_equal(fg_in, fg_out):
            return False

        row_major = _row_major_order(objects)
        if chain == row_major or chain == list(reversed(row_major)):
            return False
        if chain == _distance_from_start_order(objects, start_idx):
            return False
        centers = [_center2(objects[idx]) for idx in chain]
        if _is_monotone([r for r, _c in centers]) and _is_monotone([c for _r, c in centers]):
            return False
        if not _has_center_distance_counterexample(objects, chain):
            return False
    except (AssertionError, ValueError, IndexError):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(OBJECT_COLORS, count)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    motif = _motif_variant()
    motif_all = np.vstack(motif)
    mh = int(motif_all[:, 0].max() + 1)
    mw = int(motif_all[:, 1].max() + 1)
    if mh + 6 > height or mw + 6 > width:
        return None

    top = random.randint(2, height - mh - 3)
    left = random.randint(2, width - mw - 3)
    abs_motif = [coords + np.asarray([top, left], dtype=int) for coords in motif]

    for coords, color in zip(abs_motif, colors[:3]):
        if not _can_place_coords(occ, coords):
            return None
        _paint_object(grid, occ, coords, color)

    start_coords = abs_motif[0]
    for color in colors[3:]:
        if not _place_random_rect(grid, occ, color, start_coords):
            return None

    if not _add_start_marker(grid, start_coords):
        return None
    if _semantic_checks(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nearest-neighbor color-chain input")
