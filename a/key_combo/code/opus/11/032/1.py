from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
TEMPLATE_COLOR = 1
MARKER_COLORS = (2, 3, 4)
OBSTACLE_COLOR = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PERP_DIRS = {
    (-1, 0): ((0, -1), (0, 1)),
    (1, 0): ((0, -1), (0, 1)),
    (0, -1): ((-1, 0), (1, 0)),
    (0, 1): ((-1, 0), (1, 0)),
}

GRID_H_RANGE = (18, 23)
GRID_W_RANGE = (18, 23)
GEN_RETRIES = 500

BASE_TEMPLATE_MASKS = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
)


Coord = tuple[int, int]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _components4(mask: np.ndarray) -> list[list[Coord]]:
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[list[Coord]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        comp: list[Coord] = []
        q = deque([(r, c)])
        seen[r, c] = True
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components4(mask.astype(bool))) == 1


def _adjacent_to_obstacle(grid: np.ndarray, pos: Coord) -> bool:
    r, c = pos
    return any(
        _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == OBSTACLE_COLOR
        for nr, nc in _neighbors4(r, c)
    )


def _sightline_hits(grid: np.ndarray, marker: Coord) -> list[tuple[Coord, Coord, int]]:
    hits: list[tuple[Coord, Coord, int]] = []
    sr, sc = marker
    for dr, dc in DIRS4:
        r, c = sr + dr, sc + dc
        last_bg: Coord | None = None
        while _in_bounds(grid.shape, r, c):
            value = int(grid[r, c])
            if value == OBSTACLE_COLOR:
                if last_bg is not None:
                    hits.append(((dr, dc), last_bg, abs(r - sr) + abs(c - sc) - 1))
                break
            if value != BG:
                break
            last_bg = (r, c)
            r += dr
            c += dc
    return hits


def _trace_one_direction(grid: np.ndarray, start: Coord, first_step: Coord) -> tuple[int, Coord]:
    sr, sc = start
    dr, dc = first_step
    nr, nc = sr + dr, sc + dc
    if not _in_bounds(grid.shape, nr, nc):
        return 0, start
    if int(grid[nr, nc]) != BG or not _adjacent_to_obstacle(grid, (nr, nc)):
        return 0, start

    prev = start
    cur = (nr, nc)
    visited = {start, cur}
    length = 1

    while True:
        step = (cur[0] - prev[0], cur[1] - prev[1])
        candidates: list[Coord] = []
        for rr, cc in _neighbors4(*cur):
            if not _in_bounds(grid.shape, rr, cc):
                continue
            if (rr, cc) in visited or int(grid[rr, cc]) != BG:
                continue
            if not _adjacent_to_obstacle(grid, (rr, cc)):
                continue
            candidates.append((rr, cc))
        if not candidates:
            break
        candidates.sort(
            key=lambda pos: (
                (pos[0] - cur[0]) * step[0] + (pos[1] - cur[1]) * step[1],
                -abs((pos[0] - cur[0]) - step[0]) - abs((pos[1] - cur[1]) - step[1]),
                -pos[0],
                -pos[1],
            ),
            reverse=True,
        )
        nxt = candidates[0]
        prev, cur = cur, nxt
        visited.add(cur)
        length += 1

    return length, cur


def _trace_lengths(grid: np.ndarray, landing: Coord, direction: Coord) -> tuple[int, int]:
    step_a, step_b = PERP_DIRS[direction]
    length_a = _trace_one_direction(grid, landing, step_a)[0]
    length_b = _trace_one_direction(grid, landing, step_b)[0]
    return length_a, length_b


def _template_rel_cells(grid: np.ndarray) -> list[Coord]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == TEMPLATE_COLOR)]
    assert 3 <= len(coords) <= 6, "expected a 3-6 pixel template"
    comps = _components4(grid == TEMPLATE_COLOR)
    assert len(comps) == 1, "template must be one 4-connected component"
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    assert max(rows) - r0 + 1 <= 3 and max(cols) - c0 + 1 <= 3, "template bbox too large"
    rel = sorted((r - r0, c - c0) for r, c in coords)
    return rel


def _stamp_cells(shape: tuple[int, int], rel_cells: list[Coord], landing: Coord, k: int) -> list[Coord]:
    cells: list[Coord] = []
    lr, lc = landing
    for tr, tc in rel_cells:
        r0 = lr + tr * k
        c0 = lc + tc * k
        for rr in range(r0, r0 + k):
            for cc in range(c0, c0 + k):
                if _in_bounds(shape, rr, cc):
                    cells.append((rr, cc))
    return cells


def _marker_infos(grid: np.ndarray, rel_cells: list[Coord]) -> list[dict]:
    infos: list[dict] = []
    for color in MARKER_COLORS:
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        assert coords.shape == (1, 2), f"marker color {color} must be a single pixel"
        marker = tuple(map(int, coords[0]))
        hits = _sightline_hits(grid, marker)
        assert len(hits) == 1, "each marker must have exactly one clear cardinal sightline"
        direction, landing, _gap = hits[0]
        lengths = _trace_lengths(grid, landing, direction)
        assert lengths[0] != lengths[1], "contour lengths must not tie"
        k = max(lengths)
        assert k >= 1, "stamp scale must be positive"
        infos.append(
            {
                "color": color,
                "marker": marker,
                "direction": direction,
                "landing": landing,
                "lengths": lengths,
                "k": k,
                "distance": marker[0] + marker[1],
                "cells": _stamp_cells(grid.shape, rel_cells, landing, k),
            }
        )
    assert 2 <= len(infos) <= 3, "expected 2-3 markers"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored marker into the obstacle, read a contour length there, and stamp a scaled copy of the blue template.

    Input structure:
    - A black background contains one irregular gray obstacle, one small blue 4-connected template near a corner, and 2-3 single-pixel red/green/yellow markers.
    - Each marker has exactly one unobstructed cardinal sightline to the gray obstacle and stops on the last black cell before it.
    - The blue template occupies a bbox no larger than 3x3 and is separate from the obstacle and the markers.

    Rule:
    - For each marker, slide along its unique sightline to the landing cell just before the obstacle.
    - From that landing cell, trace the obstacle-adjacent background contour in the two perpendicular directions; let k be the larger of the two trace lengths.
    - Scale every blue template pixel to a k x k block, draw the scaled template with the landing cell as the scaled bbox's top-left corner, resolve overlaps by larger k first and then by smaller marker-to-origin Manhattan distance, and erase both the original markers and the original blue template.

    Key insight:
    - The same obstacle serves as motion barrier and as a per-marker scale lookup table: where a marker lands determines the contour distance, and that contour distance determines the stamped size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    obstacle = g == OBSTACLE_COLOR
    assert len(_components4(obstacle)) == 1, "expected one 4-connected obstacle"

    rel_cells = _template_rel_cells(g)
    infos = _marker_infos(g, rel_cells)

    out = g.copy()
    out[g == TEMPLATE_COLOR] = BG
    for info in infos:
        mr, mc = info["marker"]
        out[mr, mc] = BG

    # Higher k must be on top, so smaller k is painted first.
    # For ties, the marker closer to the top-left corner is on top, so paint farther ones first.
    for info in sorted(infos, key=lambda item: (item["k"], -item["distance"])):
        for rr, cc in info["cells"]:
            out[rr, cc] = info["color"]

    return out


def _all_template_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    seen: set[tuple[bytes, tuple[int, int]]] = set()
    for base in BASE_TEMPLATE_MASKS:
        cur = base
        for _ in range(4):
            for variant in (cur, np.fliplr(cur)):
                cropped = _crop_mask(variant)
                if not bool(cropped[0, 0]):
                    continue
                if not _is_connected(cropped):
                    continue
                key = (cropped.tobytes(), tuple(map(int, cropped.shape)))
                if key not in seen:
                    seen.add(key)
                    masks.append(cropped.copy())
            cur = np.rot90(cur)
    return masks


TEMPLATE_MASKS = _all_template_masks()


def _concave_2x2_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    total = 0
    for r in range(h - 1):
        for c in range(w - 1):
            if int(mask[r : r + 2, c : c + 2].sum()) == 3:
                total += 1
    return total


def _random_obstacle_mask() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(4, 6)
        w = random.randint(4, 6)
        mask = np.ones((h, w), dtype=bool)
        for _ in range(random.randint(1, 2)):
            side = random.choice("tblr")
            if side in "tb" and w >= 3:
                depth = random.randint(1, max(1, h - 2))
                notch_w = random.randint(1, w - 2)
                c0 = random.randint(1, w - notch_w - 1)
                r0 = 0 if side == "t" else h - depth
                mask[r0 : r0 + depth, c0 : c0 + notch_w] = False
            elif side in "lr" and h >= 3:
                depth = random.randint(1, max(1, w - 2))
                notch_h = random.randint(1, h - 2)
                r0 = random.randint(1, h - notch_h - 1)
                c0 = 0 if side == "l" else w - depth
                mask[r0 : r0 + notch_h, c0 : c0 + depth] = False
        if not (15 <= int(mask.sum()) <= 25):
            continue
        if not _is_connected(mask):
            continue
        if _concave_2x2_count(mask) < 2:
            continue
        return mask
    return None


def _place_template(mask: np.ndarray, grid_shape: tuple[int, int]) -> Coord:
    h, w = grid_shape
    th, tw = mask.shape
    inset_r = random.randint(0, 1)
    inset_c = random.randint(0, 1)
    corner = random.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        return inset_r, inset_c
    if corner == "tr":
        return inset_r, w - tw - inset_c
    if corner == "bl":
        return h - th - inset_r, inset_c
    return h - th - inset_r, w - tw - inset_c


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top_left: Coord, color: int) -> None:
    r0, c0 = top_left
    coords = np.argwhere(mask)
    for dr, dc in coords:
        grid[r0 + int(dr), c0 + int(dc)] = color


def _stamp_footprint(shape: tuple[int, int], rel_cells: list[Coord], landing: Coord, k: int) -> set[Coord]:
    return set(_stamp_cells(shape, rel_cells, landing, k))


def _enumerate_candidates(grid: np.ndarray, rel_cells: list[Coord]) -> list[dict]:
    h, w = grid.shape
    raw: list[dict] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue
            hits = _sightline_hits(grid, (r, c))
            if len(hits) != 1:
                continue
            direction, landing, gap = hits[0]
            if not (2 <= gap <= 6):
                continue
            lengths = _trace_lengths(grid, landing, direction)
            k = max(lengths)
            if lengths[0] == lengths[1] or not (1 <= k <= 4):
                continue
            raw.append(
                {
                    "marker": (r, c),
                    "landing": landing,
                    "direction": direction,
                    "gap": gap,
                    "lengths": lengths,
                    "k": k,
                    "distance": r + c,
                    "footprint": _stamp_footprint(grid.shape, rel_cells, landing, k),
                }
            )
    buckets: dict[tuple, list[dict]] = {}
    for item in raw:
        key = (item["landing"], item["direction"], item["k"], item["lengths"])
        buckets.setdefault(key, []).append(item)
    candidates: list[dict] = []
    for items in buckets.values():
        items = items[:]
        random.shuffle(items)
        candidates.extend(items[:3])
    return candidates


def _overlap_kind(combo: tuple[dict, ...]) -> str | None:
    kinds: set[str] = set()
    for a, b in itertools.combinations(combo, 2):
        if not (a["footprint"] & b["footprint"]):
            continue
        kinds.add("equal" if a["k"] == b["k"] else "diff")
    if not kinds:
        return None
    if "equal" in kinds:
        return "equal"
    return "diff"


def _combo_valid(combo: tuple[dict, ...], require_overlap: str | None) -> bool:
    markers = [item["marker"] for item in combo]
    landings = [item["landing"] for item in combo]
    ks = [item["k"] for item in combo]
    max_k = max(ks)

    if len({r for r, _ in markers}) != len(markers):
        return False
    if len({c for _, c in markers}) != len(markers):
        return False
    if len(set(landings)) != len(landings):
        return False
    if len(set(ks)) < 2 or max_k < 2:
        return False

    for a, b in itertools.combinations(combo, 2):
        dist = abs(a["landing"][0] - b["landing"][0]) + abs(a["landing"][1] - b["landing"][1])
        if dist < max_k * 3:
            return False
        if a["k"] == b["k"] and (a["footprint"] & b["footprint"]) and a["distance"] == b["distance"]:
            return False

    return _overlap_kind(combo) == require_overlap if require_overlap is not None else True


def _choose_marker_combo(candidates: list[dict]) -> tuple[dict, ...] | None:
    target_mode = random.choices(
        population=["equal", "diff", None],
        weights=[2, 3, 5],
        k=1,
    )[0]
    target_count = 3 if target_mode == "equal" else random.choice((2, 3))

    orders: list[tuple[int, str | None]] = [(target_count, target_mode)]
    for count in (3, 2):
        for mode in ("equal", "diff", None):
            pair = (count, mode)
            if pair not in orders:
                orders.append(pair)

    for count, mode in orders:
        if len(candidates) < count:
            continue
        seen: set[tuple[int, ...]] = set()
        for _ in range(400):
            idxs = tuple(sorted(random.sample(range(len(candidates)), count)))
            if idxs in seen:
                continue
            seen.add(idxs)
            combo = tuple(candidates[i] for i in idxs)
            if _combo_valid(combo, mode):
                return combo
    return None


def _render_input(
    grid_shape: tuple[int, int],
    obstacle_mask: np.ndarray,
    obstacle_pos: Coord,
    template_mask: np.ndarray,
    template_pos: Coord,
    combo: tuple[dict, ...],
) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    _paint_mask(grid, obstacle_mask, obstacle_pos, OBSTACLE_COLOR)
    _paint_mask(grid, template_mask, template_pos, TEMPLATE_COLOR)
    colors = random.sample(MARKER_COLORS, len(combo))
    for color, item in zip(colors, combo):
        r, c = item["marker"]
        grid[r, c] = color
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    obstacle_mask = grid == OBSTACLE_COLOR
    if len(_components4(obstacle_mask)) != 1:
        return False
    if not (15 <= int(obstacle_mask.sum()) <= 25):
        return False
    if _concave_2x2_count(obstacle_mask) < 2:
        return False

    template_coords = np.argwhere(grid == TEMPLATE_COLOR)
    if not (3 <= len(template_coords) <= 6):
        return False
    if len(_components4(grid == TEMPLATE_COLOR)) != 1:
        return False
    rows = template_coords[:, 0]
    cols = template_coords[:, 1]
    if int(rows.max() - rows.min() + 1) > 3 or int(cols.max() - cols.min() + 1) > 3:
        return False

    try:
        rel_cells = _template_rel_cells(grid)
        infos = _marker_infos(grid, rel_cells)
    except AssertionError:
        return False

    if len({info["marker"][0] for info in infos}) != len(infos):
        return False
    if len({info["marker"][1] for info in infos}) != len(infos):
        return False
    if len({info["k"] for info in infos}) < 2 or max(info["k"] for info in infos) < 2:
        return False

    max_k = max(info["k"] for info in infos)
    for a, b in itertools.combinations(infos, 2):
        dist = abs(a["landing"][0] - b["landing"][0]) + abs(a["landing"][1] - b["landing"][1])
        if dist < max_k * 3:
            return False
        overlap = set(a["cells"]) & set(b["cells"])
        if a["k"] == b["k"] and overlap and (a["distance"] == b["distance"]):
            return False

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return False
    if np.any(out == TEMPLATE_COLOR):
        return False

    # Verify the overlap priority rules on the actual painted result.
    for a, b in itertools.combinations(infos, 2):
        overlap = set(a["cells"]) & set(b["cells"])
        if not overlap:
            continue
        if a["k"] != b["k"]:
            top = a if a["k"] > b["k"] else b
        else:
            top = a if a["distance"] < b["distance"] else b
        for rr, cc in overlap:
            if int(out[rr, cc]) != int(top["color"]):
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    template_mask = random.choice(TEMPLATE_MASKS)
    template_pos = _place_template(template_mask, (h, w))

    obstacle_mask = _random_obstacle_mask()
    if obstacle_mask is None:
        return None
    oh, ow = obstacle_mask.shape
    min_r = 4
    max_r = h - oh - 4
    min_c = 4
    max_c = w - ow - 4
    if min_r > max_r or min_c > max_c:
        return None

    for _ in range(30):
        obstacle_pos = (random.randint(min_r, max_r), random.randint(min_c, max_c))
        skeleton = np.full((h, w), BG, dtype=int)
        _paint_mask(skeleton, obstacle_mask, obstacle_pos, OBSTACLE_COLOR)
        _paint_mask(skeleton, template_mask, template_pos, TEMPLATE_COLOR)
        rel_cells = _template_rel_cells(skeleton)
        candidates = _enumerate_candidates(skeleton, rel_cells)
        if len(candidates) < 6:
            continue
        combo = _choose_marker_combo(candidates)
        if combo is None:
            continue
        grid = _render_input((h, w), obstacle_mask, obstacle_pos, template_mask, template_pos, combo)
        if _validate_generated(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid contour-scale stamp input")
