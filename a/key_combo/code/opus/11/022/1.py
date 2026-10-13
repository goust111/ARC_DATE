from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


BG = 0
SPRITE_COLOR = 1
SEPARATOR_COLOR = 5
POINT_COLORS = (2, 3, 4, 6)
GRID_H = 14
HALF_W = 14
FULL_W = HALF_W * 2 + 1
ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_separator(g: np.ndarray, bg: int) -> int:
    cols = []
    for c in range(g.shape[1]):
        color = int(g[0, c])
        if color != bg and np.all(g[:, c] == color):
            cols.append(c)
    assert len(cols) == 1, "expected exactly one solid separator column"
    sep = int(cols[0])
    assert 2 * sep + 1 == g.shape[1], "separator must split equal-width halves"
    return sep


def _connected_component(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    out = []
    while q:
        r, c = q.popleft()
        out.append((r, c))
        for dr, dc in ORTHO:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _shape_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    nr = _normalize(coords)
    _, _, r1, c1 = _bbox(nr)
    mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
    for r, c in nr:
        mask[r, c] = True
    return mask


def _is_irregular(coords: list[tuple[int, int]]) -> bool:
    mask = _shape_mask(coords)
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(np.fliplr(mask))):
        return False
    return True


def _right_point_from_left(left_coord: tuple[int, int]) -> tuple[int, int]:
    r, c = left_coord
    return int(r), int(HALF_W - 1 - c)


def _points_are_spaced(points: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(points):
        for r2, c2 in points[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) < 3:
                return False
            if max(abs(r1 - r2), abs(c1 - c2)) < 2:
                return False
    return True


def _stamp_coords(
    sprite_coords: list[tuple[int, int]], anchor_left: tuple[int, int]
) -> list[tuple[int, int]] | None:
    ar, ac = anchor_left
    point_r, point_c = _right_point_from_left(anchor_left)
    stamped = []
    for sr, sc in sprite_coords:
        rr = point_r + (sr - ar)
        cc = point_c + (sc - ac)
        if not (0 <= rr < GRID_H and 0 <= cc < HALF_W):
            return None
        stamped.append((rr, cc))
    return stamped


def _render_output_from_hits(
    sprite_coords: list[tuple[int, int]],
    hits: list[dict],
) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    out = np.full((GRID_H, HALF_W), BG, dtype=int)
    stamp_sets: list[set[tuple[int, int]]] = []
    for hit in sorted(hits, key=lambda item: (item["point"][0], item["point"][1])):
        color = int(hit["color"])
        stamp = set(hit["stamp"])
        stamp_sets.append(stamp)
        for r, c in hit["stamp"]:
            out[r, c] = color
    return out, stamp_sets


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, FULL_W), f"expected {GRID_H}x{FULL_W} input"
    bg = _get_bg(g)
    sep = _find_separator(g, bg)
    left = g[:, :sep]
    right = g[:, sep + 1 :]

    assert int(g[0, sep]) == SEPARATOR_COLOR, "separator must be grey"
    assert not np.any(left == SEPARATOR_COLOR)
    assert not np.any(right == SEPARATOR_COLOR)

    sprite_mask = left == SPRITE_COLOR
    sprite_coords = sorted(map(tuple, np.argwhere(sprite_mask)))
    assert 5 <= len(sprite_coords) <= 8, "blue sprite must contain 5-8 cells"
    assert np.count_nonzero(left) == len(sprite_coords), "left half must only contain the blue sprite"
    comp = _connected_component(sprite_mask)
    assert len(comp) == len(sprite_coords), "sprite must be 4-connected"
    assert _is_irregular(sprite_coords), "sprite must be irregular and asymmetric"

    point_cells = sorted(
        (int(r), int(c), int(right[r, c]))
        for r, c in np.argwhere(right != bg)
    )
    assert 3 <= len(point_cells) <= len(POINT_COLORS), "right half must contain 3-4 colored points"
    assert len({color for _, _, color in point_cells}) == len(point_cells), "point colors must be unique"
    assert {color for _, _, color in point_cells}.issubset(set(POINT_COLORS))
    assert _points_are_spaced([(r, c) for r, c, _ in point_cells]), "points must be well separated"

    sprite_set = set(sprite_coords)
    hits = []
    misses = []
    for r, c, color in point_cells:
        mirrored = (r, HALF_W - 1 - c)
        if mirrored in sprite_set:
            stamp = _stamp_coords(sprite_coords, mirrored)
            assert stamp is not None, "generated hits must stamp fully in-bounds"
            hits.append(
                {
                    "point": (r, c),
                    "point_abs": (r, c + sep + 1),
                    "color": color,
                    "anchor_left": mirrored,
                    "stamp": stamp,
                }
            )
        else:
            miss_flags = set()
            if any(sr == r for sr, _ in sprite_coords):
                miss_flags.add("row")
            if any(sc == mirrored[1] for _, sc in sprite_coords):
                miss_flags.add("col")
            misses.append(
                {
                    "point": (r, c),
                    "point_abs": (r, c + sep + 1),
                    "color": color,
                    "mirrored_left": mirrored,
                    "flags": miss_flags,
                }
            )

    out, stamp_sets = _render_output_from_hits(sprite_coords, hits)
    return {
        "grid": g,
        "bg": bg,
        "sep": sep,
        "sprite": sprite_coords,
        "points": point_cells,
        "hits": hits,
        "misses": misses,
        "output": out,
        "stamp_sets": stamp_sets,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Fold the right-hand dots onto the left sprite to detect hits, then stamp recolored sprite copies back on the right half.

    Input structure:
    - The grid is split into two equal square halves by one full-height grey separator column on black background.
    - The left half contains exactly one irregular 4-connected blue sprite.
    - The right half contains several isolated single-pixel colored dots.

    Rule:
    - Mirror each right-half dot horizontally across the separator into the left-half coordinate system.
    - A dot is a hit iff its mirrored position lands exactly on a blue sprite cell.
    - For every hit, take the whole blue sprite, recolor it with that dot's color, and stamp it into the output right-half canvas aligned to the dot's original position using the hit sprite cell as the anchor pixel.
    - Ignore misses. Draw hit stamps in right-half reading order (top-to-bottom, then left-to-right), with later stamps overwriting earlier ones on overlaps.

    Key insight:
    - The fold is only a probe: it identifies which colored dots correspond to which sprite pixels, and those matched pixels become the anchor positions for the recolored stamps in the unfurled right-side space.
    """
    analysis = _analyze(np.asarray(grid, dtype=int))
    return analysis["output"]


def _grow_connected_shape(height: int, width: int, area: int) -> list[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ORTHO:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return []
        cells.add(random.choice(frontier))
    return sorted(cells)


def _generate_sprite() -> list[tuple[int, int]]:
    for _ in range(300):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        area = random.randint(5, min(8, h * w - 1))
        rel = _grow_connected_shape(h, w, area)
        if not rel:
            continue
        rel = _normalize(rel)
        if not _is_irregular(rel):
            continue
        rr0, cc0, rr1, cc1 = _bbox(rel)
        box_h = rr1 - rr0 + 1
        box_w = cc1 - cc0 + 1
        if box_h < 2 or box_w < 2:
            continue

        row_choices = list(range(2, GRID_H - box_h - 1))
        col_choices = list(range(2, HALF_W - box_w - 1))
        random.shuffle(row_choices)
        random.shuffle(col_choices)
        for r0 in row_choices:
            for c0 in col_choices:
                placed = sorted((r + r0, c + c0) for r, c in rel)
                fit_anchors = [coord for coord in placed if _stamp_coords(placed, coord) is not None]
                if len(fit_anchors) >= 2:
                    return placed
    raise RuntimeError("failed to build a valid sprite")


def _find_hit_anchor_sets(
    sprite: list[tuple[int, int]], hit_count: int, want_overlap: bool
) -> list[tuple[tuple[int, int], ...]]:
    fit = [coord for coord in sprite if _stamp_coords(sprite, coord) is not None]
    found = []
    for combo in itertools.combinations(fit, hit_count):
        right_points = [_right_point_from_left(coord) for coord in combo]
        if not _points_are_spaced(right_points):
            continue
        if len({c for _, c in combo}) != len(combo):
            continue
        if len({r for r, _ in combo}) < 2:
            continue
        stamps = [set(_stamp_coords(sprite, coord) or []) for coord in combo]
        if any(len(stamp) != len(sprite) for stamp in stamps):
            continue
        intersections = [len(a & b) for a, b in itertools.combinations(stamps, 2)]
        if want_overlap:
            if not any(0 < inter < len(sprite) for inter in intersections):
                continue
        else:
            if any(inter > 0 for inter in intersections):
                continue
        found.append(combo)
    random.shuffle(found)
    return found


def _miss_candidates(
    sprite: list[tuple[int, int]], occupied: set[tuple[int, int]]
) -> list[dict]:
    sprite_set = set(sprite)
    candidates = []
    sprite_rows = {r for r, _ in sprite}
    sprite_cols = {c for _, c in sprite}
    for r in range(GRID_H):
        for c in range(HALF_W):
            pt = (r, c)
            if pt in occupied:
                continue
            mirrored = (r, HALF_W - 1 - c)
            if mirrored in sprite_set:
                continue
            flags = set()
            if r in sprite_rows:
                flags.add("row")
            if mirrored[1] in sprite_cols:
                flags.add("col")
            if not flags:
                continue
            candidates.append({"point": pt, "flags": flags})
    random.shuffle(candidates)
    return candidates


def _choose_misses(
    sprite: list[tuple[int, int]],
    hit_points: list[tuple[int, int]],
    miss_count: int,
) -> list[tuple[int, int]]:
    occupied = set(hit_points)
    candidates = _miss_candidates(sprite, occupied)
    need = random.choice([{"row"}, {"col"}]) if miss_count == 1 else {"row", "col"}
    chosen: list[tuple[int, int]] = []
    for combo in itertools.combinations(candidates, miss_count):
        pts = [item["point"] for item in combo]
        if not _points_are_spaced(hit_points + pts):
            continue
        flags = set().union(*(item["flags"] for item in combo))
        if not need.issubset(flags):
            continue
        chosen = pts
        break
    return chosen


def _visible_contribution_checks(sprite: list[tuple[int, int]], hits: list[dict]) -> bool:
    final_out, _ = _render_output_from_hits(sprite, hits)
    for idx in range(len(hits)):
        reduced = hits[:idx] + hits[idx + 1 :]
        reduced_out, _ = _render_output_from_hits(sprite, reduced)
        if np.array_equal(final_out, reduced_out):
            return False
    visible_colors = {int(color) for color in np.unique(final_out) if int(color) != BG}
    return len(visible_colors) >= 2


def _build_grid(sprite: list[tuple[int, int]], points: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.full((GRID_H, FULL_W), BG, dtype=int)
    grid[:, HALF_W] = SEPARATOR_COLOR
    for r, c in sprite:
        grid[r, c] = SPRITE_COLOR
    for r, c, color in points:
        grid[r, c + HALF_W + 1] = color
    return grid


def _generate_case() -> np.ndarray | None:
    sprite = _generate_sprite()
    total_points = random.choice([3, 4])
    if total_points == 3:
        hit_count, miss_count = 2, 1
    else:
        hit_count, miss_count = random.choice([(2, 2), (3, 1)])
    want_overlap = random.random() < 0.5

    hit_anchor_sets = _find_hit_anchor_sets(sprite, hit_count, want_overlap)
    if not hit_anchor_sets and want_overlap:
        hit_anchor_sets = _find_hit_anchor_sets(sprite, hit_count, False)
    if not hit_anchor_sets:
        return None
    anchors = random.choice(hit_anchor_sets)
    hit_points = [_right_point_from_left(anchor) for anchor in anchors]
    misses = _choose_misses(sprite, hit_points, miss_count)
    if len(misses) != miss_count:
        return None

    all_points = hit_points + misses
    if not _points_are_spaced(all_points):
        return None
    colors = random.sample(list(POINT_COLORS), len(all_points))
    random.shuffle(colors)
    point_records = list(zip(all_points, colors))
    random.shuffle(point_records)

    points = [(r, c, color) for (r, c), color in point_records]
    grid = _build_grid(sprite, points)

    analysis = _analyze(grid)
    if len(analysis["hits"]) != hit_count or len(analysis["misses"]) != miss_count:
        return None
    if not _visible_contribution_checks(sprite, analysis["hits"]):
        return None

    miss_flags = set().union(*(miss["flags"] for miss in analysis["misses"]))
    if miss_count == 1 and not miss_flags:
        return None
    if miss_count == 2 and not {"row", "col"}.issubset(miss_flags):
        return None

    actual = main(grid)
    if not np.array_equal(actual, analysis["output"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-stamp input")
