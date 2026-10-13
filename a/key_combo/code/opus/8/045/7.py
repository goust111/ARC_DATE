from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BODY = 5
MARKER_COLORS = (1, 2, 3, 4)
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Compact directional body geometry: the body centroid lies behind the ray.
BODY_REL = {
    (0, 1): [(-2, -1), (-1, -2), (-1, -1), (0, -2), (0, -1), (1, -2), (1, -1)],
    (-1, 0): [(1, -2), (2, -1), (1, -1), (2, 0), (1, 0), (2, 1), (1, 1)],
    (0, -1): [(2, 1), (1, 2), (1, 1), (0, 2), (0, 1), (-1, 2), (-1, 1)],
    (1, 0): [(-1, 2), (-2, 1), (-1, 1), (-2, 0), (-1, 0), (-2, -1), (-1, -1)],
}

SPRITE_EXTENTS = {}
for _direction, _offsets in BODY_REL.items():
    _all = _offsets + [(0, 0)]
    SPRITE_EXTENTS[_direction] = (
        min(dr for dr, _ in _all),
        max(dr for dr, _ in _all),
        min(dc for _, dc in _all),
        max(dc for _, dc in _all),
    )

PARTITIONS = {
    5: [(1, 4), (2, 3)],
    6: [(1, 5), (2, 4), (1, 2, 3)],
}

GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _sprite_cells(marker: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = marker
    return {(r, c)} | {(r + dr, c + dc) for dr, dc in BODY_REL[direction]}


def _body_cells(marker: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = marker
    return [(r + dr, c + dc) for dr, dc in BODY_REL[direction]]


def _place_sprite(grid: np.ndarray, marker: tuple[int, int], direction: tuple[int, int], color: int) -> None:
    for rr, cc in _body_cells(marker, direction):
        grid[rr, cc] = BODY
    grid[marker] = color


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _decode_direction(body_coords: np.ndarray, marker: tuple[int, int]) -> tuple[int, int]:
    mr, mc = marker
    rel = body_coords.astype(float) - np.array([[mr, mc]], dtype=float)
    bias_r, bias_c = rel.mean(axis=0)
    if abs(bias_r) >= abs(bias_c):
        return (-1, 0) if bias_r > 0 else (1, 0)
    return (0, -1) if bias_c > 0 else (0, 1)


def _extract_records(grid: np.ndarray) -> list[dict]:
    comps = _component_coords(grid != BG)
    records = []
    for coords in comps:
        colors = [int(grid[r, c]) for r, c in coords]
        marker_cells = [(int(r), int(c), int(grid[r, c])) for r, c in coords if int(grid[r, c]) != BODY]
        body_coords = np.array([(int(r), int(c)) for r, c in coords if int(grid[r, c]) == BODY], dtype=int)
        assert len(marker_cells) == 1, "each sprite must have exactly one marker pixel"
        assert len(body_coords) == 7, "each sprite must have exactly seven grey body pixels"
        mr, mc, color = marker_cells[0]
        assert color in MARKER_COLORS, "unexpected marker color"
        assert all(v in MARKER_COLORS or v == BODY for v in colors)
        direction = _decode_direction(body_coords, (mr, mc))
        records.append(
            {
                "color": color,
                "marker": (mr, mc),
                "direction": direction,
            }
        )
    assert 4 <= len(records) <= 6, "expected 4-6 sprites"
    return records


def _trace_ray(
    grid: np.ndarray,
    marker: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> tuple[list[tuple[int, int]], bool]:
    h, w = grid.shape
    dr, dc = direction
    rr, cc = marker[0] + dr, marker[1] + dc
    cells: list[tuple[int, int]] = []
    skipped_grey = False
    while 0 <= rr < h and 0 <= cc < w and len(cells) < length:
        if int(grid[rr, cc]) == BODY:
            skipped_grey = True
        else:
            cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells, skipped_grey


def _render_from_records(grid: np.ndarray, records: list[dict]) -> tuple[np.ndarray, dict[int, int], bool, bool]:
    freqs = Counter(rec["color"] for rec in records)
    claims: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    crossing = False
    any_skip = False

    for rec in records:
        ray_cells, skipped_grey = _trace_ray(grid, rec["marker"], rec["direction"], freqs[rec["color"]])
        any_skip = any_skip or skipped_grey
        mr, mc = rec["marker"]
        for rr, cc in ray_cells:
            claims.setdefault((rr, cc), []).append((freqs[rec["color"]], -mr, -mc, rec["color"]))

    for items in claims.values():
        if len({item[3] for item in items}) >= 2:
            crossing = True

    out = grid.copy()
    for (rr, cc), items in claims.items():
        out[rr, cc] = max(items)[3]

    out[grid == BODY] = BODY
    for rec in records:
        out[rec["marker"]] = rec["color"]
    return out, {color: freqs[color] for color in sorted(freqs)}, crossing and len(claims) > 0 and any(
        len({item[3] for item in items}) >= 2 for items in claims.values()
    ), any_skip


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each grey directional sprite and cast a frequency-length ray from its marker.

    Input structure:
    - The background is black.
    - Each foreground object is an 8-connected sprite made of seven grey body pixels plus one colored marker pixel.
    - Marker colors come from a small shared palette, and the same marker color may appear in multiple sprites.

    Rule:
    - For each sprite, compute the centroid bias of the grey body relative to its marker; the ray direction is the opposite cardinal direction.
    - Count how many times each marker color appears in the whole grid; that global frequency is the ray length for every sprite of that color.
    - Starting next to the marker, paint that many cells in the ray direction, skipping grey body cells without counting them; when rays overlap, the higher-frequency color wins, with smaller marker row breaking ties.

    Key insight:
    - Direction is a local geometric cue from one sprite, but ray length and occlusion priority come from a separate global statistic: the total count of that marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    records = _extract_records(g)
    out, _freqs, _crossing, _any_skip = _render_from_records(g, records)
    return out


def _non_adjacent(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    if cells & occupied:
        return False
    blocked = set(occupied)
    for rr, cc in occupied:
        blocked.update(_neighbors8(rr, cc))
    return not (cells & blocked)


def _fits(shape: tuple[int, int], marker: tuple[int, int], direction: tuple[int, int]) -> bool:
    min_r, max_r, min_c, max_c = SPRITE_EXTENTS[direction]
    r, c = marker
    h, w = shape
    return 0 <= r + min_r and r + max_r < h and 0 <= c + min_c and c + max_c < w


def _can_place(shape: tuple[int, int], marker: tuple[int, int], direction: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    if not _fits(shape, marker, direction):
        return False
    return _non_adjacent(_sprite_cells(marker, direction), occupied)


def _choose_partition() -> tuple[list[int], list[int]]:
    total = random.randint(5, 6)
    counts = list(random.choice(PARTITIONS[total]))
    colors = random.sample(list(MARKER_COLORS), len(counts))
    return counts, colors


def _build_specs() -> tuple[list[dict], dict[int, int]]:
    counts, colors = _choose_partition()
    freqs = {color: count for color, count in zip(colors, counts)}
    specs = []
    for color, count in freqs.items():
        for _ in range(count):
            specs.append({"color": color})
    random.shuffle(specs)
    return specs, freqs


def _find_cross_pair(shape: tuple[int, int], freqs: dict[int, int], occupied: set[tuple[int, int]]) -> list[dict] | None:
    ranked = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2:
        return None
    color_a, freq_a = ranked[0]
    color_b, freq_b = ranked[1]
    h_dir = random.choice([(0, -1), (0, 1)])
    v_dir = random.choice([(-1, 0), (1, 0)])

    candidates = []
    max_dist_a = freq_a - 1 if freq_a > 1 else 1
    for dist_a in range(1, max_dist_a + 1):
        for dist_b in range(1, freq_b + 1):
            for ir in range(3, shape[0] - 3):
                for ic in range(3, shape[1] - 3):
                    marker_a = (ir - dist_a * h_dir[0], ic - dist_a * h_dir[1])
                    marker_b = (ir - dist_b * v_dir[0], ic - dist_b * v_dir[1])
                    rec_a = {"color": color_a, "marker": marker_a, "direction": h_dir}
                    rec_b = {"color": color_b, "marker": marker_b, "direction": v_dir}
                    if not _can_place(shape, marker_a, h_dir, occupied):
                        continue
                    occ_a = occupied | _sprite_cells(marker_a, h_dir)
                    if not _can_place(shape, marker_b, v_dir, occ_a):
                        continue
                    if (ir, ic) in _sprite_cells(marker_a, h_dir) or (ir, ic) in _sprite_cells(marker_b, v_dir):
                        continue
                    candidates.append((rec_a, rec_b, dist_a))
    if not candidates:
        return None
    rec_a, rec_b, dist_a = random.choice(candidates)
    rec_a["cross_distance"] = dist_a
    return [rec_a, rec_b]


def _find_blocker(
    shape: tuple[int, int],
    owner: dict,
    freqs: dict[int, int],
    blocker_color: int,
    occupied: set[tuple[int, int]],
) -> dict | None:
    owner_len = freqs[owner["color"]]
    dr, dc = owner["direction"]
    cross_distance = owner.get("cross_distance", 1)
    dist_candidates = list(range(cross_distance + 1, owner_len + 1))
    random.shuffle(dist_candidates)
    for dist in dist_candidates:
        path_r = owner["marker"][0] + dist * dr
        path_c = owner["marker"][1] + dist * dc
        if not _in_bounds(shape, path_r, path_c):
            continue
        blocker_dirs = [(-1, 0), (1, 0)] if dr == 0 else [(0, -1), (0, 1)]
        random.shuffle(blocker_dirs)
        for blocker_dir in blocker_dirs:
            if blocker_dir == (-1, 0):
                marker = (path_r - 1, path_c)
            elif blocker_dir == (1, 0):
                marker = (path_r + 1, path_c)
            elif blocker_dir == (0, -1):
                marker = (path_r, path_c - 1)
            else:
                marker = (path_r, path_c + 1)
            if not _fits(shape, marker, blocker_dir):
                continue
            cells = _sprite_cells(marker, blocker_dir)
            if (path_r, path_c) not in cells - {marker}:
                continue
            if not _non_adjacent(cells, occupied):
                continue
            # Keep the block away from the owner's own sprite for readability.
            if any(abs(rr - owner["marker"][0]) <= 1 and abs(cc - owner["marker"][1]) <= 2 for rr, cc in cells):
                continue
            return {"color": blocker_color, "marker": marker, "direction": blocker_dir}
    return None


def _random_place(shape: tuple[int, int], color: int, occupied: set[tuple[int, int]]) -> dict | None:
    h, w = shape
    options = []
    for direction in random.sample(DIRECTIONS, len(DIRECTIONS)):
        min_r, max_r, min_c, max_c = SPRITE_EXTENTS[direction]
        for r in range(-min_r, h - max_r):
            for c in range(-min_c, w - max_c):
                marker = (r, c)
                if _can_place(shape, marker, direction, occupied):
                    options.append({"color": color, "marker": marker, "direction": direction})
    if not options:
        return None
    return random.choice(options)


def _render_expected(grid: np.ndarray, records: list[dict], freqs: dict[int, int]) -> tuple[np.ndarray, bool, bool]:
    claims: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    has_overlap = False
    has_skip = False
    marker_cells = {rec["marker"] for rec in records}
    for rec in records:
        ray_cells, skipped_grey = _trace_ray(grid, rec["marker"], rec["direction"], freqs[rec["color"]])
        has_skip = has_skip or skipped_grey
        mr, mc = rec["marker"]
        for rr, cc in ray_cells:
            if (rr, cc) in marker_cells and (rr, cc) != rec["marker"]:
                return grid, False, False
            claims.setdefault((rr, cc), []).append((freqs[rec["color"]], -mr, -mc, rec["color"]))
    for items in claims.values():
        if len({item[3] for item in items}) >= 2:
            has_overlap = True
    out = grid.copy()
    for (rr, cc), items in claims.items():
        out[rr, cc] = max(items)[3]
    out[grid == BODY] = BODY
    for rec in records:
        out[rec["marker"]] = rec["color"]
    return out, has_overlap, has_skip


def _make_core_records(shape: tuple[int, int], freqs: dict[int, int]) -> list[dict] | None:
    ranked = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2 or ranked[0][1] < 3:
        return None

    owner_color = ranked[0][0]
    cross_color = ranked[1][0]
    h, w = shape
    for _ in range(300):
        cross_r = random.randint(4, h - 5)
        cross_c = random.randint(4, w - 5)
        owner_dir = random.choice([(0, 1), (0, -1)])
        cross_dir = random.choice([(-1, 0), (1, 0)])
        owner_dist = random.randint(1, ranked[0][1])
        cross_dist = random.randint(1, ranked[1][1])

        owner_marker = (cross_r - owner_dist * owner_dir[0], cross_c - owner_dist * owner_dir[1])
        cross_marker = (cross_r - cross_dist * cross_dir[0], cross_c - cross_dist * cross_dir[1])
        if not _can_place(shape, owner_marker, owner_dir, set()):
            continue
        occupied = _sprite_cells(owner_marker, owner_dir)
        if not _can_place(shape, cross_marker, cross_dir, occupied):
            continue
        occupied |= _sprite_cells(cross_marker, cross_dir)

        step_order = list(range(1, ranked[0][1] + 1))
        if owner_dist in step_order:
            step_order.remove(owner_dist)
        random.shuffle(step_order)
        step_order.append(owner_dist)

        blocker_options = []
        for step in step_order:
            target = (owner_marker[0] + step * owner_dir[0], owner_marker[1] + step * owner_dir[1])
            if not _in_bounds(shape, target[0], target[1]):
                continue
            for blocker_dir in random.sample(DIRECTIONS, len(DIRECTIONS)):
                for dr, dc in BODY_REL[blocker_dir]:
                    blocker_marker = (target[0] - dr, target[1] - dc)
                    if not _can_place(shape, blocker_marker, blocker_dir, occupied):
                        continue
                    cells = _sprite_cells(blocker_marker, blocker_dir)
                    if target in cells - {blocker_marker}:
                        blocker_options.append(
                            {"color": owner_color, "marker": blocker_marker, "direction": blocker_dir}
                        )
        if blocker_options:
            blocker = random.choice(blocker_options)
            return [
                {"color": owner_color, "marker": owner_marker, "direction": owner_dir},
                {"color": cross_color, "marker": cross_marker, "direction": cross_dir},
                blocker,
            ]
    return None


def _generate_attempt() -> np.ndarray | None:
    shape = (random.randint(14, 18), random.randint(14, 18))
    _specs, freqs = _build_specs()

    records = _make_core_records(shape, freqs)
    if records is None:
        return None
    occupied: set[tuple[int, int]] = set()
    for rec in records:
        occupied |= _sprite_cells(rec["marker"], rec["direction"])

    used_counts = Counter(rec["color"] for rec in records)
    remaining_colors = []
    for color, count in freqs.items():
        remaining_colors.extend([color] * max(0, count - used_counts[color]))
    random.shuffle(remaining_colors)

    for color in remaining_colors:
        rec = _random_place(shape, color, occupied)
        if rec is None:
            return None
        records.append(rec)
        occupied |= _sprite_cells(rec["marker"], rec["direction"])

    if len(records) != sum(freqs.values()):
        return None

    grid = np.zeros(shape, dtype=int)
    for rec in records:
        _place_sprite(grid, rec["marker"], rec["direction"], rec["color"])

    expected, has_overlap, has_skip = _render_expected(grid, records, freqs)
    if not has_overlap or not has_skip:
        return None

    extracted = _extract_records(grid)
    solved = main(grid)
    if Counter(rec["color"] for rec in extracted) != Counter(freqs):
        return None
    extracted_key = sorted((rec["color"], rec["marker"], rec["direction"]) for rec in extracted)
    original_key = sorted((rec["color"], rec["marker"], rec["direction"]) for rec in records)
    if extracted_key != original_key:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
