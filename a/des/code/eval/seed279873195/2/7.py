from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
BROWN = 9

EMITTER_COLORS = [BLUE, RED, GREEN]
TARGET_COLOR_POOL = [BLUE, RED, GREEN, MAGENTA, ORANGE, CYAN, BROWN]
MIRROR_ORIENTATIONS = ["horizontal", "vertical"]
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == int(color)
                    ):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _same_color_components(grid: np.ndarray, excluded: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            col = int(grid[r, c])
            if seen[r, c] or col in excluded:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == col
                    ):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            rows = [rr for rr, _cc in cells]
            cols = [cc for _rr, cc in cells]
            comps.append(
                {
                    "color": col,
                    "cells": cells,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _find_targets(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    targets: dict[int, tuple[int, int, int, int]] = {}
    comps = _same_color_components(grid, excluded={bg, YELLOW, GREY})
    for comp in comps:
        cells = comp["cells"]
        r0, c0, r1, c1 = comp["bbox"]
        if len(cells) != 4 or r1 - r0 != 1 or c1 - c0 != 1:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        col = int(comp["color"])
        if np.all(patch == col):
            assert col not in targets, "target colors must be unique"
            targets[col] = (int(r0), int(c0), int(r1), int(c1))
    assert targets, "expected at least one target block"
    return targets


def _find_emitter(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    h, w = grid.shape
    corners = [
        (0, 0, 1, 1, ((0, 1), (1, 0))),
        (0, w - 1, 1, -1, ((0, w - 2), (1, w - 1))),
        (h - 1, 0, -1, 1, ((h - 1, 1), (h - 2, 0))),
        (h - 1, w - 1, -1, -1, ((h - 1, w - 2), (h - 2, w - 1))),
    ]
    candidates: list[tuple[int, int, int, int, int]] = []
    for r, c, dr, dc, yellow_neighbors in corners:
        col = int(grid[r, c])
        if col in (bg, YELLOW, GREY):
            continue
        if any(_in_bounds(grid.shape, yr, yc) and int(grid[yr, yc]) == int(YELLOW) for yr, yc in yellow_neighbors):
            candidates.append((int(r), int(c), int(dr), int(dc), int(col)))
    assert len(candidates) == 1, "expected one corner emitter"
    return candidates[0]


def _parse_mirrors(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[tuple[int, int], dict]]:
    mirrors: list[dict] = []
    mirror_by_cell: dict[tuple[int, int], dict] = {}

    for cells in _components_of_color(grid, GREY):
        rows = sorted({r for r, _c in cells})
        cols = sorted({c for _r, c in cells})
        if len(rows) == 1 and 3 <= len(cells) <= 6:
            assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal mirror must be contiguous"
            orientation = "horizontal"
            r = rows[0]
            c0, c1 = cols[0], cols[-1]
            marker_candidates = [(r, c0 - 1), (r, c1 + 1)]
        elif len(cols) == 1 and 3 <= len(cells) <= 6:
            assert rows == list(range(rows[0], rows[-1] + 1)), "vertical mirror must be contiguous"
            orientation = "vertical"
            c = cols[0]
            r0, r1 = rows[0], rows[-1]
            marker_candidates = [(r0 - 1, c), (r1 + 1, c)]
        else:
            raise AssertionError("grey components must be straight mirrors")

        markers = []
        for mr, mc in marker_candidates:
            if not _in_bounds(grid.shape, mr, mc):
                continue
            col = int(grid[mr, mc])
            if col not in (bg, GREY, YELLOW):
                markers.append((int(mr), int(mc), int(col)))
        assert len(markers) == 1, "each mirror must have exactly one visible color marker"
        marker_r, marker_c, marker_color = markers[0]
        mirror = {
            "orientation": orientation,
            "cells": {(int(r), int(c)) for r, c in cells},
            "marker_pos": (int(marker_r), int(marker_c)),
            "marker_color": int(marker_color),
        }
        mirrors.append(mirror)
        for cell in mirror["cells"]:
            mirror_by_cell[cell] = mirror

    return mirrors, mirror_by_cell


def _trace_path(
    out: np.ndarray,
    mirror_by_cell: dict[tuple[int, int], dict],
    start: tuple[int, int],
    direction: tuple[int, int],
    start_color: int,
) -> int:
    r, c = start
    dr, dc = direction
    color = int(start_color)
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (int(r), int(c), int(dr), int(dc))
        assert state not in seen, "ray path must not cycle"
        seen.add(state)

        nr, nc = int(r + dr), int(c + dc)
        if not _in_bounds(out.shape, nr, nc):
            break

        mirror = mirror_by_cell.get((int(nr), int(nc)))
        if mirror is not None:
            color = int(mirror["marker_color"])
            if mirror["orientation"] == "horizontal":
                dr = -dr
            else:
                dc = -dc
            continue

        out[nr, nc] = int(color)
        r, c = int(nr), int(nc)

    return int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch a colored diagonal ray from a corner emitter, reflect it on grey mirrors,
    recolor it with the struck mirror's endpoint marker color, and expand the 2x2
    target whose color matches the ray's final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    targets = _find_targets(g, bg)
    _mirrors, mirror_by_cell = _parse_mirrors(g, bg)
    er, ec, dr, dc, emitter_color = _find_emitter(g, bg)

    out = g.copy()
    final_color = _trace_path(
        out=out,
        mirror_by_cell=mirror_by_cell,
        start=(int(er), int(ec)),
        direction=(int(dr), int(dc)),
        start_color=int(emitter_color),
    )

    assert final_color in targets, "final ray color must match a target"
    r0, c0, r1, c1 = targets[final_color]
    assert r0 > 0 and c0 > 0 and r1 + 1 < g.shape[0] and c1 + 1 < g.shape[1], "target halo must stay in bounds"
    out[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2] = int(final_color)
    return out


def _corner_state(corner: str, h: int, w: int) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    if corner == "tl":
        return (0, 0), (1, 1), [(0, 1), (1, 0)]
    if corner == "tr":
        return (0, w - 1), (1, -1), [(0, w - 2), (1, w - 1)]
    if corner == "bl":
        return (h - 1, 0), (-1, 1), [(h - 1, 1), (h - 2, 0)]
    assert corner == "br"
    return (h - 1, w - 1), (-1, -1), [(h - 1, w - 2), (h - 2, w - 1)]


def _segment_start_bounds(anchor: int, length: int, limit: int) -> tuple[int, int] | None:
    start_min = max(0, anchor - length + 1)
    start_max = min(anchor, limit - length)
    if start_min > start_max:
        return None

    interior_min = max(start_min, 1)
    interior_max = min(start_max, limit - length - 1)
    if interior_min <= interior_max:
        return (int(interior_min), int(interior_max))
    return None


def _bar_cells_through(
    h: int,
    w: int,
    orientation: str,
    r: int,
    c: int,
    length: int,
) -> list[tuple[int, int]] | None:
    if orientation == "horizontal":
        bounds = _segment_start_bounds(c, length, w)
        if bounds is None:
            return None
        start = random.randint(bounds[0], bounds[1])
        return [(int(r), int(start + j)) for j in range(length)]

    bounds = _segment_start_bounds(r, length, h)
    if bounds is None:
        return None
    start = random.randint(bounds[0], bounds[1])
    return [(int(start + j), int(c)) for j in range(length)]


def _mirror_marker_candidates(cells: list[tuple[int, int]], orientation: str) -> list[tuple[int, int]]:
    rows = sorted({r for r, _c in cells})
    cols = sorted({c for _r, c in cells})
    if orientation == "horizontal":
        r = rows[0]
        return [(int(r), int(cols[0] - 1)), (int(r), int(cols[-1] + 1))]
    c = cols[0]
    return [(int(rows[0] - 1), int(c)), (int(rows[-1] + 1), int(c))]


def _walk_segment(
    h: int,
    w: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
    seg_len: int,
    blocked: set[tuple[int, int]],
    path_cells: list[tuple[int, int]],
) -> tuple[int, int] | None:
    for _ in range(seg_len):
        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if (int(nr), int(nc)) in blocked:
            return None
        path_cells.append((int(nr), int(nc)))
        blocked.add((int(nr), int(nc)))
        r, c = int(nr), int(nc)
    return (int(r), int(c))


def _reflected_direction(orientation: str, dr: int, dc: int) -> tuple[int, int]:
    if orientation == "horizontal":
        return (-int(dr), int(dc))
    return (int(dr), -int(dc))


def _try_place_path_mirror(
    h: int,
    w: int,
    r: int,
    c: int,
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[dict, int, int] | None:
    mr, mc = int(r + dr), int(c + dc)
    if not (0 <= mr < h and 0 <= mc < w):
        return None
    if (mr, mc) in blocked:
        return None

    orientations = MIRROR_ORIENTATIONS.copy()
    random.shuffle(orientations)
    for orientation in orientations:
        new_dr, new_dc = _reflected_direction(orientation, dr, dc)
        next_r, next_c = int(r + new_dr), int(c + new_dc)
        if not (0 <= next_r < h and 0 <= next_c < w):
            continue
        if (next_r, next_c) in blocked:
            continue

        length = random.randint(3, 6)
        cells = _bar_cells_through(h, w, orientation, mr, mc, length)
        if cells is None:
            continue
        cell_set = set(cells)
        if cell_set & blocked:
            continue

        marker_candidates = _mirror_marker_candidates(cells, orientation)
        marker_options = [pos for pos in marker_candidates if pos not in blocked and _in_bounds((h, w), pos[0], pos[1])]
        if not marker_options:
            continue
        marker_pos = random.choice(marker_options)

        blocked |= cell_set
        blocked.add(marker_pos)
        return (
            {
                "orientation": orientation,
                "cells": list(cells),
                "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
            },
            int(new_dr),
            int(new_dc),
        )
    return None


def _simulate_meta(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    emitter_color: int,
    mirrors: list[dict],
    targets: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    mirror_by_cell: dict[tuple[int, int], dict] = {}
    for mirror in mirrors:
        for cell in mirror["cells"]:
            mirror_by_cell[(int(cell[0]), int(cell[1]))] = mirror

    final_color = _trace_path(out, mirror_by_cell, start, direction, emitter_color)
    tr, tc = targets[final_color]
    out[tr - 1 : tr + 3, tc - 1 : tc + 3] = int(final_color)
    return out


def _place_targets(
    h: int,
    w: int,
    target_colors: list[int],
    blocked: set[tuple[int, int]],
) -> dict[int, tuple[int, int]]:
    placements: dict[int, tuple[int, int]] = {}
    reserved = set(blocked)
    for color in target_colors:
        placed = False
        for _ in range(120):
            r = random.randint(1, h - 3)
            c = random.randint(1, w - 3)
            actual = {(int(r + dr), int(c + dc)) for dr in range(2) for dc in range(2)}
            halo = {(int(rr), int(cc)) for rr in range(r - 1, r + 3) for cc in range(c - 1, c + 3)}
            if actual & reserved:
                continue
            if halo & reserved:
                continue
            placements[int(color)] = (int(r), int(c))
            reserved |= halo
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place target halo")
    return placements


def _place_decoy_mirrors(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    target_colors: list[int],
) -> list[dict]:
    decoys: list[dict] = []
    for _ in range(random.randint(0, 3)):
        for _attempt in range(80):
            orientation = random.choice(MIRROR_ORIENTATIONS)
            length = random.randint(3, 6)
            if orientation == "horizontal":
                if w - length - 2 < 1:
                    continue
                r = random.randint(0, h - 1)
                start = random.randint(1, w - length - 1)
                cells = [(int(r), int(start + j)) for j in range(length)]
            else:
                if h - length - 2 < 1:
                    continue
                c = random.randint(0, w - 1)
                start = random.randint(1, h - length - 1)
                cells = [(int(start + j), int(c)) for j in range(length)]

            cell_set = set(cells)
            if cell_set & blocked:
                continue

            marker_candidates = _mirror_marker_candidates(cells, orientation)
            marker_options = [pos for pos in marker_candidates if pos not in blocked and _in_bounds((h, w), pos[0], pos[1])]
            if not marker_options:
                continue
            marker_pos = random.choice(marker_options)

            color = random.choice(target_colors)
            decoys.append(
                {
                    "orientation": orientation,
                    "cells": list(cells),
                    "marker_pos": (int(marker_pos[0]), int(marker_pos[1])),
                    "marker_color": int(color),
                }
            )
            blocked |= cell_set
            blocked.add(marker_pos)
            break
    return decoys


def _render_input(
    h: int,
    w: int,
    emitter_pos: tuple[int, int],
    yellow_pos: tuple[int, int],
    emitter_color: int,
    mirrors: list[dict],
    targets: dict[int, tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    er, ec = emitter_pos
    yr, yc = yellow_pos
    grid[er, ec] = int(emitter_color)
    grid[yr, yc] = int(YELLOW)

    for color, (r, c) in targets.items():
        grid[r : r + 2, c : c + 2] = int(color)

    for mirror in mirrors:
        for r, c in mirror["cells"]:
            grid[r, c] = int(GREY)
        mr, mc = mirror["marker_pos"]
        grid[mr, mc] = int(mirror["marker_color"])

    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)

    corner = random.choice(["tl", "tr", "bl", "br"])
    emitter_pos, direction, yellow_options = _corner_state(corner, h, w)
    yellow_pos = random.choice(yellow_options)
    emitter_color = random.choice(EMITTER_COLORS)
    n_reflections = random.randint(0, 4)

    blocked: set[tuple[int, int]] = {emitter_pos, yellow_pos}
    path_cells: list[tuple[int, int]] = []
    mirrors: list[dict] = []

    segment_lengths = [random.randint(2, 6) for _ in range(n_reflections)]
    r, c = emitter_pos
    dr, dc = direction

    for seg_len in segment_lengths:
        endpoint = _walk_segment(h, w, r, c, dr, dc, seg_len, blocked, path_cells)
        if endpoint is None:
            return None
        r, c = endpoint
        placed = _try_place_path_mirror(h, w, r, c, dr, dc, blocked)
        if placed is None:
            return None
        mirror, dr, dc = placed
        mirrors.append(mirror)

    final_segment_len = 0
    while True:
        nr, nc = int(r + dr), int(c + dc)
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if (nr, nc) in blocked:
            return None
        final_segment_len += 1
        path_cells.append((int(nr), int(nc)))
        blocked.add((int(nr), int(nc)))
        r, c = int(nr), int(nc)
    if final_segment_len < 2:
        return None

    used_colors = {int(emitter_color)}
    marker_colors = random.sample([c for c in TARGET_COLOR_POOL if c != int(emitter_color)], len(mirrors))
    for mirror, color in zip(mirrors, marker_colors):
        mirror["marker_color"] = int(color)
        used_colors.add(int(color))

    target_count_min = max(4, len(used_colors))
    target_count_max = min(6, len(TARGET_COLOR_POOL))
    target_count = random.randint(target_count_min, target_count_max)
    extras_pool = [c for c in TARGET_COLOR_POOL if c not in used_colors]
    extra_needed = target_count - len(used_colors)
    extra_colors = random.sample(extras_pool, extra_needed)
    target_colors = list(used_colors) + extra_colors
    random.shuffle(target_colors)

    try:
        target_positions = _place_targets(h, w, target_colors, blocked)
    except RuntimeError:
        return None

    blocked_with_targets = set(blocked)
    for r0, c0 in target_positions.values():
        for rr in range(r0 - 1, r0 + 3):
            for cc in range(c0 - 1, c0 + 3):
                blocked_with_targets.add((int(rr), int(cc)))

    decoys = _place_decoy_mirrors(h, w, blocked_with_targets, target_colors)
    all_mirrors = mirrors + decoys

    grid = _render_input(
        h=h,
        w=w,
        emitter_pos=emitter_pos,
        yellow_pos=yellow_pos,
        emitter_color=emitter_color,
        mirrors=all_mirrors,
        targets=target_positions,
    )

    expected = _simulate_meta(
        grid=grid,
        start=emitter_pos,
        direction=direction,
        emitter_color=emitter_color,
        mirrors=all_mirrors,
        targets=target_positions,
    )
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 7")
