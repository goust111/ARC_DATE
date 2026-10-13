"""Replicate each sprite by copying it along a direction implied by its opposite side band.

Input structure:
- The background is the dominant color.
- Each seed icon is a compact 3x3 multicolor sprite.
- One sprite side is adjacent to a same-color band of length 1..4, and the opposite side has a single marker pixel of the same color.

Rule:
- From each sprite-band-marker group, infer the growth direction as the side opposite the band.
- Let the band length be N; place N copies of the sprite along that direction, with one blank gap between consecutive copies (copy step = 3 + 1).
- The marker pixel is removed in output while all band pixels remain.

Key insight:
-The band simultaneously encodes the growth direction (by being on the side opposite copies) and the number of copies (its length).
"""

from collections import deque
import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    comps = []

    for sr in range(h):
        for sc in range(w):
            if visited[sr, sc] or grid[sr, sc] == bg:
                continue
            visited[sr, sc] = True
            q = deque([(sr, sc)])
            comp = []
            while q:
                r, c = q.popleft()
                comp.append((int(r), int(c)))
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if visited[nr, nc] or grid[nr, nc] == bg:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_contiguous(values):
    vals = sorted(values)
    if not vals:
        return False
    return len(vals) == (vals[-1] - vals[0] + 1)


def _infer_sprite_instruction(comp, grid: np.ndarray, bg: int):
    comp_set = set(comp)
    if len(comp_set) < 3:
        return None

    h, w = grid.shape
    min_r = min(r for r, _ in comp_set)
    max_r = max(r for r, _ in comp_set)
    min_c = min(c for _, c in comp_set)
    max_c = max(c for _, c in comp_set)

    for top in range(max(0, min_r - 1), min(h - 2, max_r + 1) + 1):
        for left in range(max(0, min_c - 1), min(w - 2, max_c + 1) + 1):
            block = [(top + dr, left + dc) for dr in range(3) for dc in range(3)]
            if any(p not in comp_set for p in block):
                continue

            block_set = set(block)
            outside = comp_set - block_set
            if not outside:
                continue

            def side_cells(kind):
                if kind == "up":
                    return {(top - 1, c) for c in range(left - 1, left + 4) if (top - 1, c) in comp_set}
                if kind == "down":
                    return {(top + 3, c) for c in range(left - 1, left + 4) if (top + 3, c) in comp_set}
                if kind == "left":
                    return {(r, left - 1) for r in range(top - 1, top + 4) if (r, left - 1) in comp_set}
                return {(r, left + 3) for r in range(top - 1, top + 4) if (r, left + 3) in comp_set}

            dir_checks = [
                {
                    "band": "up",
                    "grow": (1, 0),
                    "marker": (top + 3, left + 1),
                    "axis": "x",
                },
                {
                    "band": "down",
                    "grow": (-1, 0),
                    "marker": (top - 1, left + 1),
                    "axis": "x",
                },
                {
                    "band": "left",
                    "grow": (0, 1),
                    "marker": (top + 1, left + 3),
                    "axis": "y",
                },
                {
                    "band": "right",
                    "grow": (0, -1),
                    "marker": (top + 1, left - 1),
                    "axis": "y",
                },
            ]

            for spec in dir_checks:
                band = side_cells(spec["band"])
                if not band or len(band) > 4:
                    continue

                band_color_set = {int(grid[r, c]) for r, c in band}
                if len(band_color_set) != 1:
                    continue
                band_color = next(iter(band_color_set))
                if band_color == bg:
                    continue

                mr, mc = spec["marker"]
                marker = (mr, mc)
                if marker not in comp_set:
                    continue
                if int(grid[mr, mc]) != band_color:
                    continue

                all_outside_ok = outside == band | {marker}
                if not all_outside_ok:
                    continue

                if spec["axis"] == "x":
                    axis_vals = sorted(c for _, c in band)
                else:
                    axis_vals = sorted(r for r, _ in band)
                if not _is_contiguous(axis_vals):
                    continue

                sprite = np.array(
                    [
                        [int(grid[top + dr, left + dc]) for dc in range(3)]
                        for dr in range(3)
                    ],
                    dtype=int,
                )

                dr, dc = spec["grow"]
                if top + dr * (3 + 1) * 4 < 0:
                    pass
                if top + 3 + dr * (3 + 1) * (len(band) - 1) < 0:
                    pass

                return {
                    "top": top,
                    "left": left,
                    "dr": dr,
                    "dc": dc,
                    "step": 4,
                    "count": len(band),
                    "marker": marker,
                    "sprite": sprite,
                }

    return None


def _place_sprite(out: np.ndarray, sprite: np.ndarray, top: int, left: int):
    for dr in range(3):
        for dc in range(3):
            out[top + dr, left + dc] = sprite[dr, dc]


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    bg = _most_common_color(grid)
    out = np.array(grid, copy=True, dtype=int)

    instructions = []
    for comp in _connected_components(grid, bg):
        info = _infer_sprite_instruction(comp, grid, bg)
        if info is not None:
            instructions.append(info)

    instructions.sort(key=lambda d: (d["top"], d["left"]))

    for info in instructions:
        mr, mc = info["marker"]
        out[mr, mc] = bg
        top = info["top"]
        left = info["left"]
        dr = info["dr"]
        dc = info["dc"]
        step = info["step"]
        count = info["count"]
        sprite = info["sprite"]

        for k in range(1, count + 1):
            nt = top + dr * step * k
            nl = left + dc * step * k
            if nt < 0 or nl < 0 or nt + 3 > out.shape[0] or nl + 3 > out.shape[1]:
                continue
            _place_sprite(out, sprite, nt, nl)

    return out


def _sprite_colors(rng):
    c1, c2 = int(rng.integers(1, 10)), int(rng.integers(1, 10))
    if c2 == c1:
        c2 = (c2 % 9) + 1
    vals = [c1, c1, c2]
    rows = rng.choice(vals, size=(3, 3))
    return rows.astype(int)


def _band_segments(top: int, left: int, direction: int, cnt: int, h: int, w: int):
    if direction == 0:  # band up
        if top <= 0 or top + 3 >= h:
            return None
        band_row = top - 1
        starts = [s for s in range(left - 1, left + 3) if 0 <= s and s + cnt - 1 < w]
        return (
            "up",
            [(band_row, s + i) for s in [starts[0]] for i in range(cnt)],
            (top + 3, None),
        ) if starts else None

    if direction == 1:  # band down
        if top + 3 >= h:
            return None
        band_row = top + 3
        starts = [s for s in range(left - 1, left + 3) if 0 <= s and s + cnt - 1 < w]
        return (
            "down",
            [(band_row, s + i) for s in [starts[0]] for i in range(cnt)],
            (top - 1, None),
        ) if starts else None

    if direction == 2:  # band left
        if left <= 0 or left + 3 >= w:
            return None
        band_col = left - 1
        starts = [s for s in range(top - 1, top + 3) if 0 <= s and s + cnt - 1 < h]
        return (
            "left",
            [(s + i, band_col) for s in [starts[0]] for i in range(cnt)],
            (None, left + 3),
        ) if starts else None

    # direction == 3, band right
    if left + 3 >= w:
        return None
    band_col = left + 3
    starts = [s for s in range(top - 1, top + 3) if 0 <= s and s + cnt - 1 < h]
    if not starts:
        return None
    return (
        "right",
        [(s + i, band_col) for s in [starts[0]] for i in range(cnt)],
        (None, left - 1),
    )


def _copy_footprint(top: int, left: int, dr: int, dc: int, count: int):
    cells = []
    for k in range(count + 1):
        nt = top + dr * 4 * k
        nl = left + dc * 4 * k
        for r in range(nt, nt + 3):
            for c in range(nl, nl + 3):
                cells.append((r, c))
    return cells


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    bg = 0

    for _ in range(300):
        h = int(rng.integers(18, 31))
        w = int(rng.integers(18, 31))
        grid = np.full((h, w), bg, dtype=np.uint8)
        occupied = np.zeros((h, w), dtype=bool)

        object_count = int(rng.integers(1, 5))
        placed = 0

        for _ in range(object_count):
            good = False
            for _ in range(800):
                direction = int(rng.integers(0, 4))
                count = int(rng.integers(1, 5))

                if direction in (0, 1):
                    top = int(rng.integers(0, h - 3))
                    left = int(rng.integers(0, w - 3))
                    dr, dc = (1, 0) if direction == 0 else (-1, 0)
                    # growth-space sanity bounds
                    if direction == 0 and top - 1 < 0:
                        continue
                    if direction == 1 and top + 3 >= h:
                        continue
                    if direction == 0 and top + 3 + 4 * (count - 1) + 2 >= h:
                        continue
                    if direction == 1 and top - 4 * (count - 1) < 0:
                        continue
                    band_info = _band_segments(top, left, direction, count, h, w)
                    if band_info is None:
                        continue
                    _, band_cells, marker_fix = band_info
                    marker_row, _ = marker_fix
                else:
                    top = int(rng.integers(0, h - 3))
                    left = int(rng.integers(0, w - 3))
                    dr, dc = (0, 1) if direction == 2 else (0, -1)
                    if direction == 2 and left + 3 + 4 * (count - 1) + 2 >= w:
                        continue
                    if direction == 3 and left - 4 * (count - 1) < 0:
                        continue
                    band_info = _band_segments(top, left, direction, count, h, w)
                    if band_info is None:
                        continue
                    _, band_cells, marker_fix = band_info
                    _, marker_col = marker_fix

                # Random marker on opposite side
                if direction in (0, 1):
                    marker_row = marker_row
                    marker_col_candidates = [left, left + 1, left + 2]
                    marker_col_candidates = [x for x in marker_col_candidates if 0 <= x < w]
                    if not marker_col_candidates:
                        continue
                    marker = (marker_row, int(rng.choice(marker_col_candidates)))
                else:
                    marker_col = marker_col
                    marker_row_candidates = [top, top + 1, top + 2]
                    marker_row_candidates = [x for x in marker_row_candidates if 0 <= x < h]
                    if not marker_row_candidates:
                        continue
                    marker = (int(rng.choice(marker_row_candidates)), marker_col)

                # Copy footprint for output-scale expansion (growth direction + seeds)
                sprite = _sprite_colors(rng)
                if np.all(sprite == sprite[0, 0]):
                    sprite[0, 0] = (sprite[0, 0] % 9) + 1

                occupied_colors = set(np.unique(sprite))
                band_color = int(rng.choice([c for c in range(1, 10) if c not in occupied_colors]))

                foot = _copy_footprint(top, left, dr, dc, count)
                seed_pixels = list(band_cells) + [marker]
                all_pixels = foot + seed_pixels

                if any(r < 0 or c < 0 or r >= h or c >= w for r, c in all_pixels):
                    continue
                if any(occupied[r, c] for r, c in all_pixels):
                    continue

                for r, c in band_cells:
                    grid[r, c] = band_color
                grid[marker] = band_color
                _place_sprite(grid, sprite, top, left)

                for r, c in all_pixels:
                    occupied[r, c] = True

                good = True
                break

            if good:
                placed += 1

        if placed > 0:
            return grid

    # Fallback: always return at least one simple object.
    h = 20
    w = 20
    grid = np.full((h, w), bg, dtype=np.uint8)
    sprite = np.array([[1, 1, 2], [2, 3, 1], [1, 2, 3]], dtype=np.uint8)
    _place_sprite(grid, sprite, 8, 10)
    for c in (10, 11):
        grid[7, c] = 5
    grid[11, 11] = 5
    return grid
