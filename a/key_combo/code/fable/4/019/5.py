from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


DIR_BY_COLOR = {
    int(Color.YELLOW): (-1, 0),
    int(Color.RED): (1, 0),
    int(Color.BLUE): (0, -1),
    int(Color.GREEN): (0, 1),
}


def _to_int_array(grid: np.ndarray) -> np.ndarray:
    return np.asarray(grid, dtype=int)


def _find_cross(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    rows = [r for r in range(h) if np.all(grid[r, :] == grid[r, 0])]
    cols = [c for c in range(w) if np.all(grid[:, c] == grid[0, c])]
    if not rows or not cols:
        raise AssertionError("input must contain full-width separator row/col")

    separators = [
        (r, c, int(grid[r, c]))
        for r in rows
        for c in cols
        if int(grid[r, c]) == int(grid[0, c])
    ]
    # Prefer non-border separators that split all four quadrants.
    valid = [
        (r, c, col)
        for r, c, col in separators
        if 0 < r < h - 1 and 0 < c < w - 1
    ]
    if len(valid) != 1:
        raise AssertionError(
            f"expected exactly one solid cross; got {len(valid)} candidates"
        )

    return int(valid[0][0]), int(valid[0][1]), int(valid[0][2])


def main(grid: np.ndarray) -> np.ndarray:
    """Translate each quadrant's sprite by a direction-coded isolated-dot count.

    Input structure:
    - The grid is split by a full separator row and full separator column into 4 quadrants.
    - Each quadrant contains one single-color 4-connected sprite and several singleton dots
      of exactly one direction color.
    - Dot colors map to movement direction (yellow-up, red-down, blue-left, green-right);
      sprite color is different from all dot colors.

    Rule:
    - For each quadrant, delete all singleton dots and identify the sprite and its color.
    - Count how many dots share the same color, move the sprite by that count in the mapped direction,
      and keep the cross unchanged.

    Key insight:
    - The dots are dual-channel instructions: their color is the direction, and their quantity is
      the step distance; consuming them means they disappear in the output.
    """
    g = _to_int_array(grid)
    if g.ndim != 2:
        raise AssertionError("expected a 2D ARC grid")

    bg = get_bg(g)
    cross_r, cross_c, separator = _find_cross(g)
    h, w = g.shape

    quadrants = [
        (slice(0, cross_r), slice(0, cross_c)),
        (slice(0, cross_r), slice(cross_c + 1, w)),
        (slice(cross_r + 1, h), slice(0, cross_c)),
        (slice(cross_r + 1, h), slice(cross_c + 1, w)),
    ]

    out = g.copy()
    for q_rows, q_cols in quadrants:
        quad = g[q_rows, q_cols]
        comps = find_components(
            quad,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            min_area=1,
        )
        if not comps:
            continue

        dot_counts: dict[int, int] = {}
        sprite_comp = None
        dot_coords: list[tuple[np.ndarray, int]] = []

        for comp in comps:
            color = int(comp["color"])
            area = int(comp["area"])
            coords = comp["coords"]
            if area == 1 and color in DIR_BY_COLOR:
                dot_counts[color] = dot_counts.get(color, 0) + area
                dot_coords.append((coords, color))
            elif area >= 2 and sprite_comp is None:
                sprite_comp = comp
            elif area >= 2 and sprite_comp is not None:
                # Keep the largest movable sprite in case of accidental noise.
                if area > int(sprite_comp["area"]):
                    sprite_comp = comp

        if sprite_comp is None or len(dot_counts) != 1:
            continue

        (dot_color, dot_count), = dot_counts.items()
        if dot_color not in DIR_BY_COLOR:
            raise AssertionError(f"unexpected instruction color {dot_color}")

        dr, dc = DIR_BY_COLOR[dot_color]
        sprite_color = int(sprite_comp["color"])
        s_coords = np.asarray(sprite_comp["coords"], dtype=int)
        if s_coords.size == 0:
            continue

        # Coordinate shift from local quadrant coordinates to global.
        r0 = int(q_rows.start or 0)
        c0 = int(q_cols.start or 0)
        global_s = s_coords + np.array([r0, c0], dtype=int)
        target = global_s + np.array([dr * dot_count, dc * dot_count], dtype=int)
        tr = target[:, 0]
        tc = target[:, 1]

        if not (
            0 <= tr.min() and tr.max() < h and 0 <= tc.min() and tc.max() < w
        ):
            raise AssertionError("sprite translation leaves grid")

        # Consume dots first, then move sprite.
        for dc_coords, dcol in dot_coords:
            rr = dc_coords[:, 0] + r0
            cc = dc_coords[:, 1] + c0
            out[rr, cc] = bg

        # Clear old sprite.
        out[global_s[:, 0], global_s[:, 1]] = bg
        # Draw moved sprite (separator cross preserved by construction).
        out[tr, tc] = sprite_color

        # Safety: keep separator fixed even if a moved sprite hits it.
        out[cross_r, :] = separator
        out[:, cross_c] = separator

    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rr = coords[:, 0]
    cc = coords[:, 1]
    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def _random_connected_cells(h: int, w: int, area: int, rng: random.Random) -> np.ndarray | None:
    if area <= 0 or area > h * w:
        return None

    for _ in range(200):
        start_r = rng.randrange(h)
        start_c = rng.randrange(w)
        coords = [(start_r, start_c)]
        seen = {(start_r, start_c)}

        while len(coords) < area:
            r, c = rng.choice(coords)
            neigh = []
            if r > 0 and (r - 1, c) not in seen:
                neigh.append((r - 1, c))
            if r + 1 < h and (r + 1, c) not in seen:
                neigh.append((r + 1, c))
            if c > 0 and (r, c - 1) not in seen:
                neigh.append((r, c - 1))
            if c + 1 < w and (r, c + 1) not in seen:
                neigh.append((r, c + 1))
            if not neigh:
                break
            rr, cc = rng.choice(neigh)
            seen.add((rr, cc))
            coords.append((rr, cc))

        if len(coords) == area:
            return np.array(coords, dtype=int)

    return None


def _place_isolated_dots(
    h: int,
    w: int,
    occupied: np.ndarray,
    num: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if num <= 0:
        return []
    if num >= h * w:
        return []

    points: list[tuple[int, int]] = []
    grid_hole = occupied.copy()
    for _ in range(num):
        found = False
        for _ in range(400):
            r = rng.randrange(h)
            c = rng.randrange(w)
            if grid_hole[r, c]:
                continue
            if r > 0 and grid_hole[r - 1, c]:
                continue
            if r + 1 < h and grid_hole[r + 1, c]:
                continue
            if c > 0 and grid_hole[r, c - 1]:
                continue
            if c + 1 < w and grid_hole[r, c + 1]:
                continue
            points.append((r, c))
            grid_hole[r, c] = True
            found = True
            break
        if not found:
            return []
    return points


def _quadrant_slices(
    cross_r: int, cross_c: int, h: int, w: int
) -> list[tuple[slice, slice]]:
    return [
        (slice(0, cross_r), slice(0, cross_c)),
        (slice(0, cross_r), slice(cross_c + 1, w)),
        (slice(cross_r + 1, h), slice(0, cross_c)),
        (slice(cross_r + 1, h), slice(cross_c + 1, w)),
    ]


def generate_input() -> np.ndarray:
    bg = int(Color.BLACK)

    # Direction colors fixed by problem statement.
    direction_colors = list(DIR_BY_COLOR.keys())

    for _ in range(500):
        h = random.randint(14, 28)
        w = random.randint(14, 28)
        cross_r = random.randint(4, h - 5)
        cross_c = random.randint(4, w - 5)

        all_colors = [c for c in Color.ALL_COLORS if c != bg]
        cross_color = random.choice(all_colors)

        sprite_colors = [c for c in Color.NOT_BLACK if c not in (bg, cross_color, *direction_colors)]
        if len(sprite_colors) < 4:
            continue
        sprite_palette = random.sample(sprite_colors, 4)

        grid = np.full((h, w), bg, dtype=int)
        grid[cross_r, :] = cross_color
        grid[:, cross_c] = cross_color

        quadrants = _quadrant_slices(cross_r, cross_c, h, w)
        rng = random.Random(random.randint(0, 2**31 - 1))
        dir_colors = direction_colors.copy()
        rng.shuffle(dir_colors)

        placed = True
        for idx, (q_rows, q_cols) in enumerate(quadrants):
            q = grid[q_rows, q_cols]
            q_h, q_w = q.shape
            q_area = q_h * q_w
            if q_area < 10:
                placed = False
                break

            direction_color = dir_colors[idx]
            dr, dc = DIR_BY_COLOR[direction_color]

            # Build sprite mask by random connected walk.
            sprite_area = random.randint(2, min(10, max(2, q_area // 3)))
            sprite_coords = _random_connected_cells(q_h, q_w, sprite_area, rng)
            if sprite_coords is None:
                placed = False
                break

            r0, c0, r1, c1 = _bbox_from_coords(sprite_coords)
            if dr == -1 and r0 < 1:
                continue
            if dr == 1 and r1 > q_h - 2:
                continue
            if dc == -1 and c0 < 1:
                continue
            if dc == 1 and c1 > q_w - 2:
                continue

            max_steps = r0 if dr == -1 else q_h - 1 - r1 if dr == 1 else c0 if dc == -1 else q_w - 1 - c1
            if max_steps < 1:
                placed = False
                break

            # Keep enough free cells for dot placement.
            occupied = np.zeros((q_h, q_w), dtype=bool)
            occupied[sprite_coords[:, 0], sprite_coords[:, 1]] = True
            free_cells = int(q_area - sprite_coords.shape[0])
            if free_cells < 1:
                placed = False
                break

            max_count = min(max_steps, max(1, free_cells // 2))
            if random.random() < 0.35:
                count = min(max_steps, max_count)
            else:
                count = rng.randint(1, max_count)

            dots = _place_isolated_dots(q_h, q_w, occupied, count, rng)
            if not dots or len(dots) != count:
                placed = False
                break

            # Paint sprite.
            sprite_color = sprite_palette[idx]
            rs = q_rows.start + sprite_coords[:, 0]
            cs = q_cols.start + sprite_coords[:, 1]
            grid[rs, cs] = sprite_color

            # Paint singleton instruction dots (same color gives direction, count gives distance).
            for rr, cc in dots:
                rr_g = q_rows.start + rr
                cc_g = q_cols.start + cc
                grid[rr_g, cc_g] = direction_color

        if not placed:
            continue
        # Final deterministic sanity checks.
        try:
            _ = main(grid)
        except Exception:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input for task 5")
