from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 0
OBJECT_COLOR = 5

LEGEND_ROW_H = 4
LEGEND_ROWS = 4
LEGEND_H = LEGEND_ROW_H * LEGEND_ROWS
LEGEND_W = 5
GUARD = 1

BASE_ARROW = np.array([[1, 0], [1, 1]], dtype=bool)
ANGLES = (0, 90, 180, 270)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

SHAPE_LIBRARY: dict[int, list[np.ndarray]] = {
    1: [
        np.array([[1]], dtype=bool),
    ],
    2: [
        np.array([[1, 1]], dtype=bool),
    ],
    3: [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
    ],
    4: [
        np.array([[1, 1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
    ],
}


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _arrow_mask(angle: int) -> np.ndarray:
    return np.rot90(BASE_ARROW, -(angle // 90))


ARROW_TO_ANGLE = {_arrow_mask(angle).tobytes(): angle for angle in ANGLES}


def _nearest_int(x: float) -> int:
    return int(math.floor(x + 0.5))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, r0, c0


def _rotate_component(coords: np.ndarray, angle: int) -> np.ndarray:
    mask, r0, c0 = _mask_from_coords(coords)
    rot = np.rot90(mask, -(angle // 90))
    center_r = r0 + (mask.shape[0] - 1) / 2.0
    center_c = c0 + (mask.shape[1] - 1) / 2.0
    nr0 = _nearest_int(center_r - (rot.shape[0] - 1) / 2.0)
    nc0 = _nearest_int(center_c - (rot.shape[1] - 1) / 2.0)
    rel = np.argwhere(rot)
    out = rel + np.array([nr0, nc0], dtype=int)
    return out.astype(int)


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr, dc in DIR8:
        rr0 = max(0, dr)
        rr1 = mask.shape[0] + min(0, dr)
        cc0 = max(0, dc)
        cc1 = mask.shape[1] + min(0, dc)
        src_r0 = max(0, -dr)
        src_r1 = mask.shape[0] - max(0, dr)
        src_c0 = max(0, -dc)
        src_c1 = mask.shape[1] - max(0, dc)
        out[rr0:rr1, cc0:cc1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _parse_legend_top_left(grid: np.ndarray, bg: int) -> tuple[dict[int, int], int, int] | None:
    if grid.shape[0] < LEGEND_H + GUARD or grid.shape[1] < LEGEND_W + GUARD:
        return None
    if np.any(grid[LEGEND_H, : LEGEND_W + GUARD] != bg):
        return None
    if np.any(grid[: LEGEND_H + GUARD, LEGEND_W] != bg):
        return None

    legend = grid[:LEGEND_H, :LEGEND_W]
    colors = sorted(set(map(int, np.unique(legend))) - {bg})
    if len(colors) != 2:
        return None

    for count_color, arrow_color in (colors, colors[::-1]):
        mapping: dict[int, int] = {}
        ok = True
        for row_idx in range(LEGEND_ROWS):
            r0 = row_idx * LEGEND_ROW_H
            slot = legend[r0 : r0 + LEGEND_ROW_H, :]

            dot_col = slot[:, 0]
            if np.any((dot_col != bg) & (dot_col != count_color)):
                ok = False
                break
            count = int(np.sum(dot_col == count_color))
            if count not in (1, 2, 3, 4):
                ok = False
                break
            if not np.all(dot_col[: LEGEND_ROW_H - count] == bg):
                ok = False
                break
            if not np.all(dot_col[LEGEND_ROW_H - count :] == count_color):
                ok = False
                break

            if not np.all(slot[:, 1:3] == bg):
                ok = False
                break
            if not np.all(slot[[0, 3], 3:5] == bg):
                ok = False
                break
            patch = slot[1:3, 3:5]
            if np.any((patch != bg) & (patch != arrow_color)):
                ok = False
                break
            key = (patch == arrow_color).tobytes()
            if key not in ARROW_TO_ANGLE:
                ok = False
                break

            angle = ARROW_TO_ANGLE[key]
            if count in mapping:
                ok = False
                break
            mapping[count] = angle

        if ok and set(mapping.keys()) == {1, 2, 3, 4} and len(set(mapping.values())) == 4:
            return mapping, int(count_color), int(arrow_color)

    return None


def _solve_canonical(grid: np.ndarray) -> np.ndarray | None:
    bg = _bg_color(grid)
    parsed = _parse_legend_top_left(grid, bg)
    if parsed is None:
        return None
    mapping, count_color, arrow_color = parsed

    remaining = set(map(int, np.unique(grid))) - {bg, count_color, arrow_color}
    if len(remaining) != 1:
        return None
    object_color = int(next(iter(remaining)))

    if np.any(grid[: LEGEND_H + GUARD, : LEGEND_W + GUARD] == object_color):
        return None

    comps = _components(grid == object_color)
    if not comps:
        return None

    out = np.full_like(grid, bg)
    for comp in comps:
        area = int(len(comp))
        if area not in mapping:
            return None
        rotated = _rotate_component(comp, mapping[area])
        if len({(int(r), int(c)) for r, c in rotated}) != area:
            return None
        for r, c in rotated:
            rr = int(r)
            cc = int(c)
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]):
                return None
            if out[rr, cc] != bg:
                return None
            out[rr, cc] = object_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    According to the legend, map each grey object's pixel count to a rotation angle,
    rotate each object around its bbox center, and erase the legend.
    """
    g = np.asarray(grid, dtype=int)
    for k in range(4):
        canon = np.rot90(g, k)
        solved = _solve_canonical(canon)
        if solved is not None:
            return np.rot90(solved, -k)
    raise AssertionError("could not locate a valid corner legend")


def _sample_shape(area: int) -> np.ndarray:
    base = random.choice(SHAPE_LIBRARY[area])
    rot = np.rot90(base, random.randrange(4))
    return _trim_mask(rot)


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    rel = np.argwhere(mask)
    return rel + np.array([r0, c0], dtype=int)


def _in_bounds(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.all(coords[:, 0] >= 0)
        and np.all(coords[:, 0] < h)
        and np.all(coords[:, 1] >= 0)
        and np.all(coords[:, 1] < w)
    )


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    for r, c in coords:
        grid[int(r), int(c)] = int(color)


def _build_legend(
    grid: np.ndarray,
    mapping: dict[int, int],
    count_color: int,
    arrow_color: int,
    row_order: list[int],
) -> None:
    for row_idx, count in enumerate(row_order):
        r0 = row_idx * LEGEND_ROW_H
        grid[r0 + LEGEND_ROW_H - count : r0 + LEGEND_ROW_H, 0] = count_color
        patch = _arrow_mask(mapping[count])
        grid[r0 + 1 : r0 + 3, 3:5][patch] = arrow_color


def _place_objects(
    h: int,
    w: int,
    mapping: dict[int, int],
    areas: list[int],
    forbidden: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    occ_in = np.zeros((h, w), dtype=bool)
    occ_out = np.zeros((h, w), dtype=bool)
    blocked_in = forbidden.copy()
    blocked_out = forbidden.copy()

    for area in areas:
        angle = mapping[area]
        placed = False
        for _ in range(250):
            mask = _sample_shape(area)
            mh, mw = mask.shape
            r0 = random.randint(0, h - mh)
            c0 = random.randint(0, w - mw)
            coords_in = _coords_from_mask(mask, r0, c0)
            coords_out = _rotate_component(coords_in, angle)

            if not _in_bounds(coords_out, h, w):
                continue
            if any(blocked_in[int(r), int(c)] for r, c in coords_in):
                continue
            if any(blocked_out[int(r), int(c)] for r, c in coords_out):
                continue
            if len({(int(r), int(c)) for r, c in coords_out}) != area:
                continue

            inputs.append(coords_in)
            outputs.append(coords_out)
            occ_in[coords_in[:, 0], coords_in[:, 1]] = True
            occ_out[coords_out[:, 0], coords_out[:, 1]] = True
            blocked_in = forbidden | _dilate(occ_in)
            blocked_out = forbidden | _dilate(occ_out)
            placed = True
            break
        if not placed:
            return None

    return inputs, outputs


def _oracle_input() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(22, 30)
    w = random.randint(22, 30)

    legend_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], 2)
    count_color, arrow_color = legend_colors
    mapping = dict(zip((1, 2, 3, 4), random.sample(list(ANGLES), 4)))
    row_order = random.sample([1, 2, 3, 4], 4)

    forbidden = np.zeros((h, w), dtype=bool)
    forbidden[: LEGEND_H + GUARD, : LEGEND_W + GUARD] = True

    n_objects = random.randint(4, 7)
    areas = [1, 2, 3, 4] + random.choices([1, 2, 3, 4], k=n_objects - 4)
    random.shuffle(areas)

    placed = _place_objects(h, w, mapping, areas, forbidden)
    if placed is None:
        return None
    inputs, outputs = placed

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    _build_legend(inp, mapping, count_color, arrow_color, row_order)
    for coords in inputs:
        _paint(inp, coords, OBJECT_COLOR)
    for coords in outputs:
        _paint(out, coords, OBJECT_COLOR)

    k = random.randrange(4)
    inp = np.rot90(inp, k)
    out = np.rot90(out, k)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(300):
        pair = _oracle_input()
        if pair is None:
            continue
        inp, out = pair
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, out):
            return inp
    raise RuntimeError("failed to generate a valid input")
