from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = list(range(10))

# Generator bounds.
MIN_SIZE = 14
MAX_SIZE = 24
MIN_LEGEND_ROWS = 2
MAX_LEGEND_ROWS = 4
MAX_GEN_ATTEMPTS = 400
ROOM_KINDS = ("rect", "step_h", "step_v")


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_edge(coords: list[tuple[int, int]], height: int, width: int) -> bool:
    for r, c in coords:
        if r == 0 or c == 0 or r == height - 1 or c == width - 1:
            return True
    return False


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r_src0 = max(0, -dr)
            r_src1 = mask.shape[0] - max(0, dr)
            c_src0 = max(0, -dc)
            c_src1 = mask.shape[1] - max(0, dc)
            r_dst0 = max(0, dr)
            r_dst1 = out.shape[0] - max(0, -dr)
            c_dst0 = max(0, dc)
            c_dst1 = out.shape[1] - max(0, -dc)
            out[r_dst0:r_dst1, c_dst0:c_dst1] |= mask[r_src0:r_src1, c_src0:c_src1]
    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[np.ix_(rows, cols)]


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    color_mask = (grid != bg) & (grid != GREY)
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for coords in _components(color_mask):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width != 2 or height < 2:
            continue
        if len(coords) != height * width:
            continue
        if r0 != 0 and c0 != 0:
            continue
        candidates.append((len(coords), (r0, c0, r1, c1)))

    assert len(candidates) == 1, "expected exactly one 2-column legend on the top or left edge"
    _size, (r0, c0, r1, c1) = candidates[0]

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    mapping: dict[int, int] = {}
    for r in range(r0, r1 + 1):
        key = int(grid[r, c0])
        fill = int(grid[r, c0 + 1])
        assert key != bg and key != GREY
        assert fill != bg and fill != GREY
        assert key not in mapping
        mapping[key] = fill
    return mapping, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-enclosed region using the legend row indexed by its interior marker color.

    Input structure:
    - One dominant background color.
    - A 2-column vertical legend touches the left or top edge.
    - Several closed grey boundaries surround regions whose interiors contain
      background plus one unique non-background marker color.

    Rule:
    - Read the legend as marker-color -> fill-color.
    - Find every non-grey, non-legend component that does not touch the grid edge.
    - Each such component is an enclosed region interior; replace all of its cells
      with the mapped fill color.
    - Keep grey boundaries and the legend unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    mapping, legend_mask = _find_legend(g, bg)

    region_mask = (g != GREY) & (~legend_mask)
    out = g.copy()
    enclosed_found = 0
    for coords in _components(region_mask):
        if _touches_edge(coords, *g.shape):
            continue
        marker_colors = {int(g[r, c]) for r, c in coords if int(g[r, c]) != bg}
        assert len(marker_colors) == 1, "each enclosed region must contain exactly one marker color"
        key = next(iter(marker_colors))
        assert key in mapping, "marker color missing from legend"
        fill = mapping[key]
        for r, c in coords:
            out[r, c] = fill
        enclosed_found += 1

    assert enclosed_found > 0, "expected at least one enclosed region"
    return out


def _random_interior_shape(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        kind = rng.choice(ROOM_KINDS)
        if kind == "rect":
            mask = np.ones((rng.randint(2, 5), rng.randint(2, 6)), dtype=bool)
        elif kind == "step_h":
            h1, w1 = rng.randint(2, 5), rng.randint(2, 5)
            h2, w2 = rng.randint(2, 5), rng.randint(2, 5)
            overlap_cols = rng.randint(1, min(w1, w2) - 1)
            shift_r = rng.randint(-(h2 - 2), h1 - 2)
            shift_c = w1 - overlap_cols
            rmin = min(0, shift_r)
            rmax = max(h1, shift_r + h2)
            cmax = max(w1, shift_c + w2)
            mask = np.zeros((rmax - rmin, cmax), dtype=bool)
            mask[-rmin : -rmin + h1, :w1] = True
            mask[shift_r - rmin : shift_r - rmin + h2, shift_c : shift_c + w2] = True
        else:
            h1, w1 = rng.randint(2, 5), rng.randint(2, 5)
            h2, w2 = rng.randint(2, 5), rng.randint(2, 5)
            overlap_rows = rng.randint(1, min(h1, h2) - 1)
            shift_r = h1 - overlap_rows
            shift_c = rng.randint(-(w2 - 2), w1 - 2)
            rmax = max(h1, shift_r + h2)
            cmin = min(0, shift_c)
            cmax = max(w1, shift_c + w2)
            mask = np.zeros((rmax, cmax - cmin), dtype=bool)
            mask[:h1, -cmin : -cmin + w1] = True
            mask[shift_r : shift_r + h2, shift_c - cmin : shift_c - cmin + w2] = True

        mask = _trim_mask(mask)
        if not np.all(np.any(mask, axis=0)) or not np.all(np.any(mask, axis=1)):
            continue
        if len(_components(mask)) != 1:
            continue
        if int(mask.sum()) < 4:
            continue
        return mask
    raise RuntimeError("failed to build room interior")


def _make_room_patch(rng: random.Random, key_color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interior = np.pad(_random_interior_shape(rng), 1, constant_values=False)
    boundary = _dilate8(interior) & (~interior)
    if len(_components(boundary)) != 1:
        raise RuntimeError("room boundary must stay 4-connected")

    patch = np.zeros(interior.shape, dtype=int)
    patch[boundary] = GREY

    interior_coords = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
    max_markers = min(4, len(interior_coords))
    n_markers = rng.randint(1, max_markers)
    marker_indices = rng.sample(range(len(interior_coords)), n_markers)
    markers = np.zeros_like(interior, dtype=bool)
    for idx in marker_indices:
        r, c = interior_coords[idx]
        patch[r, c] = key_color
        markers[r, c] = True

    return patch, interior, markers


def _reserve_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _bbox_is_free(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_legend(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    index_colors: list[int],
    fill_colors: list[int],
) -> tuple[tuple[int, int, int, int], str]:
    height, width = grid.shape
    n_rows = len(index_colors)
    side = rng.choice(["left", "top"])
    if side == "left":
        r0 = rng.randint(1, height - n_rows - 1)
        c0 = 0
    else:
        r0 = 0
        c0 = rng.randint(1, width - 3)

    for i, (key, fill) in enumerate(zip(index_colors, fill_colors)):
        grid[r0 + i, c0] = key
        grid[r0 + i, c0 + 1] = fill

    bbox = (r0, c0, r0 + n_rows - 1, c0 + 1)
    _reserve_bbox(reserved, bbox, pad=1)
    return bbox, side


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(MIN_SIZE, MAX_SIZE)
    width = rng.randint(MIN_SIZE, MAX_SIZE)
    bg = rng.choice([c for c in ALL_COLORS if c != GREY])
    grid = np.full((height, width), bg, dtype=int)
    expected = grid.copy()
    reserved = np.zeros((height, width), dtype=bool)

    palette = [c for c in ALL_COLORS if c not in (bg, GREY)]
    n_rows = rng.randint(MIN_LEGEND_ROWS, MAX_LEGEND_ROWS)
    index_colors = rng.sample(palette, n_rows)
    remaining = [c for c in palette if c not in index_colors]
    fill_colors = rng.sample(remaining, n_rows)
    mapping = dict(zip(index_colors, fill_colors))

    legend_bbox, _side = _place_legend(grid, reserved, rng, index_colors, fill_colors)
    expected[legend_bbox[0] : legend_bbox[2] + 1, legend_bbox[1] : legend_bbox[3] + 1] = (
        grid[legend_bbox[0] : legend_bbox[2] + 1, legend_bbox[1] : legend_bbox[3] + 1]
    )

    n_rooms = rng.randint(1, n_rows)
    room_keys = rng.sample(index_colors, n_rooms)

    placed = 0
    for key in room_keys:
        patch, interior_mask, marker_mask = _make_room_patch(rng, key)
        ph, pw = patch.shape
        placed_here = False
        for _ in range(120):
            r0 = rng.randint(1, height - ph - 1)
            c0 = rng.randint(1, width - pw - 1)
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if not _bbox_is_free(reserved, bbox, pad=1):
                continue

            sub = grid[r0 : r0 + ph, c0 : c0 + pw]
            sub[patch == GREY] = GREY
            sub[marker_mask] = key

            exp = expected[r0 : r0 + ph, c0 : c0 + pw]
            exp[patch == GREY] = GREY
            exp[interior_mask] = mapping[key]

            _reserve_bbox(reserved, bbox, pad=1)
            placed += 1
            placed_here = True
            break
        if not placed_here:
            return None

    if placed != n_rooms:
        return None

    if _mode_color(grid) != bg:
        return None

    try:
        mapping2, legend_mask = _find_legend(grid, bg)
    except AssertionError:
        return None
    if mapping2 != mapping:
        return None

    region_mask = (grid != GREY) & (~legend_mask)
    enclosed = [comp for comp in _components(region_mask) if not _touches_edge(comp, height, width)]
    if len(enclosed) != n_rooms:
        return None
    for comp in enclosed:
        colors = {int(grid[r, c]) for r, c in comp if int(grid[r, c]) != bg}
        if len(colors) != 1:
            return None
        key = next(iter(colors))
        if key not in mapping:
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
