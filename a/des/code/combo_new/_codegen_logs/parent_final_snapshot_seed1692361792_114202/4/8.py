import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
NON_SPECIAL_COLORS = (1, 2, 3, 4, 6, 7, 8)

DECODER_COUNT_CHOICES = (3, 4)
OFFSET_RANGE = (1, 4)
OBJECT_COUNT_RANGE = (2, 4)
RECT_H_RANGE = (3, 7)
RECT_W_RANGE = (3, 7)
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
PLACEMENT_TRIES = 200
PATTERN_TRIES = 80
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == MAROON):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(sub[0, :] == MAROON):
            continue
        if not np.all(sub[-1, :] == MAROON):
            continue
        if not np.all(sub[:, 0] == MAROON):
            continue
        if not np.all(sub[:, -1] == MAROON):
            continue
        if np.any(sub[1:-1, 1:-1] == MAROON):
            continue
        candidates.append((r0, c0, r1, c1))
    assert candidates, "legend frame not found"
    return min(candidates, key=lambda box: (box[0] + box[1], box[0], box[1]))


def _read_legend(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[int, int]:
    r0, c0, r1, c1 = bbox
    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    mapping: dict[int, int] = {}
    for row in inside:
        nz = np.flatnonzero(row != BG)
        if nz.size == 0:
            continue
        start = int(nz[0])
        color = int(row[start])
        assert color not in (BG, GREY, MAROON), "invalid legend key color"
        grey_count = 0
        idx = start + 1
        while idx < len(row) and int(row[idx]) == GREY:
            grey_count += 1
            idx += 1
        assert grey_count > 0, "legend rows must encode a positive shift"
        assert color not in mapping, "legend key colors must be unique"
        mapping[color] = grey_count
    assert len(mapping) >= 2, "legend must define at least two colors"
    return mapping


def _shift_rectangle(rect: np.ndarray, strip: np.ndarray, mapping: dict[int, int], orientation: str) -> np.ndarray:
    out = rect.copy()
    if orientation == "H":
        for c, color in enumerate(strip):
            shift = mapping[int(color)] % rect.shape[0]
            out[:, c] = np.roll(rect[:, c], shift)
    else:
        for r, color in enumerate(strip):
            shift = mapping[int(color)] % rect.shape[1]
            out[r, :] = np.roll(rect[r, :], shift)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each password strip from the legend and cyclically shift the adjacent rectangle.

    Input structure:
    - The background is black.
    - A small legend sits near the upper-left corner inside a maroon rectangular frame.
    - Each non-empty legend row contains one unique key color followed immediately by one or more grey cells; the grey-run length is that color's offset value.
    - Elsewhere are several fully filled rectangular motif areas made of non-black, non-grey, non-maroon colors.
    - Each motif rectangle has a one-cell-thick password strip touching either its top edge or its left edge, and every strip cell uses one of the legend's key colors.

    Rule:
    - Read the legend to build a color-to-offset mapping by counting the consecutive grey cells to the right of each key color.
    - For a rectangle with a horizontal strip above it, shift each column downward by the offset encoded by the strip cell above that column.
    - For a rectangle with a vertical strip on its left, shift each row rightward by the offset encoded by the strip cell beside that row.
    - All shifts are cyclic within the rectangle, the legend stays unchanged, and the password strips are removed to black in the output.

    Key insight:
    - The task has two linked decoding steps: first infer numeric offsets from the framed grey-count legend, then apply those decoded values as per-column or per-row cyclic shifts for the adjacent rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    legend_bbox = _find_legend_bbox(g)
    mapping = _read_legend(g, legend_bbox)
    decoder_colors = set(mapping)

    legend_mask = np.zeros_like(g, dtype=bool)
    lr0, lc0, lr1, lc1 = legend_bbox
    legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    fg_mask = (g != BG) & (g != GREY) & (g != MAROON) & (~legend_mask)
    out = g.copy()

    for comp in _components(fg_mask):
        r0, c0, r1, c1 = _bbox(comp)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        comp_mask = np.zeros_like(sub, dtype=bool)
        for rr, cc in comp:
            comp_mask[rr - r0, cc - c0] = True
        assert comp_mask.all(), "each strip+rectangle object must occupy a full rectangle"

        top_is_strip = (
            sub.shape[0] >= 2
            and np.all(np.isin(sub[0, :], list(decoder_colors)))
            and not np.any(np.isin(sub[1:, :], list(decoder_colors)))
        )
        left_is_strip = (
            sub.shape[1] >= 2
            and np.all(np.isin(sub[:, 0], list(decoder_colors)))
            and not np.any(np.isin(sub[:, 1:], list(decoder_colors)))
        )

        assert top_is_strip ^ left_is_strip, "object orientation must be unambiguous"

        if top_is_strip:
            strip = sub[0, :]
            rect = sub[1:, :]
            shifted = _shift_rectangle(rect, strip, mapping, "H")
            out[r0, c0 : c1 + 1] = BG
            out[r0 + 1 : r1 + 1, c0 : c1 + 1] = shifted
        else:
            strip = sub[:, 0]
            rect = sub[:, 1:]
            shifted = _shift_rectangle(rect, strip, mapping, "V")
            out[r0 : r1 + 1, c0] = BG
            out[r0 : r1 + 1, c0 + 1 : c1 + 1] = shifted

    return out


def _build_legend(decoder_colors: list[int], offsets: dict[int, int]) -> np.ndarray:
    inside_h = len(decoder_colors)
    inside_w = 1 + max(offsets.values())
    legend = np.full((inside_h + 2, inside_w + 2), BG, dtype=int)
    legend[0, :] = MAROON
    legend[-1, :] = MAROON
    legend[:, 0] = MAROON
    legend[:, -1] = MAROON
    for r, color in enumerate(decoder_colors, start=1):
        legend[r, 1] = int(color)
        legend[r, 2 : 2 + offsets[int(color)]] = GREY
    return legend


def _styled_pattern(
    rng: random.Random,
    h: int,
    w: int,
    palette: list[int],
) -> np.ndarray:
    k = 2 if len(palette) == 2 else rng.choice((2, 3))
    colors = rng.sample(palette, k)
    style = rng.choice(("diag", "skew_r", "skew_c", "blocks"))
    a, b = rng.choice(((1, 1), (1, 2), (2, 1)))
    br = rng.choice((1, 2))
    bc = rng.choice((1, 2))
    phase = rng.randrange(k)

    arr = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            if style == "diag":
                idx = (r + c + phase) % k
            elif style == "skew_r":
                idx = (a * r + b * c + phase) % k
            elif style == "skew_c":
                idx = (a * (r // br) + b * c + phase) % k
            else:
                idx = (a * (r // br) + b * (c // bc) + phase) % k
            arr[r, c] = colors[idx]

    if h >= 3 and rng.random() < 0.7:
        arr[rng.randrange(h), :] = colors[(phase + 1) % k]
    if w >= 3 and rng.random() < 0.7:
        arr[:, rng.randrange(w)] = colors[(phase + 2) % k]
    return arr


def _make_object_pattern(
    rng: random.Random,
    h: int,
    w: int,
    palette: list[int],
    orientation: str,
    strip: np.ndarray,
    mapping: dict[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATTERN_TRIES):
        pattern = _styled_pattern(rng, h, w, palette)
        shifted = _shift_rectangle(pattern, strip, mapping, orientation)
        if not np.array_equal(pattern, shifted):
            return pattern, shifted
    return None


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + h + 1)
    cc1 = min(blocked.shape[1], c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    decoder_count = rng.choice(DECODER_COUNT_CHOICES)
    decoder_colors = rng.sample(list(NON_SPECIAL_COLORS), decoder_count)
    pattern_pool = [c for c in NON_SPECIAL_COLORS if c not in decoder_colors]
    if len(pattern_pool) < 2:
        return None

    offsets = {int(color): rng.randint(*OFFSET_RANGE) for color in decoder_colors}
    legend = _build_legend(decoder_colors, offsets)
    legend_h, legend_w = map(int, legend.shape)

    grid_h = rng.randint(max(GRID_H_RANGE[0], legend_h + 6), GRID_H_RANGE[1])
    grid_w = rng.randint(max(GRID_W_RANGE[0], legend_w + 6), GRID_W_RANGE[1])
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    grid[:legend_h, :legend_w] = legend
    expected[:legend_h, :legend_w] = legend
    _mark_blocked(blocked, 0, 0, legend_h, legend_w)

    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    orientations = ["H", "V"]
    while len(orientations) < object_count:
        orientations.append(rng.choice(("H", "V")))
    rng.shuffle(orientations)

    specs: list[dict[str, object]] = []
    for idx, orientation in enumerate(orientations):
        rect_h = rng.randint(*RECT_H_RANGE)
        rect_w = rng.randint(*RECT_W_RANGE)
        strip_len = rect_w if orientation == "H" else rect_h
        strip_list = [rng.choice(decoder_colors) for _ in range(strip_len)]
        if strip_len >= 2 and len(set(strip_list)) == 1 and rng.random() < 0.7:
            swap_idx = rng.randrange(strip_len)
            strip_list[swap_idx] = rng.choice([c for c in decoder_colors if c != strip_list[swap_idx]])
        strip = np.array(strip_list, dtype=int)

        base = pattern_pool[idx % len(pattern_pool)]
        others = [c for c in pattern_pool if c != base]
        need = 1 if not others else (1 if rng.random() < 0.35 else min(2, len(others)))
        palette = [base] + rng.sample(others, need)
        made = _make_object_pattern(rng, rect_h, rect_w, palette, orientation, strip, offsets)
        if made is None:
            return None
        pattern, shifted = made
        total_h = rect_h + 1 if orientation == "H" else rect_h
        total_w = rect_w if orientation == "H" else rect_w + 1
        specs.append(
            {
                "orientation": orientation,
                "rect_h": rect_h,
                "rect_w": rect_w,
                "total_h": total_h,
                "total_w": total_w,
                "strip": strip,
                "pattern": pattern,
                "shifted": shifted,
            }
        )

    placements: dict[int, tuple[int, int]] = {}
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["total_h"]) * int(specs[i]["total_w"]), reverse=True)
    for idx in order:
        spec = specs[idx]
        total_h = int(spec["total_h"])
        total_w = int(spec["total_w"])
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(0, grid_h - total_h)
            c0 = rng.randint(0, grid_w - total_w)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + total_h + 1)
            cc1 = min(grid_w, c0 + total_w + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            placements[idx] = (r0, c0)
            _mark_blocked(blocked, r0, c0, total_h, total_w)
            placed = True
            break
        if not placed:
            return None

    for idx, spec in enumerate(specs):
        r0, c0 = placements[idx]
        rect_h = int(spec["rect_h"])
        rect_w = int(spec["rect_w"])
        strip = np.asarray(spec["strip"], dtype=int)
        pattern = np.asarray(spec["pattern"], dtype=int)
        shifted = np.asarray(spec["shifted"], dtype=int)

        if spec["orientation"] == "H":
            grid[r0, c0 : c0 + rect_w] = strip
            grid[r0 + 1 : r0 + 1 + rect_h, c0 : c0 + rect_w] = pattern
            expected[r0 + 1 : r0 + 1 + rect_h, c0 : c0 + rect_w] = shifted
        else:
            grid[r0 : r0 + rect_h, c0] = strip
            grid[r0 : r0 + rect_h, c0 + 1 : c0 + 1 + rect_w] = pattern
            expected[r0 : r0 + rect_h, c0 + 1 : c0 + 1 + rect_w] = shifted

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
