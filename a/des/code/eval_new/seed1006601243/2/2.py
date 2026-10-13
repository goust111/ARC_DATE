from __future__ import annotations

import random
from collections import defaultdict, deque
from functools import lru_cache
from itertools import combinations

import numpy as np


BG = 0
FOREGROUND_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
MAX_GRID_SIZE = 30
ROW_COUNT_CHOICES = (2, 3)
ROW_GAP_RANGE = (1, 3)
COL_GAP_RANGE = (1, 3)
MARGIN_RANGE = (1, 4)
MAX_OUTER_SIDE_BY_COUNT = {2: 21, 3: 19, 4: 15}


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    assert rows.any() and cols.any()
    r = np.flatnonzero(rows)
    c = np.flatnonzero(cols)
    return np.asarray(mask[r[0] : r[-1] + 1, c[0] : c[-1] + 1], dtype=bool)


def _component_masks(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append({"color": color, "mask": mask, "area": int(mask.sum())})
    return comps


def _mask_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    mask = np.asarray(mask, dtype=bool)
    return int(mask.shape[0]), int(mask.shape[1]), mask.tobytes()


def _mask_from_key(key: tuple[int, int, bytes]) -> np.ndarray:
    h, w, data = key
    return np.frombuffer(data, dtype=bool).reshape((h, w)).copy()


@lru_cache(maxsize=None)
def _transform_keys_from_key(key: tuple[int, int, bytes]) -> tuple[tuple[int, int, bytes], ...]:
    base = _crop_mask(_mask_from_key(key))
    seen: set[tuple[int, int, bytes]] = set()
    out: list[tuple[int, int, bytes]] = []
    for rot in range(4):
        spun = np.rot90(base, rot)
        for flip in (False, True):
            transformed = np.fliplr(spun) if flip else spun
            norm_key = _mask_key(_crop_mask(transformed))
            if norm_key not in seen:
                seen.add(norm_key)
                out.append(norm_key)
    return tuple(out)


def _transform_keys(mask: np.ndarray) -> tuple[tuple[int, int, bytes], ...]:
    return _transform_keys_from_key(_mask_key(mask))


@lru_cache(maxsize=None)
def _assembly_options(
    key_a: tuple[int, int, bytes], key_b: tuple[int, int, bytes]
) -> tuple[tuple[int, int], ...]:
    mask_a = _mask_from_key(key_a)
    mask_b = _mask_from_key(key_b)
    area = int(mask_a.sum() + mask_b.sum())
    b_transforms = set(_transform_keys_from_key(key_b))
    options: set[tuple[int, int]] = set()

    candidate_rects: set[tuple[int, int]] = set()
    for rect_h in range(2, int(np.sqrt(area)) + 1):
        if area % rect_h != 0:
            continue
        rect_w = area // rect_h
        if rect_w >= 2:
            candidate_rects.add((rect_h, rect_w))
            candidate_rects.add((rect_w, rect_h))

    for ta_key in _transform_keys_from_key(key_a):
        ta = _mask_from_key(ta_key)
        h1, w1 = map(int, ta.shape)
        for rect_h, rect_w in candidate_rects:
            if h1 > rect_h or w1 > rect_w:
                continue
            for r in range(rect_h - h1 + 1):
                for c in range(rect_w - w1 + 1):
                    comp = np.ones((rect_h, rect_w), dtype=bool)
                    comp[r : r + h1, c : c + w1][ta] = False
                    comp_key = _mask_key(_crop_mask(comp))
                    if comp_key in b_transforms:
                        options.add((max(rect_h, rect_w), min(rect_h, rect_w)))
    return tuple(sorted(options))


def _assemble_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    options = _assembly_options(_mask_key(mask_a), _mask_key(mask_b))
    assert options, "same-color L pair must assemble into a rectangle"
    assert len(options) == 1, "same-color L pair must determine a unique rectangle size"
    return options[0]


def _render_nested(rectangles: list[dict]) -> np.ndarray:
    items = sorted(rectangles, key=lambda item: item["h"] * item["w"], reverse=True)
    outer_h = int(items[0]["h"])
    outer_w = int(items[0]["w"])
    out = np.full((outer_h, outer_w), BG, dtype=int)

    for i, item in enumerate(items):
        h = int(item["h"])
        w = int(item["w"])
        color = int(item["color"])
        top = (outer_h - h) // 2
        left = (outer_w - w) // 2
        out[top : top + h, left : left + w] = color

        if i + 1 == len(items):
            continue
        next_h = int(items[i + 1]["h"]) + 2
        next_w = int(items[i + 1]["w"]) + 2
        assert next_h <= h and next_w <= w
        gap_top = (outer_h - next_h) // 2
        gap_left = (outer_w - next_w) // 2
        out[gap_top : gap_top + next_h, gap_left : gap_left + next_w] = BG

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair matching-color L fragments, reassemble each pair into a solid rectangle, and nest those rectangles concentrically.

    Input structure:
    - The background is black.
    - Every non-black 4-connected object is a monochrome L shape: a filled rectangle with one corner rectangle removed.
    - Each foreground color appears in exactly two disconnected L fragments.
    - For each color, the two fragments can be rotated and/or mirrored so that together they tile one complete filled rectangle.

    Rule:
    - Group fragments by color and pair the two fragments of each color.
    - For each pair, search over rotations, reflections, and relative offsets until the two L shapes exactly tile one solid rectangle; record that rectangle's size and color.
    - Sort the recovered rectangles by area from largest to smallest and place them concentrically.
    - Leave exactly one black pixel of spacing before the next smaller rectangle, so the visible result is a nested stack of colored rectangular frames with the smallest rectangle remaining solid.

    Key insight:
    - The colors identify which fragments belong together, but the real work is geometric: each pair must first be mentally reassembled into a full rectangle before the rectangle sizes can be ordered and nested.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for comp in _component_masks(g):
        grouped[int(comp["color"])].append(comp["mask"])

    rectangles: list[dict] = []
    for color, masks in grouped.items():
        assert len(masks) == 2, "each color must contribute exactly one pair of fragments"
        rect_h, rect_w = _assemble_pair(masks[0], masks[1])
        rectangles.append({"color": int(color), "h": rect_h, "w": rect_w})

    rectangles.sort(key=lambda item: item["h"] * item["w"], reverse=True)
    for outer, inner in zip(rectangles, rectangles[1:]):
        assert outer["h"] >= inner["h"] + 4 and outer["w"] >= inner["w"] + 4
        assert (outer["h"] - inner["h"]) % 2 == 0 and (outer["w"] - inner["w"]) % 2 == 0

    return _render_nested(rectangles)


def _size_chains(count: int) -> list[tuple[int, ...]]:
    chains: list[tuple[int, ...]] = []
    max_side = MAX_OUTER_SIDE_BY_COUNT[int(count)]
    for parity in (0, 1):
        sides = [side for side in range(3, max_side + 1) if side % 2 == parity]
        for combo in combinations(sides, int(count)):
            if all(combo[i + 1] - combo[i] >= 4 for i in range(len(combo) - 1)):
                chains.append(tuple(reversed(combo)))
    return chains


CHAIN_OPTIONS = {count: _size_chains(count) for count in (2, 3, 4)}


def _make_partition_pair(side: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        band_h = random.randint(1, side - 2)
        band_top = random.randint(1, side - band_h - 1)
        cut = random.randint(1, side - 1)
        first = np.zeros((side, side), dtype=bool)
        first[:band_top, :] = True
        first[band_top : band_top + band_h, cut:] = True
        second = ~first
        if random.random() < 0.5:
            first = first.T
            second = second.T
        if random.random() < 0.5:
            first, second = second, first
        mask_a = _crop_mask(first)
        mask_b = _crop_mask(second)
        if _assembly_options(_mask_key(mask_a), _mask_key(mask_b)) == ((side, side),):
            return mask_a, mask_b
    raise RuntimeError(f"failed to construct a unique L-pair for side {side}")


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.asarray(out, dtype=bool)


def _color_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    seq = [cuts[0]]
    for a, b in zip(cuts, cuts[1:]):
        seq.append(b - a)
    seq.append(total - cuts[-1])
    return seq


def _arrange_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    pieces = patches[:]
    random.shuffle(pieces)
    piece_count = len(pieces)
    row_counts = [rows for rows in ROW_COUNT_CHOICES if rows <= piece_count]

    for _ in range(120):
        row_count = random.choice(row_counts)
        lengths = _random_composition(piece_count, row_count)
        rows: list[list[np.ndarray]] = []
        idx = 0
        for length in lengths:
            rows.append(pieces[idx : idx + length])
            idx += length

        row_heights: list[int] = []
        row_widths: list[int] = []
        row_gaps: list[list[int]] = []
        for row in rows:
            gaps = [random.randint(*COL_GAP_RANGE) for _ in range(max(0, len(row) - 1))]
            width = sum(int(p.shape[1]) for p in row) + sum(gaps)
            height = max(int(p.shape[0]) for p in row)
            row_widths.append(width)
            row_heights.append(height)
            row_gaps.append(gaps)

        vertical_gaps = [random.randint(*ROW_GAP_RANGE) for _ in range(max(0, row_count - 1))]
        core_h = sum(row_heights) + sum(vertical_gaps)
        core_w = max(row_widths)

        top_margin = random.randint(*MARGIN_RANGE)
        bottom_margin = random.randint(*MARGIN_RANGE)
        left_margin = random.randint(*MARGIN_RANGE)
        right_margin = random.randint(*MARGIN_RANGE)
        h = core_h + top_margin + bottom_margin
        w = core_w + left_margin + right_margin
        if h > MAX_GRID_SIZE or w > MAX_GRID_SIZE:
            continue

        grid = np.full((h, w), BG, dtype=int)
        row_top = top_margin
        for row_idx, (row, row_h, row_w, gaps) in enumerate(zip(rows, row_heights, row_widths, row_gaps)):
            slack = core_w - row_w
            row_left = left_margin + random.randint(0, slack)
            col = row_left
            for i, patch in enumerate(row):
                ph, pw = map(int, patch.shape)
                offset_r = row_top + random.randint(0, row_h - ph)
                grid[offset_r : offset_r + ph, col : col + pw][patch != BG] = patch[patch != BG]
                if i < len(gaps):
                    col += pw + gaps[i]
            row_top += row_h
            if row_idx < len(vertical_gaps):
                row_top += vertical_gaps[row_idx]

        return grid

    return None


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.choice((2, 3, 4))
    sides = list(random.choice(CHAIN_OPTIONS[rect_count]))
    colors = random.sample(FOREGROUND_COLORS, rect_count)
    rectangles = [{"color": color, "h": side, "w": side} for color, side in zip(colors, sides)]

    patches: list[np.ndarray] = []
    for rect in rectangles:
        mask_a, mask_b = _make_partition_pair(int(rect["h"]))
        patches.append(_color_patch(_random_transform(mask_a), int(rect["color"])))
        patches.append(_color_patch(_random_transform(mask_b), int(rect["color"])))

    patches.sort(key=lambda patch: int(np.count_nonzero(patch)), reverse=True)
    grid = _arrange_patches(patches)
    if grid is None:
        return None

    expected = _render_nested(rectangles)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
