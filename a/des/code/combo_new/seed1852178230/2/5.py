from __future__ import annotations

from collections import deque
import random

import numpy as np


NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
GEN_RETRIES = 400
REGION_RETRIES = 200
MASK_RETRIES = 200
PLACEMENT_RETRIES = 200
COLOR_POOL = tuple(range(10))
REGION_MIN_MULTI = 8
REGION_MAX_DOUBLE = 13
REGION_MAX_TRIPLE = 10
REGION_MIN_SINGLE = 10
REGION_MAX_SINGLE = 18
MOLD_BBOX_MIN = 3
MOLD_BBOX_MAX = 6
SEED_EXTRA_MAX = 2
LAYOUTS = (
    (2, 1),  # one horizontal separator
    (1, 2),  # one vertical separator
    (3, 1),  # two horizontal separators
    (1, 3),  # two vertical separators
    (2, 2),  # one horizontal + one vertical separator
)


def _most_common_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[dict[str, np.ndarray | tuple[int, int, int, int] | int]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            out.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _separator_lines(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        rows = [r for r in range(h) if bool(np.all(grid[r, :] == color))]
        cols = [c for c in range(w) if bool(np.all(grid[:, c] == color))]
        if rows or cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected exactly one separator color"
    color, rows, cols = candidates[0]
    assert 1 <= len(rows) + len(cols) <= 2, "expected one or two separator lines"
    rows = sorted(int(r) for r in rows)
    cols = sorted(int(c) for c in cols)
    return int(color), rows, cols


def _spans(length: int, cuts: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for cut in sorted(cuts):
        assert start < cut, "separator line must leave a nonempty region"
        spans.append((start, cut - 1))
        start = cut + 1
    assert start < length, "separator line must leave a final nonempty region"
    spans.append((start, length - 1))
    return spans


def _is_solid_rectangle(subgrid: np.ndarray, comp: dict, color: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    area = int(comp["area"])
    return area == (r1 - r0 + 1) * (c1 - c0 + 1) and bool(
        np.all(subgrid[r0 : r1 + 1, c0 : c1 + 1] == color)
    )


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _most_common_color(g)
    _sep_color, sep_rows, sep_cols = _separator_lines(g, bg)
    row_spans = _spans(g.shape[0], sep_rows)
    col_spans = _spans(g.shape[1], sep_cols)

    out = g.copy()
    for rr0, rr1 in row_spans:
        for cc0, cc1 in col_spans:
            sub = g[rr0 : rr1 + 1, cc0 : cc1 + 1]
            comps = _components(sub != bg)
            assert len(comps) == 2, "each region must contain exactly two objects"

            objects = []
            for comp in comps:
                coords = comp["coords"]
                colors = np.unique(sub[coords[:, 0], coords[:, 1]])
                assert colors.size == 1, "each object must be monochrome"
                color = int(colors[0])
                objects.append(
                    {
                        **comp,
                        "color": color,
                        "is_rect": _is_solid_rectangle(sub, comp, color),
                    }
                )

            seeds = [obj for obj in objects if obj["is_rect"]]
            molds = [obj for obj in objects if not obj["is_rect"]]
            assert len(seeds) == 1 and len(molds) == 1, "expected one seed rectangle and one mold object"
            seed = seeds[0]
            mold = molds[0]

            sr0, sc0, sr1, sc1 = seed["bbox"]
            mr0, mc0, mr1, mc1 = mold["bbox"]
            seed_h = sr1 - sr0 + 1
            seed_w = sc1 - sc0 + 1
            mold_h = mr1 - mr0 + 1
            mold_w = mc1 - mc0 + 1
            assert seed_h >= mold_h and seed_w >= mold_w, "seed rectangle must cover the mold bbox"

            for obj in (seed, mold):
                for r, c in obj["coords"]:
                    out[rr0 + int(r), cc0 + int(c)] = bg

            patch = np.full((mold_h, mold_w), bg, dtype=int)
            local = mold["coords"] - np.array([[mr0, mc0]])
            patch[local[:, 0], local[:, 1]] = int(seed["color"])
            out[rr0 + mr0 : rr0 + mr1 + 1, cc0 + mc0 : cc0 + mc1 + 1] = patch

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each irregular template inside a separator-defined region using the local solid rectangle's color.

    Input structure:
    - The background is the dominant color.
    - One or two full-height/full-width separator lines of a single non-background color split the grid into 2 to 4 regions.
    - Each region contains exactly two 4-connected monochrome objects: one solid rectangle and one non-rectangular shape.
    - The solid rectangle's bbox is at least as tall and wide as the irregular shape's bbox.

    Rule:
    - Treat the solid rectangle as the seed object and the irregular shape as the mold object.
    - Align the mold bbox's top-left corner to the seed bbox's top-left corner and keep only those seed pixels covered by the mold shape.
    - Erase the original seed and mold objects.
    - Place the cropped result back at the mold object's original bbox, preserving the mold shape but using the seed object's color.
    - Leave the separator lines unchanged.

    Key insight:
    - The separator lines define independent workspaces, and inside each workspace the rectangle supplies color while the irregular object supplies shape.
    """
    return _solve(grid)


def _random_band_sizes(count: int) -> list[int]:
    if count == 1:
        return [random.randint(REGION_MIN_SINGLE, REGION_MAX_SINGLE)]

    total_max = 30 - (count - 1)
    max_size = REGION_MAX_DOUBLE if count == 2 else REGION_MAX_TRIPLE
    for _ in range(200):
        total = random.randint(count * REGION_MIN_MULTI, min(total_max, count * max_size))
        sizes = [REGION_MIN_MULTI] * count
        remaining = total - count * REGION_MIN_MULTI
        caps = [max_size - REGION_MIN_MULTI] * count
        while remaining > 0:
            choices = [i for i, cap in enumerate(caps) if cap > 0]
            idx = random.choice(choices)
            add = random.randint(1, min(caps[idx], remaining))
            sizes[idx] += add
            caps[idx] -= add
            remaining -= add
        random.shuffle(sizes)
        return sizes
    raise RuntimeError("failed to sample band sizes")


def _manhattan_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    rr, cc = end
    path = [(r, c)]
    while (r, c) != (rr, cc):
        choices = []
        if r != rr:
            choices.append((r + (1 if rr > r else -1), c))
        if c != cc:
            choices.append((r, c + (1 if cc > c else -1)))
        r, c = random.choice(choices)
        path.append((r, c))
    return path


def _random_mold_mask(height: int, width: int) -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        anchors = [
            (0, random.randrange(width)),
            (height - 1, random.randrange(width)),
            (random.randrange(height), 0),
            (random.randrange(height), width - 1),
        ]
        random.shuffle(anchors)
        cells = {anchors[0]}
        for start, end in zip(anchors, anchors[1:]):
            cells.update(_manhattan_path(start, end))

        max_area = height * width - 1
        if len(cells) > max_area:
            continue

        target_hi = min(max_area, max(len(cells), int(np.ceil(0.75 * height * width))))
        target_area = random.randint(len(cells), target_hi)
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if bool(mask.all()):
            continue
        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    return None


def _boxes_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _random_positions(
    region_h: int, region_w: int, seed_h: int, seed_w: int, mold_h: int, mold_w: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    seed_rows = list(range(1, region_h - seed_h))
    seed_cols = list(range(1, region_w - seed_w))
    mold_rows = list(range(1, region_h - mold_h))
    mold_cols = list(range(1, region_w - mold_w))
    if not seed_rows or not seed_cols or not mold_rows or not mold_cols:
        return None

    for _ in range(PLACEMENT_RETRIES):
        sr = random.choice(seed_rows)
        sc = random.choice(seed_cols)
        mr = random.choice(mold_rows)
        mc = random.choice(mold_cols)
        seed_box = (sr, sc, sr + seed_h - 1, sc + seed_w - 1)
        mold_box = (mr, mc, mr + mold_h - 1, mc + mold_w - 1)
        if _boxes_touch_or_overlap(seed_box, mold_box, pad=1):
            continue
        return (sr, sc), (mr, mc)
    return None


def _generate_region(region_h: int, region_w: int, bg: int, seed_color: int, mold_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    max_mold_h = min(MOLD_BBOX_MAX, region_h - 4)
    max_mold_w = min(MOLD_BBOX_MAX, region_w - 4)
    if max_mold_h < MOLD_BBOX_MIN or max_mold_w < MOLD_BBOX_MIN:
        return None

    for _ in range(REGION_RETRIES):
        mold_h = random.randint(MOLD_BBOX_MIN, max_mold_h)
        mold_w = random.randint(MOLD_BBOX_MIN, max_mold_w)
        mold_mask = _random_mold_mask(mold_h, mold_w)
        if mold_mask is None:
            continue

        seed_h = random.randint(mold_h, min(region_h - 2, mold_h + SEED_EXTRA_MAX))
        seed_w = random.randint(mold_w, min(region_w - 2, mold_w + SEED_EXTRA_MAX))
        positions = _random_positions(region_h, region_w, seed_h, seed_w, mold_h, mold_w)
        if positions is None:
            continue
        (sr, sc), (mr, mc) = positions

        inp = np.full((region_h, region_w), bg, dtype=int)
        inp[sr : sr + seed_h, sc : sc + seed_w] = seed_color
        mold_patch = inp[mr : mr + mold_h, mc : mc + mold_w]
        mold_patch[mold_mask] = mold_color

        out = np.full((region_h, region_w), bg, dtype=int)
        out_patch = out[mr : mr + mold_h, mc : mc + mold_w]
        out_patch[mold_mask] = seed_color
        return inp, out
    return None


def _assemble_layout(row_sizes: list[int], col_sizes: list[int], bg: int, separator: int) -> tuple[np.ndarray, np.ndarray]:
    height = sum(row_sizes) + len(row_sizes) - 1
    width = sum(col_sizes) + len(col_sizes) - 1
    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    row_spans = []
    cursor = 0
    for idx, size in enumerate(row_sizes):
        row_spans.append((cursor, cursor + size - 1))
        cursor += size
        if idx < len(row_sizes) - 1:
            inp[cursor, :] = separator
            out[cursor, :] = separator
            cursor += 1

    col_spans = []
    cursor = 0
    for idx, size in enumerate(col_sizes):
        col_spans.append((cursor, cursor + size - 1))
        cursor += size
        if idx < len(col_sizes) - 1:
            inp[:, cursor] = separator
            out[:, cursor] = separator
            cursor += 1

    obj_colors = [color for color in COLOR_POOL if color not in {bg, separator}]
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            seed_color, mold_color = random.sample(obj_colors, 2)
            region = _generate_region(r1 - r0 + 1, c1 - c0 + 1, bg, seed_color, mold_color)
            if region is None:
                return None, None
            region_inp, region_out = region
            inp[r0 : r1 + 1, c0 : c1 + 1] = region_inp
            out[r0 : r1 + 1, c0 : c1 + 1] = region_out

    return inp, out


def _generate_attempt() -> np.ndarray | None:
    row_count, col_count = random.choice(LAYOUTS)
    bg, separator = random.sample(COLOR_POOL, 2)
    row_sizes = _random_band_sizes(row_count)
    col_sizes = _random_band_sizes(col_count)
    inp, expected = _assemble_layout(row_sizes, col_sizes, bg, separator)
    if inp is None or expected is None:
        return None

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    if _most_common_color(inp) != bg:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
