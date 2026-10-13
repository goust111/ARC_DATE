from __future__ import annotations

from collections import Counter
import random

import numpy as np


# --------------- small self-contained ARC helpers ---------------


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _majority_color(block: np.ndarray) -> int:
    counts = Counter(map(int, block.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a non-empty block"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique majority color"
    return int(ranked[0][0])


def _solid_component_records(grid: np.ndarray, bg: int) -> list[dict]:
    records: list[dict] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)
        assert np.all(patch != bg), "expected every non-background component to be a solid rectangle"
        records.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": patch.shape,
                "patch": patch,
            }
        )
    return records


def _is_regular_lattice(records: list[dict], shape: tuple[int, int]) -> bool:
    if not records:
        return False
    bh, bw = map(int, shape)
    starts = [(int(rec["bbox"][0]), int(rec["bbox"][1])) for rec in records]
    row_starts = sorted({r for r, _ in starts})
    col_starts = sorted({c for _, c in starts})
    if len(row_starts) < 2 or len(col_starts) < 2:
        return False
    if len(row_starts) * len(col_starts) != len(records):
        return False
    if set(starts) != {(r, c) for r in row_starts for c in col_starts}:
        return False
    if len(row_starts) > 1:
        row_steps = {row_starts[i + 1] - row_starts[i] for i in range(len(row_starts) - 1)}
        if len(row_steps) != 1 or next(iter(row_steps)) <= bh:
            return False
    if len(col_starts) > 1:
        col_steps = {col_starts[i + 1] - col_starts[i] for i in range(len(col_starts) - 1)}
        if len(col_steps) != 1 or next(iter(col_steps)) <= bw:
            return False
    return True


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(Counter(map(int, g.reshape(-1))).most_common(1)[0][0])
    records = _solid_component_records(g, bg)
    assert records, "expected non-background content"

    shape_counts = Counter(tuple(map(int, rec["shape"])) for rec in records)
    candidate_shapes = sorted(
        shape_counts.items(),
        key=lambda item: (item[1], item[0][0] * item[0][1]),
        reverse=True,
    )

    block_shape = None
    block_records: list[dict] = []
    for shape, count in candidate_shapes:
        if count < 4:
            continue
        same_shape = [rec for rec in records if tuple(map(int, rec["shape"])) == shape]
        if _is_regular_lattice(same_shape, shape):
            block_shape = tuple(map(int, shape))
            block_records = same_shape
            break
    assert block_shape is not None, "failed to identify the block lattice"

    row_starts = sorted({int(rec["bbox"][0]) for rec in block_records})
    col_starts = sorted({int(rec["bbox"][1]) for rec in block_records})
    rows, cols = len(row_starts), len(col_starts)
    guide_shape = (rows, cols)

    guide_candidates = [
        rec
        for rec in records
        if rec not in block_records and tuple(map(int, rec["shape"])) == guide_shape
    ]
    assert len(guide_candidates) == 1, "expected exactly one guide-map rectangle"
    guide = np.array(guide_candidates[0]["patch"], copy=True)
    guide_colors = {int(x) for x in guide.reshape(-1)}

    ruined_candidates = []
    for rec in block_records:
        values = {int(x) for x in rec["patch"].reshape(-1)}
        if len(values) == 1 and next(iter(values)) not in guide_colors:
            ruined_candidates.append(rec)
    assert len(ruined_candidates) == 1, "expected exactly one ruined block"
    ruined = ruined_candidates[0]

    ruined_r = row_starts.index(int(ruined["bbox"][0]))
    ruined_c = col_starts.index(int(ruined["bbox"][1]))
    target_majority = int(guide[ruined_r, ruined_c])

    donor_records = [
        rec
        for rec in block_records
        if rec is not ruined and _majority_color(rec["patch"]) == target_majority
    ]
    assert donor_records, "expected at least one intact donor block of the guided category"
    donor_patch = np.array(donor_records[0]["patch"], copy=True)
    for rec in donor_records[1:]:
        assert np.array_equal(rec["patch"], donor_patch), "same-majority donors must share one texture"

    return {
        "bg": bg,
        "rows": rows,
        "cols": cols,
        "block_shape": block_shape,
        "guide": guide,
        "guide_colors": guide_colors,
        "block_records": block_records,
        "ruined": ruined,
        "ruined_index": (ruined_r, ruined_c),
        "target_majority": target_majority,
        "donor_patch": donor_patch,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the one noise-covered lattice block by using the guide map for its category and a same-category intact block for its full texture.

    Input structure:
    - The grid contains a lattice of equal-sized non-background blocks separated by background gaps.
    - Every intact block has a unique majority color, and blocks with the same majority color share the exact same full texture.
    - A separate small guide rectangle has one cell per lattice position; its cell color equals the corresponding block's majority color.
    - Exactly one lattice block is completely overwritten by a uniform noise color that does not appear in the guide.

    Rule:
    - Identify the regular lattice of same-sized blocks and the separate guide rectangle.
    - Locate the ruined block inside the lattice and use the same row/column position in the guide to read its intended majority-color category.
    - Find another intact lattice block with that majority color and copy its entire internal texture into the ruined block.
    - Leave every other cell, including the guide map and all intact blocks, unchanged.

    Key insight:
    - The guide map only tells which category belongs at the ruined position; the visible texture must be recovered by a second hop to an intact sibling block of the same category.
    """
    info = _parse_scene(np.asarray(grid, dtype=int))
    out = np.array(grid, copy=True)
    r0, c0, r1, c1 = info["ruined"]["bbox"]
    out[r0 : r1 + 1, c0 : c1 + 1] = info["donor_patch"]
    return out


# --------------- generator ---------------

_COLORS = tuple(range(10))
_MIN_BLOCK_ROWS = 2
_MAX_BLOCK_ROWS = 4
_MIN_BLOCK_COLS = 2
_MAX_BLOCK_COLS = 4
_MIN_BLOCK_H = 3
_MAX_BLOCK_H = 4
_MIN_BLOCK_W = 3
_MAX_BLOCK_W = 4
_SEP_CHOICES = (1, 2)
_GUIDE_GAP_CHOICES = (1, 2)
_CATEGORY_COUNT_MIN = 3
_CATEGORY_COUNT_MAX = 5
_MAX_ATTEMPTS = 400


def _random_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    prev = -1
    out = []
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_dimensions() -> tuple[int, int, int, int, int, int]:
    while True:
        rows = random.randint(_MIN_BLOCK_ROWS, _MAX_BLOCK_ROWS)
        cols = random.randint(_MIN_BLOCK_COLS, _MAX_BLOCK_COLS)
        bh = random.randint(_MIN_BLOCK_H, _MAX_BLOCK_H)
        bw = random.randint(_MIN_BLOCK_W, _MAX_BLOCK_W)
        if (rows, cols) == (bh, bw):
            continue
        sep = random.choice(_SEP_CHOICES)
        gap = random.choice(_GUIDE_GAP_CHOICES)
        lattice_h = rows * bh + (rows - 1) * sep
        lattice_w = cols * bw + (cols - 1) * sep
        max_stacked_h = lattice_h + rows + gap + 2
        max_stacked_w = lattice_w + cols + gap + 2
        if max(max_stacked_h, max_stacked_w) <= 30:
            return rows, cols, bh, bw, sep, gap


def _make_texture(block_h: int, block_w: int, majority: int, usable_colors: list[int]) -> np.ndarray:
    area = block_h * block_w
    minority_max = min(4, (area - 1) // 2)
    minority_count = random.randint(2, minority_max)
    positions = random.sample([(r, c) for r in range(block_h) for c in range(block_w)], minority_count)

    accent_pool = [c for c in usable_colors if c != majority]
    accent_count = random.randint(1, min(2, len(accent_pool)))
    accents = random.sample(accent_pool, accent_count)

    patch = np.full((block_h, block_w), majority, dtype=int)
    for idx, (r, c) in enumerate(positions):
        patch[r, c] = accents[idx % len(accents)]
    assert len({int(x) for x in patch.reshape(-1)}) >= 2
    assert _majority_color(patch) == majority
    return patch


def _category_counts(n_blocks: int, n_categories: int) -> list[int]:
    counts = [1] * n_categories
    remaining = n_blocks - n_categories
    repeat_targets = random.sample(range(n_categories), min(2, remaining))
    for idx in repeat_targets:
        counts[idx] += 1
        remaining -= 1
    while remaining > 0:
        counts[random.randrange(n_categories)] += 1
        remaining -= 1
    return counts


def _split_margin(total_extra: int) -> tuple[int, int]:
    left = random.randint(1, total_extra - 1)
    return left, total_extra - left


def _place_layout(
    rows: int,
    cols: int,
    block_h: int,
    block_w: int,
    sep: int,
    gap: int,
    orientation: str,
) -> tuple[int, int, int, int, int, int]:
    lattice_h = rows * block_h + (rows - 1) * sep
    lattice_w = cols * block_w + (cols - 1) * sep
    guide_h, guide_w = rows, cols

    if orientation in ("top", "bottom"):
        occupied_h = guide_h + gap + lattice_h
        occupied_w = max(guide_w, lattice_w)
        total_h = occupied_h + random.randint(2, 30 - occupied_h)
        total_w = occupied_w + random.randint(2, 30 - occupied_w)
        top_margin, bottom_margin = _split_margin(total_h - occupied_h)
        left_margin, right_margin = _split_margin(total_w - occupied_w)
        guide_c = random.randint(left_margin, total_w - right_margin - guide_w)
        lattice_c = random.randint(left_margin, total_w - right_margin - lattice_w)
        if orientation == "top":
            guide_r = top_margin
            lattice_r = guide_r + guide_h + gap
        else:
            lattice_r = top_margin
            guide_r = lattice_r + lattice_h + gap
        return total_h, total_w, guide_r, guide_c, lattice_r, lattice_c

    occupied_h = max(guide_h, lattice_h)
    occupied_w = guide_w + gap + lattice_w
    total_h = occupied_h + random.randint(2, 30 - occupied_h)
    total_w = occupied_w + random.randint(2, 30 - occupied_w)
    top_margin, bottom_margin = _split_margin(total_h - occupied_h)
    left_margin, right_margin = _split_margin(total_w - occupied_w)
    guide_r = random.randint(top_margin, total_h - bottom_margin - guide_h)
    lattice_r = random.randint(top_margin, total_h - bottom_margin - lattice_h)
    if orientation == "left":
        guide_c = left_margin
        lattice_c = guide_c + guide_w + gap
    else:
        lattice_c = left_margin
        guide_c = lattice_c + lattice_w + gap
    return total_h, total_w, guide_r, guide_c, lattice_r, lattice_c


def _semantic_checks(
    grid: np.ndarray,
    expected_output: np.ndarray,
    textures: dict[int, np.ndarray],
    category_grid: np.ndarray,
    ruined_index: tuple[int, int],
) -> bool:
    info = _parse_scene(grid)
    if info["guide"].shape != category_grid.shape:
        return False
    if not np.array_equal(info["guide"], category_grid):
        return False
    if tuple(info["ruined_index"]) != tuple(ruined_index):
        return False
    if info["target_majority"] != int(category_grid[ruined_index]):
        return False
    if not np.array_equal(info["donor_patch"], textures[int(category_grid[ruined_index])]):
        return False
    if not np.array_equal(main(grid), expected_output):
        return False
    if np.array_equal(grid, expected_output):
        return False

    r0, c0, r1, c1 = info["ruined"]["bbox"]
    pure_fill = np.array(grid, copy=True)
    pure_fill[r0 : r1 + 1, c0 : c1 + 1] = info["target_majority"]
    if np.array_equal(pure_fill, expected_output):
        return False

    intact_majorities = {
        _majority_color(rec["patch"])
        for rec in info["block_records"]
        if rec is not info["ruined"]
    }
    if len(intact_majorities) < 2:
        return False

    target_texture = textures[int(category_grid[ruined_index])]
    if len({int(x) for x in target_texture.reshape(-1)}) < 2:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    rows, cols, block_h, block_w, sep, gap = _choose_dimensions()
    n_blocks = rows * cols
    n_categories = random.randint(_CATEGORY_COUNT_MIN, min(_CATEGORY_COUNT_MAX, n_blocks - 1))

    bg = random.choice(_COLORS)
    usable = [c for c in _COLORS if c != bg]
    category_colors = random.sample(usable, n_categories)

    textures: dict[int, np.ndarray] = {}
    for majority in category_colors:
        for _ in range(60):
            patch = _make_texture(block_h, block_w, majority, category_colors)
            if all(not np.array_equal(patch, existing) for existing in textures.values()):
                textures[majority] = patch
                break
        else:
            return None

    counts = _category_counts(n_blocks, n_categories)
    labels: list[int] = []
    for color, count in zip(category_colors, counts):
        labels.extend([color] * count)
    random.shuffle(labels)
    category_grid = np.array(labels, dtype=int).reshape(rows, cols)

    repeatable = [color for color, count in zip(category_colors, counts) if count >= 2]
    ruined_category = random.choice(repeatable)
    ruined_positions = list(zip(*np.where(category_grid == ruined_category)))
    ruined_index = random.choice(ruined_positions)

    remaining_noise = [c for c in usable if c not in set(category_colors)]
    if not remaining_noise:
        return None
    noise_color = random.choice(remaining_noise)

    orientation = random.choice(("top", "bottom", "left", "right"))
    total_h, total_w, guide_r, guide_c, lattice_r, lattice_c = _place_layout(
        rows, cols, block_h, block_w, sep, gap, orientation
    )
    grid = np.full((total_h, total_w), bg, dtype=int)

    expected = np.full_like(grid, bg)
    guide = np.array(category_grid, copy=True)
    expected[guide_r : guide_r + rows, guide_c : guide_c + cols] = guide
    grid[guide_r : guide_r + rows, guide_c : guide_c + cols] = guide

    for br in range(rows):
        for bc in range(cols):
            r0 = lattice_r + br * (block_h + sep)
            c0 = lattice_c + bc * (block_w + sep)
            majority = int(category_grid[br, bc])
            patch = textures[majority]
            expected[r0 : r0 + block_h, c0 : c0 + block_w] = patch
            if (br, bc) == ruined_index:
                grid[r0 : r0 + block_h, c0 : c0 + block_w] = noise_color
            else:
                grid[r0 : r0 + block_h, c0 : c0 + block_w] = patch

    if not _semantic_checks(grid, expected, textures, category_grid, ruined_index):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid guided-repair lattice")
