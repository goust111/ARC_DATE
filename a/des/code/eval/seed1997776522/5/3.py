from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
GEN_COLORS = (1, 2, 3, 4, 6, 7, 9)
PATTERN_FAMILIES = ("affine", "bands", "radial")
GEN_RETRIES = 400
MAX_INPUT_WIDTH = 40


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0) + 1
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes = []
    for comp in _components(mask):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        boxes.append((min(rows), min(cols), max(rows) + 1, max(cols) + 1))
    return boxes


def _downsample2(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h % 2 == 0 and w % 2 == 0, "downsample expects even dimensions"
    out = np.empty((h // 2, w // 2), dtype=int)

    for r in range(0, h, 2):
        for c in range(0, w, 2):
            block = [int(v) for v in g[r : r + 2, c : c + 2].ravel()]
            counts: dict[int, int] = {}
            for value in block:
                counts[value] = counts.get(value, 0) + 1
            top_left = block[0]
            max_count = max(counts.values())
            winners = [color for color, count in counts.items() if count == max_count]
            if len(winners) == 1:
                out[r // 2, c // 2] = winners[0]
            else:
                assert top_left in winners, "tied block must be resolved by its top-left color"
                out[r // 2, c // 2] = top_left
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the azure hole in the left panel by matching candidate patches against a 2x downsampled clue.

    Input structure
    - The background is black.
    - One full-height grey column separates a left panel and a right panel.
    - The left panel is a multicolor image with one even-sized rectangular azure damage block.
    - The right panel contains a complete 2x-downsampled copy of the left image and 3-5 full-resolution candidate patches.

    Rule
    - Locate the azure rectangle in the left panel.
    - Map that rectangle into the downsampled copy by dividing its row and column coordinates by 2.
    - Downsample each candidate patch with the same 2x2 majority rule, breaking ties by the top-left cell.
    - Exactly one candidate matches the clue region. Fill the azure rectangle with that patch.

    Output
    - Return only the repaired left panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    sep_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert len(sep_cols) == 1, "expected one grey separator column"
    sep = sep_cols[0]

    left = g[:, :sep].copy()
    right = g[:, sep + 1 :]
    left_h, left_w = left.shape
    assert left_h == h and right.shape[0] == h, "panels must share the same height"
    assert left_h % 2 == 0 and left_w % 2 == 0, "left panel must downsample by a factor of 2"

    damage_mask = left == AZURE
    assert damage_mask.any(), "expected an azure damage block"
    dr0, dc0, dr1, dc1 = _bbox(damage_mask)
    damage_h, damage_w = dr1 - dr0, dc1 - dc0
    assert damage_h % 2 == 0 and damage_w % 2 == 0, "damage size must be even"
    assert dr0 % 2 == 0 and dc0 % 2 == 0, "damage must align to the 2x2 downsampling grid"
    assert int(damage_mask.sum()) == damage_h * damage_w, "damage must be one solid rectangle"

    mask = right != BLACK
    boxes = _component_bboxes(mask)
    assert 4 <= len(boxes) <= 6, "expected one clue and 3-5 candidate rectangles"
    for r0, c0, r1, c1 in boxes:
        assert np.all(mask[r0:r1, c0:c1]), "right-panel objects must be axis-aligned rectangles"

    clue_shape = (left_h // 2, left_w // 2)
    clue_boxes = [box for box in boxes if (box[2] - box[0], box[3] - box[1]) == clue_shape]
    assert len(clue_boxes) == 1, "expected one downsampled complete copy"
    clue_box = clue_boxes[0]
    cr0, cc0, cr1, cc1 = clue_box
    assert cr0 == 0 and cr1 <= left_h // 2, "the clue copy must sit in the upper half"
    clue = right[cr0:cr1, cc0:cc1]

    candidate_boxes = [box for box in boxes if box != clue_box]
    assert 3 <= len(candidate_boxes) <= 5, "expected 3-5 candidate patches"
    for r0, c0, r1, c1 in candidate_boxes:
        assert (r1 - r0, c1 - c0) == (damage_h, damage_w), "candidate patches must match the damage size"
        assert r0 >= left_h // 2, "candidate patches must sit in the lower half"

    clue_patch = clue[dr0 // 2 : dr1 // 2, dc0 // 2 : dc1 // 2]
    matches = []
    for r0, c0, r1, c1 in candidate_boxes:
        patch = right[r0:r1, c0:c1]
        if np.array_equal(_downsample2(patch), clue_patch):
            matches.append(patch.copy())
    assert len(matches) == 1, "expected exactly one candidate patch to match the clue"

    left[dr0:dr1, dc0:dc1] = matches[0]
    return left


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _random_span(length: int, min_len: int = 1, max_len: int | None = None) -> tuple[int, int]:
    if max_len is None:
        max_len = length
    max_len = min(max_len, length)
    span = random.randint(min_len, max_len)
    start = random.randint(0, length - span)
    return start, start + span


def _partition(length: int, parts: int) -> list[tuple[int, int]]:
    cuts = sorted(random.sample(range(1, length), parts - 1))
    bounds = [0, *cuts, length]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _decorate_lowres(grid: np.ndarray, palette: list[int]) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape

    for _ in range(random.randint(1, 2)):
        color = random.choice(palette)
        if random.random() < 0.5:
            r0, r1 = _random_span(h, 1, max(1, h - 1))
            out[r0:r1, :] = color
        else:
            c0, c1 = _random_span(w, 1, max(1, w - 1))
            out[:, c0:c1] = color

    for _ in range(random.randint(1, 2)):
        r0, r1 = _random_span(h, 1, max(1, h - 1))
        c0, c1 = _random_span(w, 1, max(1, w - 1))
        out[r0:r1, c0:c1] = random.choice(palette)
    return out


def _make_affine_pattern(h: int, w: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    a, b = random.choice(((1, 1), (1, 2), (2, 1), (2, 3), (3, 1), (3, 2)))
    offset = random.randrange(k)
    base = np.array(
        [[palette[(a * r + b * c + offset) % k] for c in range(w)] for r in range(h)],
        dtype=int,
    )
    return _decorate_lowres(base, palette)


def _make_band_pattern(h: int, w: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    row_parts = random.randint(2, min(4, h))
    col_parts = random.randint(2, min(4, w))
    rows = _partition(h, row_parts)
    cols = _partition(w, col_parts)
    out = np.empty((h, w), dtype=int)
    offset = random.randrange(k)
    for ri, (r0, r1) in enumerate(rows):
        for ci, (c0, c1) in enumerate(cols):
            out[r0:r1, c0:c1] = palette[(2 * ri + ci + offset) % k]
    return _decorate_lowres(out, palette)


def _make_radial_pattern(h: int, w: int, palette: list[int]) -> np.ndarray:
    k = len(palette)
    cr = random.randrange(h)
    cc = random.randrange(w)
    offset = random.randrange(k)
    base = np.array(
        [
            [palette[(abs(r - cr) + 2 * abs(c - cc) + offset) % k] for c in range(w)]
            for r in range(h)
        ],
        dtype=int,
    )
    return _decorate_lowres(base, palette)


def _make_lowres_pattern(h: int, w: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(80):
        family = random.choice(PATTERN_FAMILIES)
        if family == "affine":
            low = _make_affine_pattern(h, w, palette)
        elif family == "bands":
            low = _make_band_pattern(h, w, palette)
        else:
            low = _make_radial_pattern(h, w, palette)

        used = {int(v) for v in np.unique(low)}
        if len(used) < 3:
            continue
        if not all(color in used for color in palette[:3]):
            continue
        return low
    return None


def _micro_block(color: int, palette: list[int]) -> np.ndarray:
    others = [value for value in palette if value != color]
    alt1 = random.choice(others)
    alt2 = random.choice(others)
    choices = [
        np.array([[color, color], [color, color]], dtype=int),
        np.array([[color, color], [color, alt1]], dtype=int),
        np.array([[color, alt1], [alt1, color]], dtype=int),
        np.array([[color, alt1], [color, alt1]], dtype=int),
        np.array([[color, alt1], [alt2, color]], dtype=int),
        np.array([[color, color], [alt1, color]], dtype=int),
    ]
    return random.choice(choices)


def _expand_lowres(low: np.ndarray, palette: list[int]) -> np.ndarray:
    h, w = low.shape
    out = np.empty((2 * h, 2 * w), dtype=int)
    for r in range(h):
        for c in range(w):
            _paste(out, _micro_block(int(low[r, c]), palette), 2 * r, 2 * c)
    return out


def _choose_damage_box(low_h: int, low_w: int) -> tuple[int, int, int, int] | None:
    max_h = min(3, (low_h - 1) // 2)
    max_w = min(3, low_w - 2)
    if max_h < 1 or max_w < 1:
        return None

    options = []
    for patch_h in range(1, max_h + 1):
        for patch_w in range(1, max_w + 1):
            if patch_h * patch_w < 2:
                continue
            for r0 in range(low_h - patch_h + 1):
                for c0 in range(low_w - patch_w + 1):
                    options.append((r0, c0, r0 + patch_h, c0 + patch_w))
    if not options:
        return None
    return random.choice(options)


def _mutate_lowres_patch(clue: np.ndarray, palette: list[int]) -> np.ndarray:
    out = clue.copy()
    cells = [(r, c) for r in range(out.shape[0]) for c in range(out.shape[1])]
    random.shuffle(cells)
    changes = random.randint(1, max(1, len(cells) // 2))
    for r, c in cells[:changes]:
        choices = [color for color in palette if color != int(out[r, c])]
        out[r, c] = random.choice(choices)
    return out


def _build_candidates(correct_patch: np.ndarray, clue_low: np.ndarray, palette: list[int], count: int) -> list[np.ndarray] | None:
    candidates = [correct_patch.copy()]
    seen_low = {tuple(int(v) for v in clue_low.ravel())}

    while len(candidates) < count:
        decoy_low = _mutate_lowres_patch(clue_low, palette)
        low_sig = tuple(int(v) for v in decoy_low.ravel())
        if low_sig in seen_low:
            continue
        seen_low.add(low_sig)
        candidates.append(_expand_lowres(decoy_low, palette))

    random.shuffle(candidates)
    return candidates


def _generate_attempt() -> np.ndarray | None:
    low_h = random.randint(5, 8)
    low_w = random.randint(5, 8)
    palette = random.sample(GEN_COLORS, random.choice((3, 4)))
    low = _make_lowres_pattern(low_h, low_w, palette)
    if low is None:
        return None

    full = _expand_lowres(low, palette)
    assert np.array_equal(_downsample2(full), low)

    damage_box_low = _choose_damage_box(low_h, low_w)
    if damage_box_low is None:
        return None
    lr0, lc0, lr1, lc1 = damage_box_low
    fr0, fc0, fr1, fc1 = 2 * lr0, 2 * lc0, 2 * lr1, 2 * lc1

    correct_patch = full[fr0:fr1, fc0:fc1].copy()
    clue_low = low[lr0:lr1, lc0:lc1].copy()
    damage_h, damage_w = correct_patch.shape

    candidate_count = random.choice((3, 4, 5))
    candidates = _build_candidates(correct_patch, clue_low, palette, candidate_count)
    if candidates is None:
        return None

    gaps = [random.randint(1, 2) for _ in range(candidate_count - 1)]
    candidate_span = candidate_count * damage_w + sum(gaps)
    right_w = max(low_w, candidate_span) + random.randint(0, 2)
    total_w = full.shape[1] + 1 + right_w
    if total_w > MAX_INPUT_WIDTH:
        return None

    canvas = np.full((full.shape[0], total_w), BLACK, dtype=int)
    left_w = full.shape[1]
    sep_col = left_w
    _paste(canvas, full, 0, 0)
    canvas[:, sep_col] = GREY

    clue_col = random.randint(0, right_w - low_w)
    _paste(canvas, low, 0, sep_col + 1 + clue_col)

    cand_top_min = low_h + 1
    cand_top_max = full.shape[0] - damage_h
    if cand_top_min > cand_top_max:
        return None
    cand_top = random.randint(cand_top_min, cand_top_max)
    cand_left = random.randint(0, right_w - candidate_span)
    cursor = cand_left
    for idx, patch in enumerate(candidates):
        _paste(canvas, patch, cand_top, sep_col + 1 + cursor)
        if idx + 1 < candidate_count:
            cursor += damage_w + gaps[idx]

    canvas[fr0:fr1, fc0:fc1] = AZURE

    expected = full.copy()
    try:
        solved = main(canvas)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
