from __future__ import annotations

import random

import numpy as np


def _apply_mapping(panel: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = panel.copy()
    original = panel
    for src, dst in mapping.items():
        out[original == src] = dst
    return out


def _infer_mapping(panel_a: np.ndarray, panel_b: np.ndarray) -> dict[int, int] | None:
    diff = panel_a != panel_b
    if not np.any(diff):
        return None

    mapping: dict[int, int] = {}
    for src_raw, dst_raw in zip(panel_a[diff].tolist(), panel_b[diff].tolist(), strict=False):
        src = int(src_raw)
        dst = int(dst_raw)
        old = mapping.get(src)
        if old is not None and old != dst:
            return None
        mapping[src] = dst

    if not mapping:
        return None
    if not np.array_equal(_apply_mapping(panel_a, mapping), panel_b):
        return None
    return mapping


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]] = []

    for panel_w in range(2, w // 3 + 1):
        separator_total = w - 3 * panel_w
        if separator_total < 2:
            continue
        for sep1_w in range(1, separator_total):
            sep2_w = separator_total - sep1_w
            sep1_c0 = panel_w
            sep2_c0 = 2 * panel_w + sep1_w
            sep1 = g[:, sep1_c0 : sep1_c0 + sep1_w]
            sep2 = g[:, sep2_c0 : sep2_c0 + sep2_w]
            sep_color = int(sep1[0, 0])
            if not np.all(sep1 == sep_color):
                continue
            if not np.all(sep2 == sep_color):
                continue

            panel1 = g[:, :panel_w]
            panel2 = g[:, sep1_c0 + sep1_w : sep2_c0]
            panel3 = g[:, sep2_c0 + sep2_w :]
            if any(np.any(panel == sep_color) for panel in (panel1, panel2, panel3)):
                continue

            mapping = _infer_mapping(panel1, panel2)
            if mapping is None:
                continue
            if not any(np.any(panel3 == src) for src in mapping):
                continue
            candidates.append((panel1, panel2, panel3, mapping))

    assert len(candidates) == 1, "expected a unique three-panel split"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a color-substitution rule from the first two panels and apply it to the third.

    Input structure
    - The grid is split into three equal left-to-right panels by two solid divider bands.
    - Panels 1 and 2 contain the same multicolor object shape, but some object colors
      are globally replaced from panel 1 to panel 2.
    - Panel 3 contains a similar object that still uses the source colors.

    Rule
    - Compare the first two panels pixel-by-pixel.
    - Record the global color changes implied by their differing pixels.
    - Apply that color mapping simultaneously to panel 3.

    Key insight
    - The first two panels are a worked example of a color substitution, while the
      third panel is the test case.
    """
    _, _, panel3, mapping = _split_panels(grid)
    return _apply_mapping(panel3, mapping)


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (6, 11)
PANEL_WIDTH_RANGE = (7, 12)
SEPARATOR_WIDTH_RANGE = (1, 2)
OBJECT_COLOR_COUNT_RANGE = (3, 5)
MAX_MAPPING_COLORS = 3
GENERATION_RETRIES = 300
STEP_CHOICES = (-1, 0, 1)


def _random_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _build_base_mask(height: int, width: int) -> np.ndarray:
    rows = [random.randrange(height)]
    for _ in range(1, width):
        row = rows[-1] + random.choice(STEP_CHOICES)
        row = max(0, min(height - 1, row))
        rows.append(row)

    mask = np.zeros((height, width), dtype=bool)
    mask[rows[0], 0] = True
    for c in range(1, width):
        r0 = rows[c - 1]
        r1 = rows[c]
        lo, hi = sorted((r0, r1))
        mask[lo : hi + 1, c] = True

    for c, row in enumerate(rows):
        if random.random() < 0.35:
            options = [rr for rr in (row - 1, row + 1) if 0 <= rr < height]
            if options:
                mask[random.choice(options), c] = True

    assert np.all(mask.any(axis=0))
    return mask


def _neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors(mask, r, c):
            if mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _mutate_mask(mask: np.ndarray) -> np.ndarray | None:
    base = mask.copy()
    h, w = mask.shape
    for _ in range(60):
        mutated = base.copy()
        changed = False
        edit_count = random.randint(1, 3)
        for _ in range(edit_count):
            do_add = random.random() < 0.6
            if do_add:
                frontier = []
                for r in range(h):
                    for c in range(w):
                        if mutated[r, c]:
                            continue
                        if any(mutated[rr, cc] for rr, cc in _neighbors(mutated, r, c)):
                            frontier.append((r, c))
                if frontier:
                    r, c = random.choice(frontier)
                    mutated[r, c] = True
                    changed = True
            else:
                removable = []
                for r, c in np.argwhere(mutated):
                    r = int(r)
                    c = int(c)
                    if int(mutated[:, c].sum()) <= 1:
                        continue
                    if sum(bool(mutated[rr, cc]) for rr, cc in _neighbors(mutated, r, c)) > 1:
                        continue
                    trial = mutated.copy()
                    trial[r, c] = False
                    if _is_connected(trial) and np.all(trial.any(axis=0)):
                        removable.append((r, c))
                if removable:
                    r, c = random.choice(removable)
                    mutated[r, c] = False
                    changed = True

        if changed and not np.array_equal(mutated, base) and _is_connected(mutated) and np.all(mutated.any(axis=0)):
            return mutated
    return None


def _paint_columns(
    mask: np.ndarray,
    bg_color: int,
    colors: list[int],
) -> np.ndarray:
    h, w = mask.shape
    grid = np.full((h, w), bg_color, dtype=int)
    segment_count = random.randint(len(colors), min(w, len(colors) + 2))
    segment_widths = _random_partition(w, segment_count)
    segment_colors = colors + [random.choice(colors) for _ in range(segment_count - len(colors))]
    random.shuffle(segment_colors)

    c0 = 0
    for width, color in zip(segment_widths, segment_colors, strict=False):
        c1 = c0 + width
        grid[:, c0:c1][mask[:, c0:c1]] = int(color)
        c0 = c1
    return grid


def _sample_palette(count: int, forbidden: set[int]) -> list[int]:
    pool = [color for color in range(10) if color not in forbidden]
    return random.sample(pool, count)


def _build_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    sep1_w = random.randint(*SEPARATOR_WIDTH_RANGE)
    sep2_w = random.randint(*SEPARATOR_WIDTH_RANGE)

    sep_color = 1
    bg_choices = [color for color in range(10) if color != sep_color]
    bg_color = random.choice(bg_choices)

    object_color_count = random.randint(*OBJECT_COLOR_COUNT_RANGE)
    object_colors = _sample_palette(object_color_count, {bg_color, sep_color})
    max_map = min(MAX_MAPPING_COLORS, object_color_count - 1)
    map_count = random.randint(1, max_map)
    source_colors = random.sample(object_colors, map_count)
    target_colors = _sample_palette(map_count, {bg_color, sep_color, *object_colors})
    mapping = {src: dst for src, dst in zip(source_colors, target_colors, strict=False)}

    mask12 = _build_base_mask(height, panel_w)
    mask3 = _mutate_mask(mask12)
    if mask3 is None:
        return None

    panel1 = _paint_columns(mask12, bg_color, object_colors)
    panel2 = _apply_mapping(panel1, mapping)
    panel3 = _paint_columns(mask3, bg_color, object_colors)
    if not all(np.any(panel3 == src) for src in source_colors):
        return None

    grid = np.full((height, panel_w * 3 + sep1_w + sep2_w), sep_color, dtype=int)
    grid[:, :panel_w] = panel1
    grid[:, panel_w + sep1_w : 2 * panel_w + sep1_w] = panel2
    grid[:, 2 * panel_w + sep1_w + sep2_w :] = panel3

    try:
        expected = _apply_mapping(panel3, mapping)
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
