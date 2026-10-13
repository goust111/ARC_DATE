from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
GREY = 5
SOURCE_COLORS = (1, 2, 3, 4)
SOURCE_SIZE = 3
TEMPLATE_SIZE = 5
OUTPUT_SIZE = 15
MAX_GENERATION_ATTEMPTS = 300


BASE_TEMPLATES = (
    (
        "####.",
        "#....",
        ".....",
        ".....",
        ".....",
    ),
    (
        "###..",
        ".#...",
        ".#...",
        ".....",
        ".....",
    ),
    (
        "###..",
        "..#..",
        ".##..",
        ".....",
        ".....",
    ),
    (
        "#....",
        "#....",
        "####.",
        ".....",
        ".....",
    ),
    (
        ".#...",
        ".#...",
        ".###.",
        ".###.",
        ".....",
    ),
)


def _mask_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _neighbors_diag(r: int, c: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for dr in (-1, 1):
        for dc in (-1, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out.append((rr, cc))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected4(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(r, c, mask.shape):
            if bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _is_asymmetric(mask: np.ndarray) -> bool:
    base = mask.tobytes()
    same_count = 0
    cur = mask.copy()
    for _ in range(4):
        for variant in (cur, np.fliplr(cur)):
            if variant.shape == mask.shape and variant.tobytes() == base:
                same_count += 1
        cur = np.rot90(cur)
    return same_count == 1


def _diag_noise_mask(mask: np.ndarray) -> np.ndarray:
    noise = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        orth = any(bool(mask[rr, cc]) for rr, cc in _neighbors4(r, c, mask.shape))
        diag = any(bool(mask[rr, cc]) for rr, cc in _neighbors_diag(r, c, mask.shape))
        if (not orth) and diag:
            noise[r, c] = True
    return noise


def _fill_gap_mask(mask: np.ndarray) -> np.ndarray:
    fills = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                continue
            horizontal = 0 < c < w - 1 and bool(mask[r, c - 1]) and bool(mask[r, c + 1])
            vertical = 0 < r < h - 1 and bool(mask[r - 1, c]) and bool(mask[r + 1, c])
            if horizontal or vertical:
                fills[r, c] = True
    return fills


def _clean_template(mask: np.ndarray) -> np.ndarray:
    erased = mask & ~_diag_noise_mask(mask)
    return erased | _fill_gap_mask(erased)


def _erase_only(mask: np.ndarray) -> np.ndarray:
    return mask & ~_diag_noise_mask(mask)


def _fill_only(mask: np.ndarray) -> np.ndarray:
    return mask | _fill_gap_mask(mask)


def _gap_candidates(mask: np.ndarray) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        horizontal = 0 < c < w - 1 and bool(mask[r, c - 1]) and bool(mask[r, c + 1])
        vertical = 0 < r < h - 1 and bool(mask[r - 1, c]) and bool(mask[r + 1, c])
        if horizontal or vertical:
            tag = ("h" if horizontal else "") + ("v" if vertical else "")
            out.append((r, c, tag))
    return out


def _noise_slot_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if bool(mask[r, c]):
                continue
            orth = any(bool(mask[rr, cc]) for rr, cc in _neighbors4(r, c, mask.shape))
            diag = any(bool(mask[rr, cc]) for rr, cc in _neighbors_diag(r, c, mask.shape))
            if (not orth) and diag:
                out.append((r, c))
    return out


def _stamp_output(source: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    for r in range(SOURCE_SIZE):
        for c in range(SOURCE_SIZE):
            color = int(source[r, c])
            if color == BG:
                continue
            block = out[r * TEMPLATE_SIZE : (r + 1) * TEMPLATE_SIZE, c * TEMPLATE_SIZE : (c + 1) * TEMPLATE_SIZE]
            block[template_mask] = color
    return out


def _split_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mask = (grid != BG) & (grid != GREY)
    source_coords = np.argwhere(source_mask)
    assert source_coords.size > 0, "expected a colored 3x3 source panel"
    sr0, sc0, sr1, sc1 = _bbox(source_coords)
    assert (sr1 - sr0 + 1, sc1 - sc0 + 1) == (SOURCE_SIZE, SOURCE_SIZE), "source panel must occupy a 3x3 bbox"

    template_r0 = sr0 + SOURCE_SIZE + 1
    template_c0 = sc0
    assert template_r0 + TEMPLATE_SIZE <= grid.shape[0], "template panel must fit below the source panel"
    assert template_c0 + TEMPLATE_SIZE <= grid.shape[1], "template panel must fit to the right of the source panel"
    assert np.all(grid[sr1 + 1, sc0 : sc0 + SOURCE_SIZE] == BG), "expected one blank separator row below the source panel"

    source = grid[sr0 : sr0 + SOURCE_SIZE, sc0 : sc0 + SOURCE_SIZE].copy()
    template = grid[template_r0 : template_r0 + TEMPLATE_SIZE, template_c0 : template_c0 + TEMPLATE_SIZE].copy()

    assert set(map(int, np.unique(template))).issubset({BG, GREY}), "template panel may only contain background and grey"
    assert len(set(map(int, np.unique(source))) - {BG}) >= 1, "source panel must contain foreground pixels"
    return source, template


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean a noisy grey stamp template, then stamp its restored shape using the colors from the source panel.

    Input structure:
    - The grid is mostly black background.
    - A 3x3 source panel of colored pixels occupies the upper-left corner of the active layout.
    - One blank row below it begins a 5x5 grey template panel sharing the same left edge; some grey pixels are diagonal-only noise and some true template cells are missing as 1-cell horizontal or vertical gaps.

    Rule:
    - Locate the 3x3 source panel and the 5x5 template panel.
    - In the template, erase every grey pixel with no 4-neighbor grey but at least one diagonal grey neighbor.
    - Then fill every black cell that lies exactly between two grey cells in the same row or in the same column.
    - Build a 15x15 output as a 3x3 array of 5x5 blocks, and for each non-black source pixel stamp the cleaned template into the matching block recolored to that source pixel's color.

    Key insight:
    - The grey template shown in the input is not the real stamp: diagonal-only greys are noise and 1-cell straight-line holes must be restored before any recolored stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    source, template = _split_input(g)
    cleaned_mask = _clean_template(template == GREY)

    assert 3 <= int(cleaned_mask.sum()) <= 8, "cleaned template must be a small stamp"
    assert _is_connected4(cleaned_mask), "cleaned template must be 4-connected"
    assert _is_asymmetric(cleaned_mask), "cleaned template must be asymmetric"

    return _stamp_output(source, cleaned_mask)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    cur = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        cur = np.fliplr(cur)
    return cur.copy()


def _find_corruption_plan(clean_mask: np.ndarray, target_noise_count: int) -> np.ndarray | None:
    gaps = _gap_candidates(clean_mask)
    slots = _noise_slot_candidates(clean_mask)
    if len(gaps) < target_noise_count or len(slots) < target_noise_count:
        return None

    gap_combos = list(itertools.combinations(gaps, target_noise_count))
    random.shuffle(gap_combos)
    slot_combos = list(itertools.combinations(slots, target_noise_count))
    random.shuffle(slot_combos)

    for gap_combo in gap_combos:
        tags = "".join(tag for _r, _c, tag in gap_combo)
        if target_noise_count >= 2 and not ("h" in tags and "v" in tags):
            continue

        removed = clean_mask.copy()
        for r, c, _tag in gap_combo:
            removed[r, c] = False

        fills = _fill_gap_mask(removed)
        if not all(bool(fills[r, c]) for r, c, _tag in gap_combo):
            continue

        for slot_combo in slot_combos:
            noisy = removed.copy()
            for r, c in slot_combo:
                noisy[r, c] = True
            if not np.array_equal(_clean_template(noisy), clean_mask):
                continue
            noise_mask = _diag_noise_mask(noisy)
            if int(noise_mask.sum()) != target_noise_count:
                continue
            if any(not bool(noise_mask[r, c]) for r, c in slot_combo):
                continue
            return noisy
    return None


def _sample_template_pair(target_noise_count: int) -> tuple[np.ndarray, np.ndarray]:
    templates = [_mask_from_rows(rows) for rows in BASE_TEMPLATES]
    for _ in range(120):
        clean_mask = _random_transform(random.choice(templates))
        if not _is_connected4(clean_mask) or not _is_asymmetric(clean_mask):
            continue
        noisy_mask = _find_corruption_plan(clean_mask, target_noise_count)
        if noisy_mask is None:
            continue
        if np.array_equal(noisy_mask, clean_mask):
            continue
        return clean_mask, noisy_mask
    raise RuntimeError(f"failed to sample a template supporting {target_noise_count} noise cells")


def _sample_source_panel() -> np.ndarray:
    positions = [(r, c) for r in range(SOURCE_SIZE) for c in range(SOURCE_SIZE)]
    for _ in range(120):
        count = random.randint(2, 4)
        chosen = random.sample(positions, count)
        coords = np.array(chosen, dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (SOURCE_SIZE, SOURCE_SIZE):
            continue

        distinct = random.randint(2, min(len(SOURCE_COLORS), count))
        palette = random.sample(list(SOURCE_COLORS), distinct)
        colors = palette + [random.choice(palette) for _ in range(count - distinct)]
        random.shuffle(colors)

        source = np.zeros((SOURCE_SIZE, SOURCE_SIZE), dtype=int)
        for (r, c), color in zip(chosen, colors):
            source[r, c] = int(color)
        return source
    raise RuntimeError("failed to sample a valid source panel")


def _render_input(source: np.ndarray, noisy_mask: np.ndarray) -> np.ndarray:
    top = random.randint(0, 4)
    left = random.randint(0, 4)
    bottom = random.randint(0, 6)
    right = random.randint(0, 6)

    height = top + SOURCE_SIZE + 1 + TEMPLATE_SIZE + bottom
    width = left + TEMPLATE_SIZE + right
    grid = np.zeros((height, width), dtype=int)

    grid[top : top + SOURCE_SIZE, left : left + SOURCE_SIZE] = source
    template_r0 = top + SOURCE_SIZE + 1
    template_c0 = left
    grid[template_r0 : template_r0 + TEMPLATE_SIZE, template_c0 : template_c0 + TEMPLATE_SIZE][noisy_mask] = GREY
    return grid


def _validate_example(inp: np.ndarray, source: np.ndarray, clean_mask: np.ndarray, noisy_mask: np.ndarray, target_noise_count: int) -> bool:
    if int(_diag_noise_mask(noisy_mask).sum()) != target_noise_count:
        return False
    if not np.array_equal(_clean_template(noisy_mask), clean_mask):
        return False
    if not _is_connected4(clean_mask) or not _is_asymmetric(clean_mask):
        return False

    expected = _stamp_output(source, clean_mask)
    noisy_out = _stamp_output(source, noisy_mask)
    erase_only_out = _stamp_output(source, _erase_only(noisy_mask))
    fill_only_out = _stamp_output(source, _fill_only(noisy_mask))

    if np.array_equal(noisy_out, expected):
        return False
    if np.array_equal(erase_only_out, expected):
        return False
    if np.array_equal(fill_only_out, expected):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        target_noise_count = random.choices((1, 2, 3), weights=(3, 4, 3), k=1)[0]
        clean_mask, noisy_mask = _sample_template_pair(target_noise_count)
        source = _sample_source_panel()
        inp = _render_input(source, noisy_mask)
        if _validate_example(inp, source, clean_mask, noisy_mask, target_noise_count):
            return inp
    raise RuntimeError("failed to generate a valid noisy-template stamping input")
