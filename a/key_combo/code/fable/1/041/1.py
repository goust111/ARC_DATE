from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]
Sample = tuple[int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _tile_fill(shape: tuple[int, int], motif: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    height, width = map(int, shape)
    motif_h, motif_w = map(int, motif.shape)
    off_r, off_c = map(int, offset)
    rows = (np.arange(height) - off_r) % motif_h
    cols = (np.arange(width) - off_c) % motif_w
    return motif[rows[:, None], cols[None, :]]


def _candidate_offsets(motif: np.ndarray, samples: list[Sample]) -> list[tuple[int, int]]:
    motif_h, motif_w = map(int, motif.shape)
    candidates: list[tuple[int, int]] = []
    for off_r in range(motif_h):
        for off_c in range(motif_w):
            if all(int(motif[(r - off_r) % motif_h, (c - off_c) % motif_w]) == color for r, c, color in samples):
                candidates.append((off_r, off_c))
    return candidates


def _find_separator(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    candidates: list[tuple[str, int, int]] = []

    for r in range(height):
        color = int(g[r, 0])
        if color != bg and bool(np.all(g[r] == color)):
            candidates.append(("row", r, color))
    for c in range(width):
        color = int(g[0, c])
        if color != bg and bool(np.all(g[:, c] == color)):
            candidates.append(("col", c, color))

    assert len(candidates) == 1, "expected one full non-background separator line"
    axis, index, color = candidates[0]
    return axis, int(index), int(color)


def _split_panels(grid: np.ndarray, axis: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    if axis == "row":
        first = g[:index, :]
        second = g[index + 1 :, :]
    else:
        first = g[:, :index]
        second = g[:, index + 1 :]

    assert first.size > 0 and second.size > 0, "separator must split the grid into two non-empty panels"
    area_first = int(first.shape[0] * first.shape[1])
    area_second = int(second.shape[0] * second.shape[1])
    assert area_first != area_second, "motif panel and canvas panel must have different sizes"
    if area_first < area_second:
        return np.asarray(first, dtype=int), np.asarray(second, dtype=int)
    return np.asarray(second, dtype=int), np.asarray(first, dtype=int)


def _extract_motif(panel: np.ndarray, bg: int) -> np.ndarray:
    mask = np.asarray(panel, dtype=int) != bg
    box = _bbox(mask)
    assert box is not None, "motif panel must contain one rectangular seed motif"
    motif = _crop_bbox(panel, box)
    assert bool(np.all(motif != bg)), "motif must be a full rectangle without background holes"
    assert bool(np.array_equal(mask, _mask_from_bbox(mask.shape, box))), "motif panel should contain only the seed motif"
    return motif


def _mask_from_bbox(shape: tuple[int, int], box: BBox) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _collect_samples(canvas: np.ndarray, bg: int) -> list[Sample]:
    coords = np.argwhere(np.asarray(canvas, dtype=int) != bg)
    samples = [(int(r), int(c), int(canvas[r, c])) for r, c in coords]
    assert samples, "canvas panel must contain sparse sample pixels"
    return samples


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the unique wallpaper phase from sparse samples and tile the seed motif across the canvas.

    Input structure:
    - A single full separator line splits the grid into a small motif panel and a larger mostly blank canvas panel.
    - The small panel contains one non-background rectangular motif placed in a corner.
    - The large panel contains only background plus a few colored sample pixels taken from one hidden periodic tiling of that motif.

    Rule:
    - Crop the rectangular motif from the small panel.
    - Consider every horizontal and vertical phase offset of the infinite tiling induced by that motif.
    - Keep the unique offset consistent with all sample pixels, then fill the whole large panel with that phased tiling.
    - Output only the tiled canvas region, excluding the motif panel and the separator line.

    Key insight:
    - Each sample is just a phase constraint on the same repeated wallpaper; a single point stays ambiguous because motif colors repeat, but their intersection fixes one offset.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    axis, index, _sep_color = _find_separator(g, bg)
    motif_panel, canvas_panel = _split_panels(g, axis, index)
    motif = _extract_motif(motif_panel, bg)
    samples = _collect_samples(canvas_panel, bg)

    candidates = _candidate_offsets(motif, samples)
    assert len(candidates) == 1, "sample constraints must determine a unique wallpaper phase"
    return _tile_fill(canvas_panel.shape, motif, candidates[0])


# --------------- generator ---------------

PALETTE = tuple(range(10))
MOTIF_SIDE_RANGE = (2, 5)
CANVAS_SIDE_RANGE = (6, 20)
PANEL_EXTRA_MAX = 3
SAMPLE_SIZE_PREFERENCES = (2, 3, 4, 5)
MAX_ATTEMPTS = 300


def _has_nontrivial_tiling_period(motif: np.ndarray) -> bool:
    motif_h, motif_w = map(int, motif.shape)
    for dr in range(motif_h):
        for dc in range(motif_w):
            if dr == 0 and dc == 0:
                continue
            if bool(np.array_equal(motif, np.roll(np.roll(motif, dr, axis=0), dc, axis=1))):
                return True
    return False


def _gen_motif(colors: list[int]) -> np.ndarray | None:
    motif_h = random.randint(*MOTIF_SIDE_RANGE)
    motif_w = random.randint(*MOTIF_SIDE_RANGE)
    area = motif_h * motif_w
    max_colors = min(len(colors), area // 2)
    if max_colors < 2:
        return None

    n_colors = random.randint(2, max_colors)
    motif_colors = random.sample(colors, n_colors)
    counts = [2] * n_colors
    for _ in range(area - 2 * n_colors):
        counts[random.randrange(n_colors)] += 1

    cells: list[int] = []
    for color, count in zip(motif_colors, counts):
        cells.extend([int(color)] * int(count))
    random.shuffle(cells)
    motif = np.asarray(cells, dtype=int).reshape(motif_h, motif_w)

    if _has_nontrivial_tiling_period(motif):
        return None
    if len(set(map(int, motif.reshape(-1)))) < 2:
        return None
    return motif


def _choose_canvas_dims(motif: np.ndarray) -> tuple[int, int]:
    motif_h, motif_w = map(int, motif.shape)
    canvas_h = random.randint(max(CANVAS_SIDE_RANGE[0], motif_h + 1), CANVAS_SIDE_RANGE[1])
    canvas_w = random.randint(max(CANVAS_SIDE_RANGE[0], motif_w + 1), CANVAS_SIDE_RANGE[1])
    return int(canvas_h), int(canvas_w)


def _spaced_subset(coords: list[tuple[int, int]], k: int, gap: int) -> list[tuple[int, int]] | None:
    order = coords[:]
    random.shuffle(order)
    chosen: list[tuple[int, int]] = []
    for r, c in order:
        if all(max(abs(r - rr), abs(c - cc)) > gap for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == k:
                return chosen
    return None


def _is_minimally_unique(motif: np.ndarray, samples: list[Sample], true_offset: tuple[int, int]) -> bool:
    if _candidate_offsets(motif, samples) != [true_offset]:
        return False
    return all(len(_candidate_offsets(motif, samples[:i] + samples[i + 1 :])) > 1 for i in range(len(samples)))


def _find_sample_set(motif: np.ndarray, canvas: np.ndarray, true_offset: tuple[int, int]) -> list[Sample] | None:
    height, width = map(int, canvas.shape)
    coords = [(r, c) for r in range(height) for c in range(width)]

    preferred = list(SAMPLE_SIZE_PREFERENCES)
    random.shuffle(preferred)
    for k in preferred:
        if k > len(coords):
            continue
        for gap in (2, 1, 0):
            for _ in range(400):
                picked = _spaced_subset(coords, k, gap)
                if picked is None:
                    continue
                samples = [(r, c, int(canvas[r, c])) for r, c in picked]
                if len({color for _, _, color in samples}) < 2:
                    continue
                if _is_minimally_unique(motif, samples, true_offset):
                    return samples
    return None


def _panel_origin(panel_shape: tuple[int, int], motif_shape: tuple[int, int]) -> tuple[int, int]:
    panel_h, panel_w = map(int, panel_shape)
    motif_h, motif_w = map(int, motif_shape)
    top = 0 if panel_h == motif_h or random.random() < 0.5 else panel_h - motif_h
    left = 0 if panel_w == motif_w or random.random() < 0.5 else panel_w - motif_w
    return int(top), int(left)


def _compose_input(
    motif: np.ndarray,
    samples: list[Sample],
    canvas_shape: tuple[int, int],
    bg: int,
    sep: int,
) -> np.ndarray:
    canvas_h, canvas_w = map(int, canvas_shape)
    layout = random.choice(("top", "bottom", "left", "right"))
    motif_h, motif_w = map(int, motif.shape)

    if layout in {"top", "bottom"}:
        panel_h = random.randint(motif_h, min(motif_h + PANEL_EXTRA_MAX, canvas_h - 1))
        panel_w = canvas_w
        panel = np.full((panel_h, panel_w), bg, dtype=int)
        start_r, start_c = _panel_origin(panel.shape, motif.shape)
        panel[start_r : start_r + motif_h, start_c : start_c + motif_w] = motif

        canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
        for r, c, color in samples:
            canvas[r, c] = color
        separator = np.full((1, canvas_w), sep, dtype=int)

        if layout == "top":
            return np.vstack((panel, separator, canvas))
        return np.vstack((canvas, separator, panel))

    panel_h = canvas_h
    panel_w = random.randint(motif_w, min(motif_w + PANEL_EXTRA_MAX, canvas_w - 1))
    panel = np.full((panel_h, panel_w), bg, dtype=int)
    start_r, start_c = _panel_origin(panel.shape, motif.shape)
    panel[start_r : start_r + motif_h, start_c : start_c + motif_w] = motif

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    for r, c, color in samples:
        canvas[r, c] = color
    separator = np.full((canvas_h, 1), sep, dtype=int)

    if layout == "left":
        return np.hstack((panel, separator, canvas))
    return np.hstack((canvas, separator, panel))


def _gen_attempt() -> np.ndarray | None:
    bg, sep, *color_pool = random.sample(PALETTE, 6)
    motif = _gen_motif(color_pool)
    if motif is None:
        return None

    canvas_shape = _choose_canvas_dims(motif)
    true_offset = (random.randrange(motif.shape[0]), random.randrange(motif.shape[1]))
    expected = _tile_fill(canvas_shape, motif, true_offset)
    samples = _find_sample_set(motif, expected, true_offset)
    if samples is None:
        return None

    grid = _compose_input(motif, samples, canvas_shape, bg, sep)
    if max(grid.shape) > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    bg_check = _get_bg(grid)
    axis, index, _ = _find_separator(grid, bg_check)
    motif_panel, canvas_panel = _split_panels(grid, axis, index)
    motif_check = _extract_motif(motif_panel, bg_check)
    samples_check = _collect_samples(canvas_panel, bg_check)
    if not np.array_equal(motif_check, motif):
        return None
    if sorted(samples_check) != sorted(samples):
        return None
    if not _is_minimally_unique(motif_check, samples_check, true_offset):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wallpaper-phase ARC input")
