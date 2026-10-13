from __future__ import annotations

import random

import numpy as np


ENTRY_H = 5
ENTRY_W = 6
ENTRY_GAP = 1

PATTERN_BANK: tuple[np.ndarray, ...] = (
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=bool),
)


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        if r1 != r0:
            pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        if c1 != c0:
            pts.append((r, c1))
    return pts


def _is_frame_mask(mask: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
        return False
    coords = np.argwhere(mask)
    rr = coords[:, 0]
    cc = coords[:, 1]
    on_border = (rr == r0) | (rr == r1) | (cc == c0) | (cc == c1)
    if not np.all(on_border):
        return False
    return bool(
        mask[r0, c0 : c1 + 1].any()
        and mask[r1, c0 : c1 + 1].any()
        and mask[r0 : r1 + 1, c0].any()
        and mask[r0 : r1 + 1, c1].any()
    )


def _parse_legend_entry(mask: np.ndarray) -> tuple[int, np.ndarray]:
    r0, c0, r1, c1 = _bbox_from_mask(mask)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    if sub.shape != (ENTRY_H, ENTRY_W):
        raise ValueError(f"unexpected legend entry shape: {sub.shape}")
    count = int(sub[:, 0].sum())
    pattern = sub[-3:, -3:].copy()
    if not (1 <= count <= 5):
        raise ValueError("legend count must be between 1 and 5")
    return count, pattern


def _tile_pattern(height: int, width: int, pattern: np.ndarray) -> np.ndarray:
    reps_r = (height + pattern.shape[0] - 1) // pattern.shape[0]
    reps_c = (width + pattern.shape[1] - 1) // pattern.shape[1]
    return np.tile(pattern, (reps_r, reps_c))[:height, :width]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame interior with the legend pattern keyed by its gap count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]

    legend_map: dict[int, np.ndarray] = {}
    frames: list[tuple[int, tuple[int, int, int, int]]] = []

    for color in colors:
        mask = g == color
        if _is_frame_mask(mask):
            frames.append((color, _bbox_from_mask(mask)))
        else:
            count, pattern = _parse_legend_entry(mask)
            if count in legend_map:
                raise ValueError("duplicate legend entry count")
            legend_map[count] = pattern

    out = g.copy()
    for color, (r0, c0, r1, c1) in frames:
        gaps = sum(1 for r, c in _border_positions(r0, c0, r1, c1) if int(g[r, c]) == bg)
        if gaps not in legend_map:
            raise ValueError(f"missing legend entry for gap count {gaps}")
        pattern = legend_map[gaps]
        if r1 - r0 <= 1 or c1 - c0 <= 1:
            continue
        tiled = _tile_pattern(r1 - r0 - 1, c1 - c0 - 1, pattern)
        region = out[r0 + 1 : r1, c0 + 1 : c1]
        region[:] = bg
        region[tiled] = color

    return out


GEN_ATTEMPTS = 400
FRAME_COUNT_OPTIONS = (3, 4)
GRID_H_RANGE = (25, 32)
FIELD_W_RANGE = (18, 25)
FIELD_RIGHT_GAP_RANGE = (3, 5)
RIGHT_MARGIN_RANGE = (0, 1)
FRAME_H_RANGE = (5, 8)
FRAME_W_RANGE = (6, 11)
FRAME_PADDING = 1
FRAME_PLACE_TRIES = 300


def _legend_patch(count: int, pattern: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full((ENTRY_H, ENTRY_W), bg, dtype=int)
    patch[:count, 0] = color
    patch[-3:, -3:][pattern] = color
    return patch


def _frame_patch(height: int, width: int, color: int, gap_count: int, bg: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    border = (
        [(0, c) for c in range(1, width - 1)]
        + [(height - 1, c) for c in range(1, width - 1)]
        + [(r, 0) for r in range(1, height - 1)]
        + [(r, width - 1) for r in range(1, height - 1)]
    )
    chosen = random.sample(border, gap_count)
    for r, c in chosen:
        patch[r, c] = bg
    return patch


def _place_frames(
    grid: np.ndarray,
    field_right: int,
    specs: list[tuple[int, int, int, int]],
    bg: int,
) -> bool:
    h, _ = grid.shape
    occupied = np.zeros((h, field_right), dtype=bool)

    for _, frame_h, frame_w, color, gap_count in sorted(
        [(frame_h * frame_w, frame_h, frame_w, color, gap_count) for frame_h, frame_w, color, gap_count in specs],
        reverse=True,
    ):
        patch = _frame_patch(frame_h, frame_w, color, gap_count, bg)
        placed = False

        for _ in range(FRAME_PLACE_TRIES):
            if h - frame_h - 1 < 1 or field_right - frame_w - 1 < 1:
                return False
            r0 = random.randint(1, h - frame_h - 1)
            c0 = random.randint(1, field_right - frame_w - 1)
            rr0 = max(0, r0 - FRAME_PADDING)
            cc0 = max(0, c0 - FRAME_PADDING)
            rr1 = min(h, r0 + frame_h + FRAME_PADDING)
            cc1 = min(field_right, c0 + frame_w + FRAME_PADDING)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            mask = patch != bg
            grid[r0 : r0 + frame_h, c0 : c0 + frame_w][mask] = patch[mask]
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n_frames = random.choice(FRAME_COUNT_OPTIONS)
    counts = random.sample([1, 2, 3, 4, 5], n_frames)
    palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 2 * n_frames)
    frame_colors = palette[:n_frames]
    legend_colors = palette[n_frames:]

    patterns = random.sample(list(PATTERN_BANK), n_frames)
    legend_map = {count: pattern for count, pattern in zip(counts, patterns)}

    legend_height = ENTRY_H * n_frames + ENTRY_GAP * (n_frames - 1)
    grid_h = random.randint(max(GRID_H_RANGE[0], legend_height + 2), GRID_H_RANGE[1])
    field_w = random.randint(FIELD_W_RANGE[0], FIELD_W_RANGE[1])
    right_gap = random.randint(*FIELD_RIGHT_GAP_RANGE)
    right_margin = random.randint(*RIGHT_MARGIN_RANGE)
    grid_w = field_w + right_gap + ENTRY_W + right_margin
    legend_left = grid_w - ENTRY_W - right_margin
    field_right = legend_left - right_gap
    if field_right < max(FRAME_W_RANGE) + 2:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)

    legend_top = random.randint(1, grid_h - legend_height - 1)
    legend_order = counts[:]
    random.shuffle(legend_order)
    color_for_count = {count: color for count, color in zip(legend_order, legend_colors)}
    for idx, count in enumerate(legend_order):
        r0 = legend_top + idx * (ENTRY_H + ENTRY_GAP)
        patch = _legend_patch(count, legend_map[count], color_for_count[count], bg)
        grid[r0 : r0 + ENTRY_H, legend_left : legend_left + ENTRY_W] = patch

    frame_specs: list[tuple[int, int, int, int]] = []
    for count, color in zip(counts, frame_colors):
        frame_h = random.randint(*FRAME_H_RANGE)
        frame_w = random.randint(*FRAME_W_RANGE)
        frame_specs.append((frame_h, frame_w, color, count))

    if not _place_frames(grid, field_right, frame_specs, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
