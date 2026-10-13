from __future__ import annotations

from collections import Counter, deque
import math
import random

import numpy as np


BACKGROUND = 0
ERASED = 5
PAYLOAD_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

COPY_COUNTS = (5, 7)
SPRITE_HEIGHT_RANGE = (3, 6)
SPRITE_WIDTH_RANGE = (3, 6)
SPRITE_COLOR_COUNT_RANGE = (3, 4)
LAYOUT_ROW_OPTIONS = (1, 2, 3)
GAP_RANGE = (1, 3)
OUTER_MARGIN_MAX = 2
MAX_GENERATION_ATTEMPTS = 400
MAX_VISIBILITY_ATTEMPTS = 200
MAX_SPRITE_ATTEMPTS = 80


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            comps.append(
                {
                    "bbox": bbox,
                    "crop": g[r0 : r1 + 1, c0 : c1 + 1].copy(),
                }
            )

    comps.sort(key=lambda comp: tuple(comp["bbox"][:2]))
    return comps


def _extract_crops(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg)
    assert len(comps) >= 3, "expected several repeated copies"
    crops = [np.asarray(comp["crop"], dtype=int) for comp in comps]
    shapes = {crop.shape for crop in crops}
    assert len(shapes) == 1, "all copies must share one bounding-box size"
    return crops


def _majority_consensus(crops: list[np.ndarray]) -> np.ndarray:
    h, w = crops[0].shape
    consensus = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            votes = [int(crop[r, c]) for crop in crops if int(crop[r, c]) != ERASED]
            assert votes, "every position needs at least one visible vote"
            counts = Counter(votes)
            best = max(counts.values())
            winners = [color for color, n in counts.items() if n == best]
            assert len(winners) == 1, "consensus must be unique at every cell"
            consensus[r, c] = winners[0]
    return consensus


def _traitor_index(crops: list[np.ndarray], consensus: np.ndarray) -> int:
    mismatching = []
    for idx, crop in enumerate(crops):
        bad = (crop != ERASED) & (crop != consensus)
        if np.any(bad):
            mismatching.append(idx)
    assert len(mismatching) == 1, "there must be exactly one non-grey outlier copy"
    return mismatching[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the majority-consensus sprite, find the single contradictory copy, and return its repaired crop.

    Input structure:
    - The background contains several separated sprite copies with identical bounding-box size.
    - Each copy is a multicolor rectangle whose grey cells are erased wildcards.
    - Exactly one copy also contains one or more visible non-grey pixels that disagree with the shared original sprite.

    Rule:
    - Crop all non-background copies and align them by their common bounding-box coordinates.
    - For each cell position, take the unique majority among non-grey colors to recover the consensus original sprite.
    - Identify the only copy whose visible non-grey cells contradict that consensus, then return that copy's crop with grey cells filled from the consensus while preserving its contradictory non-grey pixels.

    Key insight:
    - Grey erasures remove direct certainty, so the original sprite must be reconstructed cell-by-cell by majority vote before the outlier copy can be identified and selectively repaired.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    crops = _extract_crops(g)
    consensus = _majority_consensus(crops)
    idx = _traitor_index(crops, consensus)
    traitor = crops[idx]
    return np.where(traitor == ERASED, consensus, traitor)


def _make_sprite(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    area = height * width
    assert len(colors) <= area
    for _ in range(MAX_SPRITE_ATTEMPTS):
        flat = colors[:] + [random.choice(colors) for _ in range(area - len(colors))]
        random.shuffle(flat)
        sprite = np.array(flat, dtype=int).reshape(height, width)

        if len(set(map(int, sprite.ravel()))) < 2:
            continue
        if not any(len(set(map(int, row))) >= 2 for row in sprite):
            continue
        if not any(len(set(map(int, col))) >= 2 for col in sprite.T):
            continue
        if height > 1 and len({tuple(map(int, row)) for row in sprite}) == 1:
            continue
        if width > 1 and len({tuple(map(int, col)) for col in sprite.T}) == 1:
            continue
        if np.array_equal(sprite, np.fliplr(sprite)) and np.array_equal(sprite, np.flipud(sprite)):
            continue
        return sprite
    return None


def _sample_honest_visibility(
    honest_count: int,
    area: int,
    mismatch_positions: set[int],
) -> np.ndarray | None:
    min_visible = max(2, honest_count - 2)
    max_greys_per_copy = max(2, math.ceil(area * 0.45))
    mismatch_list = sorted(mismatch_positions)

    for _ in range(MAX_VISIBILITY_ATTEMPTS):
        vis = np.zeros((honest_count, area), dtype=bool)
        for pos in range(area):
            if pos in mismatch_positions:
                choices = list(range(min_visible, honest_count))
            else:
                choices = list(range(min_visible, honest_count + 1))
            k = random.choice(choices)
            visible = random.sample(range(honest_count), k)
            vis[visible, pos] = True

        grey_counts = area - vis.sum(axis=1)
        if np.any(grey_counts == 0):
            continue
        if np.any(grey_counts > max_greys_per_copy):
            continue
        if any(int(vis[:, pos].sum()) < 2 for pos in range(area)):
            continue
        if any(all(bool(vis[i, pos]) for pos in mismatch_list) for i in range(honest_count)):
            continue
        witness_sets = {tuple(np.flatnonzero(vis[:, pos]).tolist()) for pos in mismatch_list}
        if len(witness_sets) == 1:
            continue
        return vis
    return None


def _choose_layout(count: int, sprite_h: int, sprite_w: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    candidates = []
    for rows in LAYOUT_ROW_OPTIONS:
        if rows > count:
            continue
        cols = math.ceil(count / rows)
        for gap_r in range(GAP_RANGE[0], GAP_RANGE[1] + 1):
            for gap_c in range(GAP_RANGE[0], GAP_RANGE[1] + 1):
                core_h = rows * sprite_h + (rows - 1) * gap_r
                core_w = cols * sprite_w + (cols - 1) * gap_c
                if core_h > 30 or core_w > 30:
                    continue
                candidates.append((rows, cols, gap_r, gap_c, core_h, core_w))

    if not candidates:
        return None

    rows, cols, gap_r, gap_c, core_h, core_w = random.choice(candidates)
    top = random.randint(0, min(OUTER_MARGIN_MAX, 30 - core_h))
    left = random.randint(0, min(OUTER_MARGIN_MAX, 30 - core_w))
    bottom = random.randint(0, min(OUTER_MARGIN_MAX, 30 - core_h - top))
    right = random.randint(0, min(OUTER_MARGIN_MAX, 30 - core_w - left))
    canvas_h = core_h + top + bottom
    canvas_w = core_w + left + right

    slots = [
        (top + r * (sprite_h + gap_r), left + c * (sprite_w + gap_c))
        for r in range(rows)
        for c in range(cols)
    ]
    random.shuffle(slots)
    positions = sorted(slots[:count])
    return canvas_h, canvas_w, positions


def _render_input(canvas_h: int, canvas_w: int, positions: list[tuple[int, int]], crops: list[np.ndarray]) -> np.ndarray:
    grid = np.full((canvas_h, canvas_w), BACKGROUND, dtype=int)
    for (r0, c0), crop in zip(positions, crops):
        h, w = crop.shape
        grid[r0 : r0 + h, c0 : c0 + w] = crop
    return grid


def _semantic_checks(
    inp: np.ndarray,
    original: np.ndarray,
    expected: np.ndarray,
    traitor_idx: int,
    mismatch_positions: set[int],
    honest_visibility: np.ndarray,
) -> bool:
    try:
        crops = _extract_crops(inp)
        consensus = _majority_consensus(crops)
        found_idx = _traitor_index(crops, consensus)
        solved = main(inp)
    except Exception:
        return False

    if found_idx != traitor_idx:
        return False
    if not np.array_equal(consensus, original):
        return False
    if not np.array_equal(solved, expected):
        return False

    traitor = crops[traitor_idx]
    if not np.any(traitor == ERASED):
        return False
    if not np.any((traitor != ERASED) & (traitor != original)):
        return False
    if np.array_equal(traitor, expected):
        return False
    if np.array_equal(original, expected):
        return False

    honest_seen = 0
    mismatch_list = sorted(mismatch_positions)
    honest_mask_row = 0
    for idx, crop in enumerate(crops):
        if idx == traitor_idx:
            continue
        honest_seen += 1
        if np.any((crop != ERASED) & (crop != original)):
            return False
        if not np.any(crop == ERASED):
            return False
        if all(bool(honest_visibility[honest_mask_row, pos]) for pos in mismatch_list):
            return False
        honest_mask_row += 1

    return honest_seen == honest_visibility.shape[0]


def _gen_attempt() -> np.ndarray | None:
    copy_count = random.choice(COPY_COUNTS)
    sprite_h = random.randint(*SPRITE_HEIGHT_RANGE)
    sprite_w = random.randint(*SPRITE_WIDTH_RANGE)
    palette_size = random.randint(*SPRITE_COLOR_COUNT_RANGE)
    colors = random.sample(PAYLOAD_COLORS, palette_size)
    sprite = _make_sprite(sprite_h, sprite_w, colors)
    if sprite is None:
        return None

    layout = _choose_layout(copy_count, sprite_h, sprite_w)
    if layout is None:
        return None
    canvas_h, canvas_w, positions = layout

    area = sprite_h * sprite_w
    max_mismatches = min(4, area - 1)
    if max_mismatches < 2:
        return None
    mismatch_count = random.randint(2, max_mismatches)
    mismatch_positions = set(random.sample(range(area), mismatch_count))

    traitor_idx = random.randrange(copy_count)
    honest_count = copy_count - 1
    honest_visibility = _sample_honest_visibility(honest_count, area, mismatch_positions)
    if honest_visibility is None:
        return None

    flat_original = sprite.reshape(-1)
    non_mismatch_positions = [idx for idx in range(area) if idx not in mismatch_positions]
    if not non_mismatch_positions:
        return None
    grey_count = random.randint(1, min(len(non_mismatch_positions), max(2, area // 3)))
    traitor_greys = set(random.sample(non_mismatch_positions, grey_count))

    traitor_flat = flat_original.copy()
    for pos in mismatch_positions:
        options = [color for color in colors if color != int(flat_original[pos])]
        traitor_flat[pos] = random.choice(options)
    for pos in traitor_greys:
        traitor_flat[pos] = ERASED
    traitor_crop = traitor_flat.reshape(sprite_h, sprite_w)

    crops: list[np.ndarray] = []
    honest_row = 0
    for idx in range(copy_count):
        if idx == traitor_idx:
            crops.append(traitor_crop.copy())
            continue
        crop = np.where(honest_visibility[honest_row].reshape(sprite_h, sprite_w), sprite, ERASED)
        crops.append(crop.astype(int))
        honest_row += 1

    inp = _render_input(canvas_h, canvas_w, positions, crops)
    expected = np.where(traitor_crop == ERASED, sprite, traitor_crop)
    if not _semantic_checks(inp, sprite, expected, traitor_idx, mismatch_positions, honest_visibility):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
