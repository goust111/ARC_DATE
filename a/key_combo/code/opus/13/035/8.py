from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 2
INPUT_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
SEED_BBOX_OPTIONS = ((2, 2), (2, 3), (3, 2))
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tile_patch(patch: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    rr = np.arange(int(out_h))[:, None] % patch.shape[0]
    cc = np.arange(int(out_w))[None, :] % patch.shape[1]
    return patch[rr, cc]


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    counts = Counter(map(int, g[g != BACKGROUND].flat))
    assert counts, "expected non-background structure"

    seed_entries: list[tuple[int, np.ndarray]] = []
    scatter_component_count: Counter[int] = Counter()
    scatter_coords: list[tuple[int, int, int]] = []

    for color in sorted(counts):
        comps = _components(g == int(color))
        for comp in comps:
            if comp.shape[0] == 1:
                r, c = map(int, comp[0])
                scatter_component_count[int(color)] += 1
                scatter_coords.append((r, c, int(color)))
            else:
                seed_entries.append((int(color), comp))

    assert len(seed_entries) == 1, "expected exactly one multi-pixel seed component"
    seed_color, seed_coords = seed_entries[0]
    assert counts[seed_color] == int(seed_coords.shape[0]), "seed color must belong only to the seed component"

    scatter_colors = sorted(scatter_component_count)
    assert 2 <= len(scatter_colors) <= 3, "expected 2-3 scatter colors"
    assert seed_color not in scatter_component_count, "seed color must not be reused by scatter pixels"

    anchor_colors = [color for color in scatter_colors if scatter_component_count[color] == 1]
    assert len(anchor_colors) == 1, "expected exactly one uniquely frequent scatter color"
    anchor_color = int(anchor_colors[0])

    for color in scatter_colors:
        count = int(scatter_component_count[color])
        if color == anchor_color:
            assert count == 1
        else:
            assert 2 <= count <= 4

    anchor_pos = next((r, c) for r, c, color in scatter_coords if color == anchor_color)
    seed_box = _bbox(seed_coords)
    r0, c0, r1, c1 = seed_box
    patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert patch.shape in SEED_BBOX_OPTIONS
    assert np.all((patch == BACKGROUND) | (patch == seed_color))
    assert len(_components(patch == seed_color)) == 1

    return {
        "anchor": (int(anchor_pos[0]), int(anchor_pos[1])),
        "anchor_color": anchor_color,
        "scatter_coords": [(int(r), int(c)) for r, c, _color in scatter_coords],
        "seed_color": seed_color,
        "seed_bbox": seed_box,
        "seed_patch": patch,
    }


def _render_output(shape: tuple[int, int], anchor: tuple[int, int], seed_patch: np.ndarray) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    anchor_r, anchor_c = map(int, anchor)
    seed_h, seed_w = map(int, seed_patch.shape)
    top = anchor_r - seed_h
    bottom = anchor_r + seed_h
    left = anchor_c - seed_w
    right = anchor_c + seed_w
    assert top >= 0 and left >= 0 and bottom < shape[0] and right < shape[1]

    out[top, left : right + 1] = FRAME_COLOR
    out[bottom, left : right + 1] = FRAME_COLOR
    out[top : bottom + 1, left] = FRAME_COLOR
    out[top : bottom + 1, right] = FRAME_COLOR
    out[top + 1 : bottom, left + 1 : right] = _tile_patch(seed_patch, 2 * seed_h - 1, 2 * seed_w - 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the uniquely frequent scatter color to locate a center, then build and fill a red frame from the separate seed pattern.

    Input structure:
    - The background is black, and most foreground pixels are isolated single-cell scatter markers.
    - These scatter markers use 2-3 colors; exactly one scatter color appears once, while the others appear 2-4 times.
    - One separate monochrome 4-connected seed pattern occupies a 2x2, 2x3, or 3x2 bounding box and is not color-shared with the scatter markers.

    Rule:
    - Find the uniquely frequent scatter color and use its lone pixel as the anchor center.
    - Find the seed pattern's bounding box, let its height be H and width be W, then draw a red hollow frame of size (2H+1) by (2W+1) centered on the anchor.
    - Erase all input foreground pixels, tile the seed bounding-box contents (including black holes) from the frame interior's top-left corner across the interior, and keep the frame red.

    Key insight:
    - Three independent clues must be fused: frequency identifies where the result goes, the seed bbox determines the frame size, and the seed bbox contents determine the interior fill.
    """
    info = _analyze_input(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid).shape, info["anchor"], info["seed_patch"])


def _seed_masks(height: int, width: int) -> list[np.ndarray]:
    area = int(height * width)
    masks = []
    for bits in range(1, 1 << area):
        if bits.bit_count() < 3 or bits.bit_count() > min(6, area):
            continue
        mask = np.zeros((height, width), dtype=bool)
        for idx in range(area):
            if (bits >> idx) & 1:
                mask[idx // width, idx % width] = True
        if mask.all():
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if len(_components(mask)) != 1:
            continue
        masks.append(mask.astype(int))
    assert masks
    return masks


SEED_MASKS = {dims: _seed_masks(*dims) for dims in SEED_BBOX_OPTIONS}


def _expanded_cells(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set(coords)
    for r, c in list(coords):
        for nr, nc in _neighbors8(r, c):
            expanded.add((int(nr), int(nc)))
    return expanded


def _frame_cells(anchor: tuple[int, int], seed_shape: tuple[int, int]) -> set[tuple[int, int]]:
    ar, ac = map(int, anchor)
    seed_h, seed_w = map(int, seed_shape)
    top = ar - seed_h
    bottom = ar + seed_h
    left = ac - seed_w
    right = ac + seed_w
    cells = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def _sample_scatter_positions(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    anchor: tuple[int, int, int],
    counts_by_color: dict[int, int],
    blocked: set[tuple[int, int]],
    frame_box: tuple[int, int, int, int],
) -> list[tuple[int, int, int]] | None:
    anchor_r, anchor_c, anchor_color = map(int, anchor)
    taken = set(blocked)
    if (anchor_r, anchor_c) in taken:
        return None
    if any((nr, nc) in taken for nr, nc in _neighbors8(anchor_r, anchor_c)):
        return None
    taken.add((anchor_r, anchor_c))
    scatter: list[tuple[int, int, int]] = [(anchor_r, anchor_c, anchor_color)]
    top, left, bottom, right = frame_box
    prefer_inside = bool(rng.integers(0, 2))

    for color, count in counts_by_color.items():
        for index in range(count):
            candidates = []
            for r in range(rows):
                for c in range(cols):
                    if (r, c) in taken:
                        continue
                    if any((nr, nc) in taken for nr, nc in _neighbors8(r, c)):
                        continue
                    candidates.append((r, c))
            if not candidates:
                return None

            if prefer_inside and len(scatter) == 1 and index == 0:
                inside = [
                    cell
                    for cell in candidates
                    if top + 1 <= cell[0] <= bottom - 1 and left + 1 <= cell[1] <= right - 1
                ]
                if inside:
                    candidates = inside
            choice = candidates[int(rng.integers(0, len(candidates)))]
            scatter.append((int(choice[0]), int(choice[1]), int(color)))
            taken.add(choice)
    return scatter


def _validate_case(
    grid: np.ndarray,
    *,
    anchor: tuple[int, int],
    seed_bbox: tuple[int, int, int, int],
    seed_patch: np.ndarray,
    scatter: list[tuple[int, int, int]],
) -> bool:
    try:
        info = _analyze_input(grid)
    except AssertionError:
        return False

    if info["anchor"] != anchor:
        return False
    if info["seed_bbox"] != seed_bbox:
        return False
    if not np.array_equal(info["seed_patch"], seed_patch):
        return False

    out = main(grid)
    expected = _render_output(grid.shape, anchor, seed_patch)
    if not np.array_equal(out, expected):
        return False

    for r, c, _color in scatter:
        if out[r, c] != BACKGROUND:
            return False

    sr0, sc0, sr1, sc1 = seed_bbox
    if np.any(out[sr0 : sr1 + 1, sc0 : sc1 + 1] != BACKGROUND):
        return False

    seed_h, seed_w = seed_patch.shape
    top = anchor[0] - seed_h
    left = anchor[1] - seed_w
    bottom = anchor[0] + seed_h
    right = anchor[1] + seed_w
    interior = out[top + 1 : bottom, left + 1 : right]
    if not np.array_equal(interior, _tile_patch(seed_patch, 2 * seed_h - 1, 2 * seed_w - 1)):
        return False

    border = _frame_cells(anchor, seed_patch.shape)
    red_count = int(np.count_nonzero(out == FRAME_COLOR))
    if red_count != len(border):
        return False
    if any(out[r, c] != FRAME_COLOR for r, c in border):
        return False

    return True


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    rows = int(rng.integers(15, 21))
    cols = int(rng.integers(15, 21))

    seed_h, seed_w = SEED_BBOX_OPTIONS[int(rng.integers(0, len(SEED_BBOX_OPTIONS)))]
    seed_mask = SEED_MASKS[(seed_h, seed_w)][int(rng.integers(0, len(SEED_MASKS[(seed_h, seed_w)])))]

    n_scatter_colors = 2 if rng.random() < 0.5 else 3
    palette = list(rng.choice(INPUT_COLORS, size=n_scatter_colors + 1, replace=False))
    seed_color = int(palette[0])
    scatter_colors = [int(color) for color in palette[1:]]
    anchor_color = scatter_colors[int(rng.integers(0, len(scatter_colors)))]

    seed_patch = (seed_mask * seed_color).astype(int)
    anchor_margin = 4
    anchor_r = int(rng.integers(anchor_margin, rows - anchor_margin))
    anchor_c = int(rng.integers(anchor_margin, cols - anchor_margin))
    anchor = (anchor_r, anchor_c)

    top = anchor_r - seed_h
    left = anchor_c - seed_w
    bottom = anchor_r + seed_h
    right = anchor_c + seed_w
    if top < 0 or left < 0 or bottom >= rows or right >= cols:
        return None

    frame_cells = _frame_cells(anchor, seed_patch.shape)
    forbidden = _expanded_cells(frame_cells)

    seed_positions = []
    for sr0 in range(rows - seed_h + 1):
        for sc0 in range(cols - seed_w + 1):
            seed_cells = {
                (sr0 + int(r), sc0 + int(c))
                for r, c in np.argwhere(seed_patch != BACKGROUND)
            }
            if seed_cells & forbidden:
                continue
            if any(any((nr, nc) in forbidden for nr, nc in _neighbors8(r, c)) for r, c in seed_cells):
                continue
            seed_positions.append((sr0, sc0))
    if not seed_positions:
        return None

    sr0, sc0 = seed_positions[int(rng.integers(0, len(seed_positions)))]
    seed_bbox = (sr0, sc0, sr0 + seed_h - 1, sc0 + seed_w - 1)
    seed_cells = {(sr0 + int(r), sc0 + int(c)) for r, c in np.argwhere(seed_patch != BACKGROUND)}
    occupied = _expanded_cells(seed_cells)

    counts_by_color = {}
    for color in scatter_colors:
        if color != anchor_color:
            counts_by_color[int(color)] = int(rng.integers(2, 5))

    scatter = _sample_scatter_positions(
        rng,
        rows,
        cols,
        (anchor_r, anchor_c, anchor_color),
        counts_by_color,
        occupied,
        (top, left, bottom, right),
    )
    if scatter is None:
        return None

    grid = np.zeros((rows, cols), dtype=int)
    for r, c, color in scatter:
        grid[r, c] = int(color)
    for r, c in np.argwhere(seed_patch != BACKGROUND):
        grid[sr0 + int(r), sc0 + int(c)] = seed_color

    if not _validate_case(grid, anchor=anchor, seed_bbox=seed_bbox, seed_patch=seed_patch, scatter=scatter):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
