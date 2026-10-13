from __future__ import annotations

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
MAROON = 9
TILE_COLORS = (1, 2, 4, 6, 7, 8)
MASKS = [(l, u, r) for l in (0, 1) for u in (0, 1) for r in (0, 1)]
CORNER_BY_INDEX = ((0, 0), (0, 2), (2, 0), (2, 2), (0, 0), (0, 2), (2, 0), (2, 2))
GEN_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_divider_col(grid: np.ndarray, bg: int) -> int:
    preferred = np.flatnonzero(np.all(grid == MAROON, axis=0))
    if preferred.size == 1:
        return int(preferred[0])

    candidates = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) != bg and np.all(col == col[0]):
            candidates.append(c)
    assert len(candidates) == 1, f"expected one full-height divider, got {candidates}"
    return int(candidates[0])


def _extract_templates(left: np.ndarray, bg: int) -> dict[tuple[int, int, int], np.ndarray]:
    h, w = left.shape
    templates: dict[tuple[int, int, int], np.ndarray] = {}
    for r in range(3, h):
        for c in range(w - 2):
            label = left[r, c : c + 3]
            if not np.all(np.isin(label, (GREEN, GREY))):
                continue
            if c > 0 and int(left[r, c - 1]) != bg:
                continue
            if c + 3 < w and int(left[r, c + 3]) != bg:
                continue
            tile = left[r - 3 : r, c : c + 3].copy()
            if not np.any(tile != bg):
                continue
            mask = tuple(int(v == GREEN) for v in label.tolist())
            assert mask not in templates, f"duplicate template for mask {mask}"
            templates[mask] = tile
    assert templates, "failed to parse any labeled templates"
    return templates


def _sorted_seed_positions(canvas: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(canvas == GREEN)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return [tuple(map(int, coords[i])) for i in order]


def _neighbor_mask(canvas: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int, int]:
    w = canvas.shape[1]
    return (
        int(c - 1 >= 0 and int(canvas[r, c - 1]) != bg),
        int(r - 1 >= 0 and int(canvas[r - 1, c]) != bg),
        int(c + 1 < w and int(canvas[r, c + 1]) != bg),
    )


def _stamp_tile(canvas: np.ndarray, tile: np.ndarray, center: tuple[int, int], bg: int) -> None:
    h, w = canvas.shape
    r0, c0 = center
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            rr = r0 + dr
            cc = c0 + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            value = int(tile[dr + 1, dc + 1])
            if value != bg:
                canvas[rr, cc] = value


def _solve_canvas(canvas: np.ndarray, templates: dict[tuple[int, int, int], np.ndarray], bg: int) -> np.ndarray:
    out = canvas.copy()
    for seed in _sorted_seed_positions(canvas):
        mask = _neighbor_mask(out, seed[0], seed[1], bg)
        assert mask in templates, f"missing template for mask {mask}"
        _stamp_tile(out, templates[mask], seed, bg)
    return out


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    divider_col = _find_divider_col(g, bg)
    templates = _extract_templates(g[:, :divider_col], bg)
    return _solve_canvas(g[:, divider_col + 1 :], templates, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the labeled 3x3 texture that matches each seed's local neighborhood and grow the picture step by step.

    Input structure:
    - The background is black.
    - A full-height maroon divider splits the grid into a left template area and a right main canvas.
    - In the template area, each example consists of a 3x3 colored texture with an attached 1x3 label row underneath.
    - In each label row, green marks mean "left neighbor exists", "upper neighbor exists", and "right neighbor exists"; grey marks absent directions.
    - The main canvas contains isolated green seed pixels on background.

    Rule:
    - Read the green seeds on the main canvas in row-major order.
    - For the current seed, test whether the left, upper, and right adjacent cells are already non-background on the current canvas.
    - Match that three-bit neighborhood pattern to the unique label row in the template area.
    - Stamp the corresponding 3x3 texture centered on the seed; only the texture's non-background pixels are drawn, so earlier pixels can remain visible.
    - Each stamped texture immediately affects later seeds.
    - Return only the right-side main canvas, excluding the divider and the template area.

    Key insight:
    - The label row is a direction code for the seed's visible neighborhood, and the seeds must be processed in a fixed order because earlier growth changes later neighborhood tests.
    """
    return _solve_grid(np.asarray(grid, dtype=int))


def _build_tile(mask: tuple[int, int, int], rng: np.random.Generator, idx: int) -> np.ndarray:
    base = int(rng.choice(np.array(TILE_COLORS, dtype=int)))
    accent_pool = np.array([c for c in TILE_COLORS if c != base], dtype=int)
    accent = int(rng.choice(accent_pool))

    tile = np.full((3, 3), base, dtype=int)
    tile[1, 1] = accent
    tile[2, 1] = accent
    if mask[0]:
        tile[1, 0] = BLACK
    if mask[1]:
        tile[0, 1] = BLACK
    if mask[2]:
        tile[1, 2] = BLACK

    cr, cc = CORNER_BY_INDEX[idx]
    tile[cr, cc] = accent
    if sum(mask) >= 2:
        tile[2, cc] = accent
    return tile


def _build_templates(rng: np.random.Generator) -> dict[tuple[int, int, int], np.ndarray]:
    order = MASKS.copy()
    rng.shuffle(order)
    templates = {}
    for idx, mask in enumerate(order):
        templates[mask] = _build_tile(mask, rng, idx)
    return templates


def _sample_row_positions(rng: np.random.Generator, height: int) -> list[int]:
    base = np.array([0, 1, 1, 1, 0], dtype=int)
    extra = int(height - 19)
    if extra > 0:
        base += rng.multinomial(extra, [0.2] * 5)
    rows = [int(base[0])]
    for gap in base[1:4]:
        rows.append(rows[-1] + 4 + int(gap))
    return rows


def _sample_col_positions(rng: np.random.Generator, left_width: int) -> list[int]:
    base = np.array([0, 1, 0], dtype=int)
    extra = int(left_width - 7)
    if extra > 0:
        base += rng.multinomial(extra, [1 / 3] * 3)
    c0 = int(base[0])
    c1 = c0 + 3 + int(base[1])
    return [c0, c1]


def _grow_seed_component(
    rng: np.random.Generator,
    gh: int,
    gw: int,
    target: int,
) -> set[tuple[int, int]]:
    cells = {(int(rng.integers(gh)), int(rng.integers(gw)))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < gh and 0 <= nc < gw and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if frontier and (rng.random() < 0.85 or len(cells) < 4):
            cells.add(frontier[int(rng.integers(len(frontier)))])
            continue
        empties = [(r, c) for r in range(gh) for c in range(gw) if (r, c) not in cells]
        if not empties:
            break
        cells.add(empties[int(rng.integers(len(empties)))])
    return cells


def _build_seed_canvas(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    gh = int(rng.integers(4, 8))
    gw = int(rng.integers(4, 8))
    target = int(rng.integers(6, min(15, gh * gw) + 1))
    cells = _grow_seed_component(rng, gh, gw, target)
    top = int(rng.integers(1, height - gh))
    left = int(rng.integers(1, width - gw))

    canvas = np.full((height, width), BLACK, dtype=int)
    for r, c in cells:
        canvas[top + r, left + c] = GREEN
    return canvas


def _seed_only_masks(canvas: np.ndarray) -> list[tuple[int, int, int]]:
    seeds = _sorted_seed_positions(canvas)
    seed_set = set(seeds)
    return [
        (
            int((r, c - 1) in seed_set),
            int((r - 1, c) in seed_set),
            int((r, c + 1) in seed_set),
        )
        for r, c in seeds
    ]


def _quality_ok(
    seed_canvas: np.ndarray,
    output: np.ndarray,
    templates: dict[tuple[int, int, int], np.ndarray],
) -> bool:
    seeds = _sorted_seed_positions(seed_canvas)
    if len(seeds) < 6:
        return False

    masks_seen = []
    canvas = seed_canvas.copy()
    for seed in seeds:
        mask = _neighbor_mask(canvas, seed[0], seed[1], BLACK)
        masks_seen.append(mask)
        _stamp_tile(canvas, templates[mask], seed, BLACK)
    if not np.array_equal(canvas, output):
        return False

    distinct = set(masks_seen)
    if len(distinct) < 4:
        return False

    bits = np.array(masks_seen, dtype=int)
    for i in range(3):
        if int(bits[:, i].min()) == int(bits[:, i].max()):
            return False

    initial = _seed_only_masks(seed_canvas)
    if not any(a != b for a, b in zip(masks_seen, initial)):
        return False

    if np.count_nonzero(output != BLACK) > output.size * 0.65:
        return False
    return True


def _build_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    height = int(rng.integers(20, 27))
    left_width = int(rng.integers(7, 11))
    right_width = int(rng.integers(10, 30 - left_width))

    templates = _build_templates(rng)
    seed_canvas = _build_seed_canvas(rng, height, right_width)
    output = _solve_canvas(seed_canvas, templates, BLACK)
    if not _quality_ok(seed_canvas, output, templates):
        return None

    grid = np.full((height, left_width + 1 + right_width), BLACK, dtype=int)
    rows = _sample_row_positions(rng, height)
    cols = _sample_col_positions(rng, left_width)
    panel_masks = MASKS.copy()
    rng.shuffle(panel_masks)
    idx = 0
    for r in rows:
        for c in cols:
            mask = panel_masks[idx]
            grid[r : r + 3, c : c + 3] = templates[mask]
            grid[r + 3, c : c + 3] = np.array([GREEN if bit else GREY for bit in mask], dtype=int)
            idx += 1

    divider_col = left_width
    grid[:, divider_col] = MAROON
    grid[:, divider_col + 1 :] = seed_canvas

    if not np.array_equal(_solve_grid(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
