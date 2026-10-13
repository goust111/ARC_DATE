from __future__ import annotations

import random

import numpy as np


STAMP_COLOR = 1
GRID_MIN = 10
GRID_MAX = 30
STAMP_COUNT_OPTIONS = (1, 2, 3, 4)
STAMP_H_RANGE = (3, 6)
STAMP_W_RANGE = (3, 6)
STAMP_FILL_RATIO = (0.40, 0.75)
MIN_STAMP_AREA = 5
MASK_ATTEMPTS = 120
GENERATION_ATTEMPTS = 300
OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in OFFSETS_4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


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
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _stamp_info(
    grid: np.ndarray, bg: int, cells: list[tuple[int, int]]
) -> tuple[tuple[int, int, int, int], np.ndarray, int]:
    r0, c0, r1, c1 = _bbox(cells)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = np.zeros_like(patch, dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True

    colors, counts = np.unique(patch[mask], return_counts=True)
    assert int(mask.sum()) >= 3, "each stamp must have at least three occupied cells"
    assert not np.any(patch[mask] == bg), "stamp cells must be non-background"

    base_colors = [int(color) for color, count in zip(colors, counts, strict=False) if int(count) > 1]
    ink_colors = [int(color) for color, count in zip(colors, counts, strict=False) if int(count) == 1]
    assert len(base_colors) == 1, "each stamp must have exactly one repeated body color"
    assert len(ink_colors) == 1, "each stamp must have exactly one single-pixel ink color"

    return (r0, c0, r1, c1), mask, ink_colors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each stamped shape downward and repaint the mirrored copy with its embedded ink color.

    Input structure:
    - A dominant background contains several 4-connected stamp components made of one repeated body color plus one embedded single-pixel ink color.
    - Each stamp occupies one tight bounding box, and different stamps' bounding boxes do not overlap horizontally.
    - The rectangle directly below every stamp bounding box is empty background and is tall enough to hold a vertically flipped copy of the stamp shape.

    Rule:
    - Keep the original stamps unchanged.
    - For each non-background connected component, extract its full occupied mask inside its bounding box, including the embedded ink pixel.
    - Flip that occupied mask vertically and place it immediately below the original bounding box with the same column alignment.
    - Paint every occupied cell of the mirrored copy with the stamp's ink color.

    Key insight:
    - The shape to copy comes from the whole stamp geometry, while the color of the copy comes only from the unique non-body pixel embedded inside that stamp.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _dominant_color(g)

    components = _components(g != bg)
    assert components, "expected at least one stamp"

    stamp_records = []
    for cells in components:
        bbox, mask, ink = _stamp_info(g, bg, cells)
        stamp_records.append((bbox, mask, ink))

    stamp_records.sort(key=lambda item: item[0][1])
    for i in range(len(stamp_records) - 1):
        _, c0, _, c1 = stamp_records[i][0]
        _, next_c0, _, _ = stamp_records[i + 1][0]
        assert next_c0 > c1, "stamp bounding boxes must not overlap horizontally"

    out = g.copy()
    for (r0, c0, r1, c1), mask, ink in stamp_records:
        stamp_h = r1 - r0 + 1
        stamp_w = c1 - c0 + 1
        dest_r0 = r1 + 1
        dest_r1 = dest_r0 + stamp_h

        assert dest_r1 <= h, "mirrored stamp must stay inside the grid"
        assert stamp_w == mask.shape[1]
        assert np.all(g[dest_r0:dest_r1, c0 : c1 + 1] == bg), "stamping zone must start blank"

        mirrored = np.flipud(mask)
        dest = out[dest_r0:dest_r1, c0 : c1 + 1].copy()
        dest[mirrored] = ink
        out[dest_r0:dest_r1, c0 : c1 + 1] = dest

    return out


def _split_nonnegative(rng: random.Random, total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[rng.randrange(parts)] += 1
    return values


def _dot_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            test = mask.copy()
            test[r, c] = False
            if len(_components(test)) == 1:
                candidates.append((r, c))
    return candidates


def _make_stamp_mask(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(MASK_ATTEMPTS):
        box_h = rng.randint(*STAMP_H_RANGE)
        box_w = rng.randint(*STAMP_W_RANGE)
        area_lo = max(MIN_STAMP_AREA, int(np.ceil(box_h * box_w * STAMP_FILL_RATIO[0])))
        area_hi = min(box_h * box_w - 1, int(np.floor(box_h * box_w * STAMP_FILL_RATIO[1])))
        if area_hi < area_lo:
            continue

        target = rng.randint(area_lo, area_hi)
        seed = (rng.randrange(box_h), rng.randrange(box_w))
        cells = {seed}
        frontier = set(_neighbors4(seed[0], seed[1], box_h, box_w))
        while frontier and len(cells) < target:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], box_h, box_w):
                if nb not in cells:
                    frontier.add(nb)

        if len(cells) != target:
            continue

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows, cols = np.where(mask)
        mask = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        if mask.shape[0] < STAMP_H_RANGE[0] or mask.shape[1] < STAMP_W_RANGE[0]:
            continue
        if np.all(mask):
            continue
        if not np.any(mask[1:-1, 1:-1]):
            continue

        candidates = _dot_candidates(mask)
        if candidates:
            return mask, candidates
    return None


def _expected_output(
    inp: np.ndarray,
    stamps: list[dict[str, int | np.ndarray | tuple[int, int]]],
) -> np.ndarray:
    out = inp.copy()
    for stamp in stamps:
        r0 = int(stamp["r0"])
        c0 = int(stamp["c0"])
        ink = int(stamp["ink"])
        mask = np.asarray(stamp["mask"], dtype=bool)
        stamp_h, stamp_w = mask.shape
        dest = out[r0 + stamp_h : r0 + 2 * stamp_h, c0 : c0 + stamp_w].copy()
        dest[np.flipud(mask)] = ink
        out[r0 + stamp_h : r0 + 2 * stamp_h, c0 : c0 + stamp_w] = dest
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg_choices = [color for color in range(10) if color != STAMP_COLOR]
    bg = rng.choice(bg_choices)
    ink_pool = [color for color in range(10) if color not in (bg, STAMP_COLOR)]

    stamp_count = rng.choice(STAMP_COUNT_OPTIONS)
    shapes: list[tuple[np.ndarray, tuple[int, int]]] = []
    widths: list[int] = []
    heights: list[int] = []
    for _ in range(stamp_count):
        made = _make_stamp_mask(rng)
        if made is None:
            return None
        mask, candidates = made
        dot = rng.choice(candidates)
        shapes.append((mask, dot))
        heights.append(int(mask.shape[0]))
        widths.append(int(mask.shape[1]))

    min_w = max(GRID_MIN, sum(widths) + max(0, stamp_count - 1))
    min_h = max(GRID_MIN, max(2 * height for height in heights))
    if min_w > GRID_MAX or min_h > GRID_MAX:
        return None

    grid_w = rng.randint(min_w, min(GRID_MAX, min_w + 8))
    grid_h = rng.randint(min_h, min(GRID_MAX, min_h + 8))

    extra_space = grid_w - (sum(widths) + max(0, stamp_count - 1))
    margins = _split_nonnegative(rng, extra_space, stamp_count + 1)
    cols: list[int] = []
    cursor = margins[0]
    for i, width in enumerate(widths):
        cols.append(cursor)
        cursor += width
        if i < stamp_count - 1:
            cursor += 1 + margins[i + 1]
    assert cursor + margins[-1] == grid_w

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    used_inks: list[int] = []
    stamps: list[dict[str, int | np.ndarray | tuple[int, int]]] = []
    for i, ((mask, dot), c0) in enumerate(zip(shapes, cols, strict=False)):
        stamp_h, stamp_w = mask.shape
        r0 = rng.randint(0, grid_h - 2 * stamp_h)
        if used_inks and rng.random() < 0.35:
            ink = rng.choice(used_inks)
        else:
            ink = rng.choice(ink_pool)
        used_inks.append(ink)

        patch = np.full((stamp_h, stamp_w), bg, dtype=int)
        patch[mask] = STAMP_COLOR
        patch[dot] = ink

        target = inp[r0 : r0 + stamp_h, c0 : c0 + stamp_w]
        if np.any(target != bg):
            return None
        if np.any(inp[r0 + stamp_h : r0 + 2 * stamp_h, c0 : c0 + stamp_w] != bg):
            return None

        inp[r0 : r0 + stamp_h, c0 : c0 + stamp_w] = patch
        stamps.append({"r0": r0, "c0": c0, "mask": mask, "ink": ink})

    values, counts = np.unique(inp, return_counts=True)
    if int(values[int(np.argmax(counts))]) != bg:
        return None

    expected = _expected_output(inp, stamps)
    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-and-ink input")
