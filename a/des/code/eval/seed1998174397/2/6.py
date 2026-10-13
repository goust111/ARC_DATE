import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PALETTE_COLORS = (1, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 300
SHAPE_ATTEMPTS = 200
PLACE_ATTEMPTS = 200


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _erode4(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    if h < 3 or w < 3:
        return out
    out[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return out


def _find_palette(grid: np.ndarray) -> tuple[int, list[int]]:
    row = np.asarray(grid[-1], dtype=int)
    w = row.shape[0]
    candidates: list[tuple[int, list[int]]] = []
    start = None
    for c in range(w + 1):
        value = int(row[c]) if c < w else BLACK
        if value != BLACK and start is None:
            start = c
        elif value == BLACK and start is not None:
            colors = row[start:c].tolist()
            if len(colors) == 5 and len(set(int(x) for x in colors)) == 5:
                candidates.append((start, [int(x) for x in colors]))
            start = None
    assert len(candidates) == 1, "expected exactly one 5-color palette on the bottom row"
    return candidates[0]


def _interior_cells(sub_grey_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = sub_grey_mask.shape
    inside: list[tuple[int, int]] = []
    for comp in _components(~sub_grey_mask):
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in comp):
            inside.extend(comp)
    return inside


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed grey boundary by looking up the count of red pixels inside it.

    The bottom row contains a 5-color palette. The Nth palette color corresponds to
    regions containing N red pixels. Region borders are recolored with that palette
    color, and the interior red pixels are cleared to black.
    """
    g = np.asarray(grid, dtype=int)
    _, palette = _find_palette(g)

    grey_mask = g == GREY
    pal_start, _ = _find_palette(g)
    grey_mask[-1, pal_start : pal_start + 5] = False

    out = g.copy()
    for comp in _components(grey_mask):
        r0, c0, r1, c1 = _bbox(comp)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        enclosed = _interior_cells(sub == GREY)
        if not enclosed:
            continue
        red_count = sum(1 for rr, cc in enclosed if int(sub[rr, cc]) == RED)
        assert 1 <= red_count <= 5, "each enclosed region must contain 1-5 red pixels"
        fill_color = int(palette[red_count - 1])
        for r, c in comp:
            out[r, c] = fill_color
        for rr, cc in enclosed:
            if int(sub[rr, cc]) == RED:
                out[r0 + rr, c0 + cc] = BLACK
    return out


def _shape_masks(rng: random.Random, red_count: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = rng.randint(6, 10)
        w = rng.randint(6, 11)
        mask = np.ones((h, w), dtype=bool)

        notch_count = rng.randint(1, 5)
        for _ in range(notch_count):
            trial = mask.copy()
            side = rng.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                if h <= 3 or w <= 3:
                    continue
                depth = rng.randint(1, min(2, h - 2))
                span = rng.randint(1, w - 2)
                start = rng.randint(0, w - span)
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                trial[rows, start : start + span] = False
            else:
                if h <= 3 or w <= 3:
                    continue
                depth = rng.randint(1, min(2, w - 2))
                span = rng.randint(1, h - 2)
                start = rng.randint(0, h - span)
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                trial[start : start + span, cols] = False
            if trial.sum() >= max(18, red_count + 10):
                mask = trial

        if rng.random() < 0.5:
            mask = np.flipud(mask)
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.35:
            mask = mask.T

        if len(_components(mask)) != 1:
            continue

        interior = _erode4(mask)
        if int(interior.sum()) < red_count + 1:
            continue
        if len(_components(interior)) != 1:
            continue

        boundary = mask & ~interior
        if len(_components(boundary)) != 1:
            continue
        return boundary, interior
    return None


def _pick_red_cells(
    rng: random.Random, interior_cells: list[tuple[int, int]], count: int
) -> list[tuple[int, int]]:
    shuffled = interior_cells[:]
    rng.shuffle(shuffled)
    chosen: list[tuple[int, int]] = []
    for cell in shuffled:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return rng.sample(interior_cells, count)


def _expected_output(
    grid: np.ndarray,
    palette: list[int],
    regions: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    for region in regions:
        count = int(region["count"])
        color = int(palette[count - 1])
        for r, c in region["boundary"]:
            out[r, c] = color
        for r, c in region["red"]:
            out[r, c] = BLACK
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(13, 22)
    w = rng.randint(16, 28)
    region_count = rng.randint(2, 5)

    grid = np.zeros((h, w), dtype=int)
    palette = rng.sample(list(PALETTE_COLORS), 5)
    pal_start = rng.randint(0, w - 5)
    grid[-1, pal_start : pal_start + 5] = palette

    counts = list(range(1, 6))
    rng.shuffle(counts)
    counts = counts[:region_count]

    region_specs = []
    for count in counts:
        shape = _shape_masks(rng, count)
        if shape is None:
            return None
        boundary, interior = shape
        region_specs.append(
            {
                "count": count,
                "boundary_mask": boundary,
                "interior_mask": interior,
                "area": int(boundary.size),
            }
        )
    region_specs.sort(key=lambda spec: int(spec["area"]), reverse=True)

    occupied = np.zeros((h, w), dtype=bool)
    occupied[h - 2 :, :] = True
    placed_regions: list[dict[str, object]] = []

    for spec in region_specs:
        boundary = np.asarray(spec["boundary_mask"], dtype=bool)
        interior = np.asarray(spec["interior_mask"], dtype=bool)
        count = int(spec["count"])
        sh, sw = boundary.shape
        if h - sh - 2 < 0 or w - sw < 0:
            return None

        placed = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = rng.randint(0, h - sh - 2)
            c0 = rng.randint(0, w - sw)

            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h, r0 + sh + 1)
            pc1 = min(w, c0 + sw + 1)
            if occupied[pr0:pr1, pc0:pc1].any():
                continue

            boundary_cells = [(r0 + r, c0 + c) for r, c in zip(*np.where(boundary))]
            interior_cells = [(r0 + r, c0 + c) for r, c in zip(*np.where(interior))]
            red_cells = _pick_red_cells(rng, interior_cells, count)

            for r, c in boundary_cells:
                grid[r, c] = GREY
            for r, c in red_cells:
                grid[r, c] = RED

            occupied[pr0:pr1, pc0:pc1] = True
            placed_regions.append(
                {
                    "count": count,
                    "boundary": boundary_cells,
                    "red": red_cells,
                }
            )
            placed = True
            break

        if not placed:
            return None

    if not placed_regions:
        return None

    expected = _expected_output(grid, palette, placed_regions)
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
