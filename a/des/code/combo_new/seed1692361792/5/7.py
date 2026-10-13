import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
MAROON = 9

FRAME_TO_MODE = {
    RED: "lr",
    BLUE: "ud",
    GREEN: "rot",
}

DRAW_COLORS = (4, 5, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 400


def _same_color_components(grid: np.ndarray, colors: set[int]) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    color = int(grid[comp[0][0], comp[0][1]])
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected = 2 * (height + width) - 4
    if len(comp) != expected:
        return None

    for r in range(r0, r1 + 1):
        if int(grid[r, c0]) != color or int(grid[r, c1]) != color:
            return None
    for c in range(c0, c1 + 1):
        if int(grid[r0, c]) != color or int(grid[r1, c]) != color:
            return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    frames: list[tuple[int, int, int, int, int]] = []
    for comp in _same_color_components(grid, set(FRAME_TO_MODE)):
        bbox = _frame_bbox(grid, comp)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        frames.append((int(grid[comp[0][0], comp[0][1]]), r0, c0, r1, c1))
    frames.sort(key=lambda item: (item[1], item[2], item[3], item[4]))
    return frames


def _paired_view(arr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "lr":
        return arr[:, ::-1]
    if mode == "ud":
        return arr[::-1, :]
    if mode == "rot":
        return arr[::-1, ::-1]
    raise ValueError(mode)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair erased cells inside framed regions by using the symmetry encoded by each frame color.

    Input structure:
    - Background is black.
    - The grid contains several disjoint 1-pixel-thick rectangular frames whose border color is red, blue, or green.
    - Each frame encloses a sparse pattern drawn with one or two non-black colors, except that 1 to 3 cells have been erased to black.

    Rule:
    - Detect every hollow rectangular border and read its color as a symmetry instruction.
    - Red frames enforce left-right mirror symmetry across the interior vertical midline, blue frames enforce top-bottom mirror symmetry across the interior horizontal midline, and green frames enforce 180-degree rotational symmetry about the interior center.
    - Wherever an interior black cell has a non-black symmetric partner, restore that cell and mark the repaired position with maroon.

    Key insight: The border color is metadata: it does not belong to the pattern itself, but tells you which symmetry to apply to recover the missing pixels.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame_color, r0, c0, r1, c1 in _find_frames(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        paired = _paired_view(inner, FRAME_TO_MODE[frame_color])
        missing = (inner == BLACK) & (paired != BLACK)
        repaired = inner.copy()
        repaired[missing] = MAROON
        out[r0 + 1 : r1, c0 + 1 : c1] = repaired

    return out


def _transform(mode: str, r: int, c: int, h: int, w: int) -> tuple[int, int]:
    if mode == "lr":
        return r, w - 1 - c
    if mode == "ud":
        return h - 1 - r, c
    if mode == "rot":
        return h - 1 - r, w - 1 - c
    raise ValueError(mode)


def _orbits(h: int, w: int, mode: str) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rr, cc = _transform(mode, r, c, h, w)
            orbit = sorted({(r, c), (rr, cc)})
            for cell in orbit:
                seen.add(cell)
            out.append(orbit)
    return out


def _choose_fill_count(n_orbits: int, n_missing: int) -> int:
    low = min(n_orbits, max(3, n_missing + 2))
    high = min(n_orbits, max(low, int(np.ceil(0.7 * n_orbits))))
    return random.randint(low, high)


def _build_region(frame_color: int) -> tuple[np.ndarray, np.ndarray]:
    mode = FRAME_TO_MODE[frame_color]
    inner_h = random.randint(4, 8)
    inner_w = random.randint(4, 8)
    orbits = _orbits(inner_h, inner_w, mode)
    mutable_ids = [idx for idx, orbit in enumerate(orbits) if len(orbit) > 1]
    assert mutable_ids

    n_missing = random.randint(1, min(3, len(mutable_ids)))
    fill_count = _choose_fill_count(len(orbits), n_missing)

    repair_ids = random.sample(mutable_ids, n_missing)
    remaining_ids = [idx for idx in range(len(orbits)) if idx not in repair_ids]
    extra_ids = random.sample(remaining_ids, fill_count - n_missing)
    filled_ids = repair_ids + extra_ids
    random.shuffle(filled_ids)

    n_colors = random.randint(1, 2)
    colors = random.sample(DRAW_COLORS, n_colors)
    orbit_colors = [colors[idx % n_colors] for idx in range(len(filled_ids))]
    random.shuffle(orbit_colors)

    clean = np.zeros((inner_h, inner_w), dtype=int)
    for orbit_id, color in zip(filled_ids, orbit_colors):
        for r, c in orbits[orbit_id]:
            clean[r, c] = color

    inp_inner = clean.copy()
    out_inner = clean.copy()
    for orbit_id in repair_ids:
        erased_r, erased_c = random.choice(orbits[orbit_id])
        inp_inner[erased_r, erased_c] = BLACK
        out_inner[erased_r, erased_c] = MAROON

    region_h = inner_h + 2
    region_w = inner_w + 2
    inp_region = np.full((region_h, region_w), frame_color, dtype=int)
    out_region = np.full((region_h, region_w), frame_color, dtype=int)
    inp_region[1:-1, 1:-1] = inp_inner
    out_region[1:-1, 1:-1] = out_inner
    return inp_region, out_region


def _sample_canvas_size(n_regions: int) -> tuple[int, int]:
    lower = 20 if n_regions == 4 else 16
    return random.randint(lower, 30), random.randint(lower, 30)


def _place_regions(
    grid: np.ndarray,
    expected: np.ndarray,
    regions: list[tuple[np.ndarray, np.ndarray]],
) -> bool:
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)

    for inp_region, out_region in sorted(regions, key=lambda item: item[0].shape[0] * item[0].shape[1], reverse=True):
        rh, rw = inp_region.shape
        if rh + 2 > h or rw + 2 > w:
            return False

        placed = False
        for _ in range(120):
            r0 = random.randint(1, h - rh - 1)
            c0 = random.randint(1, w - rw - 1)
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + rh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + rw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            grid[r0 : r0 + rh, c0 : c0 + rw] = inp_region
            expected[r0 : r0 + rh, c0 : c0 + rw] = out_region
            occupied[r0 : r0 + rh, c0 : c0 + rw] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _generation_attempt() -> np.ndarray | None:
    n_regions = random.randint(3, 4)
    frame_colors = [RED, BLUE, GREEN]
    if n_regions == 4:
        frame_colors.append(random.choice((RED, BLUE, GREEN)))
    random.shuffle(frame_colors)

    regions = [_build_region(frame_color) for frame_color in frame_colors]
    h, w = _sample_canvas_size(n_regions)
    inp = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    if not _place_regions(inp, expected, regions):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid symmetry-repair input")
