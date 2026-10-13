from __future__ import annotations

import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _outer_ring(block: np.ndarray) -> np.ndarray:
    if block.shape[0] == 1:
        return block.reshape(-1)
    return np.concatenate(
        (block[0, :], block[-1, :], block[1:-1, 0], block[1:-1, -1])
    )


def _decode_seed(block: np.ndarray, bg: int) -> tuple[int, int]:
    n, m = block.shape
    assert n == m and n % 2 == 1 and n >= 3, "seed sprite must be an odd square"
    assert not np.any(block == bg), "seed sprite must be fully filled"

    colors = list(map(int, np.unique(block)))
    assert len(colors) == 2, "seed sprite must use exactly two colors"

    border_vals = np.unique(_outer_ring(block))
    assert border_vals.size == 1, "seed sprite outer ring must identify the border color"
    border = int(border_vals[0])
    assert border != bg

    marker_colors = [color for color in colors if color != border]
    assert len(marker_colors) == 1, "seed sprite must have one marker color"
    marker = int(marker_colors[0])
    assert np.any(block == marker), "marker color must appear"

    return border, marker


def _paste_clipped(dst: np.ndarray, src: np.ndarray, top: int, left: int) -> None:
    h, w = dst.shape
    sh, sw = src.shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(h, top + sh)
    c1 = min(w, left + sw)
    if r0 >= r1 or c0 >= c1:
        return
    dst[r0:r1, c0:c1] = src[r0 - top : r1 - top, c0 - left : c1 - left]


def _expand_seed(
    out: np.ndarray,
    block: np.ndarray,
    marker: int,
    top: int,
    left: int,
) -> None:
    n = block.shape[0]
    center = n // 2
    for rr, cc in np.argwhere(block == marker):
        copy_top = top + (int(rr) - center) * n
        copy_left = left + (int(cc) - center) * n
        _paste_clipped(out, block, copy_top, copy_left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each seed sprite according to the spatial arrangement of its own marker pixels.

    Each foreground component is an odd-sized square sprite with a uniform outer ring
    (the border color) and a second color used as the marker/blueprint color.  Every
    marker pixel spawns a full sprite copy at a macro offset of one sprite-width per
    cell from the seed's center; border-color blueprint cells contribute only background.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components4(g != bg)
    assert comps, "expected at least one seed sprite"

    seeds: list[tuple[int, int, np.ndarray, int]] = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        assert block.shape[0] == block.shape[1], "foreground component must be square"
        assert len(comp) == block.size, "seed sprite must fill its bounding square"
        _border, marker = _decode_seed(block, bg)
        seeds.append((r0, c0, block.copy(), marker))

    out = np.full_like(g, bg)
    for r0, c0, block, marker in sorted(seeds, key=lambda item: (item[0], item[1])):
        _expand_seed(out, block, marker, r0, c0)
    return out


# --------------- generator ---------------

GEN_BG = 8
GEN_FOREGROUND = [0, 1, 2, 3, 4, 5, 6, 7, 9]
GEN_SEED_COUNT_CHOICES = [1, 2, 3]
GEN_SIZE_CHOICES = [5, 7]
GEN_MARKER_COUNT_RANGE = (2, 5)
GEN_MAX_ATTEMPTS = 400


def _grid_shape(rng: random.Random, count: int) -> tuple[int, int]:
    if count == 1:
        lo, hi = 18, 30
    elif count == 2:
        lo, hi = 22, 30
    else:
        lo, hi = 28, 30
    return rng.randint(lo, hi), rng.randint(lo, hi)


def _choose_sizes(rng: random.Random, count: int) -> list[int]:
    sizes: list[int] = []
    for _ in range(count):
        if count == 3:
            sizes.append(5)
        else:
            sizes.append(rng.choice(GEN_SIZE_CHOICES))
    return sizes


def _central_window(n: int) -> list[tuple[int, int]]:
    center = n // 2
    return [(center + dr, center + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _make_pattern(
    rng: random.Random, n: int, border: int, marker: int
) -> tuple[np.ndarray, list[tuple[int, int]], bool]:
    positions = _central_window(n)
    center = (n // 2, n // 2)
    include_center = rng.choice([False, True])
    selectable = [pos for pos in positions if include_center or pos != center]

    marker_total = rng.randint(GEN_MARKER_COUNT_RANGE[0], GEN_MARKER_COUNT_RANGE[1])
    if include_center:
        others = [pos for pos in selectable if pos != center]
        chosen = [center] + rng.sample(others, marker_total - 1)
    else:
        chosen = rng.sample(selectable, marker_total)

    block = np.full((n, n), border, dtype=int)
    for rr, cc in chosen:
        block[rr, cc] = marker
    return block, chosen, include_center


def _input_cells(top: int, left: int, n: int) -> set[tuple[int, int]]:
    return {(top + rr, left + cc) for rr in range(n) for cc in range(n)}


def _output_cells(
    block: np.ndarray,
    marker_positions: list[tuple[int, int]],
    top: int,
    left: int,
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    n = block.shape[0]
    center = n // 2
    cells: set[tuple[int, int]] = set()
    for rr, cc in marker_positions:
        copy_top = top + (rr - center) * n
        copy_left = left + (cc - center) * n
        r0 = max(0, copy_top)
        c0 = max(0, copy_left)
        r1 = min(h, copy_top + n)
        c1 = min(w, copy_left + n)
        for r in range(r0, r1):
            for c in range(c0, c1):
                cells.add((r, c))
    return cells


def _is_clipped(
    marker_positions: list[tuple[int, int]], n: int, top: int, left: int, h: int, w: int
) -> bool:
    center = n // 2
    for rr, cc in marker_positions:
        copy_top = top + (rr - center) * n
        copy_left = left + (cc - center) * n
        if copy_top < 0 or copy_left < 0 or copy_top + n > h or copy_left + n > w:
            return True
    return False


def _dilate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    grown: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    grown.add((nr, nc))
    return grown


def _candidate_positions(
    marker_positions: list[tuple[int, int]], n: int, h: int, w: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    clipped: list[tuple[int, int]] = []
    unclipped: list[tuple[int, int]] = []
    for top in range(h - n + 1):
        for left in range(w - n + 1):
            pos = (top, left)
            if _is_clipped(marker_positions, n, top, left, h, w):
                clipped.append(pos)
            else:
                unclipped.append(pos)
    return clipped, unclipped


def _place_specs(
    rng: random.Random, specs: list[dict[str, object]], h: int, w: int
) -> list[dict[str, object]] | None:
    reserved: set[tuple[int, int]] = set()
    placed: list[dict[str, object]] = []

    ordered_specs = sorted(
        specs,
        key=lambda spec: (-int(spec["n"]), -len(spec["marker_positions"])),
    )

    for spec in ordered_specs:
        n = int(spec["n"])
        block = np.asarray(spec["block"], dtype=int)
        marker_positions = list(spec["marker_positions"])
        want_clip = bool(spec["want_clip"])
        clipped, unclipped = _candidate_positions(marker_positions, n, h, w)

        pools: list[list[tuple[int, int]]] = []
        if want_clip and clipped:
            pools = [clipped, unclipped]
        elif not want_clip and unclipped:
            pools = [unclipped, clipped]
        else:
            pools = [clipped or unclipped]

        placed_here = False
        for pool in pools:
            order = pool[:]
            rng.shuffle(order)
            for top, left in order:
                in_cells = _input_cells(top, left, n)
                out_cells = _output_cells(block, marker_positions, top, left, h, w)
                footprint = in_cells | out_cells
                if not footprint or footprint & reserved:
                    continue
                placed_spec = dict(spec)
                placed_spec["top"] = top
                placed_spec["left"] = left
                placed.append(placed_spec)
                reserved |= _dilate(footprint, h, w)
                placed_here = True
                break
            if placed_here:
                break
        if not placed_here:
            return None

    return placed


def _build_grid(specs: list[dict[str, object]], h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), GEN_BG, dtype=int)
    for spec in specs:
        block = np.asarray(spec["block"], dtype=int)
        top = int(spec["top"])
        left = int(spec["left"])
        n = block.shape[0]
        grid[top : top + n, left : left + n] = block
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice(GEN_SEED_COUNT_CHOICES)
    h, w = _grid_shape(rng, count)

    colors = GEN_FOREGROUND[:]
    rng.shuffle(colors)
    needed_colors = 2 * count
    if needed_colors > len(colors):
        return None

    sizes = _choose_sizes(rng, count)
    specs: list[dict[str, object]] = []
    for idx in range(count):
        border = colors[2 * idx]
        marker = colors[2 * idx + 1]
        n = sizes[idx]
        block, marker_positions, include_center = _make_pattern(rng, n, border, marker)
        specs.append(
            {
                "n": n,
                "block": block,
                "marker_positions": marker_positions,
                "want_clip": rng.choice([False, True]),
                "include_center": include_center,
            }
        )

    placed = _place_specs(rng, specs, h, w)
    if placed is None:
        return None

    grid = _build_grid(placed, h, w)

    comps = _components4(grid != GEN_BG)
    if len(comps) != count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
