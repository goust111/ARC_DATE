# Variant of 1818057f
# Dimension: D6 — '+'字的嵌入约束 (meta)
# Change: '+'字总是嵌入在更大的前景连通分量中（size 6-61），从不以孤立5像素十字出现... → 嵌入深度编码目标颜色

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis
# a) Concrete behavior:
#    The input still shows one foreground color on a red background, and the local
#    '+' test is unchanged: a center pixel with 4 orthogonal foreground neighbors.
#    The visible change happens only in the output color. Instead of repainting every
#    detected '+' to one universal teal, the solver must first measure the 4-connected
#    foreground component containing that '+'. Components with size < 20 map to green,
#    sizes 20..39 map to teal, and sizes >= 40 map to orange.
#    Example trace: in the source task, a 5-pixel '+' with a 9-cell tail and a 5-pixel
#    '+' with a 43-cell tail both became teal. In this variant they are still detected
#    by the same local predicate, but the first one becomes green and the second becomes
#    orange because their enclosing components have different total sizes.
# b) Complexity calibration:
#    The rule uses only three size bands so it stays learnable from a few examples, but
#    each generated grid is forced to contain all three bands so the color split is
#    actually observable. The generator keeps the components one-color in the input and
#    varies only their overall extent, which gives a human-visible clue for the mapping.
# c) Hidden assumptions in the original main():
#    - Picking the smallest non-bg color as `fg` was harmless because the input used
#      exactly one foreground color. That remains true here; the generator preserves it.
#    - Writing literal 8 to every matched '+' was safe only because the output color was
#      constant. In the variant this must become a size-dependent lookup.
#    - The original scan could have recolored overlapping '+' footprints multiple times.
#      That is still harmless only if overlaps do not create conflicting colors; the
#      generator therefore keeps one '+' per component and separates components.
# d) Placement legality:
#    Each target component is a '+' cross plus a compact one-cell-wide tail. The tail is
#    built as a self-avoiding path, so the component stays 4-connected, never goes out of
#    bounds, and does not create accidental extra '+' centers. Placement requires a
#    1-cell orthogonal moat between components so component sizes remain independent.
# e) Visual observability:
#    Color choice depends only on visible component size, never on color ids or hidden
#    ordering. Every sample includes small, medium, and large enclosing components, so
#    the solver's three-way branching can be inferred from 2-4 examples.
# f) Visual cleanliness:
#    Large components are compact snakes rather than noisy blobs, and unrelated distractor
#    pieces are tiny separated paths. That keeps each '+' and its surrounding component
#    easy to parse by eye.
# g) Color inferability:
#    Only three output colors are introduced: green for small components, teal for medium
#    components, and orange for large components. The component-size cue repeats within
#    and across samples, so the mapping is supported structurally instead of arbitrarily.
#
# Solver assumption audit:
# - `main()` assumes exactly one non-bg color in the input; `generate_input()` fixes bg=2
#   and fg=4 and never introduces extra input colors.
# - `main()` measures component size on the unrecolored foreground mask; the generator
#   keeps components disconnected by at least one orthogonal background cell.
# - `main()` detects '+' motifs from the original grid rather than incrementally from the
#   output; the generator keeps exactly one '+' center per target component.


SMALL_COLOR = 3   # green
MEDIUM_COLOR = 8  # teal
LARGE_COLOR = 7   # orange
ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
PLUS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def _color_for_size(size: int) -> int:
    if size < 20:
        return SMALL_COLOR
    if size < 40:
        return MEDIUM_COLOR
    return LARGE_COLOR


def _band_for_size(size: int) -> str:
    if size < 20:
        return "small"
    if size < 40:
        return "medium"
    return "large"


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    H, W = mask.shape
    comp_ids = np.full((H, W), -1, dtype=int)
    sizes: list[int] = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or comp_ids[r, c] != -1:
                continue
            cid = len(sizes)
            stack = [(r, c)]
            comp_ids[r, c] = cid
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                for dr, dc in ORTHO:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and comp_ids[nr, nc] == -1:
                        comp_ids[nr, nc] = cid
                        stack.append((nr, nc))
            sizes.append(size)
    return comp_ids, sizes


def _plus_centers_in_grid(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO):
                centers.append((r, c))
    return centers


def _plus_centers_in_coords(coords: set[tuple[int, int]]) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for r, c in coords:
        if all((r + dr, c + dc) in coords for dr, dc in ORTHO):
            centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect foreground '+' motifs and recolor each one by the size of its enclosing component.

    Input structure:
    - bg is the most common color.
    - One foreground color forms several 4-connected components on the bg.
    - Target '+' shapes are embedded inside larger foreground components rather than isolated.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Measure the size of the 4-connected foreground component containing that center.
    - Recolor the 5 pixels of that '+' to green when size < 20, teal when 20 <= size < 40,
      and orange when size >= 40.
    - All other pixels remain unchanged.

    Key insight: The local '+' detector is unchanged from the source task, but the output
    color now depends on a nonlocal property: the total size of the connected foreground
    region surrounding the detected motif.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    mask = g == fg
    comp_ids, comp_sizes = _label_components(mask)
    H, W = g.shape

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in ORTHO):
                color = _color_for_size(comp_sizes[int(comp_ids[r, c])])
                for dr, dc in PLUS:
                    out[r + dr, c + dc] = color
    return out


def _rotate_coords(coords: set[tuple[int, int]], turns: int) -> set[tuple[int, int]]:
    out = set(coords)
    for _ in range(turns % 4):
        out = {(c, -r) for r, c in out}
    return out


def _compact_path(length: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if length <= 0:
        return []

    coords = [(0, 0)]
    remaining = length - 1
    row, col = 0, 0
    direction = 1

    while remaining > 0:
        max_run = min(6, remaining)
        min_run = 1 if remaining == 1 else min(2, max_run)
        run = int(rng.integers(min_run, max_run + 1))
        for _ in range(run):
            col += direction
            coords.append((row, col))
        remaining -= run
        if remaining == 0:
            break

        connector = 2 if remaining >= 2 else 1
        for _ in range(connector):
            row += 1
            coords.append((row, col))
        remaining -= connector
        direction *= -1

    return coords


def _sample_size(rng: np.random.Generator, band: str) -> int:
    if band == "small":
        return int(rng.integers(8, 15))
    if band == "medium":
        return int(rng.integers(20, 29))
    if band == "large":
        return int(rng.integers(40, 49))
    raise ValueError(f"unknown band: {band}")


def _make_plus_component(target_size: int, rng: np.random.Generator) -> set[tuple[int, int]] | None:
    if target_size <= 5:
        return None

    for _ in range(40):
        coords = set(PLUS)
        for pr, pc in _compact_path(target_size - 5, rng):
            coords.add((pr, pc + 2))
        coords = _rotate_coords(coords, int(rng.integers(0, 4)))
        centers = _plus_centers_in_coords(coords)
        if centers == [(0, 0)]:
            return coords
    return None


def _make_distractor(length: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    coords = set(_compact_path(length, rng))
    return _rotate_coords(coords, int(rng.integers(0, 4)))


def _can_place_component(
    grid: np.ndarray,
    coords: set[tuple[int, int]],
    center_r: int,
    center_c: int,
    bg: int,
) -> bool:
    H, W = grid.shape
    global_cells = {(center_r + r, center_c + c) for r, c in coords}

    for gr, gc in global_cells:
        if not (0 <= gr < H and 0 <= gc < W):
            return False
        if int(grid[gr, gc]) != bg:
            return False

    for gr, gc in global_cells:
        for dr, dc in ORTHO:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in global_cells and int(grid[nr, nc]) != bg:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    coords: set[tuple[int, int]],
    value: int,
    bg: int,
    rng: np.random.Generator,
    *,
    center_needs_margin: bool,
) -> bool:
    H, W = grid.shape
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    low_r = -min(rows)
    high_r = H - 1 - max(rows)
    low_c = -min(cols)
    high_c = W - 1 - max(cols)

    if center_needs_margin:
        low_r = max(low_r, 1)
        high_r = min(high_r, H - 2)
        low_c = max(low_c, 1)
        high_c = min(high_c, W - 2)

    if low_r > high_r or low_c > high_c:
        return False

    origins = [(r, c) for r in range(low_r, high_r + 1) for c in range(low_c, high_c + 1)]
    rng.shuffle(origins)
    for center_r, center_c in origins:
        if not _can_place_component(grid, coords, center_r, center_c, bg):
            continue
        for dr, dc in coords:
            grid[center_r + dr, center_c + dc] = value
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(14, 21))
    W = int(rng.integers(14, 21))
    bg, fg = 2, 4
    area = H * W
    target_fg = int(area * rng.uniform(0.28, 0.36))

    sizes = [
        _sample_size(rng, "small"),
        _sample_size(rng, "medium"),
        _sample_size(rng, "large"),
    ]
    while sum(sizes) < target_fg and len(sizes) < 6:
        remaining = target_fg - sum(sizes)
        bands = ["small", "small", "medium", "medium"]
        if remaining >= 40 and area >= 260 and sum(size >= 40 for size in sizes) < 2:
            bands.append("large")
        sizes.append(_sample_size(rng, str(rng.choice(bands))))

    if sum(sizes) >= area // 2:
        return None

    components: list[tuple[int, set[tuple[int, int]]]] = []
    for size in sizes:
        coords = _make_plus_component(size, rng)
        if coords is None:
            return None
        components.append((size, coords))

    grid = np.full((H, W), bg, dtype=int)
    for _, coords in sorted(components, key=lambda item: item[0], reverse=True):
        if not _place_component(grid, coords, fg, bg, rng, center_needs_margin=True):
            return None

    distractor_budget = max(0, target_fg - int(np.sum(grid != bg)))
    distractor_tries = 0
    while distractor_budget >= 2 and distractor_tries < 6:
        distractor_tries += 1
        length = int(rng.integers(1, min(5, distractor_budget + 1)))
        coords = _make_distractor(length, rng)
        if _plus_centers_in_coords(coords):
            continue
        if _place_component(grid, coords, fg, bg, rng, center_needs_margin=False):
            distractor_budget -= length

    inp = grid.copy()
    out = main(inp)
    centers = _plus_centers_in_grid(inp, fg)
    if len(centers) < 3:
        return None

    mask = inp == fg
    comp_ids, comp_sizes = _label_components(mask)
    seen_bands = {_band_for_size(comp_sizes[int(comp_ids[r, c])]) for r, c in centers}
    if seen_bands != {"small", "medium", "large"}:
        return None

    added_colors = set(map(int, out.flatten())) - {bg, fg}
    if added_colors != {SMALL_COLOR, MEDIUM_COLOR, LARGE_COLOR}:
        return None

    if not np.array_equal(out, main(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
