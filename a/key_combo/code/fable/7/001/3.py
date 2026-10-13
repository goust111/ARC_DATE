from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_7_3", "/home/lixinyu/snap/new/solvers/arc_utils.py"
).load_module()

Color = _ARC_UTILS.Color
colors_present = _ARC_UTILS.colors_present
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds
components_mask = _ARC_UTILS.components_mask


DIVIDER_COLOR = Color.GREY
ANCHOR_COLOR = Color.YELLOW
OUTPUT_COLOR = Color.PINK
INPUT_PALETTE = [
    Color.BLACK,
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]
GEN_RETRIES = 400


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert 3 <= height <= 30 and 1 <= width <= 30

    divider_rows = [r for r in range(height) if np.all(g[r, :] == DIVIDER_COLOR)]
    assert len(divider_rows) == 1, "expected one full-width grey divider row"
    divider_r = divider_rows[0]
    assert divider_r > 0 and divider_r < height - 1

    top = g[:divider_r, :]
    bottom = g[divider_r + 1 :, :]
    assert top.shape == bottom.shape, "the divider must split the grid into equal halves"

    top_anchor_coords = np.argwhere(top == ANCHOR_COLOR)
    bottom_anchor_coords = np.argwhere(bottom == ANCHOR_COLOR)
    assert top_anchor_coords.shape[0] == 1, "expected one top yellow anchor"
    assert bottom_anchor_coords.shape[0] == 1, "expected one bottom yellow anchor"
    top_anchor = tuple(map(int, top_anchor_coords[0]))
    bottom_anchor = tuple(map(int, bottom_anchor_coords[0]))

    shared = (set(colors_present(top)) & set(colors_present(bottom))) - {ANCHOR_COLOR}
    assert len(shared) == 1, "expected exactly one shared non-anchor color"
    bg = int(next(iter(shared)))

    assert not np.any(top == DIVIDER_COLOR)
    assert not np.any(bottom == DIVIDER_COLOR)

    top_fg_colors = colors_present(top, exclude={bg, ANCHOR_COLOR})
    bottom_fg_colors = colors_present(bottom, exclude={bg, ANCHOR_COLOR})
    assert len(top_fg_colors) == 1, "expected one top foreground color"
    assert len(bottom_fg_colors) == 1, "expected one bottom foreground color"

    top_mask = top == int(top_fg_colors[0])
    bottom_mask = bottom == int(bottom_fg_colors[0])
    assert not top_mask[top_anchor]
    assert not bottom_mask[bottom_anchor]

    return {
        "grid": g,
        "bg": bg,
        "top_mask": top_mask,
        "bottom_mask": bottom_mask,
        "top_anchor": top_anchor,
        "bottom_anchor": bottom_anchor,
        "shape": top.shape,
    }


def _translate_mask(mask: np.ndarray, delta: tuple[int, int], out_shape: tuple[int, int]) -> np.ndarray:
    dr, dc = map(int, delta)
    out = np.zeros(out_shape, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r) + dr
        cc = int(c) + dc
        if in_bounds(out_shape, rr, cc):
            out[rr, cc] = True
    return out


def _semantic_oracle(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_input(grid)
    shape = parsed["shape"]
    bg = parsed["bg"]
    tr, tc = parsed["top_anchor"]
    br, bc = parsed["bottom_anchor"]
    delta = (tr - br, tc - bc)

    top_set = {tuple(map(int, rc)) for rc in np.argwhere(parsed["top_mask"])}
    aligned_bottom_set = set()
    for r, c in np.argwhere(parsed["bottom_mask"]):
        rr = int(r) + delta[0]
        cc = int(c) + delta[1]
        if in_bounds(shape, rr, cc):
            aligned_bottom_set.add((rr, cc))

    out = np.full(shape, bg, dtype=int)
    for r, c in top_set ^ aligned_bottom_set:
        out[r, c] = OUTPUT_COLOR
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Align the two half-region masks by their yellow anchors, then emit the foreground symmetric difference.

    Input structure:
    - A full-width gray divider row splits the grid into same-sized top and bottom regions on a shared background.
    - Each region contains exactly one yellow anchor pixel plus one non-yellow foreground pattern color.
    - Yellow is an anchor marker only and is not part of either foreground mask.

    Rule:
    - Find the unique yellow anchor in each half and translate the bottom half so the two anchors coincide in the top-half coordinate system.
    - Compare only the two foreground masks after that alignment.
    - Return one half-sized grid, coloring pink exactly where one aligned mask is foreground and the other is not; output background everywhere else.

    Key insight:
    - The raw top and bottom positions are a decoy: the yellow pixels define the alignment offset, and the answer is the anchor-aligned XOR of the two foreground masks.
    """
    parsed = _parse_input(input_grid)
    tr, tc = parsed["top_anchor"]
    br, bc = parsed["bottom_anchor"]
    delta = (tr - br, tc - bc)
    aligned_bottom = _translate_mask(parsed["bottom_mask"], delta, parsed["shape"])

    out = np.full(parsed["shape"], parsed["bg"], dtype=int)
    out[parsed["top_mask"] ^ aligned_bottom] = OUTPUT_COLOR
    return out


def _neighbors4(shape: tuple[int, int], cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = cell
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _edge_distance(shape: tuple[int, int], cell: tuple[int, int]) -> int:
    h, w = shape
    r, c = cell
    return min(r, c, h - 1 - r, w - 1 - c)


def _choose_anchor(cells: list[tuple[int, int]], shape: tuple[int, int]) -> tuple[int, int]:
    edge_cells = [cell for cell in cells if _edge_distance(shape, cell) <= 1]
    if edge_cells and random.random() < 0.45:
        return random.choice(edge_cells)
    return random.choice(cells)


def _candidate_deltas(h: int, w: int) -> list[tuple[int, int]]:
    min_visible = max(8, (h * w) // 5)
    out = []
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            visible_area = (h - abs(dr)) * (w - abs(dc))
            if visible_area >= min_visible:
                out.append((dr, dc))
    return out


def _split_total(total: int, parts: int) -> list[int]:
    sizes = [1] * parts
    for _ in range(total - parts):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _grow_component(
    shape: tuple[int, int],
    seed: tuple[int, int],
    size: int,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    comp = {seed}
    while len(comp) < size:
        frontier = set()
        for cell in comp:
            frontier.update(_neighbors4(shape, cell))
        frontier -= comp
        frontier -= blocked
        if not frontier:
            return None
        comp.add(random.choice(list(frontier)))
    return comp


def _random_mask(shape: tuple[int, int], forbidden: tuple[int, int], target_size: int) -> np.ndarray | None:
    h, w = shape
    all_cells = [(r, c) for r in range(h) for c in range(w) if (r, c) != forbidden]
    if target_size <= 0 or target_size > len(all_cells):
        return None

    component_count = min(
        random.choices([1, 2, 3], weights=[5, 3, 1], k=1)[0], target_size
    )
    component_sizes = _split_total(target_size, component_count)

    occupied: set[tuple[int, int]] = set()
    for size in component_sizes:
        candidates = [cell for cell in all_cells if cell not in occupied]
        if not candidates:
            return None
        random.shuffle(candidates)

        comp = None
        for seed in candidates[: min(len(candidates), 25)]:
            grown = _grow_component(shape, seed, size, occupied | {forbidden})
            if grown is not None:
                comp = grown
                break
        if comp is None:
            return None
        occupied |= comp

    mask = np.zeros(shape, dtype=bool)
    for r, c in occupied:
        mask[r, c] = True
    return mask


def _choose_size_bounds(shape: tuple[int, int]) -> tuple[int, int]:
    area = shape[0] * shape[1]
    lo = max(5, area // 10)
    hi = min(area - 2, max(lo + 2, area // 4))
    return lo, hi


def _build_masks(
    shape: tuple[int, int],
    top_anchor: tuple[int, int],
    bottom_anchor: tuple[int, int],
    delta: tuple[int, int],
    require_clip: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = shape
    size_lo, size_hi = _choose_size_bounds(shape)

    for _ in range(120):
        top_size = random.randint(size_lo, size_hi)
        bottom_size = random.randint(size_lo, size_hi)
        top_mask = _random_mask(shape, top_anchor, top_size)
        bottom_mask = _random_mask(shape, bottom_anchor, bottom_size)
        if top_mask is None or bottom_mask is None:
            continue

        clipped_sources = [
            (int(r), int(c))
            for r, c in np.argwhere(bottom_mask)
            if not in_bounds(shape, int(r) + delta[0], int(c) + delta[1])
        ]
        if require_clip and not clipped_sources:
            continue

        aligned_bottom = _translate_mask(bottom_mask, delta, shape)
        overlap = int(np.count_nonzero(top_mask & aligned_bottom))
        top_only = int(np.count_nonzero(top_mask & ~aligned_bottom))
        bottom_only = int(np.count_nonzero(aligned_bottom & ~top_mask))
        xor_count = top_only + bottom_only
        if min(overlap, top_only, bottom_only) == 0:
            continue
        if xor_count < 3 or xor_count > (h * w) // 2:
            continue

        top_components = len(components_mask(top_mask, connectivity=4))
        bottom_components = len(components_mask(bottom_mask, connectivity=4))
        if top_components > 4 or bottom_components > 4:
            continue

        if delta != (0, 0):
            naive_xor = top_mask ^ bottom_mask
            aligned_xor = top_mask ^ aligned_bottom
            if np.array_equal(naive_xor, aligned_xor):
                continue

        return top_mask, bottom_mask
    return None


def _render_input(
    shape: tuple[int, int],
    bg: int,
    top_color: int,
    bottom_color: int,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    top_anchor: tuple[int, int],
    bottom_anchor: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    top = np.full(shape, bg, dtype=int)
    bottom = np.full(shape, bg, dtype=int)
    top[top_mask] = int(top_color)
    bottom[bottom_mask] = int(bottom_color)
    top[top_anchor] = ANCHOR_COLOR
    bottom[bottom_anchor] = ANCHOR_COLOR

    grid = np.full((2 * h + 1, w), bg, dtype=int)
    grid[:h, :] = top
    grid[h, :] = DIVIDER_COLOR
    grid[h + 1 :, :] = bottom
    return grid


def _generate_case() -> np.ndarray | None:
    h = random.randint(6, 14)
    w = random.randint(6, 14)
    shape = (h, w)

    bg, top_color, bottom_color = random.sample(INPUT_PALETTE, 3)
    aligned_anchors = random.random() < 0.25

    if aligned_anchors:
        all_cells = [(r, c) for r in range(h) for c in range(w)]
        top_anchor = _choose_anchor(all_cells, shape)
        bottom_anchor = top_anchor
        delta = (0, 0)
        require_clip = False
    else:
        delta = random.choice(_candidate_deltas(h, w))
        feasible_top = [
            (r, c)
            for r in range(max(0, delta[0]), min(h - 1, h - 1 + delta[0]) + 1)
            for c in range(max(0, delta[1]), min(w - 1, w - 1 + delta[1]) + 1)
        ]
        if not feasible_top:
            return None
        top_anchor = _choose_anchor(feasible_top, shape)
        bottom_anchor = (top_anchor[0] - delta[0], top_anchor[1] - delta[1])
        require_clip = random.random() < 0.6

    masks = _build_masks(shape, top_anchor, bottom_anchor, delta, require_clip)
    if masks is None:
        return None
    top_mask, bottom_mask = masks

    grid = _render_input(
        shape, bg, top_color, bottom_color, top_mask, bottom_mask, top_anchor, bottom_anchor
    )

    try:
        expected = _semantic_oracle(grid)
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchor-aligned xor input")
