import random
from typing import List, Sequence, Tuple

import numpy as np


BLACK = 0
FRAME_COLOR = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_DEPTH = 2
MAX_DEPTH = 4
MIN_GROUPS = 1
MAX_GROUPS = 3
GRID_H_MAX = 32
GRID_W_MAX = 34
MAX_GENERATION_ATTEMPTS = 400

BBox = Tuple[int, int, int, int]


def _neighbors4(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> BBox:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _frame_border_cells(box: BBox) -> set[Tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: set[Tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected:
        return False
    return set(map(tuple, coords.tolist())) == _frame_border_cells((r0, c0, r1, c1))


def _contains(outer: BBox, inner: BBox) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1


def _find_palette(grid: np.ndarray) -> Tuple[int, int, int, List[int]]:
    h, w = grid.shape
    candidates: List[Tuple[int, int, int, int, List[int]]] = []

    for row in (0, h - 1):
        c = 0
        while c < w:
            if int(grid[row, c]) == BLACK:
                c += 1
                continue
            c0 = c
            colors: List[int] = []
            while c < w and int(grid[row, c]) != BLACK:
                colors.append(int(grid[row, c]))
                c += 1
            c1 = c - 1
            if len(colors) < 2:
                continue
            if len(set(colors)) != len(colors):
                continue
            candidates.append((len(colors), row, c0, c1, colors))

    if not candidates:
        raise ValueError("palette not found")

    candidates.sort(reverse=True)
    _width, row, c0, c1, colors = candidates[0]
    return row, c0, c1, colors


def _extract_scene(grid: np.ndarray) -> Tuple[List[int], List[Tuple[BBox, np.ndarray]], Tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("expected a non-empty 2D grid")

    row, c0, c1, palette = _find_palette(g)
    mask = g != BLACK
    mask[row, c0 : c1 + 1] = False

    frames: List[Tuple[BBox, np.ndarray]] = []
    for coords in _components(mask):
        if not _is_frame_component(coords):
            raise ValueError("unexpected non-frame component")
        frames.append((_bbox(coords), coords))

    if not frames:
        raise ValueError("no frames found")

    return palette, frames, (row, c0, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each rectangular frame according to its nesting depth, using the
    edge palette's left-to-right colors as depth 1..k.
    """
    g = np.asarray(grid, dtype=int)
    palette, frames, _palette_box = _extract_scene(g)

    out = g.copy()
    boxes = [box for box, _coords in frames]
    for box, coords in frames:
        depth = 1 + sum(_contains(other, box) for other in boxes if other != box)
        if depth > len(palette):
            raise ValueError("palette shorter than maximum nesting depth")
        color = int(palette[depth - 1])
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _draw_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _make_chain(depth: int, rng: random.Random) -> List[BBox]:
    inner_h = rng.randint(3, 6)
    inner_w = rng.randint(3, 7)
    boxes: List[BBox] = [(0, 0, inner_h - 1, inner_w - 1)]
    cur_h = inner_h
    cur_w = inner_w

    for _ in range(depth - 1):
        top_gap = rng.randint(1, 3)
        bottom_gap = rng.randint(1, 3)
        left_gap = rng.randint(1, 3)
        right_gap = rng.randint(1, 3)
        shift_r = top_gap + 1
        shift_c = left_gap + 1
        shifted = [
            (r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c)
            for r0, c0, r1, c1 in boxes
        ]
        cur_h += top_gap + bottom_gap + 2
        cur_w += left_gap + right_gap + 2
        boxes = [(0, 0, cur_h - 1, cur_w - 1)] + shifted

    return boxes


def _outer_size(chain: Sequence[BBox]) -> Tuple[int, int]:
    r0, c0, r1, c1 = chain[0]
    return r1 - r0 + 1, c1 - c0 + 1


def _boxes_too_close(a: BBox, b: BBox, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_groups(
    chains: Sequence[Sequence[BBox]],
    h: int,
    w: int,
    palette_row: int,
    rng: random.Random,
) -> List[Tuple[int, int]]:
    order = sorted(range(len(chains)), key=lambda i: _outer_size(chains[i])[0] * _outer_size(chains[i])[1], reverse=True)
    placed_boxes: List[Tuple[int, int, int, int]] = []
    offsets: List[Tuple[int, int] | None] = [None] * len(chains)

    for idx in order:
        outer_h, outer_w = _outer_size(chains[idx])
        min_r = 2 if palette_row == 0 else 1
        max_r = h - outer_h - (2 if palette_row == h - 1 else 1)
        min_c = 1
        max_c = w - outer_w - 2
        if max_r < min_r or max_c < min_c:
            raise ValueError("canvas too small for placement")

        placed = False
        for _ in range(200):
            r0 = rng.randint(min_r, max_r)
            c0 = rng.randint(min_c, max_c)
            candidate = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_too_close(candidate, other, pad=1) for other in placed_boxes):
                continue
            placed_boxes.append(candidate)
            offsets[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            raise ValueError("failed to place all groups")

    return [(int(r), int(c)) for r, c in offsets if r is not None and c is not None]


def _expected_depths(chains: Sequence[Sequence[BBox]]) -> List[int]:
    depths: List[int] = []
    for chain in chains:
        depths.extend(range(1, len(chain) + 1))
    return depths


def _gen_attempt(rng: random.Random) -> np.ndarray:
    max_depth = rng.randint(MIN_DEPTH, MAX_DEPTH)
    group_count = rng.randint(MIN_GROUPS, MAX_GROUPS)
    palette_row_top = rng.random() < 0.5
    palette_row = 0 if palette_row_top else -1

    depths = [max_depth]
    while len(depths) < group_count:
        depths.append(rng.randint(1, max_depth))
    rng.shuffle(depths)

    chains = [_make_chain(depth, rng) for depth in depths]
    outer_heights = sorted((_outer_size(chain)[0] for chain in chains), reverse=True)
    outer_widths = sorted((_outer_size(chain)[1] for chain in chains), reverse=True)

    base_h = max(outer_heights[0] + 4, sum(outer_heights[: min(2, len(outer_heights))]) + 2)
    base_w = max(
        outer_widths[0] + 4,
        sum(outer_widths[: min(2, len(outer_widths))]) + 3,
        max_depth + 3,
    )
    h = base_h + rng.randint(0, 6)
    w = base_w + rng.randint(0, 8)
    if h > GRID_H_MAX or w > GRID_W_MAX:
        return None

    try:
        offsets = _place_groups(chains, h, w, 0 if palette_row_top else h - 1, rng)
    except ValueError:
        return None

    grid = np.zeros((h, w), dtype=int)
    palette = rng.sample(PALETTE_COLORS, max_depth)
    start_c = rng.randint(0, w - max_depth)
    row = 0 if palette_row_top else h - 1
    grid[row, start_c : start_c + max_depth] = np.array(palette, dtype=int)

    for chain, (off_r, off_c) in zip(chains, offsets):
        for box in chain:
            r0, c0, r1, c1 = box
            _draw_frame(grid, (r0 + off_r, c0 + off_c, r1 + off_r, c1 + off_c), FRAME_COLOR)

    try:
        found_palette, frames, _ = _extract_scene(grid)
    except ValueError:
        return None

    found_depths = sorted(
        1 + sum(_contains(other_box, box) for other_box, _ in frames if other_box != box)
        for box, _ in frames
    )
    if found_palette != palette:
        return None
    if found_depths != sorted(_expected_depths(chains)):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
