from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
NOISE_COLORS = [3, 4, 5, 6, 7, 8, 9]

MAX_GENERATION_ATTEMPTS = 500
MIN_GRID_SIZE = 16
MAX_GRID_SIZE = 28


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r0, c0 in np.argwhere(grid == int(color)):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))
    return components


def _solid_2x2_blocks(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    for coords in _color_components(grid, color):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        is_block = (
            int(coords.shape[0]) == 4
            and int(r1 - r0) == 1
            and int(c1 - c0) == 1
            and bool(np.all(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(color)))
        )
        assert is_block, f"color {color} components must be isolated solid 2x2 blocks"
        blocks.append((int(r0), int(c0)))
    return sorted(blocks)


def _block_rows(block: tuple[int, int]) -> set[int]:
    r, _c = block
    return {int(r), int(r) + 1}


def _block_cols(block: tuple[int, int]) -> set[int]:
    _r, c = block
    return {int(c), int(c) + 1}


def _is_reached_by_block(target: tuple[int, int], source: tuple[int, int]) -> bool:
    return bool(_block_rows(target) & _block_rows(source) or _block_cols(target) & _block_cols(source))


def _activation_closure(
    fuse: tuple[int, int], dormant_blocks: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    active: set[tuple[int, int]] = {tuple(fuse)}
    dormant = [tuple(b) for b in dormant_blocks]

    changed = True
    while changed:
        changed = False
        active_rows: set[int] = set()
        active_cols: set[int] = set()
        for block in active:
            active_rows.update(_block_rows(block))
            active_cols.update(_block_cols(block))

        for block in dormant:
            if block in active:
                continue
            if _block_rows(block) & active_rows or _block_cols(block) & active_cols:
                active.add(block)
                changed = True
    return active


def _cross_mask(shape: tuple[int, int], blocks: set[tuple[int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for r, c in blocks:
        mask[int(r), :] = True
        mask[int(r) + 1, :] = True
        mask[:, int(c)] = True
        mask[:, int(c) + 1] = True
    return mask


def _erase_cross(output: np.ndarray, block: tuple[int, int]) -> None:
    h, w = output.shape
    r, c = map(int, block)
    for rr in (r, r + 1):
        if 0 <= rr < h:
            keep = output[rr, :] == RED
            output[rr, ~keep] = BLACK
    for cc in (c, c + 1):
        if 0 <= cc < w:
            keep = output[:, cc] == RED
            output[~keep, cc] = BLACK


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trigger chained cross erasures from the unique blue 2x2 fuse block.

    Input structure:
    - The grid is filled with noisy colors other than black, blue, and red.
    - Exactly one isolated blue 2x2 block is the fuse, and isolated black 2x2
      blocks are dormant bombs.
    - Red singleton pixels are protected pixels and may lie anywhere outside
      the 2x2 blocks.

    Rule:
    - Activate the blue fuse first. For every activated block, turn all
      non-red pixels in its two rows and two columns black.
    - A dormant black 2x2 block activates when any one of its four cells lies
      in the rows or columns of any already activated block.
    - Repeat this propagation to a fixed point. Dormant blocks outside this
      row/column reachability closure do not trigger, and red pixels never
      change.

    Key insight:
    - The result is the transitive closure of row/column reachability among
      2x2 blocks, not an independent cross erase for every black block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fuses = _solid_2x2_blocks(g, BLUE)
    assert len(fuses) == 1, "expected exactly one blue 2x2 fuse"
    dormant = _solid_2x2_blocks(g, BLACK)

    active = _activation_closure(fuses[0], dormant)
    out = g.copy()
    for block in sorted(active):
        _erase_cross(out, block)
    return out


def _top_left_positions(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]


def _expanded_rect_overlaps(a: tuple[int, int], b: tuple[int, int], padding: int = 1) -> bool:
    ar0, ac0 = int(a[0]) - padding, int(a[1]) - padding
    ar1, ac1 = int(a[0]) + 1 + padding, int(a[1]) + 1 + padding
    br0, bc0 = int(b[0]), int(b[1])
    br1, bc1 = int(b[0]) + 1, int(b[1]) + 1
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _can_place_block(pos: tuple[int, int], blocks: list[tuple[int, int]], h: int, w: int) -> bool:
    r, c = map(int, pos)
    if not (1 <= r <= h - 3 and 1 <= c <= w - 3):
        return False
    for block in blocks:
        if _expanded_rect_overlaps((r, c), block, padding=1):
            return False
    return True


def _choose_aligned_block(
    anchor: tuple[int, int],
    axis: str,
    blocks: list[tuple[int, int]],
    h: int,
    w: int,
    *,
    forbidden_direct: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    candidates = []
    for pos in _top_left_positions(h, w):
        if not _can_place_block(pos, blocks, h, w):
            continue
        if axis == "row" and not (_block_rows(pos) & _block_rows(anchor)):
            continue
        if axis == "col" and not (_block_cols(pos) & _block_cols(anchor)):
            continue
        if forbidden_direct is not None and _is_reached_by_block(pos, forbidden_direct):
            continue
        candidates.append(pos)
    if not candidates:
        return None
    return random.choice(candidates)


def _choose_silent_block(
    blocks: list[tuple[int, int]],
    h: int,
    w: int,
    active_rows: set[int],
    active_cols: set[int],
) -> tuple[int, int] | None:
    candidates = []
    for pos in _top_left_positions(h, w):
        if not _can_place_block(pos, blocks, h, w):
            continue
        if _block_rows(pos) & active_rows:
            continue
        if _block_cols(pos) & active_cols:
            continue
        candidates.append(pos)
    if not candidates:
        return None
    return random.choice(candidates)


def _block_cell_mask(shape: tuple[int, int], blocks: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in blocks:
        mask[int(r) : int(r) + 2, int(c) : int(c) + 2] = True
    return mask


def _render_grid(
    h: int,
    w: int,
    fuse: tuple[int, int],
    dormant: list[tuple[int, int]],
    red_cells: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.random.choice(np.array(NOISE_COLORS, dtype=int), size=(h, w)).astype(int)

    fr, fc = fuse
    grid[fr : fr + 2, fc : fc + 2] = BLUE
    for r, c in dormant:
        grid[r : r + 2, c : c + 2] = BLACK
    for r, c in red_cells:
        grid[int(r), int(c)] = RED
    return grid


def _choose_red_cells(
    shape: tuple[int, int],
    fuse: tuple[int, int],
    dormant: list[tuple[int, int]],
    active: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    block_mask = _block_cell_mask(shape, [fuse] + dormant)
    cross = _cross_mask(shape, active)

    cross_choices = [tuple(map(int, p)) for p in np.argwhere(cross & ~block_mask)]
    outside_choices = [tuple(map(int, p)) for p in np.argwhere(~cross & ~block_mask)]
    if len(cross_choices) < 2 or not outside_choices:
        return None

    random.shuffle(cross_choices)
    random.shuffle(outside_choices)
    red_cells = cross_choices[: random.randint(2, min(5, len(cross_choices)))]
    red_cells.append(outside_choices[0])

    all_free = [tuple(map(int, p)) for p in np.argwhere(~block_mask)]
    random.shuffle(all_free)
    target_count = random.randint(4, min(12, max(4, len(all_free))))
    used = set(red_cells)
    for cell in all_free:
        if len(used) >= target_count:
            break
        used.add(cell)
    return sorted(used)


def _semantic_validate(grid: np.ndarray) -> bool:
    try:
        fuse_list = _solid_2x2_blocks(grid, BLUE)
        dormant = _solid_2x2_blocks(grid, BLACK)
        if len(fuse_list) != 1 or len(dormant) < 3:
            return False
        fuse = fuse_list[0]
        active = _activation_closure(fuse, dormant)
        active_dormant = active - {fuse}
        direct = {b for b in dormant if _is_reached_by_block(b, fuse)}
        transitive = active_dormant - direct
        silent = set(dormant) - active
        if not direct or not transitive or not silent:
            return False

        cross = _cross_mask(grid.shape, active)
        red = grid == RED
        if not bool((red & cross).any()):
            return False

        out = main(grid)
        if not np.array_equal(out[red], grid[red]):
            return False
        if not bool(np.all(out[cross & ~red] == BLACK)):
            return False
        if not np.array_equal(out[~cross], grid[~cross]):
            return False
        if not bool(np.any((grid != BLACK) & (grid != RED) & cross)):
            return False
        return True
    except AssertionError:
        return False


def _build_case() -> np.ndarray | None:
    h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    all_positions = _top_left_positions(h, w)
    random.shuffle(all_positions)

    fuse = all_positions[0]
    blocks = [fuse]
    first_axis = random.choice(["row", "col"])
    second_axis = "col" if first_axis == "row" else "row"

    direct = _choose_aligned_block(fuse, first_axis, blocks, h, w)
    if direct is None:
        return None
    blocks.append(direct)

    transitive = _choose_aligned_block(direct, second_axis, blocks, h, w, forbidden_direct=fuse)
    if transitive is None:
        return None
    blocks.append(transitive)

    dormant: list[tuple[int, int]] = [direct, transitive]
    active_chain = [fuse, direct, transitive]

    for _ in range(random.randint(0, 2)):
        parent = random.choice(active_chain[1:])
        axis = random.choice(["row", "col"])
        extra = _choose_aligned_block(parent, axis, blocks, h, w)
        if extra is not None:
            blocks.append(extra)
            dormant.append(extra)
            active_chain.append(extra)

    active = _activation_closure(fuse, dormant)
    active_rows: set[int] = set()
    active_cols: set[int] = set()
    for block in active:
        active_rows.update(_block_rows(block))
        active_cols.update(_block_cols(block))

    for _ in range(random.randint(1, 3)):
        silent = _choose_silent_block(blocks, h, w, active_rows, active_cols)
        if silent is None:
            return None
        blocks.append(silent)
        dormant.append(silent)

    final_active = _activation_closure(fuse, dormant)
    if not set(active_chain).issubset(final_active):
        return None
    if not any(b not in final_active for b in dormant):
        return None

    red_cells = _choose_red_cells((h, w), fuse, dormant, final_active)
    if red_cells is None:
        return None

    grid = _render_grid(h, w, fuse, dormant, red_cells)
    if not _semantic_validate(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
