import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BG = 0
GEN_RETRIES = 500
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (15, 26)
CHAIN_LEN_RANGE = (2, 4)
SPAN_W_RANGE = (4, 7)
TERRAIN_HEIGHT_RANGE = (2, 6)
STACK_GAP_RANGE = (1, 2)
TOP_MARGIN = 2


@dataclass
class Block:
    color: int
    row: int
    col: int
    mask: np.ndarray
    cells: np.ndarray
    bottoms: np.ndarray


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _mask_from_coords(coords: np.ndarray) -> tuple[int, int, np.ndarray]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return r0, c0, mask


def _bottom_cells(mask: np.ndarray) -> np.ndarray:
    cells = []
    for c in range(mask.shape[1]):
        rows = np.where(mask[:, c])[0]
        if rows.size:
            cells.append((int(rows.max()), c))
    return np.array(cells, dtype=int)


def _infer_terrain_color(grid: np.ndarray) -> int:
    bottom_colors = {int(v) for v in grid[-1] if int(v) != BG}
    assert len(bottom_colors) == 1, "expected exactly one terrain color touching the bottom edge"
    return next(iter(bottom_colors))


def _parse_blocks(grid: np.ndarray) -> tuple[int, list[Block]]:
    terrain_color = _infer_terrain_color(grid)
    colors = sorted(set(map(int, grid.ravel())) - {BG, terrain_color})
    blocks: list[Block] = []
    for color in colors:
        for coords in _components_of_color(grid, color):
            row, col, mask = _mask_from_coords(coords)
            cells = np.argwhere(mask)
            blocks.append(
                Block(
                    color=color,
                    row=row,
                    col=col,
                    mask=mask,
                    cells=cells.astype(int),
                    bottoms=_bottom_cells(mask),
                )
            )
    blocks.sort(key=lambda block: (block.row, block.col, block.color))
    return terrain_color, blocks


def _paint(grid: np.ndarray, block: Block, row: int, col: int, color: int) -> None:
    grid[row + block.cells[:, 0], col + block.cells[:, 1]] = color


def _drop_distance(block: Block, occupied: np.ndarray, support: np.ndarray) -> int | None:
    h = occupied.shape[0]
    occupied_other = occupied.copy()
    occupied_other[block.row + block.cells[:, 0], block.col + block.cells[:, 1]] = False
    max_drop = h - 1 - int(block.row + block.cells[:, 0].max())
    for dist in range(1, max_drop + 1):
        shifted_rows = block.row + dist + block.cells[:, 0]
        shifted_cols = block.col + block.cells[:, 1]
        if occupied_other[shifted_rows, shifted_cols].any():
            break
        fully_supported = True
        for br, bc in block.bottoms:
            rr = block.row + dist + br
            cc = block.col + bc
            if rr == h - 1:
                continue
            if not support[rr + 1, cc]:
                fully_supported = False
                break
        if fully_supported:
            return dist
    return None


def _simulate_settling(
    grid: np.ndarray, *, require_unique: bool
) -> tuple[np.ndarray, list[int], list[int]]:
    out = np.asarray(grid, dtype=int).copy()
    assert out.ndim == 2, "expected a 2D grid"
    terrain_color, blocks = _parse_blocks(out)
    support = out == terrain_color
    settled_colors: list[int] = []
    movable_counts: list[int] = []
    remaining = list(blocks)

    while True:
        occupied = out != BG
        movable = []
        for idx, block in enumerate(remaining):
            dist = _drop_distance(block, occupied, support)
            if dist is not None:
                movable.append((idx, dist))
        movable_counts.append(len(movable))
        if not movable:
            break
        if require_unique:
            assert len(movable) == 1, "expected a unique movable block in each chain step"
        idx, dist = movable[0]
        block = remaining.pop(idx)
        _paint(out, block, block.row, block.col, BG)
        block.row += dist
        _paint(out, block, block.row, block.col, block.color)
        support[block.row + block.cells[:, 0], block.col + block.cells[:, 1]] = True
        settled_colors.append(block.color)

    return out, settled_colors, movable_counts


def main(grid: np.ndarray) -> np.ndarray:
    """Let only the uniquely complementary floating block fall, merge into the terrain, and repeat until the chain stops.

    Input structure:
    - The background is black.
    - One irregular monochrome terrain touches the bottom edge and provides the initial support profile.
    - Several monochrome floating blocks hang above the terrain; some are part of a forced chain and at least one is a near-fit distractor.

    Rule:
    - Repeatedly find the floating block that can move straight down without crossing occupied cells and reach a position where every bottom-face cell is immediately supported by terrain, settled blocks, or the bottom edge.
    - In each valid puzzle state there is exactly one such block; move it to that resting position.
    - Merge each settled block into the support, continue the chain, and leave every non-fitting distractor suspended.

    Key insight:
    - A block does not stop at first contact; it moves only when its whole bottom contour is simultaneously supported, and each settled block creates the next complementary socket.
    """
    out, _, _ = _simulate_settling(np.asarray(grid, dtype=int), require_unique=True)
    return out


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    assert coords.size > 0
    target = {tuple(map(int, p)) for p in coords}
    q = deque([tuple(map(int, coords[0]))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nb in target and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(target)


def _terrain_heights(width: int) -> list[int]:
    lo, hi = TERRAIN_HEIGHT_RANGE
    heights = [random.randint(lo, hi)]
    for _ in range(1, width):
        step = random.choice([-1, 0, 1])
        heights.append(max(lo, min(hi, heights[-1] + step)))
    if max(heights) == min(heights):
        heights[random.randrange(width)] = min(hi, heights[0] + 1)
    return heights


def _layer_mask(prev_profile: list[int], next_profile: list[int]) -> np.ndarray:
    top = min(next_profile)
    h = max(prev_profile) - top
    w = len(prev_profile)
    mask = np.zeros((h, w), dtype=bool)
    for c in range(w):
        mask[next_profile[c] - top : prev_profile[c] - top, c] = True
    return mask


def _build_chain_profiles(base_profile: list[int], chain_len: int) -> list[list[int]] | None:
    profiles = [list(base_profile)]
    for _ in range(chain_len):
        prev = profiles[-1]
        for _ in range(80):
            delta = [random.choice([1, 2]) for _ in prev]
            if len(set(delta)) == 1:
                delta[random.randrange(len(delta))] = 3 - delta[0]
            next_profile = [p - d for p, d in zip(prev, delta)]
            if min(next_profile) < TOP_MARGIN:
                continue
            mask = _layer_mask(prev, next_profile)
            if _connected(mask):
                profiles.append(next_profile)
                break
        else:
            return None
    return profiles


def _distractor_mask(target_profile: list[int]) -> np.ndarray | None:
    width = len(target_profile)
    for _ in range(60):
        profile = list(target_profile)
        idx = random.randrange(width)
        if width > 1 and random.random() < 0.6:
            neighbor = idx - 1 if idx > 0 and (idx == width - 1 or random.random() < 0.5) else idx + 1
            profile[idx] = target_profile[neighbor]
        else:
            profile[idx] += random.choice([-1, 1])
        if profile == target_profile or min(profile) < 1:
            continue
        top = min(profile)
        mask = np.zeros((max(profile) - top + 1, width), dtype=bool)
        for c, bottom in enumerate(profile):
            mask[: bottom - top + 1, c] = True
        return mask
    return None


def _draw_terrain(grid: np.ndarray, terrain_rows: list[int], color: int) -> None:
    for c, top in enumerate(terrain_rows):
        grid[top:, c] = color


def _make_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    chain_len = random.randint(*CHAIN_LEN_RANGE)
    span_w = random.randint(*SPAN_W_RANGE)
    if span_w + 4 > w:
        return None

    terrain_heights = _terrain_heights(w)
    terrain_rows = [h - height for height in terrain_heights]

    chain_start = random.randint(1, w - span_w - 2)
    base_profile = terrain_rows[chain_start : chain_start + span_w]
    profiles = _build_chain_profiles(base_profile, chain_len)
    if profiles is None:
        return None

    chain_masks = [_layer_mask(profiles[i - 1], profiles[i]) for i in range(1, len(profiles))]
    gaps = [random.randint(*STACK_GAP_RANGE) for _ in range(max(0, chain_len - 1))]
    stack_height = sum(mask.shape[0] for mask in chain_masks) + sum(gaps)
    max_stack_top = min(profiles[-1]) - stack_height - 1
    if max_stack_top < 1:
        return None
    stack_top = random.randint(1, max_stack_top)

    terrain_color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    block_palette = [c for c in range(1, 10) if c != terrain_color]
    random.shuffle(block_palette)
    if len(block_palette) < chain_len + 1:
        return None

    grid = np.zeros((h, w), dtype=int)
    _draw_terrain(grid, terrain_rows, terrain_color)

    chain_colors = []
    cursor = stack_top
    for rev_idx, mask in enumerate(reversed(chain_masks)):
        color = block_palette.pop()
        chain_colors.append(color)
        grid[cursor : cursor + mask.shape[0], chain_start : chain_start + span_w][mask] = color
        if rev_idx < len(gaps):
            cursor += mask.shape[0] + gaps[rev_idx]

    chain_colors.reverse()

    distractor_starts = [
        start
        for start in range(1, w - span_w - 1)
        if start + span_w <= chain_start - 1 or start >= chain_start + span_w + 1
    ]
    if not distractor_starts:
        return None
    distractor_start = random.choice(distractor_starts)
    distractor_profile = terrain_rows[distractor_start : distractor_start + span_w]
    distractor = _distractor_mask(distractor_profile)
    if distractor is None:
        return None
    max_distractor_top = min(distractor_profile) - distractor.shape[0] - 1
    if max_distractor_top < 1:
        return None
    distractor_top = random.randint(1, max_distractor_top)
    distractor_color = block_palette.pop()
    grid[
        distractor_top : distractor_top + distractor.shape[0],
        distractor_start : distractor_start + span_w,
    ][distractor] = distractor_color

    try:
        out, settled_colors, movable_counts = _simulate_settling(grid, require_unique=True)
    except AssertionError:
        return None

    if settled_colors != chain_colors:
        return None
    if len(settled_colors) != chain_len:
        return None
    if movable_counts[-1] != 0 or any(count != 1 for count in movable_counts[:-1]):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-reaction seating puzzle")
