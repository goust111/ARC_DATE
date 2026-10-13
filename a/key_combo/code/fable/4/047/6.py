from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
SIDES = ("up", "down", "left", "right")
PALETTE = tuple(range(10))
MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _edge_signature(patch: np.ndarray, side: str, bg: int) -> set[tuple[int, int]]:
    sig: set[tuple[int, int]] = set()
    if side == "up":
        edge = patch[0, :]
    elif side == "down":
        edge = patch[-1, :]
    elif side == "left":
        edge = patch[:, 0]
    else:
        edge = patch[:, -1]
    for offset, color in enumerate(edge):
        color = int(color)
        if color != int(bg):
            sig.add((int(offset), color))
    return sig


def _extract_blocks(grid: np.ndarray, bg: int) -> tuple[list[dict], int]:
    comps = _find_components(grid != bg)
    assert comps, "expected at least one block"
    blocks = []
    sizes = set()
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w, "expected square block bounding boxes"
        sizes.add(h)
        blocks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )
    assert len(sizes) == 1, "expected all blocks to share one block size"
    k = int(next(iter(sizes)))
    return blocks, k


def _build_adjacency(blocks: list[dict], bg: int) -> list[dict[str, int]]:
    n = len(blocks)
    signatures = [{side: _edge_signature(block["patch"], side, bg) for side in SIDES} for block in blocks]
    adjacency: list[dict[str, int]] = [dict() for _ in range(n)]
    for i in range(n):
        for side in SIDES:
            matches = []
            want = signatures[i][side]
            if not want:
                continue
            for j in range(n):
                if i == j:
                    continue
                other = signatures[j][OPPOSITE[side]]
                if want & other:
                    matches.append(j)
            assert len(matches) <= 1, f"ambiguous {side} match for block {i}"
            if matches:
                adjacency[i][side] = matches[0]
    for i, nbrs in enumerate(adjacency):
        for side, j in nbrs.items():
            assert adjacency[j].get(OPPOSITE[side]) == i, "edge matches must be reciprocal"
    return adjacency


def _reconstruct_positions(adjacency: list[dict[str, int]], anchor_idx: int) -> dict[int, tuple[int, int]]:
    positions = {int(anchor_idx): (0, 0)}
    q = deque([int(anchor_idx)])
    while q:
        i = q.popleft()
        r0, c0 = positions[i]
        for side, j in adjacency[i].items():
            dr, dc = DIRS[side]
            candidate = (r0 + dr, c0 + dc)
            if j in positions:
                assert positions[j] == candidate, "inconsistent 2D layout"
                continue
            positions[j] = candidate
            q.append(j)
    assert len(positions) == len(adjacency), "expected one connected assembly"
    assert len(set(positions.values())) == len(positions), "two blocks cannot occupy the same layout cell"
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered square sprite blocks by edge-pixel interlocks and keep the unique edge-touching anchor fixed.

    Input structure:
    - A background grid contains several separated k×k multicolor blocks.
    - Exactly one block touches the outer grid border; that block is the anchor.
    - Neighboring blocks are encoded by matching foreground pixels on opposite block edges.

    Rule:
    - Split the foreground into the scattered square blocks and read each block's four edge signatures.
    - Whenever two opposite edges share same-color foreground pixels at the same offsets, treat those blocks as adjacent in that direction.
    - Recover the unique 2D block layout from those adjacencies, keep the anchor block in place, clear the old floating copies, and paste every other block into its assembled position with edge-to-edge contact.

    Key insight:
    - The edge matches define a 2D adjacency graph rather than a single sequence, and the only absolute position comes from the unique block already pinned to the canvas border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    blocks, k = _extract_blocks(g, bg)

    h, w = g.shape
    anchor_candidates = []
    for idx, block in enumerate(blocks):
        r0, c0, r1, c1 = block["bbox"]
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            anchor_candidates.append(idx)
    assert len(anchor_candidates) == 1, "expected exactly one border-touching anchor block"
    anchor_idx = int(anchor_candidates[0])

    adjacency = _build_adjacency(blocks, bg)
    layout = _reconstruct_positions(adjacency, anchor_idx)
    anchor_r, anchor_c, _anchor_r1, _anchor_c1 = blocks[anchor_idx]["bbox"]

    out = np.full_like(g, int(bg))
    for idx, block in enumerate(blocks):
        lr, lc = layout[idx]
        r0 = anchor_r + lr * k
        c0 = anchor_c + lc * k
        patch = block["patch"]
        ph, pw = patch.shape
        assert 0 <= r0 and r0 + ph <= h and 0 <= c0 and c0 + pw <= w, "assembled block leaves the grid"
        window = out[r0 : r0 + ph, c0 : c0 + pw]
        assert np.all(window == bg), "assembled blocks may touch but not overlap"
        mask = patch != bg
        window[mask] = patch[mask]
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((r - min_r, c - min_c) for r, c in cells)


def _grow_shape(n_blocks: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < n_blocks:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS.values():
                cand = (r + dr, c + dc)
                if cand not in cells:
                    frontier.add(cand)
        cells.add(random.choice(tuple(frontier)))
    return _normalize_cells(cells)


def _valid_anchor_choices(cells: list[tuple[int, int]]) -> list[tuple[int, str]]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    choices = []
    for side, wanted in (
        ("up", min_r),
        ("down", max_r),
    ):
        idxs = [i for i, (r, _c) in enumerate(cells) if r == wanted]
        if len(idxs) == 1:
            choices.append((idxs[0], side))
    for side, wanted in (
        ("left", min_c),
        ("right", max_c),
    ):
        idxs = [i for i, (_r, c) in enumerate(cells) if c == wanted]
        if len(idxs) == 1:
            choices.append((idxs[0], side))
    return choices


def _shape_adjacency(cells: list[tuple[int, int]]) -> list[dict[str, int]]:
    lookup = {cell: i for i, cell in enumerate(cells)}
    adjacency: list[dict[str, int]] = [dict() for _ in cells]
    for i, (r, c) in enumerate(cells):
        for side, (dr, dc) in DIRS.items():
            j = lookup.get((r + dr, c + dc))
            if j is not None:
                adjacency[i][side] = j
    return adjacency


def _draw_path(patch: np.ndarray, start: tuple[int, int], center: tuple[int, int], color: int, side: str) -> None:
    r, c = start
    cr, cc = center
    patch[r, c] = int(color)
    if side == "up":
        step = 1 if cr >= r else -1
        for rr in range(r, cr + step, step):
            patch[rr, c] = int(color)
        step = 1 if cc >= c else -1
        for cc2 in range(c, cc + step, step):
            patch[cr, cc2] = int(color)
    elif side == "down":
        step = -1 if cr <= r else 1
        for rr in range(r, cr + step, step):
            patch[rr, c] = int(color)
        step = 1 if cc >= c else -1
        for cc2 in range(c, cc + step, step):
            patch[cr, cc2] = int(color)
    elif side == "left":
        step = 1 if cc >= c else -1
        for cc2 in range(c, cc + step, step):
            patch[r, cc2] = int(color)
        step = 1 if cr >= r else -1
        for rr in range(r, cr + step, step):
            patch[rr, cc] = int(color)
    else:
        step = -1 if cc <= c else 1
        for cc2 in range(c, cc + step, step):
            patch[r, cc2] = int(color)
        step = 1 if cr >= r else -1
        for rr in range(r, cr + step, step):
            patch[rr, cc] = int(color)


def _edge_coord(side: str, offset: int, k: int) -> tuple[int, int]:
    if side == "up":
        return 0, offset
    if side == "down":
        return k - 1, offset
    if side == "left":
        return offset, 0
    return offset, k - 1


def _make_patch(side_tokens: dict[str, tuple[int, int]], k: int, bg: int) -> np.ndarray:
    patch = np.full((k, k), int(bg), dtype=int)
    center = (random.randint(1, k - 2), random.randint(1, k - 2))
    sides = list(SIDES)
    random.shuffle(sides)
    for side in sides:
        offset, color = side_tokens[side]
        _draw_path(patch, _edge_coord(side, offset, k), center, color, side)

    extras = random.randint(0, k)
    fg_colors = [color for _offset, color in side_tokens.values()]
    alt_colors = [color for color in PALETTE if color != bg]
    for _ in range(extras):
        frontier = []
        for r in range(1, k - 1):
            for c in range(1, k - 1):
                if patch[r, c] != bg:
                    continue
                if (
                    patch[r - 1, c] != bg
                    or patch[r + 1, c] != bg
                    or patch[r, c - 1] != bg
                    or patch[r, c + 1] != bg
                ):
                    frontier.append((r, c))
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        patch[rr, cc] = random.choice(fg_colors)

    used = sorted({int(v) for v in patch[patch != bg]})
    if len(used) < 2:
        rr, cc = center
        choices = [color for color in alt_colors if color != int(patch[rr, cc])]
        patch[rr, cc] = random.choice(choices)
    return patch


def _allocate_side_tokens(shape_adj: list[dict[str, int]], k: int, bg: int) -> list[dict[str, tuple[int, int]]]:
    pool = [(offset, color) for offset in range(1, k - 1) for color in PALETTE if color != bg]
    random.shuffle(pool)
    side_tokens: list[dict[str, tuple[int, int]]] = [dict() for _ in shape_adj]
    for i, nbrs in enumerate(shape_adj):
        for side, j in nbrs.items():
            if side in side_tokens[i]:
                continue
            if i < j:
                token = pool.pop()
                side_tokens[i][side] = token
                side_tokens[j][OPPOSITE[side]] = token
    for tokens in side_tokens:
        for side in SIDES:
            if side not in tokens:
                tokens[side] = pool.pop()
    return side_tokens


def _boxes_too_close(a: tuple[int, int], b: tuple[int, int], k: int, pad: int = 1) -> bool:
    ar, ac = a
    br, bc = b
    return not (
        ar + k - 1 + pad < br
        or br + k - 1 + pad < ar
        or ac + k - 1 + pad < bc
        or bc + k - 1 + pad < ac
    )


def _assembled_positions(
    cells: list[tuple[int, int]],
    anchor_side: str,
    shape_h: int,
    shape_w: int,
    k: int,
    grid_h: int,
    grid_w: int,
) -> dict[int, tuple[int, int]]:
    layout_h = shape_h * k
    layout_w = shape_w * k
    if anchor_side == "up":
        base_r = 0
        base_c = random.randint(1, grid_w - layout_w - 1)
    elif anchor_side == "down":
        base_r = grid_h - layout_h
        base_c = random.randint(1, grid_w - layout_w - 1)
    elif anchor_side == "left":
        base_r = random.randint(1, grid_h - layout_h - 1)
        base_c = 0
    else:
        base_r = random.randint(1, grid_h - layout_h - 1)
        base_c = grid_w - layout_w
    return {i: (base_r + r * k, base_c + c * k) for i, (r, c) in enumerate(cells)}


def _scatter_input(
    patches: list[np.ndarray],
    target_positions: dict[int, tuple[int, int]],
    anchor_idx: int,
    bg: int,
    grid_h: int,
    grid_w: int,
    k: int,
) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    placed: dict[int, tuple[int, int]] = {anchor_idx: target_positions[anchor_idx]}
    anchor_r, anchor_c = target_positions[anchor_idx]
    grid[anchor_r : anchor_r + k, anchor_c : anchor_c + k] = patches[anchor_idx]

    order = [i for i in range(len(patches)) if i != anchor_idx]
    random.shuffle(order)
    forbidden_targets = list(target_positions.values())

    for idx in order:
        candidates = []
        for r0 in range(1, grid_h - k):
            for c0 in range(1, grid_w - k):
                pos = (r0, c0)
                if any(_boxes_too_close(pos, other, k) for other in placed.values()):
                    continue
                if any(_boxes_too_close(pos, target, k) for target in forbidden_targets):
                    continue
                candidates.append(pos)
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + k, c0 : c0 + k] = patches[idx]
        placed[idx] = (r0, c0)
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_blocks = random.randint(3, 5)
    k = random.randint(5, 6)
    bg = random.choice(PALETTE)

    cells: list[tuple[int, int]] | None = None
    anchor_choices: list[tuple[int, str]] = []
    for _ in range(80):
        candidate = _grow_shape(n_blocks)
        rows = {r for r, _ in candidate}
        cols = {c for _, c in candidate}
        choices = _valid_anchor_choices(candidate)
        if len(rows) >= 2 and len(cols) >= 2 and choices:
            cells = candidate
            anchor_choices = choices
            break
    if cells is None:
        return None

    anchor_idx, anchor_side = random.choice(anchor_choices)
    shape_adj = _shape_adjacency(cells)
    side_tokens = _allocate_side_tokens(shape_adj, k, bg)
    patches = [_make_patch(tokens, k, bg) for tokens in side_tokens]

    shape_h = max(r for r, _ in cells) + 1
    shape_w = max(c for _, c in cells) + 1
    layout_h = shape_h * k
    layout_w = shape_w * k

    min_h = max(layout_h + 1, k + 6)
    min_w = max(layout_w + 1, k + 6)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)

    orth_margin_ok = (
        (anchor_side in {"up", "down"} and grid_w - layout_w >= 2)
        or (anchor_side in {"left", "right"} and grid_h - layout_h >= 2)
    )
    if not orth_margin_ok:
        return None

    target_positions = _assembled_positions(cells, anchor_side, shape_h, shape_w, k, grid_h, grid_w)
    out = np.full((grid_h, grid_w), int(bg), dtype=int)
    for idx, patch in enumerate(patches):
        r0, c0 = target_positions[idx]
        out[r0 : r0 + k, c0 : c0 + k] = patch

    inp = _scatter_input(patches, target_positions, anchor_idx, bg, grid_h, grid_w, k)
    if inp is None:
        return None
    try:
        produced = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
