from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BACKGROUND = 0
PIVOT = 5
FRAME_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
_GENERATION_COUNT = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive (r0, c0, r1, c1) bbox, using the arc_utils (r, c) convention."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("a hollow rectangle must be at least 3 by 3")
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of absolute (r, c) coordinates."""
    todo = mask.copy().astype(bool)
    h, w = todo.shape
    result: list[np.ndarray] = []
    while bool(todo.any()):
        sr, sc = map(int, np.argwhere(todo)[0])
        todo[sr, sc] = False
        queue = deque([(sr, sc)])
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and todo[rr, cc]:
                    todo[rr, cc] = False
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _unique_order(colors: tuple[int, ...], edges: set[tuple[int, int]]) -> tuple[int, ...] | None:
    above = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for lower, higher in edges:
        if lower == higher or lower not in above or higher not in above:
            return None
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    used: set[int] = set()
    while len(order) < len(colors):
        available = [c for c in colors if c not in used and indegree[c] == 0]
        if len(available) != 1:
            return None
        color = available[0]
        order.append(color)
        used.add(color)
        for higher in above[color]:
            indegree[higher] -= 1
    return tuple(order)


def _parse_frames(
    grid: np.ndarray, region: np.ndarray, colors: tuple[int, ...]
) -> tuple[tuple[int, ...], dict[int, np.ndarray], set[tuple[int, int]]] | None:
    """Reconstruct and validate the complete stack inside one proposed frame region."""
    if bool(np.any(region & (grid == PIVOT))):
        return None
    if set(map(int, np.unique(grid[region]))) - {BACKGROUND, *colors}:
        return None

    masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = region & (grid == color)
        box = _bbox(visible)
        if box is None:
            return None
        try:
            mask = _frame_mask(grid.shape, box)
        except ValueError:
            return None
        if not bool(np.all(mask[visible])) or bool(np.any(mask & ~region)):
            return None
        masks[color] = mask

    union = np.zeros(grid.shape, dtype=bool)
    for mask in masks.values():
        union |= mask
    if not np.array_equal(union & region, region & (grid != BACKGROUND)):
        return None

    edges: set[tuple[int, int]] = set()
    for r, c in np.argwhere(union):
        r, c = int(r), int(c)
        covering = [color for color in colors if masks[color][r, c]]
        shown = int(grid[r, c])
        if shown not in covering:
            return None
        if len(covering) >= 2:
            edges.update((color, shown) for color in covering if color != shown)

    order = _unique_order(colors, edges)
    if order is None:
        return None

    # A second, stronger check: repainting the inferred frames in layer order
    # must reproduce every pixel in the proposed frame region.
    rebuilt = np.zeros_like(grid)
    for color in order:
        rebuilt[masks[color]] = color
    if not np.array_equal(rebuilt[region], grid[region]):
        return None
    return order, masks, edges


def _rotate_offset(dr: int, dc: int, turns: int) -> tuple[int, int]:
    """Rotate a row/column offset counterclockwise in screen coordinates."""
    for _ in range(turns % 4):
        dr, dc = -dc, dr
    return dr, dc


def _parse_sprites(
    grid: np.ndarray, region: np.ndarray, colors: tuple[int, ...]
) -> dict[int, tuple[np.ndarray, tuple[int, int]]] | None:
    """Find the unique same-color sprite adjacent to each unique gray pivot."""
    allowed = {BACKGROUND, PIVOT, *colors}
    if set(map(int, np.unique(grid[region]))) - allowed:
        return None

    pivots = [tuple(map(int, p)) for p in np.argwhere(region & (grid == PIVOT))]
    if len(pivots) != len(colors):
        return None
    if any(abs(r1 - r2) + abs(c1 - c2) <= 1 for i, (r1, c1) in enumerate(pivots)
           for r2, c2 in pivots[i + 1 :]):
        return None

    by_color: dict[int, np.ndarray] = {}
    pivot_for_color: dict[int, tuple[int, int]] = {}
    for color in colors:
        comps = _components(region & (grid == color))
        if len(comps) != 1:
            return None
        coords = comps[0]
        adjacent = []
        cell_set = {tuple(map(int, cell)) for cell in coords}
        for pivot in pivots:
            pr, pc = pivot
            if any((pr + dr, pc + dc) in cell_set for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                adjacent.append(pivot)
        if len(adjacent) != 1:
            return None
        by_color[color] = coords
        pivot_for_color[color] = adjacent[0]

    if len(set(pivot_for_color.values())) != len(colors):
        return None
    return {color: (by_color[color], pivot_for_color[color]) for color in colors}


def _transform_candidate(
    grid: np.ndarray,
    order: tuple[int, ...],
    sprites: dict[int, tuple[np.ndarray, tuple[int, int]]],
) -> np.ndarray | None:
    out = grid.copy()
    for coords, _pivot in sprites.values():
        out[coords[:, 0], coords[:, 1]] = BACKGROUND

    occupied = out != BACKGROUND
    targets: dict[int, np.ndarray] = {}
    h, w = grid.shape
    for depth, color in enumerate(order):
        coords, (pr, pc) = sprites[color]
        rotated = []
        for r, c in coords:
            dr, dc = int(r) - pr, int(c) - pc
            rr, cc = _rotate_offset(dr, dc, depth)
            rotated.append((pr + rr, pc + cc))
        target = np.asarray(rotated, dtype=int)
        if not bool(np.all((0 <= target[:, 0]) & (target[:, 0] < h)
                           & (0 <= target[:, 1]) & (target[:, 1] < w))):
            return None
        if bool(np.any(occupied[target[:, 0], target[:, 1]])):
            return None
        occupied[target[:, 0], target[:, 1]] = True
        targets[color] = target

    for color, target in targets.items():
        out[target[:, 0], target[:, 1]] = color
    return out


def _candidate_regions(grid: np.ndarray):
    """Yield both role assignments around every full-black row or column."""
    h, w = grid.shape
    for axis, size in ((0, h), (1, w)):
        for cut in range(1, size - 1):
            is_blank = bool(np.all(grid[cut, :] == BACKGROUND)) if axis == 0 else bool(
                np.all(grid[:, cut] == BACKGROUND)
            )
            if not is_blank:
                continue
            before = np.zeros(grid.shape, dtype=bool)
            after = np.zeros(grid.shape, dtype=bool)
            if axis == 0:
                before[:cut, :] = True
                after[cut + 1 :, :] = True
            else:
                before[:, :cut] = True
                after[:, cut + 1 :] = True
            yield before, after
            yield after, before


def _analyze(grid: np.ndarray) -> list[dict]:
    if grid.ndim != 2 or not (1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30):
        return []
    values = set(map(int, np.unique(grid)))
    colors = tuple(sorted(values - {BACKGROUND, PIVOT}))
    if not 2 <= len(colors) <= len(FRAME_PALETTE) or PIVOT not in values:
        return []

    analyses: list[dict] = []
    seen: set[tuple[bytes, tuple[int, ...]]] = set()
    for frame_region, sprite_region in _candidate_regions(grid):
        frame_data = _parse_frames(grid, frame_region, colors)
        if frame_data is None:
            continue
        order, masks, edges = frame_data
        sprites = _parse_sprites(grid, sprite_region, colors)
        if sprites is None:
            continue
        output = _transform_candidate(grid, order, sprites)
        if output is None:
            continue
        key = (frame_region.tobytes(), order)
        if key in seen:
            continue
        seen.add(key)
        analyses.append({
            "output": output,
            "order": order,
            "masks": masks,
            "edges": edges,
            "sprites": sprites,
        })
    return analyses


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer hollow-frame layer depths and rotate each matching sprite around its gray pivot.

    Input structure:
    - A black grid has, on one side, several differently colored one-cell-thick hollow rectangular frames whose overlaps show only the upper frame.
    - The other side has one asymmetric monochrome sprite of every frame color and one orthogonally adjacent gray pivot pixel per sprite.
    - One or more full-black rows or columns separate the frame area from the sprite area.

    Rule:
    - Reconstruct each frame from the bbox of its visible pixels within the frame area; at every reconstructed overlap, the visible color is above every hidden color.
    - Require these constraints to give one bottom-to-top order, assign depths 0, 1, ... in that order, and rotate each same-color sprite counterclockwise by 90 degrees times its depth about its own gray pivot.
    - Keep every pivot and the entire overlaid-frame area exactly as shown in the input.

    Key insight:
    - The rectangles are depth counters rather than output objects: their occlusion order supplies the integer quarter-turn used by the spatially separate same-color pivoted sprite.
    """
    g = np.asarray(grid, dtype=int)
    analyses = _analyze(g)
    assert analyses, "input must have a valid separated frame stack and pivoted sprite set"

    first = analyses[0]["output"]
    first_order = analyses[0]["order"]
    assert all(item["order"] == first_order for item in analyses), "layer order must be uniquely determined"
    assert all(np.array_equal(item["output"], first) for item in analyses), "output must be uniquely determined"
    return first.copy()


# ------------------------------ generator ------------------------------


def _random_frame_stack(
    h: int, frame_w: int, colors: list[int]
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    n = len(colors)
    shift_r = random.choice((-1, 1))
    shift_c = random.choice((-1, 1))
    travel = n - 1
    box_h = random.randint(7, min(12, h - travel - 2))
    box_w = random.randint(5, min(8, frame_w - travel - 2))

    if shift_r > 0:
        r0 = random.randint(1, h - box_h - travel - 1)
    else:
        r0 = random.randint(1 + travel, h - box_h - 1)
    if shift_c > 0:
        c0 = random.randint(1, frame_w - box_w - travel - 1)
    else:
        c0 = random.randint(1 + travel, frame_w - box_w - 1)

    canvas = np.zeros((h, frame_w), dtype=int)
    masks: list[np.ndarray] = []
    boxes = []
    for i in range(n):
        rr = r0 + i * shift_r
        cc = c0 + i * shift_c
        box = (rr, cc, rr + box_h - 1, cc + box_w - 1)
        boxes.append(box)
        masks.append(_frame_mask(canvas.shape, box))

    # Layer depth is deliberately independent of geometric position. Reject the
    # two monotone assignments so bbox position cannot replace occlusion reading.
    permutations = [p for p in itertools.permutations(range(n))
                    if p != tuple(range(n)) and p != tuple(reversed(range(n)))]
    assignment = random.choice(permutations)
    layer_masks = [masks[i] for i in assignment]
    layer_boxes = [boxes[i] for i in assignment]
    for color, mask in zip(colors, layer_masks):
        canvas[mask] = color

    # Occlusion may hide crossings, but it must never destroy a frame's bbox clues.
    if any(_bbox(canvas == color) != box for color, box in zip(colors, layer_boxes)):
        return None
    if any(not bool(np.any(layer_masks[i] & layer_masks[j]))
           for i in range(n) for j in range(i + 1, n)):
        return None
    cover_count = sum(mask.astype(int) for mask in masks)
    overlap_fraction = float(np.sum(cover_count >= 2)) / float(np.sum(cover_count >= 1))
    if overlap_fraction > 0.32:
        return None
    return canvas, layer_masks


def _random_asymmetric_offsets() -> set[tuple[int, int]]:
    """Grow a small 4-connected shape adjacent to (0, 0), distinct in all rotations."""
    for _ in range(100):
        cells = {random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))}
        target = random.randint(3, 6)
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    p = (r + dr, c + dc)
                    if p != (0, 0) and -2 <= p[0] <= 2 and -2 <= p[1] <= 2 and p not in cells:
                        frontier.add(p)
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target:
            continue
        rotations = []
        for turns in range(4):
            rotations.append(frozenset(_rotate_offset(r, c, turns) for r, c in cells))
        if len(set(rotations)) == 4:
            return cells
    raise RuntimeError("failed to create an asymmetric sprite")


def _sprite_pivots(h: int, start_c: int, sprite_w: int, n: int) -> list[tuple[int, int]]:
    row_edges = (0, h // 2, h)
    col_edges = (start_c, start_c + sprite_w // 2, start_c + sprite_w)
    slots = [(ri, ci) for ri in range(2) for ci in range(2)]
    random.shuffle(slots)
    pivots = []
    for ri, ci in slots[:n]:
        rlo, rhi = row_edges[ri], row_edges[ri + 1]
        clo, chi = col_edges[ci], col_edges[ci + 1]
        # Radius two for every rotation, plus one visibly empty cell between slots.
        pivots.append((random.randint(rlo + 3, rhi - 4), random.randint(clo + 3, chi - 4)))
    random.shuffle(pivots)
    return pivots


def _semantic_check(inp: np.ndarray, expected: np.ndarray, intended_order: tuple[int, ...]) -> bool:
    analyses = _analyze(inp)
    if not analyses:
        return False
    if any(item["order"] != intended_order for item in analyses):
        return False
    if any(not np.array_equal(item["output"], expected) for item in analyses):
        return False
    item = analyses[0]
    if len(item["edges"]) < len(intended_order) - 1:
        return False

    # Semantic, not merely oracle, checks: every nonzero depth changes its sprite,
    # and every generated shape distinguishes all four quarter-turns (including CW/CCW).
    for depth, color in enumerate(intended_order):
        coords, (pr, pc) = item["sprites"][color]
        source = frozenset(map(tuple, coords.tolist()))
        variants = set()
        for turns in range(4):
            variants.add(frozenset((pr + _rotate_offset(int(r) - pr, int(c) - pc, turns)[0],
                                    pc + _rotate_offset(int(r) - pr, int(c) - pc, turns)[1])
                                   for r, c in coords))
        if len(variants) != 4:
            return False
        if depth > 0:
            rotated = frozenset((pr + _rotate_offset(int(r) - pr, int(c) - pc, depth)[0],
                                 pc + _rotate_offset(int(r) - pr, int(c) - pc, depth)[1])
                                for r, c in coords)
            if rotated == source:
                return False
    return not np.array_equal(inp, expected)


def _gen_attempt(n: int) -> np.ndarray | None:
    h = random.randint(18, 24)
    frame_w = random.randint(10, 12)
    separator_w = random.randint(2, 3)
    sprite_w = random.randint(14, min(16, 30 - frame_w - separator_w))
    w = frame_w + separator_w + sprite_w

    order = random.sample(FRAME_PALETTE, n)
    stack = _random_frame_stack(h, frame_w, order)
    if stack is None:
        return None
    frame_canvas, _masks = stack

    inp = np.zeros((h, w), dtype=int)
    inp[:, :frame_w] = frame_canvas
    sprite_start = frame_w + separator_w
    pivots = _sprite_pivots(h, sprite_start, sprite_w, n)

    expected = inp.copy()
    offsets_by_color: dict[int, set[tuple[int, int]]] = {}
    pivot_by_color: dict[int, tuple[int, int]] = {}
    for color, (pr, pc) in zip(random.sample(order, n), pivots):
        offsets = _random_asymmetric_offsets()
        offsets_by_color[color] = offsets
        pivot_by_color[color] = (pr, pc)
        inp[pr, pc] = PIVOT
        expected[pr, pc] = PIVOT
        for dr, dc in offsets:
            inp[pr + dr, pc + dc] = color

    depth = {color: i for i, color in enumerate(order)}
    for color, offsets in offsets_by_color.items():
        pr, pc = pivot_by_color[color]
        source_coords = np.argwhere(inp == color)
        sprite_coords = [tuple(map(int, p)) for p in source_coords if int(p[1]) >= sprite_start]
        if not any(abs(r - pr) + abs(c - pc) == 1 for r, c in sprite_coords):
            return None
        for r, c in sprite_coords:
            expected[r, c] = BACKGROUND
        for dr, dc in offsets:
            rr, cc = _rotate_offset(dr, dc, depth[color])
            expected[pr + rr, pc + cc] = color

    # Rotate the whole scene to cover left/right/top/bottom layouts. A proper
    # rotation commutes with the requested CCW sprite rotation.
    quarter_turns = random.randrange(4)
    if quarter_turns:
        inp = np.rot90(inp, quarter_turns).copy()
        expected = np.rot90(expected, quarter_turns).copy()

    intended_order = tuple(order)
    try:
        if not _semantic_check(inp, expected, intended_order):
            return None
        if not np.array_equal(main(inp), expected):
            return None
    except (AssertionError, ValueError, IndexError):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate diverse, oracle- and semantics-checked instances within 30x30."""
    global _GENERATION_COUNT
    # Every three calls include two four-layer cases (all depths 0/1/2/3) and
    # one three-layer case, so object count varies without losing rare-depth coverage.
    n = (4, 3, 4)[_GENERATION_COUNT % 3]
    _GENERATION_COUNT += 1
    for _ in range(300):
        result = _gen_attempt(n)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid layered-knob input")
