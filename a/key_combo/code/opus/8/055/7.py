import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
X_SIZE = 5
BLOCK_SIZE = 3
BOX_STEP = 6
MAX_SIDE = 30
MAX_GEN_ATTEMPTS = 400

X_TO_BLOCK = {
    (0, 0): (0, 0),
    (1, 1): (0, 1),
    (0, 4): (0, 2),
    (3, 1): (1, 0),
    (2, 2): (1, 1),
    (1, 3): (1, 2),
    (4, 0): (2, 0),
    (3, 3): (2, 1),
    (4, 4): (2, 2),
}
X_RELATIVE_CELLS = frozenset(X_TO_BLOCK)
CORNER_POSITIONS = ((0, 0), (0, 2), (2, 2), (2, 0))
EDGE_AND_CENTER_POSITIONS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))

PATTERNS = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 3)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (0, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 4)),
    ((0, 0), (1, 0), (0, 1), (0, 3)),
)
_pattern_order = list(range(len(PATTERNS)))
random.shuffle(_pattern_order)
_pattern_cursor = 0


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components8(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _fold_x(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    assert r1 - r0 + 1 == X_SIZE and c1 - c0 + 1 == X_SIZE, "expected a 5x5 X bbox"
    rel_coords = {tuple(map(int, rc)) for rc in np.argwhere(grid[r0 : r1 + 1, c0 : c1 + 1] != bg)}
    assert rel_coords == X_RELATIVE_CELLS, "expected exactly the 9 X-diagonal cells"

    block = np.full((BLOCK_SIZE, BLOCK_SIZE), bg, dtype=int)
    for (xr, xc), (br, bc) in X_TO_BLOCK.items():
        color = int(grid[r0 + xr, c0 + xc])
        assert color != bg, "expected colored pixels on every X position"
        block[br, bc] = color

    box = grid[r0 : r1 + 1, c0 : c1 + 1]
    for rr in range(X_SIZE):
        for cc in range(X_SIZE):
            if (rr, cc) in X_RELATIVE_CELLS:
                continue
            assert int(box[rr, cc]) == bg, "expected background off the X diagonals"
    return block


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    assert bg == BG, "expected black background"

    components = _components8(g, bg)
    assert 2 <= len(components) <= 4, "expected 2-4 separate X objects"

    objects = []
    for comp in components:
        assert len(comp) == 9, "each X must have 9 colored cells"
        bbox = _bbox(comp)
        block = _fold_x(g, bbox, bg)
        objects.append({"bbox": bbox, "folded": block})

    return sorted(objects, key=lambda obj: obj["bbox"])


def _adjacency_counts(bboxes: list[tuple[int, int, int, int]]) -> list[int]:
    counts = [0] * len(bboxes)
    for i in range(len(bboxes)):
        r0a, c0a, r1a, c1a = bboxes[i]
        for j in range(i + 1, len(bboxes)):
            r0b, c0b, r1b, c1b = bboxes[j]
            horizontal = r0a == r0b and r1a == r1b and (c1a + 2 == c0b or c1b + 2 == c0a)
            vertical = c0a == c0b and c1a == c1b and (r1a + 2 == r0b or r1b + 2 == r0a)
            if horizontal or vertical:
                counts[i] += 1
                counts[j] += 1
    return counts


def _rotate_corners(block: np.ndarray, steps: int) -> np.ndarray:
    out = np.array(block, dtype=int, copy=True)
    for _ in range(steps % 4):
        prev = out.copy()
        out[0, 0] = prev[2, 0]
        out[0, 2] = prev[0, 0]
        out[2, 2] = prev[0, 2]
        out[2, 0] = prev[2, 2]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each 5x5 colored X into a centered 3x3 block, then rotate that block's corner colors by its folded-neighbor count.

    Input structure:
    - The background is black.
    - The foreground consists of 2-4 separate multicolor 5x5 X shapes whose nine
      colored cells lie exactly on the two diagonals of their 5x5 bounding box.
    - Different X boxes are disjoint and may be one background cell apart
      horizontally or vertically.

    Rule:
    - Fold every 5x5 X into a 3x3 block with the fixed diagonal-to-edge mapping
      from the puzzle statement, then place that 3x3 block back into the same
      5x5 window with centers aligned.
    - Count how many other X boxes sit exactly one background cell away in the
      four orthogonal directions with the same row-span or column-span; after the
      fold removes each X's discarded outer ring, those are the side-neighboring
      folded blocks.
    - Rotate only the folded block's four corner colors clockwise by that count.
      Edge-middle cells and the center stay fixed.

    Key insight:
    - The X's nine colors are preserved but remapped into a compact 3x3 role
      layout; only after identifying which folded blocks become orthogonal
      neighbors can the correct corner rotation amount be determined.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = _extract_objects(g)
    bboxes = [obj["bbox"] for obj in objects]
    counts = _adjacency_counts(bboxes)
    assert any(count > 0 for count in counts), "expected at least one folded adjacency"

    out = np.full(g.shape, bg, dtype=int)
    for obj, turns in zip(objects, counts):
        block = _rotate_corners(obj["folded"], turns)
        r0, c0, _r1, _c1 = obj["bbox"]
        rr, cc = r0 + 1, c0 + 1
        assert np.all(out[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] == bg), "folded blocks must not overlap"
        out[rr : rr + BLOCK_SIZE, cc : cc + BLOCK_SIZE] = block
    return out


def _next_pattern() -> tuple[tuple[int, int], ...]:
    global _pattern_cursor
    if _pattern_cursor > 0 and _pattern_cursor % len(_pattern_order) == 0:
        random.shuffle(_pattern_order)
    idx = _pattern_order[_pattern_cursor % len(_pattern_order)]
    _pattern_cursor += 1
    return PATTERNS[idx]


def _transform_pattern(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    points = [(int(r), int(c)) for r, c in coords]
    for _ in range(random.randrange(4)):
        points = [(c, -r) for r, c in points]
    if random.choice((False, True)):
        points = [(r, -c) for r, c in points]
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return tuple(sorted((r - min_r, c - min_c) for r, c in points))


def _coarse_counts(coords: tuple[tuple[int, int], ...]) -> list[int]:
    coord_set = set(coords)
    counts = []
    for r, c in coords:
        total = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            total += (r + dr, c + dc) in coord_set
        counts.append(total)
    return counts


def _split_slack(base: int) -> tuple[int, int]:
    slack = MAX_SIDE - base
    assert slack >= 0
    lead = random.randint(0, min(4, slack))
    trail = random.randint(0, min(4, slack - lead))
    return lead, trail


def _make_block(center_color: int, turns: int, used_outputs: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(200):
        block = np.empty((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
        corners = random.sample(PALETTE, 4)
        block[0, 0], block[0, 2], block[2, 2], block[2, 0] = corners
        block[1, 1] = int(center_color)
        for pos in ((0, 1), (1, 0), (1, 2), (2, 1)):
            block[pos] = random.choice(PALETTE)
        rotated = _rotate_corners(block, turns)
        sig = tuple(int(v) for v in rotated.ravel())
        if sig in used_outputs:
            continue
        if turns % 4 != 0 and np.array_equal(rotated, block):
            continue
        used_outputs.add(sig)
        return block
    raise RuntimeError("failed to build a distinctive block")


def _block_to_x(block: np.ndarray) -> np.ndarray:
    patch = np.zeros((X_SIZE, X_SIZE), dtype=int)
    for (xr, xc), (br, bc) in X_TO_BLOCK.items():
        patch[xr, xc] = int(block[br, bc])
    return patch


def _attempt() -> np.ndarray | None:
    coarse = _transform_pattern(_next_pattern())
    counts = _coarse_counts(coarse)
    if len(coarse) >= 3 and len(set(counts)) < 2:
        return None

    max_r = max(r for r, _ in coarse)
    max_c = max(c for _, c in coarse)
    base_h = max_r * BOX_STEP + X_SIZE
    base_w = max_c * BOX_STEP + X_SIZE
    if base_h > MAX_SIDE or base_w > MAX_SIDE:
        return None
    top_margin, bottom_margin = _split_slack(base_h)
    left_margin, right_margin = _split_slack(base_w)
    grid = np.zeros((base_h + top_margin + bottom_margin, base_w + left_margin + right_margin), dtype=int)

    center_colors = random.sample(PALETTE, len(coarse))
    used_outputs: set[tuple[int, ...]] = set()
    for idx, ((cr, cc), turns) in enumerate(zip(coarse, counts)):
        block = _make_block(center_colors[idx], turns, used_outputs)
        x_patch = _block_to_x(block)
        r0 = top_margin + cr * BOX_STEP
        c0 = left_margin + cc * BOX_STEP
        grid[r0 : r0 + X_SIZE, c0 : c0 + X_SIZE] = x_patch

    objects = _extract_objects(grid)
    parsed_counts = _adjacency_counts([obj["bbox"] for obj in objects])
    if parsed_counts != counts:
        return None
    if not any(turns > 0 for turns in counts):
        return None
    if len(coarse) >= 3 and len(set(parsed_counts)) < 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    changed_blocks = 0
    output_signatures = set()
    for obj, turns in zip(objects, parsed_counts):
        rotated = _rotate_corners(obj["folded"], turns)
        output_signatures.add(tuple(int(v) for v in rotated.ravel()))
        if turns % 4 != 0 and not np.array_equal(rotated, obj["folded"]):
            changed_blocks += 1
    if changed_blocks == 0:
        return None
    if len(output_signatures) != len(objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
