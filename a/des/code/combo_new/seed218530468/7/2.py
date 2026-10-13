from __future__ import annotations

from collections import deque
import numpy as np


BG = 0
PIPE = 5
RED = 2
BLUE = 1
GREEN = 3
MAROON = 9

NODE_COLORS = (RED, BLUE, GREEN)
SHAPE_COLORS = (4, 6, 7, 8)
SPECIAL_COLORS = (BG, PIPE, RED, BLUE, GREEN, MAROON)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_RETRIES = 400

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in CARDINALS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _midpoint_neighbors(r0: int, c0: int, h: int, w: int) -> list[tuple[int, int]]:
    return [
        (r0 + 1, c0 - 1),
        (r0 + 1, c0 + 3),
        (r0 - 1, c0 + 1),
        (r0 + 3, c0 + 1),
    ]


def _slot_attachment_neighbors(r0: int, c0: int) -> list[tuple[int, int]]:
    return [
        (r0 + 2, c0 - 1),
        (r0 + 2, c0 + 5),
        (r0 - 1, c0 + 2),
        (r0 + 5, c0 + 2),
    ]


def _find_shape(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    mask = ~np.isin(grid, SPECIAL_COLORS)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for pts in _components(mask):
        r0, c0, r1, c1 = _bbox(pts)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        attachment = []
        for ar, ac in _midpoint_neighbors(r0, c0, 3, 3):
            if 0 <= ar < grid.shape[0] and 0 <= ac < grid.shape[1] and int(grid[ar, ac]) == PIPE:
                attachment.append((ar, ac))
        if len(attachment) == 1:
            candidates.append(((r0, c0), attachment[0]))
    assert len(candidates) == 1, f"expected exactly one 3x3 input shape, got {len(candidates)}"
    return candidates[0]


def _find_slot(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            sub = grid[r0 : r0 + 5, c0 : c0 + 5]
            border_ok = bool(
                np.all(sub[0, :] == MAROON)
                and np.all(sub[-1, :] == MAROON)
                and np.all(sub[:, 0] == MAROON)
                and np.all(sub[:, -1] == MAROON)
            )
            if not border_ok:
                continue
            if not np.all(sub[1:4, 1:4] == BG):
                continue
            attachment = []
            for ar, ac in _slot_attachment_neighbors(r0, c0):
                if 0 <= ar < h and 0 <= ac < w and int(grid[ar, ac]) == PIPE:
                    attachment.append((ar, ac))
            if len(attachment) == 1:
                candidates.append(((r0, c0), attachment[0]))
    assert len(candidates) == 1, f"expected exactly one output slot, got {len(candidates)}"
    return candidates[0]


def _path_between(grid: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    passable = np.isin(grid, (PIPE, RED, BLUE, GREEN))
    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nxt in _neighbors4(cur[0], cur[1], grid.shape):
            if not passable[nxt] or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    assert end in prev, "pipe path from input shape to output slot must exist"
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _apply_node_ops(shape: np.ndarray, node_colors: list[int]) -> np.ndarray:
    out = shape.copy()
    for color in node_colors:
        if color == RED:
            out = np.rot90(out, -1)
        elif color == BLUE:
            out = np.fliplr(out)
        elif color == GREEN:
            out = np.flipud(out)
        else:
            raise AssertionError(color)
    return out.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move a 3x3 sprite through a pipe network and apply node-driven geometric transforms in traversal order.

    Input structure:
    - The background is black, with a one-cell-wide grey pipe network made of horizontal and vertical segments.
    - Colored intersection nodes lie on the pipe network; red, blue, and green nodes encode different transforms.
    - One pipe endpoint touches a 3x3 colored sprite, while another pipe endpoint touches a maroon 5x5 frame whose 3x3 interior is empty.

    Rule:
    - Start from the pipe endpoint adjacent to the 3x3 sprite and follow the unique pipe path to the endpoint adjacent to the maroon frame.
    - Each time the path passes through a colored node, transform the carried 3x3 sprite: red = rotate 90 degrees clockwise, blue = horizontal flip, green = vertical flip.
    - Clear the original sprite to background and write the final transformed 3x3 sprite into the empty interior of the maroon frame.

    Key insight:
    The task separates into two independent steps: recover the ordered node sequence from the pipe path, then compose the corresponding geometric transforms on the sprite.
    """
    g = np.asarray(grid, dtype=int)
    (shape_r0, shape_c0), start = _find_shape(g)
    (slot_r0, slot_c0), end = _find_slot(g)
    path = _path_between(g, start, end)
    node_colors = [int(g[r, c]) for r, c in path if int(g[r, c]) in NODE_COLORS]

    sprite = g[shape_r0 : shape_r0 + 3, shape_c0 : shape_c0 + 3].copy()
    transformed = _apply_node_ops(sprite, node_colors)

    out = g.copy()
    out[shape_r0 : shape_r0 + 3, shape_c0 : shape_c0 + 3] = BG
    out[slot_r0 + 1 : slot_r0 + 4, slot_c0 + 1 : slot_c0 + 4] = transformed
    return out


_SHAPE_MASKS = [
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=bool),
]


def _apply_symmetry(arr: np.ndarray, code: int) -> np.ndarray:
    if code == 0:
        return arr.copy()
    if code == 1:
        return np.rot90(arr, -1).copy()
    if code == 2:
        return np.rot90(arr, 2).copy()
    if code == 3:
        return np.rot90(arr, 1).copy()
    if code == 4:
        return np.fliplr(arr).copy()
    if code == 5:
        return np.flipud(arr).copy()
    if code == 6:
        return arr.T.copy()
    raise AssertionError(code)


def _random_sprite(rng: np.random.Generator) -> np.ndarray:
    mask = _SHAPE_MASKS[int(rng.integers(len(_SHAPE_MASKS)))].copy()
    mask = _apply_symmetry(mask.astype(int), int(rng.integers(7))).astype(bool)
    pts = list(zip(*np.where(mask)))
    n_colors = int(rng.integers(2, min(len(SHAPE_COLORS), len(pts)) + 1))
    palette = rng.choice(np.array(SHAPE_COLORS, dtype=int), size=n_colors, replace=False)

    sprite = np.full((3, 3), BG, dtype=int)
    for r, c in pts:
        sprite[r, c] = int(palette[int(rng.integers(n_colors))])

    # Break accidental symmetry from repeated colors.
    if np.array_equal(sprite, np.rot90(sprite, -1)) or np.array_equal(sprite, np.fliplr(sprite)) or np.array_equal(
        sprite, np.flipud(sprite)
    ):
        for idx, (r, c) in enumerate(pts):
            sprite[r, c] = int(SHAPE_COLORS[idx % len(SHAPE_COLORS)])
    return sprite


def _cells_on_segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        return [(r, c) for c in range(c0, c1 + 1)]
    if a[1] == b[1]:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        return [(r, c) for r in range(r0, r1 + 1)]
    raise AssertionError("segment must be axis-aligned")


def _step_toward(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    dr = 0 if src[0] == dst[0] else (1 if dst[0] > src[0] else -1)
    dc = 0 if src[1] == dst[1] else (1 if dst[1] > src[1] else -1)
    return dr, dc


def _paint_path_cells(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    color: int,
    allowed_overlap: set[tuple[int, int]],
) -> bool:
    for cell in cells:
        value = int(grid[cell])
        if value != BG and cell not in allowed_overlap:
            return False
    for cell in cells:
        grid[cell] = color
    return True


def _branch_options(
    grid: np.ndarray,
    node: tuple[int, int],
    used_dirs: set[tuple[int, int]],
    rng: np.random.Generator,
) -> list[list[tuple[int, int]]]:
    choices: list[list[tuple[int, int]]] = []
    dirs = [d for d in CARDINALS if d not in used_dirs]
    rng.shuffle(dirs)
    for dr, dc in dirs:
        max_len = 0
        cells: list[tuple[int, int]] = []
        for length in range(1, 4):
            nr = node[0] + dr * length
            nc = node[1] + dc * length
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                break
            if int(grid[nr, nc]) != BG:
                break
            cells.append((nr, nc))
            max_len = length
        if max_len:
            use_len = int(rng.integers(1, max_len + 1))
            choices.append(cells[:use_len])
    return choices


def _sample_even_rows(
    rng: np.random.Generator,
    row_choices: np.ndarray,
    start_row: int,
    end_row: int,
    node_count: int,
) -> list[int] | None:
    rows: list[int] = []
    prev = start_row
    n_even = node_count // 2
    for idx in range(n_even):
        if node_count % 2 == 0 and idx == n_even - 1:
            row = end_row
            if row == prev:
                return None
        else:
            forbid = {prev}
            if node_count % 2 == 1 and idx == n_even - 1:
                forbid.add(end_row)
            candidates = [int(r) for r in row_choices if int(r) not in forbid]
            if not candidates:
                return None
            row = int(candidates[int(rng.integers(len(candidates)))])
        rows.append(row)
        prev = row
    return rows


def _sample_odd_cols(
    rng: np.random.Generator,
    start_col: int,
    end_col: int,
    node_count: int,
) -> list[int] | None:
    n_odd = (node_count + 1) // 2
    interior = list(range(start_col + 2, end_col))
    if node_count % 2 == 1:
        interior = [c for c in interior if c < end_col]
        need = n_odd - 1
        if len(interior) < need:
            return None
        picks = sorted(int(c) for c in rng.choice(np.array(interior, dtype=int), size=need, replace=False))
        return picks + [end_col]
    interior = [c for c in interior if c < end_col]
    if len(interior) < n_odd:
        return None
    return sorted(int(c) for c in rng.choice(np.array(interior, dtype=int), size=n_odd, replace=False))


def _decorate_slot(grid: np.ndarray, r0: int, c0: int):
    grid[r0 : r0 + 5, c0] = MAROON
    grid[r0 : r0 + 5, c0 + 4] = MAROON
    grid[r0, c0 : c0 + 5] = MAROON
    grid[r0 + 4, c0 : c0 + 5] = MAROON


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 30))
    w = int(rng.integers(20, 30))
    grid = np.full((h, w), BG, dtype=int)

    sprite = _random_sprite(rng)
    node_count = int(rng.integers(3, 6))
    base_colors = [RED, BLUE, GREEN]
    while len(base_colors) < node_count:
        base_colors.append(int(rng.choice(np.array(NODE_COLORS, dtype=int))))
    rng.shuffle(base_colors)

    shape_r0 = int(rng.integers(2, h - 5))
    shape_c0 = int(rng.integers(1, 4))
    slot_r0 = int(rng.integers(2, h - 7))
    slot_c0 = int(w - 6 - rng.integers(1, 4))
    if slot_c0 - (shape_c0 + 3) < 6:
        return None

    grid[shape_r0 : shape_r0 + 3, shape_c0 : shape_c0 + 3] = sprite
    _decorate_slot(grid, slot_r0, slot_c0)

    start = (shape_r0 + 1, shape_c0 + 3)
    end = (slot_r0 + 2, slot_c0 - 1)
    if int(grid[start]) != BG or int(grid[end]) != BG:
        return None

    row_choices = np.arange(2, h - 2, dtype=int)
    even_rows = _sample_even_rows(rng, row_choices, start[0], end[0], node_count)
    odd_cols = _sample_odd_cols(rng, start[1], end[1], node_count)
    if even_rows is None or odd_cols is None:
        return None

    anchors = [start]
    odd_idx = 0
    even_idx = 0
    for idx in range(1, node_count + 1):
        if idx % 2 == 1:
            anchors.append((anchors[-1][0], odd_cols[odd_idx]))
            odd_idx += 1
        else:
            anchors.append((even_rows[even_idx], anchors[-1][1]))
            even_idx += 1
    anchors.append(end)

    for a, b in zip(anchors, anchors[1:]):
        cells = _cells_on_segment(a, b)
        if not _paint_path_cells(grid, cells, PIPE, {a, b}):
            return None

    for idx in range(1, len(anchors) - 1):
        node = anchors[idx]
        prev_anchor = anchors[idx - 1]
        next_anchor = anchors[idx + 1]
        used_dirs = {_step_toward(node, prev_anchor), _step_toward(node, next_anchor)}
        options = _branch_options(grid, node, used_dirs, rng)
        if not options:
            return None
        rng.shuffle(options)
        branch_count = 1
        if len(options) >= 2 and rng.random() < 0.35:
            branch_count = 2
        chosen: list[list[tuple[int, int]]] = []
        used_first_cells: set[tuple[int, int]] = set()
        for cells in options:
            if cells[0] in used_first_cells:
                continue
            chosen.append(cells)
            used_first_cells.add(cells[0])
            if len(chosen) == branch_count:
                break
        if len(chosen) != branch_count:
            return None
        for cells in chosen:
            if not _paint_path_cells(grid, cells, PIPE, set()):
                return None

    for idx, color in enumerate(base_colors, start=1):
        node_r, node_c = anchors[idx]
        grid[node_r, node_c] = int(color)

    sym_code = int(rng.integers(7))
    grid = _apply_symmetry(grid, sym_code)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    (shape_r0, shape_c0), _ = _find_shape(grid)
    (slot_r0, slot_c0), _ = _find_slot(grid)
    input_sprite = grid[shape_r0 : shape_r0 + 3, shape_c0 : shape_c0 + 3]
    output_sprite = out[slot_r0 + 1 : slot_r0 + 4, slot_c0 + 1 : slot_c0 + 4]
    if np.array_equal(input_sprite, output_sprite):
        return None
    if np.count_nonzero(output_sprite) < 3:
        return None
    if len({int(v) for v in grid.flatten() if int(v) in NODE_COLORS}) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {MAX_RETRIES} attempts")
