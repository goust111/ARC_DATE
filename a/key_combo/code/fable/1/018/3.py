from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)

DIR_NAME = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}

TEMPLATE_CATALOG = [
    (RIGHT, UP),
    (LEFT, UP),
    (DOWN, RIGHT, UP),
    (DOWN, LEFT, UP),
    (UP, RIGHT, UP),
    (UP, LEFT, UP),
    (RIGHT, DOWN, LEFT, UP),
    (RIGHT, UP, LEFT, UP),
    (LEFT, DOWN, RIGHT, UP),
    (LEFT, UP, RIGHT, UP),
]

TEMPLATE_WEIGHTS = [1, 1, 2, 2, 2, 2, 3, 2, 3, 2]

GEN_ATTEMPTS = 500


def _column_to_block(col: int, sep_cols: tuple[int, int], width: int) -> int | None:
    c1, c2 = map(int, sep_cols)
    if not (0 <= int(col) < int(width)):
        return None
    if col < c1:
        return 0
    if c1 < col < c2:
        return 1
    if c2 < col:
        return 2
    return None


def _reflect(direction: tuple[int, int], mirror_kind: str) -> tuple[int, int]:
    if mirror_kind == "/":
        mapping = {UP: RIGHT, RIGHT: UP, DOWN: LEFT, LEFT: DOWN}
    elif mirror_kind == "\\":
        mapping = {UP: LEFT, LEFT: UP, DOWN: RIGHT, RIGHT: DOWN}
    else:
        raise AssertionError("unknown mirror kind")
    if direction not in mapping:
        raise AssertionError("ray must be axis-aligned")
    return mapping[direction]


def _mirror_kind(in_dir: tuple[int, int], out_dir: tuple[int, int]) -> str:
    if (in_dir, out_dir) in {(UP, RIGHT), (RIGHT, UP), (DOWN, LEFT), (LEFT, DOWN)}:
        return "/"
    if (in_dir, out_dir) in {(UP, LEFT), (LEFT, UP), (DOWN, RIGHT), (RIGHT, DOWN)}:
        return "\\"
    raise AssertionError("mirror turn must be a 90-degree reflection")


def _arrow_cells_from_tip(tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, tip)
    if direction == UP:
        rod = [(r + i, c) for i in range(4)]
        bar = [(r + 1, c - 1), (r + 1, c), (r + 1, c + 1)]
    elif direction == DOWN:
        rod = [(r - i, c) for i in range(4)]
        bar = [(r - 1, c - 1), (r - 1, c), (r - 1, c + 1)]
    elif direction == LEFT:
        rod = [(r, c + i) for i in range(4)]
        bar = [(r - 1, c + 1), (r, c + 1), (r + 1, c + 1)]
    elif direction == RIGHT:
        rod = [(r, c - i) for i in range(4)]
        bar = [(r - 1, c - 1), (r, c - 1), (r + 1, c - 1)]
    else:
        raise AssertionError("invalid arrow direction")
    cells = sorted(set(rod + bar))
    assert len(cells) == 6
    return cells


def _decode_arrow(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = map(int, mask.shape)
    if int(mask.sum()) != 6:
        return None

    if (h, w) == (4, 3):
        row_counts = mask.sum(axis=1).astype(int)
        col_counts = mask.sum(axis=0).astype(int)
        if sorted(row_counts.tolist()) != [1, 1, 1, 3]:
            return None
        if sorted(col_counts.tolist()) != [1, 1, 4]:
            return None
        rod_col = int(np.argmax(col_counts))
        bar_rows = np.flatnonzero(row_counts == 3)
        if len(bar_rows) != 1:
            return None
        bar_row = int(bar_rows[0])
        if bar_row not in (1, 2):
            return None
        expected = np.zeros_like(mask, dtype=bool)
        expected[:, rod_col] = True
        expected[bar_row, :] = True
        if not np.array_equal(mask, expected):
            return None
        if bar_row == 1:
            return (0, rod_col), UP
        return (3, rod_col), DOWN

    if (h, w) == (3, 4):
        row_counts = mask.sum(axis=1).astype(int)
        col_counts = mask.sum(axis=0).astype(int)
        if sorted(row_counts.tolist()) != [1, 1, 4]:
            return None
        if sorted(col_counts.tolist()) != [1, 1, 1, 3]:
            return None
        rod_row = int(np.argmax(row_counts))
        bar_cols = np.flatnonzero(col_counts == 3)
        if len(bar_cols) != 1:
            return None
        bar_col = int(bar_cols[0])
        if bar_col not in (1, 2):
            return None
        expected = np.zeros_like(mask, dtype=bool)
        expected[rod_row, :] = True
        expected[:, bar_col] = True
        if not np.array_equal(mask, expected):
            return None
        if bar_col == 1:
            return (rod_row, 0), LEFT
        return (rod_row, 3), RIGHT

    return None


def _find_layout(grid: np.ndarray) -> tuple[int, int, tuple[int, int]]:
    h, w = map(int, grid.shape)
    candidates: list[tuple[int, int, tuple[int, int]]] = []
    for r in range(1, h - 1):
        if not np.all(grid[r] == grid[r, 0]):
            continue
        sep_color = int(grid[r, 0])
        sep_cols = tuple(c for c in range(w) if np.all(grid[:r, c] == sep_color))
        if len(sep_cols) != 2:
            continue
        c1, c2 = map(int, sep_cols)
        widths = (c1, c2 - c1 - 1, w - c2 - 1)
        if min(widths) <= 0:
            continue
        if widths[0] != widths[1] or widths[1] != widths[2]:
            continue
        candidates.append((r, sep_color, (c1, c2)))
    assert len(candidates) == 1, "expected one horizontal separator layout"
    return candidates[0]


def _find_arrow(lower: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], int]:
    bg = int(get_bg(lower))
    comps = find_components(lower, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    matches: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for comp in comps:
        if int(comp["area"]) != 6:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        cropped = crop_bbox(lower, (r0, c0, r1, c1))
        color = int(comp["color"])
        mask = cropped == color
        decoded = _decode_arrow(mask)
        if decoded is None:
            continue
        (tr, tc), direction = decoded
        matches.append(((r0 + tr, c0 + tc), direction, color))
    assert len(matches) == 1, "expected one asymmetric T arrow"
    return matches[0]


def _find_mirrors(lower: np.ndarray) -> tuple[dict[tuple[int, int], str], list[set[tuple[int, int]]]]:
    bg = int(get_bg(lower))
    comps = find_components(
        lower,
        bg=bg,
        colors=[Color.RED],
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    mirror_map: dict[tuple[int, int], str] = {}
    mirror_components: list[set[tuple[int, int]]] = []
    for comp in comps:
        coords = comp["coords"]
        diffs = {int(r - c) for r, c in coords}
        sums = {int(r + c) for r, c in coords}
        if len(diffs) == 1 and len(sums) > 1:
            kind = "\\"
        elif len(sums) == 1 and len(diffs) > 1:
            kind = "/"
        else:
            raise AssertionError("red mirror must be a diagonal line")
        cells = {tuple(map(int, rc)) for rc in coords.tolist()}
        mirror_components.append(cells)
        for cell in cells:
            mirror_map[cell] = kind
    assert mirror_components, "expected at least one mirror"
    return mirror_map, mirror_components


def _trace_ray(
    lower: np.ndarray,
    tip: tuple[int, int],
    direction: tuple[int, int],
    mirror_map: dict[tuple[int, int], str],
) -> tuple[int, list[tuple[int, int]]]:
    h, w = map(int, lower.shape)
    r, c = map(int, tip)
    dr, dc = map(int, direction)
    visited: set[tuple[int, int, int, int]] = set()
    hits: list[tuple[int, int]] = []

    while True:
        state = (r, c, dr, dc)
        assert state not in visited, "ray entered a loop"
        visited.add(state)

        nr, nc = r + dr, c + dc
        if nr < 0:
            assert dr == -1, "ray may leave only upward"
            return nc, hits
        assert 0 <= nr < h and 0 <= nc < w, "ray left the launch zone sideways or downward"

        r, c = nr, nc
        if (r, c) in mirror_map:
            dr, dc = _reflect((dr, dc), mirror_map[(r, c)])
            hits.append((r, c))


def _solve(grid: np.ndarray) -> tuple[int, list[np.ndarray], dict]:
    g = np.asarray(grid, dtype=int)
    sep_row, sep_color, sep_cols = _find_layout(g)
    c1, c2 = map(int, sep_cols)
    blocks = [
        g[:sep_row, :c1].copy(),
        g[:sep_row, c1 + 1 : c2].copy(),
        g[:sep_row, c2 + 1 :].copy(),
    ]
    lower = g[sep_row + 1 :, :]
    tip, direction, arrow_color = _find_arrow(lower)
    mirror_map, mirror_components = _find_mirrors(lower)
    exit_col, hits = _trace_ray(lower, tip, direction, mirror_map)
    selected = _column_to_block(exit_col, sep_cols, g.shape[1])
    assert selected is not None, "ray exited through a separator column"
    return selected, blocks, {
        "sep_row": int(sep_row),
        "sep_color": int(sep_color),
        "sep_cols": tuple(map(int, sep_cols)),
        "lower": lower,
        "tip": tuple(map(int, tip)),
        "direction": direction,
        "arrow_color": int(arrow_color),
        "mirror_map": mirror_map,
        "mirror_components": mirror_components,
        "hits": hits,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the arrow's reflected ray to select one of three candidate pattern blocks.

    Input structure:
    - The upper region contains three equal-width rectangular pattern blocks arranged side by side.
    - Two vertical separator bars split the blocks, and one horizontal separator bar splits them from a lower launch zone.
    - The launch zone contains one six-cell monochrome asymmetric T arrow and one or more red diagonal mirror lines.

    Rule:
    - Determine the arrow direction from which end of the four-cell shaft lies closest to the three-cell crossbar.
    - Start a ray from that tip and move one grid cell at a time in the pointing direction.
    - Whenever the ray enters a red diagonal mirror, reflect it by ninety degrees according to the mirror slash.
    - The ray eventually leaves the launch zone upward; output the top block whose column interval contains that exit column.

    Key insight:
    - The mirrors do not mark the answer directly; they convert the arrow into an optical pointer whose final upward exit chooses the block.
    """
    selected, blocks, _ = _solve(np.asarray(grid, dtype=int))
    return blocks[selected].copy()


def _block_palette(exclude: set[int]) -> list[int]:
    return [c for c in Color.ALL_COLORS if c not in exclude]


def _make_block(height: int, width: int, palette: list[int]) -> np.ndarray:
    colors = random.sample(palette, k=min(len(palette), random.randint(3, 4)))
    while len(colors) < 4:
        colors.append(colors[-1])
    base, c1, c2, c3 = colors[:4]
    rr, cc = np.indices((height, width))
    block = np.full((height, width), base, dtype=int)
    mode = random.choice(["checker", "frame", "bands", "quadrants", "diamond"])

    if mode == "checker":
        block[(rr + cc + random.randint(0, 1)) % 2 == 0] = c1
        block[random.randrange(height), :] = c2
        block[:, random.randrange(width)] = c3
    elif mode == "frame":
        block[0, :] = c1
        block[-1, :] = c1
        block[:, 0] = c1
        block[:, -1] = c1
        block[1:-1, 1:-1] = base
        n = min(height, width)
        idx = np.arange(n)
        block[idx, idx] = c2
        block[idx, width - 1 - idx] = c3
    elif mode == "bands":
        period = random.choice([2, 3])
        block[(rr + random.randint(0, period - 1)) % period == 0] = c1
        block[(cc + random.randint(0, period - 1)) % period == 0] = c2
        r0 = random.randint(1, max(1, height - 3))
        c0 = random.randint(1, max(1, width - 3))
        block[r0 : min(height - 1, r0 + 2), c0 : min(width - 1, c0 + 2)] = c3
    elif mode == "quadrants":
        sr = random.randint(1, height - 2)
        sc = random.randint(1, width - 2)
        block[:sr, :sc] = c1
        block[:sr, sc:] = c2
        block[sr:, :sc] = c3
        block[sr:, sc:] = base
        block[sr, :] = c1
        block[:, sc] = c2
    else:
        cr = random.randint(1, height - 2)
        cc0 = random.randint(1, width - 2)
        radius = random.randint(1, min(height, width) // 2)
        diamond = np.abs(rr - cr) + np.abs(cc - cc0) <= radius
        block[diamond] = c1
        block[np.abs(rr - cc) <= 0] = c2
        block[np.abs(rr + cc - (width - 1)) <= 0] = c3

    if np.all(block == block[0, 0]):
        block[height // 2, width // 2] = c1
        block[height // 2, max(0, width // 2 - 1)] = c2
    return block


def _distinct_blocks(height: int, width: int, exclude: set[int]) -> list[np.ndarray]:
    palette = _block_palette(exclude)
    for _ in range(200):
        blocks = [_make_block(height, width, palette) for _ in range(3)]
        keys = {b.tobytes() for b in blocks}
        if len(keys) == 3:
            return blocks
    raise RuntimeError("failed to generate distinct blocks")


def _walk_segment(start: tuple[int, int], direction: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    r, c = map(int, start)
    dr, dc = map(int, direction)
    return [(r + dr * i, c + dc * i) for i in range(1, steps + 1)]


def _trace_with_mirrors(
    lower: np.ndarray,
    tip: tuple[int, int],
    direction: tuple[int, int],
    mirror_components: list[set[tuple[int, int]]],
) -> tuple[int, list[tuple[int, int]]]:
    mirror_map: dict[tuple[int, int], str] = {}
    for cells in mirror_components:
        diffs = {r - c for r, c in cells}
        sums = {r + c for r, c in cells}
        if len(diffs) == 1 and len(sums) > 1:
            kind = "\\"
        elif len(sums) == 1 and len(diffs) > 1:
            kind = "/"
        else:
            raise AssertionError("invalid mirror component")
        for cell in cells:
            mirror_map[cell] = kind
    return _trace_ray(lower, tip, direction, mirror_map)


def _safe_selected_block(
    lower: np.ndarray,
    tip: tuple[int, int],
    direction: tuple[int, int],
    mirror_components: list[set[tuple[int, int]]],
    sep_cols: tuple[int, int],
    width: int,
) -> int | None:
    try:
        exit_col, _ = _trace_with_mirrors(lower, tip, direction, mirror_components)
    except AssertionError:
        return None
    return _column_to_block(exit_col, sep_cols, width)


def _distance_gt_one(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) > 1


def _choose_mirror_cells(
    hit: tuple[int, int],
    kind: str,
    shape: tuple[int, int],
    forbidden_path: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
    existing_red: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = map(int, shape)
    dr, dc = (1, 1) if kind == "\\" else (1, -1)
    options: list[set[tuple[int, int]]] = []
    for length in (2, 3, 4):
        for before in range(length):
            after = length - 1 - before
            cells = {
                (int(hit[0] + dr * t), int(hit[1] + dc * t))
                for t in range(-before, after + 1)
            }
            if not all(0 <= r < h and 0 <= c < w for r, c in cells):
                continue
            if any(cell != hit and cell in forbidden_path for cell in cells):
                continue
            if any(cell in arrow_cells for cell in cells):
                continue
            if any(not _distance_gt_one(cell, other) for cell in cells for other in existing_red):
                continue
            options.append(cells)
    if not options:
        return None
    return random.choice(options)


def _build_path(
    launch_h: int,
    width: int,
    exit_col: int,
    directions: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], list[tuple[int, int]], list[set[tuple[int, int]]]] | None:
    mirror_count = len(directions) - 1
    for _ in range(300):
        points: list[tuple[int, int] | None] = [None] * (mirror_count + 1)
        points[-1] = (random.randint(2, launch_h - 3), int(exit_col))
        ok = True
        for idx in range(mirror_count - 1, -1, -1):
            length = random.randint(2, 6)
            tr, tc = points[idx + 1]
            dr, dc = directions[idx]
            origin = (int(tr - dr * length), int(tc - dc * length))
            if not (0 <= origin[0] < launch_h and 0 <= origin[1] < width):
                ok = False
                break
            points[idx] = origin
        if not ok:
            continue

        tip = points[0]
        mirrors = [tuple(map(int, p)) for p in points[1:]]
        arrow_cells = set(_arrow_cells_from_tip(tip, directions[0]))
        if not all(0 <= r < launch_h and 0 <= c < width for r, c in arrow_cells):
            continue

        seen: set[tuple[int, int]] = set()
        path_cells: list[tuple[int, int]] = []
        cur = tip
        for nxt, direction in zip(mirrors, directions[:-1]):
            steps = abs(int(nxt[0]) - int(cur[0])) + abs(int(nxt[1]) - int(cur[1]))
            cells = _walk_segment(cur, direction, steps)
            if any(cell in seen for cell in cells):
                ok = False
                break
            seen.update(cells)
            path_cells.extend(cells)
            cur = nxt
        if not ok:
            continue

        final_cells = [(r, cur[1]) for r in range(cur[0] - 1, -1, -1)]
        if any(cell in seen for cell in final_cells):
            continue
        if any(cell in arrow_cells for cell in path_cells + final_cells):
            continue

        forbidden_path = set(path_cells + final_cells) - set(mirrors)
        mirror_components: list[set[tuple[int, int]]] = []
        existing_red: set[tuple[int, int]] = set()
        for hit, in_dir, out_dir in zip(mirrors, directions[:-1], directions[1:]):
            cells = _choose_mirror_cells(
                hit,
                _mirror_kind(in_dir, out_dir),
                (launch_h, width),
                forbidden_path,
                arrow_cells,
                existing_red,
            )
            if cells is None:
                ok = False
                break
            mirror_components.append(cells)
            existing_red.update(cells)
        if not ok:
            continue

        return tip, mirrors, mirror_components
    return None


def _render_launch_zone(
    launch_h: int,
    width: int,
    tip: tuple[int, int],
    direction: tuple[int, int],
    arrow_color: int,
    mirror_components: list[set[tuple[int, int]]],
) -> np.ndarray:
    lower = np.full((launch_h, width), Color.BLACK, dtype=int)
    for cells in mirror_components:
        for r, c in cells:
            lower[r, c] = Color.RED
    for r, c in _arrow_cells_from_tip(tip, direction):
        lower[r, c] = int(arrow_color)
    return lower


def _assemble_grid(
    blocks: list[np.ndarray],
    lower: np.ndarray,
    sep_color: int,
) -> np.ndarray:
    top_h, block_w = map(int, blocks[0].shape)
    launch_h, width = map(int, lower.shape)
    c1 = block_w
    c2 = block_w * 2 + 1
    grid = np.full((top_h + 1 + launch_h, width), Color.BLACK, dtype=int)
    grid[:top_h, :c1] = blocks[0]
    grid[:top_h, c1] = int(sep_color)
    grid[:top_h, c1 + 1 : c2] = blocks[1]
    grid[:top_h, c2] = int(sep_color)
    grid[:top_h, c2 + 1 :] = blocks[2]
    grid[top_h, :] = int(sep_color)
    grid[top_h + 1 :, :] = lower
    return grid


def _expected_block(
    blocks: list[np.ndarray],
    target_idx: int,
) -> np.ndarray:
    return blocks[int(target_idx)].copy()


def _gen_attempt() -> np.ndarray | None:
    block_w = random.randint(5, 8)
    top_h = random.randint(5, 8)
    width = block_w * 3 + 2
    max_launch_h = 29 - top_h
    if max_launch_h < 8:
        return None
    launch_h = random.randint(8, min(18, max_launch_h))

    sep_color, arrow_color = random.sample(
        [c for c in Color.NOT_BLACK if c != Color.RED],
        2,
    )
    blocks = _distinct_blocks(top_h, block_w, {Color.RED, sep_color, arrow_color})
    target_idx = random.randrange(3)
    block_starts = [0, block_w + 1, 2 * block_w + 2]
    exit_col = block_starts[target_idx] + random.randrange(block_w)

    directions = random.choices(TEMPLATE_CATALOG, weights=TEMPLATE_WEIGHTS, k=1)[0]
    built = _build_path(launch_h, width, exit_col, directions)
    if built is None:
        return None
    tip, hits, mirror_components = built

    sep_cols = (block_w, 2 * block_w + 1)
    tip_block = _column_to_block(tip[1], sep_cols, width)
    if tip_block is not None and tip_block == target_idx:
        return None

    lower = _render_launch_zone(launch_h, width, tip, directions[0], arrow_color, mirror_components)

    try:
        traced_exit, traced_hits = _trace_with_mirrors(lower, tip, directions[0], mirror_components)
    except AssertionError:
        return None
    if traced_exit != exit_col or traced_hits != hits:
        return None

    selected = _column_to_block(traced_exit, sep_cols, width)
    if selected != target_idx:
        return None

    if _safe_selected_block(lower, tip, directions[0], [], sep_cols, width) == target_idx:
        return None

    for i in range(len(mirror_components)):
        reduced = [cells for j, cells in enumerate(mirror_components) if j != i]
        if _safe_selected_block(lower, tip, directions[0], reduced, sep_cols, width) == target_idx:
            return None

    grid = _assemble_grid(blocks, lower, sep_color)
    expected = _expected_block(blocks, target_idx)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
