import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("T", "B", "L", "R")
RETRY_LIMIT = 400

PERIMETER_LAYOUTS = (
    (2, 2),
    (2, 2),
    (2, 3),
    (2, 3),
    (3, 2),
    (3, 2),
    (1, 4),
    (4, 1),
    (1, 5),
    (5, 1),
    (1, 6),
    (6, 1),
)
MATCHED_ONLY_LAYOUTS = ((2, 2),)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _separator_color(grid: np.ndarray, bg: int) -> int:
    best_score = None
    best_color = None
    for value in np.unique(grid):
        color = int(value)
        if color == bg:
            continue
        row_hits = int(np.all(grid == color, axis=1).sum())
        col_hits = int(np.all(grid == color, axis=0).sum())
        if row_hits == 0 and col_hits == 0:
            continue
        score = (row_hits + col_hits, row_hits > 0, col_hits > 0)
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
    assert best_color is not None, "expected full-row or full-column separator lines"
    return best_color


def _segments_from_lines(lines: np.ndarray, size: int) -> list[tuple[int, int]]:
    if lines.size == 0:
        return [(0, size)]

    segments: list[tuple[int, int]] = []
    start = 0
    for idx in sorted(map(int, lines.tolist())):
        if start < idx:
            segments.append((start, idx))
        start = idx + 1
    if start < size:
        segments.append((start, size))
    return segments


def _side_color(line: np.ndarray, bg: int, sep: int) -> int | None:
    core = line[1:-1] if line.size > 2 else line
    values = np.unique(core)
    if len(values) != 1:
        return None
    color = int(values[0])
    if color in (bg, sep):
        return None
    return color


def _panel_side_colors(panel: np.ndarray, bg: int, sep: int) -> dict[str, int | None]:
    return {
        "T": _side_color(panel[0, :], bg, sep),
        "B": _side_color(panel[-1, :], bg, sep),
        "L": _side_color(panel[:, 0], bg, sep),
        "R": _side_color(panel[:, -1], bg, sep),
    }


def _pair_neighbors(side_maps: list[dict[str, int | None]]) -> list[dict[str, int | None]]:
    neighbors = [{side: None for side in SIDES} for _ in side_maps]

    def connect(src_side: str, dst_side: str) -> None:
        src_by_color: dict[int, list[int]] = {}
        dst_by_color: dict[int, list[int]] = {}
        for idx, side_map in enumerate(side_maps):
            src_color = side_map[src_side]
            dst_color = side_map[dst_side]
            if src_color is not None:
                src_by_color.setdefault(int(src_color), []).append(idx)
            if dst_color is not None:
                dst_by_color.setdefault(int(dst_color), []).append(idx)

        for color, src_ids in src_by_color.items():
            dst_ids = dst_by_color.get(color, [])
            if len(src_ids) != 1 or len(dst_ids) != 1:
                continue
            src_id = src_ids[0]
            dst_id = dst_ids[0]
            if src_id == dst_id:
                continue
            prev_src = neighbors[src_id][src_side]
            prev_dst = neighbors[dst_id][dst_side]
            assert prev_src in (None, dst_id)
            assert prev_dst in (None, src_id)
            neighbors[src_id][src_side] = dst_id
            neighbors[dst_id][dst_side] = src_id

    connect("R", "L")
    connect("B", "T")
    return neighbors


def _layout_coords(neighbors: list[dict[str, int | None]]) -> tuple[dict[int, tuple[int, int]], int, int]:
    moves = {"T": (-1, 0), "B": (1, 0), "L": (0, -1), "R": (0, 1)}
    roots = [idx for idx, nbs in enumerate(neighbors) if nbs["L"] is None and nbs["T"] is None]
    assert roots, "expected a top-left panel"

    root = roots[0]
    coords = {root: (0, 0)}
    queue: deque[int] = deque([root])

    while queue:
        idx = queue.popleft()
        row, col = coords[idx]
        for side, other in neighbors[idx].items():
            if other is None:
                continue
            dr, dc = moves[side]
            wanted = (row + dr, col + dc)
            if other in coords:
                assert coords[other] == wanted, "inconsistent handshake layout"
                continue
            coords[other] = wanted
            queue.append(other)

    assert len(coords) == len(neighbors), "panel graph must be connected"
    min_row = min(row for row, _ in coords.values())
    min_col = min(col for _, col in coords.values())
    normalized = {idx: (row - min_row, col - min_col) for idx, (row, col) in coords.items()}

    row_count = 1 + max(row for row, _ in normalized.values())
    col_count = 1 + max(col for _, col in normalized.values())
    assert len(set(normalized.values())) == len(neighbors), "panel positions must be unique"
    assert row_count * col_count == len(neighbors), "handshakes must reconstruct a full rectangle"
    return normalized, row_count, col_count


def _crop_present_bars(panel: np.ndarray, side_map: dict[str, int | None]) -> np.ndarray:
    top = 1 if side_map["T"] is not None else 0
    bottom = 1 if side_map["B"] is not None else 0
    left = 1 if side_map["L"] is not None else 0
    right = 1 if side_map["R"] is not None else 0
    return panel[top : panel.shape[0] - bottom, left : panel.shape[1] - right]


def _crop_full_frame(panel: np.ndarray) -> np.ndarray:
    assert panel.shape[0] >= 3 and panel.shape[1] >= 3, "frame crop needs a one-cell border"
    return panel[1:-1, 1:-1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble shuffled separator-grid panels by matching same-color edge bars and then discard the bars.

    Input structure:
    - The background is black, and full single-cell grey rows or columns split the
      canvas into 4 to 6 rectangular panels.
    - Each panel contains a colored interior pattern that stays away from the panel
      boundary.
    - Some panel sides carry a continuous one-cell-thick edge bar whose color is
      neither black nor grey.

    Rule:
    - Read each panel's top, bottom, left, and right edge-bar colors.
    - A right-side color matching another panel's left-side color means those two
      panels are horizontal neighbors; likewise bottom-to-top matches give vertical
      neighbors.
    - Reconstruct the unique rectangular panel layout, remove the edge bars, and
      tile only the interior content without grey separator gaps.

    Key insight:
    - The colored edge bars are not decoration; they are handshake labels that
      encode the true 2D adjacency graph independently of the panels' scrambled
      positions in the input grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep = _separator_color(g, bg)
    row_blocks = _segments_from_lines(np.flatnonzero(np.all(g == sep, axis=1)), g.shape[0])
    col_blocks = _segments_from_lines(np.flatnonzero(np.all(g == sep, axis=0)), g.shape[1])

    panels: list[np.ndarray] = []
    for r0, r1 in row_blocks:
        for c0, c1 in col_blocks:
            panels.append(g[r0:r1, c0:c1])
    assert 4 <= len(panels) <= 6, "expected 4 to 6 separator-defined panels"

    side_maps = [_panel_side_colors(panel, bg, sep) for panel in panels]
    neighbors = _pair_neighbors(side_maps)
    coords, out_rows, out_cols = _layout_coords(neighbors)

    actual_crops = [_crop_present_bars(panel, side_map) for panel, side_map in zip(panels, side_maps)]
    actual_shapes = {tuple(crop.shape) for crop in actual_crops}
    if len(actual_shapes) == 1:
        pieces = actual_crops
    else:
        pieces = [_crop_full_frame(panel) for panel in panels]
        frame_shapes = {tuple(piece.shape) for piece in pieces}
        assert len(frame_shapes) == 1, "all panels must share one interior size after bar removal"

    piece_h, piece_w = pieces[0].shape
    out = np.full((out_rows * piece_h, out_cols * piece_w), bg, dtype=int)
    for idx, piece in enumerate(pieces):
        row, col = coords[idx]
        out[row * piece_h : (row + 1) * piece_h, col * piece_w : (col + 1) * piece_w] = piece
    return out


def _factor_pairs(n_panels: int) -> list[tuple[int, int]]:
    pairs = []
    for rows in range(1, n_panels + 1):
        if n_panels % rows == 0:
            pairs.append((rows, n_panels // rows))
    return pairs


def _size_options(out_rows: int, out_cols: int, in_rows: int, in_cols: int, mode: str) -> list[tuple[int, int]]:
    border = 2 if mode == "perimeter" else 1
    options = []
    for inner_h in range(3, 7):
        for inner_w in range(3, 7):
            panel_h = inner_h + border
            panel_w = inner_w + border
            input_h = in_rows * panel_h + max(0, in_rows - 1)
            input_w = in_cols * panel_w + max(0, in_cols - 1)
            output_h = out_rows * inner_h
            output_w = out_cols * inner_w
            if max(input_h, input_w, output_h, output_w) <= 30:
                options.append((inner_h, inner_w))
    return options


def _choose_pair_colors(
    pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    pattern_by_coord: dict[tuple[int, int], int],
    rng: random.Random,
) -> dict[tuple[tuple[int, int], tuple[int, int]], int] | None:
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    assignments: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

    def rec(pos: int, used: set[int]) -> bool:
        if pos == len(shuffled):
            return True
        pair = shuffled[pos]
        forbidden = {pattern_by_coord[pair[0]], pattern_by_coord[pair[1]]}
        candidates = [color for color in PALETTE if color not in forbidden and color not in used]
        rng.shuffle(candidates)
        for color in candidates:
            assignments[pair] = color
            if rec(pos + 1, used | {color}):
                return True
            del assignments[pair]
        return False

    if not rec(0, set()):
        return None
    return assignments


def _choose_outer_colors(
    out_rows: int,
    out_cols: int,
    pattern_by_coord: dict[tuple[int, int], int],
    horizontal_colors: set[int],
    vertical_colors: set[int],
    rng: random.Random,
) -> dict[str, int] | None:
    top_patterns = {pattern_by_coord[(0, col)] for col in range(out_cols)}
    bottom_patterns = {pattern_by_coord[(out_rows - 1, col)] for col in range(out_cols)}
    left_patterns = {pattern_by_coord[(row, 0)] for row in range(out_rows)}
    right_patterns = {pattern_by_coord[(row, out_cols - 1)] for row in range(out_rows)}

    top_candidates = [c for c in PALETTE if c not in vertical_colors and c not in top_patterns]
    bottom_candidates = [c for c in PALETTE if c not in vertical_colors and c not in bottom_patterns]
    left_candidates = [c for c in PALETTE if c not in horizontal_colors and c not in left_patterns]
    right_candidates = [c for c in PALETTE if c not in horizontal_colors and c not in right_patterns]
    rng.shuffle(top_candidates)
    rng.shuffle(bottom_candidates)
    rng.shuffle(left_candidates)
    rng.shuffle(right_candidates)

    for top in top_candidates:
        for bottom in bottom_candidates:
            if bottom == top:
                continue
            for left in left_candidates:
                for right in right_candidates:
                    if right == left:
                        continue
                    return {"T": top, "B": bottom, "L": left, "R": right}
    return None


def _shape_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    if height == 1 or width == 1:
        mask = np.zeros((height, width), dtype=bool)
        length = rng.randint(1, max(height, width))
        if height == 1:
            start = rng.randint(0, width - length)
            mask[0, start : start + length] = True
        else:
            start = rng.randint(0, height - length)
            mask[start : start + length, 0] = True
        return mask

    kind = rng.choice(("rect", "blob", "ell", "plus"))

    if kind == "rect":
        h = rng.randint(1, height)
        w = rng.randint(1, width)
        r0 = rng.randint(0, height - h)
        c0 = rng.randint(0, width - w)
        mask = np.zeros((height, width), dtype=bool)
        mask[r0 : r0 + h, c0 : c0 + w] = True
        return mask

    if kind == "plus":
        mask = np.zeros((height, width), dtype=bool)
        cr = rng.randrange(height)
        cc = rng.randrange(width)
        up = rng.randint(0, cr)
        down = rng.randint(0, height - 1 - cr)
        left = rng.randint(0, cc)
        right = rng.randint(0, width - 1 - cc)
        mask[cr - up : cr + down + 1, cc] = True
        mask[cr, cc - left : cc + right + 1] = True
        return mask

    if kind == "ell":
        mask = np.zeros((height, width), dtype=bool)
        r0 = rng.randint(0, height - 2)
        c0 = rng.randint(0, width - 2)
        arm_h = rng.randint(1, height - r0)
        arm_w = rng.randint(1, width - c0)
        mask[r0 : r0 + arm_h, c0] = True
        mask[r0 + arm_h - 1, c0 : c0 + arm_w] = True
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        return mask

    target = rng.randint(1, height * width)
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for row, col in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for row, col in cells:
        mask[row, col] = True
    return mask


def _make_interior(inner_h: int, inner_w: int, color: int, rng: random.Random) -> np.ndarray:
    interior = np.full((inner_h, inner_w), BG, dtype=int)
    usable_h = inner_h - 2
    usable_w = inner_w - 2
    mask = _shape_mask(usable_h, usable_w, rng)
    interior[1:-1, 1:-1][mask] = color
    return interior


def _render_panel(interior: np.ndarray, side_map: dict[str, int | None]) -> np.ndarray:
    top = 1 if side_map["T"] is not None else 0
    bottom = 1 if side_map["B"] is not None else 0
    left = 1 if side_map["L"] is not None else 0
    right = 1 if side_map["R"] is not None else 0

    panel_h = interior.shape[0] + top + bottom
    panel_w = interior.shape[1] + left + right
    panel = np.full((panel_h, panel_w), BG, dtype=int)
    r0 = top
    c0 = left
    panel[r0 : r0 + interior.shape[0], c0 : c0 + interior.shape[1]] = interior

    if side_map["L"] is not None:
        panel[:, 0] = int(side_map["L"])
    if side_map["R"] is not None:
        panel[:, -1] = int(side_map["R"])
    if side_map["T"] is not None:
        panel[0, :] = int(side_map["T"])
    if side_map["B"] is not None:
        panel[-1, :] = int(side_map["B"])
    return panel


def _assemble_grid(patches: list[np.ndarray], rows: int, cols: int) -> np.ndarray:
    patch_h, patch_w = patches[0].shape
    grid_h = rows * patch_h + max(0, rows - 1)
    grid_w = cols * patch_w + max(0, cols - 1)
    grid = np.full((grid_h, grid_w), SEP, dtype=int)

    idx = 0
    for row in range(rows):
        r0 = row * (patch_h + 1)
        for col in range(cols):
            c0 = col * (patch_w + 1)
            grid[r0 : r0 + patch_h, c0 : c0 + patch_w] = patches[idx]
            idx += 1
    return grid


def _build_expected(interiors: dict[tuple[int, int], np.ndarray], rows: int, cols: int) -> np.ndarray:
    inner_h, inner_w = next(iter(interiors.values())).shape
    out = np.full((rows * inner_h, cols * inner_w), BG, dtype=int)
    for row in range(rows):
        for col in range(cols):
            patch = interiors[(row, col)]
            out[row * inner_h : (row + 1) * inner_h, col * inner_w : (col + 1) * inner_w] = patch
    return out


def _sample_layout_and_sizes(rng: random.Random) -> tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    mode = rng.choice(("perimeter", "matched"))
    out_layout = rng.choice(PERIMETER_LAYOUTS if mode == "perimeter" else MATCHED_ONLY_LAYOUTS)
    n_panels = out_layout[0] * out_layout[1]
    viable = []
    for in_layout in _factor_pairs(n_panels):
        sizes = _size_options(out_layout[0], out_layout[1], in_layout[0], in_layout[1], mode)
        if sizes:
            viable.append((in_layout, sizes))
    if not viable:
        return None

    alt = [item for item in viable if item[0] != out_layout]
    in_layout, sizes = rng.choice(alt if alt and rng.random() < 0.7 else viable)
    weights = [inner_h * inner_w for inner_h, inner_w in sizes]
    inner_h, inner_w = rng.choices(sizes, weights=weights, k=1)[0]
    return mode, out_layout, in_layout, (inner_h, inner_w)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    sampled = _sample_layout_and_sizes(rng)
    if sampled is None:
        return None
    mode, out_layout, in_layout, inner_size = sampled
    out_rows, out_cols = out_layout
    n_panels = out_rows * out_cols
    inner_h, inner_w = inner_size

    pattern_colors = rng.sample(PALETTE, n_panels)
    coords = [(row, col) for row in range(out_rows) for col in range(out_cols)]
    pattern_by_coord = {coord: color for coord, color in zip(coords, pattern_colors)}

    horizontal_pairs = [((row, col), (row, col + 1)) for row in range(out_rows) for col in range(out_cols - 1)]
    vertical_pairs = [((row, col), (row + 1, col)) for row in range(out_rows - 1) for col in range(out_cols)]
    horizontal = _choose_pair_colors(horizontal_pairs, pattern_by_coord, rng)
    vertical = _choose_pair_colors(vertical_pairs, pattern_by_coord, rng)
    if horizontal is None or vertical is None:
        return None

    outer_colors = None
    if mode == "perimeter":
        outer_colors = _choose_outer_colors(
            out_rows,
            out_cols,
            pattern_by_coord,
            set(horizontal.values()),
            set(vertical.values()),
            rng,
        )
        if outer_colors is None:
            return None

    side_by_coord = {coord: {side: None for side in SIDES} for coord in coords}
    for left_coord, right_coord in horizontal_pairs:
        color = horizontal[(left_coord, right_coord)]
        side_by_coord[left_coord]["R"] = color
        side_by_coord[right_coord]["L"] = color
    for top_coord, bottom_coord in vertical_pairs:
        color = vertical[(top_coord, bottom_coord)]
        side_by_coord[top_coord]["B"] = color
        side_by_coord[bottom_coord]["T"] = color

    if outer_colors is not None:
        for col in range(out_cols):
            side_by_coord[(0, col)]["T"] = outer_colors["T"]
            side_by_coord[(out_rows - 1, col)]["B"] = outer_colors["B"]
        for row in range(out_rows):
            side_by_coord[(row, 0)]["L"] = outer_colors["L"]
            side_by_coord[(row, out_cols - 1)]["R"] = outer_colors["R"]

    interiors = {
        coord: _make_interior(inner_h, inner_w, pattern_by_coord[coord], rng)
        for coord in coords
    }
    panels = {coord: _render_panel(interiors[coord], side_by_coord[coord]) for coord in coords}

    panel_shapes = {tuple(panel.shape) for panel in panels.values()}
    if len(panel_shapes) != 1:
        return None

    expected = _build_expected(interiors, out_rows, out_cols)
    order = coords[:]
    rng.shuffle(order)
    if order == coords:
        order = order[1:] + order[:1]
    patches = [panels[coord] for coord in order]
    inp = _assemble_grid(patches, in_layout[0], in_layout[1])

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid handshake-panel ARC input")
