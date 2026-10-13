import random
from typing import List, Sequence, Tuple

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
PROTO_COLORS = (1, 2, 3, 6, 7, 8, 9)
GEN_ATTEMPTS = 1000


def _bbox(mask: np.ndarray) -> Tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _cluster_axis(values: Sequence[int], expected_groups: int) -> List[List[int]]:
    vals = sorted(set(int(v) for v in values))
    assert vals, "expected at least one occupied coordinate"
    groups: List[List[int]] = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] >= 3:
            groups.append([v])
        else:
            groups[-1].append(v)
    assert len(groups) == expected_groups, (groups, expected_groups)
    return groups


def _stretch(proto: np.ndarray, row_factors: Sequence[int], col_factors: Sequence[int]) -> np.ndarray:
    out_h = sum(int(v) for v in row_factors)
    out_w = sum(int(v) for v in col_factors)
    out = np.zeros((out_h, out_w), dtype=int)

    row_offsets = [0]
    for v in row_factors:
        row_offsets.append(row_offsets[-1] + int(v))
    col_offsets = [0]
    for v in col_factors:
        col_offsets.append(col_offsets[-1] + int(v))

    for i in range(proto.shape[0]):
        for j in range(proto.shape[1]):
            out[row_offsets[i] : row_offsets[i + 1], col_offsets[j] : col_offsets[j + 1]] = int(proto[i, j])
    return out


def _parse_instance(grid: np.ndarray) -> Tuple[np.ndarray, List[int], List[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = [int(r) for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(sep_rows) == 1, "expected exactly one full grey separator row"
    sep = sep_rows[0]
    assert 0 < sep < g.shape[0] - 1, "separator must split top and bottom panels"

    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.size > 0 and bottom.size > 0
    assert not np.any(top == YELLOW), "yellow should only appear in anchor panel"

    proto_box = _bbox(top != BLACK)
    assert proto_box is not None, "missing prototype"
    proto = _crop(top, proto_box)
    assert 3 <= proto.shape[0] <= 4 and 3 <= proto.shape[1] <= 4, proto.shape
    assert not np.any(proto == BLACK), "prototype must occupy its compact bbox"
    assert not np.any(proto == GREY), "grey is reserved for separator"

    anchor_coords = np.argwhere(bottom == YELLOW)
    assert len(anchor_coords) > 0, "missing anchors"
    assert np.all((bottom == BLACK) | (bottom == YELLOW)), "bottom panel may only contain black and yellow"

    row_groups = _cluster_axis(anchor_coords[:, 0], proto.shape[0])
    col_groups = _cluster_axis(anchor_coords[:, 1], proto.shape[1])

    row_factors: List[int] = []
    for rows in row_groups:
        row_set = set(rows)
        count = sum(1 for r, _c in anchor_coords if int(r) in row_set)
        assert 1 <= count <= 3
        row_factors.append(int(count))

    col_factors: List[int] = []
    for cols in col_groups:
        col_set = set(cols)
        count = sum(1 for _r, c in anchor_coords if int(c) in col_set)
        assert 1 <= count <= 3
        col_factors.append(int(count))

    assert sum(row_factors) == len(anchor_coords) == sum(col_factors)
    assert any(v != row_factors[0] for v in row_factors[1:]) or any(
        v != col_factors[0] for v in col_factors[1:]
    ), "stretch factors must not be uniform on both axes"
    return proto, row_factors, col_factors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the compact prototype by row-band and column-band anchor counts.

    Input structure:
    - The background is black, with a single full grey separator row splitting the grid into a top prototype panel and a bottom anchor panel.
    - The top panel contains one compact colored prototype whose tight bounding box is 3x3 or 4x4.
    - The bottom panel contains only sparse yellow anchor pixels; their row coordinates naturally cluster into as many row bands as the prototype has rows, and their column coordinates cluster into as many column bands as the prototype has columns.

    Rule:
    - Crop the prototype from the top panel and let its size be R x C.
    - Group the yellow anchors in the bottom panel into R row bands and C column bands by splitting only across gaps of at least two blank rows or columns.
    - Count anchors in each row band to get row stretch factors H_i, and count anchors in each column band to get column stretch factors W_j.
    - Output only the stretched prototype: prototype cell (i, j) becomes an H_i x W_j rectangle of the same color, and all such rectangles are packed tightly in row-major order.

    Key insight:
    - Anchor coordinates only identify which row band and column band each anchor belongs to; the actual stretch amount comes from how many anchors fall in each band, not from the physical band span.
    """
    proto, row_factors, col_factors = _parse_instance(grid)
    return _stretch(proto, row_factors, col_factors)


def _sample_counts(rng: random.Random, length: int) -> List[int]:
    while True:
        counts = [rng.randint(1, 3) for _ in range(length)]
        if any(v > 1 for v in counts):
            return counts


def _sample_degree_sequences(rng: random.Random, rows: int, cols: int) -> Tuple[List[int], List[int]]:
    for _ in range(500):
        row_counts = _sample_counts(rng, rows)
        total = sum(row_counts)
        if not (cols <= total <= 3 * cols):
            continue
        col_counts = [1] * cols
        remaining = total - cols
        while remaining > 0:
            j = rng.randrange(cols)
            if col_counts[j] < 3:
                col_counts[j] += 1
                remaining -= 1
        if not any(v > 1 for v in col_counts):
            continue
        if all(v == row_counts[0] for v in row_counts) and all(v == col_counts[0] for v in col_counts):
            continue
        return row_counts, col_counts
    raise RuntimeError("failed to sample feasible degree sequences")


def _sample_count_matrix(
    rng: random.Random, row_counts: Sequence[int], col_counts: Sequence[int]
) -> np.ndarray:
    rows = len(row_counts)
    cols = len(col_counts)
    remaining_rows = list(map(int, row_counts))
    remaining_cols = list(map(int, col_counts))
    mat = np.zeros((rows, cols), dtype=int)

    while sum(remaining_rows) > 0:
        candidate_rows = [i for i, v in enumerate(remaining_rows) if v > 0]
        candidate_cols = [j for j, v in enumerate(remaining_cols) if v > 0]
        i = rng.choice(candidate_rows)
        j = rng.choice(candidate_cols)
        mat[i, j] += 1
        remaining_rows[i] -= 1
        remaining_cols[j] -= 1

    assert np.all(mat.sum(axis=1) == np.asarray(row_counts))
    assert np.all(mat.sum(axis=0) == np.asarray(col_counts))
    return mat


def _build_prototype(rng: random.Random, rows: int, cols: int) -> np.ndarray:
    mode = rng.choice(["mix", "cross", "blocks"])
    colors = rng.sample(PROTO_COLORS, rng.randint(2, min(4, len(PROTO_COLORS))))
    proto = np.zeros((rows, cols), dtype=int)

    if mode == "mix":
        row_colors = [rng.choice(colors) for _ in range(rows)]
        col_colors = [rng.choice(colors) for _ in range(cols)]
        for r in range(rows):
            for c in range(cols):
                proto[r, c] = row_colors[r] if (r + c) % 2 == 0 else col_colors[c]
    elif mode == "cross":
        base = colors[0]
        proto[:] = base
        accent1 = colors[1]
        accent2 = colors[-1]
        proto[rows // 2, :] = accent1
        proto[:, cols // 2] = accent2
        proto[rows // 2, cols // 2] = rng.choice(colors)
        for r in range(rows):
            proto[r, rng.randrange(cols)] = rng.choice(colors)
    else:
        split_r = rng.randint(1, rows - 1)
        split_c = rng.randint(1, cols - 1)
        fill = [rng.choice(colors) for _ in range(4)]
        proto[:split_r, :split_c] = fill[0]
        proto[:split_r, split_c:] = fill[1]
        proto[split_r:, :split_c] = fill[2]
        proto[split_r:, split_c:] = fill[3]
        proto[rng.randrange(rows), rng.randrange(cols)] = rng.choice(colors)

    if len(set(map(int, proto.ravel()))) == 1:
        rr = rng.randrange(rows)
        cc = rng.randrange(cols)
        proto[rr, cc] = rng.choice([c for c in PROTO_COLORS if c != int(proto[rr, cc])])
    return proto


def _place_anchors(
    rng: random.Random,
    count_matrix: np.ndarray,
    row_spans: Sequence[Tuple[int, int]],
    col_spans: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    anchors: List[Tuple[int, int]] = []
    for i in range(count_matrix.shape[0]):
        rows = list(range(row_spans[i][0], row_spans[i][1] + 1))
        for j in range(count_matrix.shape[1]):
            need = int(count_matrix[i, j])
            if need == 0:
                continue
            cols = list(range(col_spans[j][0], col_spans[j][1] + 1))
            cells = [(r, c) for r in rows for c in cols]
            assert need <= len(cells)
            anchors.extend(rng.sample(cells, need))
    assert len(anchors) == int(count_matrix.sum())
    assert len(set(anchors)) == len(anchors)
    return anchors


def _occupied_band_width(anchors: Sequence[Tuple[int, int]], band: Tuple[int, int], axis: int) -> int:
    vals = sorted({coord[axis] for coord in anchors if band[0] <= coord[axis] <= band[1]})
    if not vals:
        return 0
    return vals[-1] - vals[0] + 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(3, 4)
    cols = rng.randint(3, 4)
    row_counts, col_counts = _sample_degree_sequences(rng, rows, cols)
    count_matrix = _sample_count_matrix(rng, row_counts, col_counts)

    band_heights = [2 if row_counts[i] >= 2 and rng.random() < 0.75 else 1 for i in range(rows)]
    band_widths = [2 if col_counts[j] >= 2 and rng.random() < 0.75 else 1 for j in range(cols)]

    if not any(h == 2 for h in band_heights):
        band_heights[rng.choice([i for i, v in enumerate(row_counts) if v >= 2])] = 2
    if not any(w == 2 for w in band_widths):
        band_widths[rng.choice([j for j, v in enumerate(col_counts) if v >= 2])] = 2

    for i in range(rows):
        for j in range(cols):
            if count_matrix[i, j] > band_heights[i] * band_widths[j]:
                return None

    top_margin = rng.randint(1, 3)
    proto_bottom_margin = rng.randint(1, 3)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)

    bottom_top_margin = rng.randint(1, 3)
    bottom_left_margin = rng.randint(1, 3)
    bottom_right_margin = rng.randint(1, 3)
    bottom_bottom_margin = rng.randint(1, 3)

    row_gap_choices = [rng.randint(2, 3) for _ in range(rows - 1)]
    col_gap_choices = [rng.randint(2, 3) for _ in range(cols - 1)]

    row_spans: List[Tuple[int, int]] = []
    cursor = bottom_top_margin
    for i in range(rows):
        row_spans.append((cursor, cursor + band_heights[i] - 1))
        cursor += band_heights[i]
        if i + 1 < rows:
            cursor += row_gap_choices[i]
    bottom_height = cursor + bottom_bottom_margin

    col_spans: List[Tuple[int, int]] = []
    cursor = bottom_left_margin
    for j in range(cols):
        col_spans.append((cursor, cursor + band_widths[j] - 1))
        cursor += band_widths[j]
        if j + 1 < cols:
            cursor += col_gap_choices[j]
    bottom_width = cursor + bottom_right_margin

    anchors = _place_anchors(rng, count_matrix, row_spans, col_spans)
    if not any(_occupied_band_width(anchors, span, 0) >= 2 for span in row_spans):
        return None
    if not any(_occupied_band_width(anchors, span, 1) >= 2 for span in col_spans):
        return None

    proto = _build_prototype(rng, rows, cols)
    top_height = top_margin + rows + proto_bottom_margin
    grid_height = top_height + 1 + bottom_height
    grid_width = max(left_margin + cols + right_margin, bottom_width)
    if max(grid_height, grid_width) > 30:
        return None

    grid = np.zeros((grid_height, grid_width), dtype=int)
    sep = top_height
    grid[sep, :] = GREY

    proto_col = rng.randint(left_margin - 1 if left_margin > 0 else 0, grid_width - cols - right_margin)
    proto_row = top_margin
    grid[proto_row : proto_row + rows, proto_col : proto_col + cols] = proto

    for r, c in anchors:
        grid[sep + 1 + r, c] = YELLOW

    expected = _stretch(proto, row_counts, col_counts)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    parsed_proto, parsed_rows, parsed_cols = _parse_instance(grid)
    if not np.array_equal(parsed_proto, proto):
        return None
    if parsed_rows != list(map(int, row_counts)):
        return None
    if parsed_cols != list(map(int, col_counts)):
        return None

    if not np.any(actual != BLACK):
        return None
    if max(actual.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
