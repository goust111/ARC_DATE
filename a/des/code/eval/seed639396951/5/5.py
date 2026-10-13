import random
from collections import Counter, deque

import numpy as np


GREY = 5
PALETTE = tuple(range(10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_RETRIES = 400
PLACEMENT_RETRIES = 300
CANVAS_MAX = 30
PIECE_SIZES = (5, 7)
DIMENSION_OPTIONS = (
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (4, 1),
    (5, 1),
)


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).reshape(-1), minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _extract_pieces_and_marker(grid: np.ndarray) -> tuple[int, int, list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    piece_candidates = []
    marker_candidates = []
    for comp in _components(g != bg):
        if comp.shape[0] == 1:
            r, c = map(int, comp[0])
            marker_candidates.append((r, c, int(g[r, c])))
            continue
        r0, c0, r1, c1 = _bbox(comp)
        side_h = int(r1 - r0 + 1)
        side_w = int(c1 - c0 + 1)
        if side_h != side_w or side_h < 3 or side_h % 2 == 0:
            continue
        piece_candidates.append({"bbox": (r0, c0, r1, c1), "coords": comp, "side": side_h})

    assert piece_candidates, "expected square puzzle pieces"
    side = Counter(piece["side"] for piece in piece_candidates).most_common(1)[0][0]
    pieces = []
    for piece in piece_candidates:
        if int(piece["side"]) != int(side):
            continue
        r0, c0, r1, c1 = piece["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        mid = side // 2
        pieces.append(
            {
                "patch": patch,
                "top": int(patch[0, mid]),
                "bottom": int(patch[-1, mid]),
                "left": int(patch[mid, 0]),
                "right": int(patch[mid, -1]),
            }
        )

    assert pieces, "expected at least one valid piece"
    assert marker_candidates, "expected a bottom-left marker"
    marker_candidates.sort(key=lambda item: (-int(item[0]), int(item[1])))
    marker_color = int(marker_candidates[0][2])
    return bg, marker_color, pieces, int(side)


def _unique_unused_match(
    pieces: list[dict], used: set[int], side_name: str, color: int
) -> int | None:
    matches = [idx for idx, piece in enumerate(pieces) if idx not in used and int(piece[side_name]) == int(color)]
    if not matches:
        return None
    assert len(matches) == 1, f"ambiguous {side_name} match for color {color}"
    return int(matches[0])


def _reconstruct_layout(pieces: list[dict], marker_color: int) -> list[list[int]]:
    start_candidates = [idx for idx, piece in enumerate(pieces) if int(piece["left"]) == int(marker_color)]
    assert len(start_candidates) == 1, "marker must identify exactly one start piece"
    start = int(start_candidates[0])

    first_row = [start]
    used_in_row = {start}
    while True:
        nxt = _unique_unused_match(pieces, used_in_row, "left", int(pieces[first_row[-1]]["right"]))
        if nxt is None:
            break
        first_row.append(nxt)
        used_in_row.add(nxt)

    all_used = set(first_row)
    columns = []
    for top_idx in first_row:
        column = [top_idx]
        cur = top_idx
        while True:
            nxt = _unique_unused_match(pieces, all_used, "top", int(pieces[cur]["bottom"]))
            if nxt is None:
                break
            column.append(nxt)
            all_used.add(nxt)
            cur = nxt
        columns.append(column)

    row_count = len(columns[0])
    assert all(len(col) == row_count for col in columns), "columns must have equal height"
    layout = [[int(columns[c][r]) for c in range(len(columns))] for r in range(row_count)]

    for r in range(row_count):
        for c in range(len(columns) - 1):
            left_piece = pieces[layout[r][c]]
            right_piece = pieces[layout[r][c + 1]]
            assert int(left_piece["right"]) == int(right_piece["left"]), "broken horizontal assembly"
    for r in range(row_count - 1):
        for c in range(len(columns)):
            upper = pieces[layout[r][c]]
            lower = pieces[layout[r + 1][c]]
            assert int(upper["bottom"]) == int(lower["top"]), "broken vertical assembly"

    assert len(all_used) == len(pieces), "assembly must use every piece"
    return layout


def _assemble_layout(layout: list[list[np.ndarray]], bg: int) -> np.ndarray:
    rows = len(layout)
    cols = len(layout[0])
    side = int(layout[0][0].shape[0])
    step = side - 1
    out = np.full((rows * step + 1, cols * step + 1), int(bg), dtype=int)
    for r in range(rows):
        for c in range(cols):
            patch = np.asarray(layout[r][c], dtype=int)
            r0 = r * step
            c0 = c * step
            region = out[r0 : r0 + side, c0 : c0 + side]
            overlap = region != int(bg)
            assert not np.any(overlap & (region != patch)), "shared borders must agree"
            region[~overlap] = patch[~overlap]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble equal-sized framed square pieces by matching the midpoint colors on their borders.

    A singleton marker near the bottom-left corner identifies the top-left piece via its left interface color.
    Horizontal matches use right->left midpoint colors, vertical matches use bottom->top midpoint colors, and
    touching pieces share one border row/column in the output.
    """
    bg, marker_color, pieces, _side = _extract_pieces_and_marker(grid)
    index_layout = _reconstruct_layout(pieces, marker_color)
    patch_layout = [[pieces[idx]["patch"] for idx in row] for row in index_layout]
    return _assemble_layout(patch_layout, bg)


def _random_pattern(side: int, bg: int, rng: random.Random) -> np.ndarray:
    colors = [c for c in PALETTE if c not in {bg, GREY}]
    rng.shuffle(colors)
    inner = np.full((side - 2, side - 2), int(colors[0]), dtype=int)
    accents = colors[1 : 1 + rng.randint(1, min(3, len(colors) - 1))]
    modes = ["diag", "anti", "vstripe", "hstripe", "block", "corners", "cross"]
    for color in accents:
        mode = rng.choice(modes)
        if mode == "diag":
            for i in range(inner.shape[0]):
                inner[i, i % inner.shape[1]] = int(color)
        elif mode == "anti":
            for i in range(inner.shape[0]):
                inner[i, inner.shape[1] - 1 - (i % inner.shape[1])] = int(color)
        elif mode == "vstripe":
            inner[:, rng.randrange(inner.shape[1])] = int(color)
        elif mode == "hstripe":
            inner[rng.randrange(inner.shape[0]), :] = int(color)
        elif mode == "block":
            h = rng.randint(1, inner.shape[0])
            w = rng.randint(1, inner.shape[1])
            r0 = rng.randint(0, inner.shape[0] - h)
            c0 = rng.randint(0, inner.shape[1] - w)
            inner[r0 : r0 + h, c0 : c0 + w] = int(color)
        elif mode == "corners":
            inner[0, 0] = int(color)
            inner[0, -1] = int(color)
            inner[-1, 0] = int(color)
            inner[-1, -1] = int(color)
        else:
            mid = inner.shape[0] // 2
            inner[mid, :] = int(color)
            inner[:, inner.shape[1] // 2] = int(color)
    if np.unique(inner).size == 1:
        inner[rng.randrange(inner.shape[0]), rng.randrange(inner.shape[1])] = int(colors[-1])
    return inner


def _make_piece(
    side: int,
    top: int,
    bottom: int,
    left: int,
    right: int,
    bg: int,
    rng: random.Random,
) -> np.ndarray:
    patch = np.full((side, side), int(GREY), dtype=int)
    patch[1:-1, 1:-1] = _random_pattern(side, bg, rng)
    mid = side // 2
    patch[0, mid] = int(top)
    patch[-1, mid] = int(bottom)
    patch[mid, 0] = int(left)
    patch[mid, -1] = int(right)
    return patch


def _dimension_choices_for_side(side: int) -> list[tuple[int, int]]:
    valid = []
    for rows, cols in DIMENSION_OPTIONS:
        h_needed = rows * max(0, cols - 1)
        v_needed = max(0, rows - 1) * cols
        out_h = rows * (side - 1) + 1
        out_w = cols * (side - 1) + 1
        if h_needed <= 4 and v_needed <= 4 and out_h <= CANVAS_MAX and out_w <= CANVAS_MAX:
            valid.append((rows, cols))
    return valid


def _build_layout(rows: int, cols: int, side: int, bg: int, rng: random.Random) -> tuple[list[list[np.ndarray]], int]:
    colors = [c for c in PALETTE if c not in {bg, GREY}]

    h_needed = rows * max(0, cols - 1)
    h_colors = colors[:]
    rng.shuffle(h_colors)
    h_match = h_colors[:h_needed]
    h_rest = h_colors[h_needed:]
    start_left = int(h_rest[0])
    other_left_choices = [int(c) for c in h_rest[1:3]] or [int(h_rest[1])]
    right_choices = [int(c) for c in h_rest[3:]] or [int(h_rest[-1])]

    left_side = [[0] * cols for _ in range(rows)]
    right_side = [[0] * cols for _ in range(rows)]
    idx = 0
    for r in range(rows):
        left_side[r][0] = int(start_left if r == 0 else rng.choice(other_left_choices))
        for c in range(cols - 1):
            color = int(h_match[idx])
            idx += 1
            right_side[r][c] = color
            left_side[r][c + 1] = color
        right_side[r][cols - 1] = int(rng.choice(right_choices))

    v_colors = colors[:]
    rng.shuffle(v_colors)
    v_needed = max(0, rows - 1) * cols
    v_match = v_colors[:v_needed]
    v_rest = v_colors[v_needed:]
    top_choices = [int(c) for c in v_rest[:2]]
    bottom_choices = [int(c) for c in v_rest[2:4]] or [int(v_rest[-1])]

    top_side = [[0] * cols for _ in range(rows)]
    bottom_side = [[0] * cols for _ in range(rows)]
    idx = 0
    for c in range(cols):
        top_side[0][c] = int(rng.choice(top_choices))
    for r in range(rows - 1):
        for c in range(cols):
            color = int(v_match[idx])
            idx += 1
            bottom_side[r][c] = color
            top_side[r + 1][c] = color
    for c in range(cols):
        bottom_side[rows - 1][c] = int(rng.choice(bottom_choices))

    layout = []
    for r in range(rows):
        row = []
        for c in range(cols):
            row.append(
                _make_piece(
                    side=side,
                    top=int(top_side[r][c]),
                    bottom=int(bottom_side[r][c]),
                    left=int(left_side[r][c]),
                    right=int(right_side[r][c]),
                    bg=bg,
                    rng=rng,
                )
            )
        layout.append(row)
    return layout, int(start_left)


def _scatter_layout(layout: list[list[np.ndarray]], marker_color: int, bg: int, rng: random.Random) -> np.ndarray | None:
    patches = [patch for row in layout for patch in row]
    side = int(patches[0].shape[0])
    count = len(patches)
    min_h = max(22, side * 3 + 2)
    min_w = max(22, side * 3 + 2)

    for _ in range(PLACEMENT_RETRIES):
        h = rng.randint(min_h, CANVAS_MAX)
        w = rng.randint(min_w, CANVAS_MAX)
        if h * w < count * side * side + 180:
            continue
        grid = np.full((h, w), int(bg), dtype=int)
        blocked = np.zeros((h, w), dtype=bool)
        for rr, cc in ((h - 1, 0), (h - 2, 0), (h - 1, 1), (h - 2, 1)):
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True

        order = list(range(count))
        rng.shuffle(order)
        ok = True
        for idx in order:
            patch = patches[idx]
            placed = False
            for _ in range(200):
                r0 = rng.randint(0, h - side)
                c0 = rng.randint(0, w - side)
                r1 = r0 + side - 1
                c1 = c0 + side - 1
                pad_r0 = max(0, r0 - 1)
                pad_c0 = max(0, c0 - 1)
                pad_r1 = min(h - 1, r1 + 1)
                pad_c1 = min(w - 1, c1 + 1)
                if blocked[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1].any():
                    continue
                grid[r0 : r1 + 1, c0 : c1 + 1] = patch
                blocked[pad_r0 : pad_r1 + 1, pad_c0 : pad_c1 + 1] = True
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        grid[h - 1, 0] = int(marker_color)
        if _get_bg(grid) != int(bg):
            continue
        return grid
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    side = int(rng.choice(PIECE_SIZES))
    rows, cols = rng.choice(_dimension_choices_for_side(side))
    bg = int(rng.choice([c for c in PALETTE if c != GREY]))

    layout, marker_color = _build_layout(rows, cols, side, bg, rng)
    grid = _scatter_layout(layout, marker_color, bg, rng)
    if grid is None:
        return None

    expected = _assemble_layout(layout, bg)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
