import numpy as np
from collections import deque


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    pieces: list[dict] = []
    for coords in _components(grid != 0):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape[0] >= 3 and patch.shape[1] >= 3, "each fragment needs seams and interior"

        inner = patch[1:-1, 1:-1].copy()
        pieces.append(
            {
                "inner": inner,
                "h": int(inner.shape[0]),
                "w": int(inner.shape[1]),
                "top": tuple(int(x) for x in patch[0, 1:-1]),
                "bottom": tuple(int(x) for x in patch[-1, 1:-1]),
                "left": tuple(int(x) for x in patch[1:-1, 0]),
                "right": tuple(int(x) for x in patch[1:-1, -1]),
            }
        )
    assert pieces, "expected at least one fragment"
    return pieces


def _unique_relation(pieces: list[dict], src_side: str, dst_side: str, size_key: str) -> dict[int, int | None]:
    rel: dict[int, int | None] = {}
    for i, piece in enumerate(pieces):
        matches = [
            j
            for j, other in enumerate(pieces)
            if i != j and piece[size_key] == other[size_key] and piece[src_side] == other[dst_side]
        ]
        assert len(matches) <= 1, f"ambiguous {src_side}->{dst_side} seam match"
        rel[i] = matches[0] if matches else None
    return rel


def _try_layout(
    start: int,
    pieces: list[dict],
    right_of: dict[int, int | None],
    left_of: dict[int, int | None],
    down_of: dict[int, int | None],
    up_of: dict[int, int | None],
) -> list[list[int]] | None:
    rows: list[list[int]] = []
    used: set[int] = set()
    row_start = start

    while row_start is not None:
        if row_start in used:
            return None
        row: list[int] = []
        cur = row_start
        while cur is not None:
            if cur in used or cur in row:
                return None
            row.append(cur)
            cur = right_of[cur]
        rows.append(row)
        used.update(row)
        row_start = down_of[row_start]

    if not rows:
        return None

    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return None

    if len(used) != len(pieces):
        return None

    row_heights = [pieces[row[0]]["h"] for row in rows]
    col_widths = [pieces[rows[0][c]]["w"] for c in range(width)]

    for r, row in enumerate(rows):
        for c, idx in enumerate(row):
            if pieces[idx]["h"] != row_heights[r] or pieces[idx]["w"] != col_widths[c]:
                return None

            if left_of[idx] != (row[c - 1] if c > 0 else None):
                return None
            if right_of[idx] != (row[c + 1] if c + 1 < width else None):
                return None
            if up_of[idx] != (rows[r - 1][c] if r > 0 else None):
                return None
            if down_of[idx] != (rows[r + 1][c] if r + 1 < len(rows) else None):
                return None

    return rows


def _piece_layout(
    pieces: list[dict],
    right_of: dict[int, int | None],
    left_of: dict[int, int | None],
    down_of: dict[int, int | None],
    up_of: dict[int, int | None],
) -> list[list[int]]:
    starts = [i for i in range(len(pieces)) if left_of[i] is None and up_of[i] is None]
    assert starts, "expected a top-left fragment"

    layouts = []
    for start in starts:
        layout = _try_layout(start, pieces, right_of, left_of, down_of, up_of)
        if layout is not None:
            layouts.append(layout)

    assert len(layouts) == 1, "fragment seams must determine a unique rectangle"
    return layouts[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered rectangular fragments by matching their seam-marker sequences.

    Each fragment is a rectangle whose interior is the true picture piece, surrounded by
    one-cell seam markers on the four sides. Adjacent fragments in the original image
    share identical seam sequences on their touching sides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pieces = _extract_pieces(g)
    right_of = _unique_relation(pieces, "right", "left", "h")
    left_of = _unique_relation(pieces, "left", "right", "h")
    down_of = _unique_relation(pieces, "bottom", "top", "w")
    up_of = _unique_relation(pieces, "top", "bottom", "w")

    for i in range(len(pieces)):
        if right_of[i] is not None:
            assert left_of[right_of[i]] == i
        if left_of[i] is not None:
            assert right_of[left_of[i]] == i
        if down_of[i] is not None:
            assert up_of[down_of[i]] == i
        if up_of[i] is not None:
            assert down_of[up_of[i]] == i

    layout = _piece_layout(pieces, right_of, left_of, down_of, up_of)
    row_heights = [pieces[row[0]]["h"] for row in layout]
    col_widths = [pieces[layout[0][c]]["w"] for c in range(len(layout[0]))]

    out = np.zeros((sum(row_heights), sum(col_widths)), dtype=int)
    r_off = 0
    for r, row in enumerate(layout):
        c_off = 0
        for c, idx in enumerate(row):
            inner = pieces[idx]["inner"]
            h, w = inner.shape
            out[r_off : r_off + h, c_off : c_off + w] = inner
            c_off += w
        r_off += row_heights[r]
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

LAYOUT_OPTIONS = ((2, 2), (2, 3), (3, 2), (3, 3))
SIZE_MIN = 2
SIZE_MAX = 5  # exclusive
SUPER_CANVAS = 30
PLACEMENT_TRIES = 400
MAX_GENERATION_ATTEMPTS = 300


def _sample_sizes(rng: np.random.Generator, count: int) -> list[int]:
    return [int(x) for x in rng.integers(SIZE_MIN, SIZE_MAX, size=count)]


def _make_pattern(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    palette = rng.choice(np.arange(1, 10), size=int(rng.integers(3, 6)), replace=False)
    mode = ("noise", "checker", "bands", "blocks")[int(rng.integers(4))]

    if mode == "noise":
        arr = rng.choice(palette, size=(h, w))
    elif mode == "checker":
        br = int(rng.integers(1, min(3, h) + 1))
        bc = int(rng.integers(1, min(3, w) + 1))
        rr = (np.arange(h)[:, None] // br).astype(int)
        cc = (np.arange(w)[None, :] // bc).astype(int)
        arr = palette[(rr + cc + int(rng.integers(len(palette)))) % len(palette)]
    elif mode == "bands":
        row_colors = rng.choice(palette, size=h)
        col_colors = rng.choice(palette, size=w)
        take_rows = rng.random((h, w)) < 0.6
        arr = np.where(take_rows, row_colors[:, None], col_colors[None, :])
    else:
        arr = np.full((h, w), int(palette[0]), dtype=int)
        for _ in range(int(rng.integers(3, 7))):
            color = int(palette[int(rng.integers(len(palette)))])
            r0 = int(rng.integers(0, h))
            r1 = int(rng.integers(r0 + 1, h + 1))
            c0 = int(rng.integers(0, w))
            c1 = int(rng.integers(c0 + 1, w + 1))
            arr[r0:r1, c0:c1] = color
        if h >= 2 and rng.random() < 0.5:
            arr[int(rng.integers(0, h)), :] = int(palette[int(rng.integers(len(palette)))])
        if w >= 2 and rng.random() < 0.5:
            arr[:, int(rng.integers(0, w))] = int(palette[int(rng.integers(len(palette)))])

    arr = np.asarray(arr, dtype=int)
    needed = min(3, len(palette))
    for color in palette:
        if len(set(map(int, arr.reshape(-1)))) >= needed:
            break
        rr = int(rng.integers(0, h))
        cc = int(rng.integers(0, w))
        arr[rr, cc] = int(color)
    return arr


def _sample_unique_sequence(rng: np.random.Generator, length: int, used: set[tuple[int, ...]]) -> tuple[int, ...]:
    for _ in range(500):
        seq = tuple(int(x) for x in rng.integers(1, 10, size=length))
        if length > 1 and len(set(seq)) == 1:
            continue
        if seq not in used:
            used.add(seq)
            return seq
    raise RuntimeError("failed to sample a unique seam sequence")


def _build_patches(
    rng: np.random.Generator,
    image: np.ndarray,
    row_heights: list[int],
    col_widths: list[int],
) -> list[np.ndarray]:
    rows = len(row_heights)
    cols = len(col_widths)
    top = [[None] * cols for _ in range(rows)]
    bottom = [[None] * cols for _ in range(rows)]
    left = [[None] * cols for _ in range(rows)]
    right = [[None] * cols for _ in range(rows)]

    used_lr: set[tuple[int, ...]] = set()
    used_tb: set[tuple[int, ...]] = set()

    for r in range(rows):
        for c in range(cols - 1):
            seq = _sample_unique_sequence(rng, row_heights[r], used_lr)
            right[r][c] = seq
            left[r][c + 1] = seq
        left[r][0] = _sample_unique_sequence(rng, row_heights[r], used_lr)
        right[r][cols - 1] = _sample_unique_sequence(rng, row_heights[r], used_lr)

    for c in range(cols):
        for r in range(rows - 1):
            seq = _sample_unique_sequence(rng, col_widths[c], used_tb)
            bottom[r][c] = seq
            top[r + 1][c] = seq
        top[0][c] = _sample_unique_sequence(rng, col_widths[c], used_tb)
        bottom[rows - 1][c] = _sample_unique_sequence(rng, col_widths[c], used_tb)

    r_edges = np.cumsum([0] + row_heights)
    c_edges = np.cumsum([0] + col_widths)
    patches: list[np.ndarray] = []

    for r in range(rows):
        for c in range(cols):
            inner = image[r_edges[r] : r_edges[r + 1], c_edges[c] : c_edges[c + 1]]
            h, w = inner.shape
            patch = np.zeros((h + 2, w + 2), dtype=int)
            patch[1:-1, 1:-1] = inner
            patch[0, 1:-1] = np.array(top[r][c], dtype=int)
            patch[-1, 1:-1] = np.array(bottom[r][c], dtype=int)
            patch[1:-1, 0] = np.array(left[r][c], dtype=int)
            patch[1:-1, -1] = np.array(right[r][c], dtype=int)
            patches.append(patch)
    return patches


def _scatter_patches(rng: np.random.Generator, patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.zeros((SUPER_CANVAS, SUPER_CANVAS), dtype=int)
    occupied = np.zeros((SUPER_CANVAS, SUPER_CANVAS), dtype=bool)

    for idx in rng.permutation(len(patches)):
        patch = patches[int(idx)]
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = int(rng.integers(0, SUPER_CANVAS - ph + 1))
            c0 = int(rng.integers(0, SUPER_CANVAS - pw + 1))
            rs = max(0, r0 - 1)
            re = min(SUPER_CANVAS, r0 + ph + 1)
            cs = max(0, c0 - 1)
            ce = min(SUPER_CANVAS, c0 + pw + 1)
            if occupied[rs:re, cs:ce].any():
                continue
            canvas[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(canvas[r0 : r0 + ph, c0 : c0 + pw], patch)
            occupied[r0 : r0 + ph, c0 : c0 + pw] = True
            placed = True
            break
        if not placed:
            return None

    coords = np.argwhere(canvas != 0)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = max(0, int(r0) - int(rng.integers(0, 3)))
    c0 = max(0, int(c0) - int(rng.integers(0, 3)))
    r1 = min(SUPER_CANVAS - 1, int(r1) + int(rng.integers(0, 3)))
    c1 = min(SUPER_CANVAS - 1, int(c1) + int(rng.integers(0, 3)))
    return canvas[r0 : r1 + 1, c0 : c1 + 1]


def _gen_attempt() -> np.ndarray | None:
    layout = LAYOUT_OPTIONS[int(_RNG.integers(len(LAYOUT_OPTIONS)))]
    row_heights = _sample_sizes(_RNG, layout[0])
    col_widths = _sample_sizes(_RNG, layout[1])
    image = _make_pattern(_RNG, sum(row_heights), sum(col_widths))
    patches = _build_patches(_RNG, image, row_heights, col_widths)
    grid = _scatter_patches(_RNG, patches)
    if grid is None:
        return None
    if not np.array_equal(main(grid), image):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt()
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seam-reassembly puzzle")
