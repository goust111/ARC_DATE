from __future__ import annotations

import random

import numpy as np


BLACK = 0
GEN_RETRIES = 400
PIECE_COUNT_RANGE = (2, 6)
FACE_HEIGHT_RANGE = (2, 5)
MAX_EXTRA_WIDTH = 5
PATTERN_STYLES = (
    "checker",
    "vstripe",
    "hstripe",
    "quadrants",
    "diagonal",
    "parity",
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble labeled puzzle pieces into their hidden left-to-right canvas.

    Input structure:
    - The background is Black.
    - Several same-sized non-overlapping rectangular piece blocks sit above the last row, separated by Black cells.
    - Each block has an upper picture half and a lower solid-color label half of equal height.
    - The bottom row contains a left-to-right strip of unique single-pixel label colors.

    Rule:
    - Read the non-Black pixels in the bottom row from left to right.
    - For each strip color, find the rectangle whose lower half is filled with that color.
    - Extract that rectangle's upper half and concatenate all such picture halves horizontally in strip order.

    Key insight:
    - The lower solid half is not output content; it is a visible label that links each picture fragment to the bottom-row ordering instruction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2, "expected at least one content row and one strip row"

    order = [int(v) for v in g[-1] if int(v) != BLACK]
    assert order, "bottom strip must contain at least one label color"
    assert len(order) == len(set(order)), "bottom strip label colors must be unique"

    pieces: dict[int, np.ndarray] = {}
    expected_shape: tuple[int, int] | None = None
    for pts in _components(g[:-1] != BLACK):
        r0, c0 = map(int, pts.min(axis=0))
        r1, c1 = map(int, pts.max(axis=0))
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert bool((patch != BLACK).all()), "piece regions must be solid non-Black rectangles"
        ph, pw = map(int, patch.shape)
        assert ph % 2 == 0 and ph > 0, "piece height must split evenly into front and label halves"

        face = patch[: ph // 2].copy()
        label_half = patch[ph // 2 :]
        label_colors = np.unique(label_half)
        assert label_colors.size == 1, "piece label half must be a single solid color"
        label = int(label_colors[0])
        assert label != BLACK, "piece label color must be non-Black"
        assert label not in pieces, "piece label colors must be unique"

        if expected_shape is None:
            expected_shape = (ph, pw)
        else:
            assert (ph, pw) == expected_shape, "all pieces must share the same rectangle size"
        pieces[label] = face

    assert pieces, "expected at least one piece rectangle"
    assert set(order) == set(pieces), "bottom strip colors must match the available piece labels"
    return np.concatenate([pieces[label] for label in order], axis=1)


def _make_pattern(height: int, width: int, palette: list[int], rng: random.Random) -> np.ndarray:
    for _ in range(80):
        style = rng.choice(PATTERN_STYLES)
        n_colors = rng.randint(2, min(4, len(palette)))
        colors = rng.sample(palette, n_colors)
        arr = np.zeros((height, width), dtype=int)

        if style == "checker":
            for r in range(height):
                for c in range(width):
                    arr[r, c] = colors[(r + c) % n_colors]
        elif style == "vstripe":
            for c in range(width):
                arr[:, c] = colors[c % n_colors]
        elif style == "hstripe":
            for r in range(height):
                arr[r, :] = colors[r % n_colors]
        elif style == "quadrants":
            split_r = rng.randint(1, height - 1)
            split_c = rng.randint(1, width - 1)
            quad_colors = [colors[i % n_colors] for i in range(4)]
            arr[:split_r, :split_c] = quad_colors[0]
            arr[:split_r, split_c:] = quad_colors[1]
            arr[split_r:, :split_c] = quad_colors[2]
            arr[split_r:, split_c:] = quad_colors[3]
        elif style == "diagonal":
            for r in range(height):
                for c in range(width):
                    arr[r, c] = colors[(r + 2 * c) % n_colors]
        elif style == "parity":
            for r in range(height):
                for c in range(width):
                    arr[r, c] = colors[((r % 2) * 2 + (c % 2)) % n_colors]
        else:
            raise AssertionError("unknown pattern style")

        if np.unique(arr).size >= 2:
            return arr
    raise RuntimeError("failed to construct a non-degenerate front pattern")


def _row_counts(piece_count: int, rng: random.Random) -> list[int]:
    if piece_count == 1 or rng.random() < 0.5:
        return [piece_count]
    split = rng.randint(1, piece_count - 1)
    counts = [split, piece_count - split]
    rng.shuffle(counts)
    return counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.randint(*PIECE_COUNT_RANGE)
    row_counts = _row_counts(piece_count, rng)
    row_total = len(row_counts)
    max_row_count = max(row_counts)

    max_piece_width = min(
        6,
        30 // piece_count,
        (30 - 2 - (max_row_count - 1)) // max_row_count,
    )
    if max_piece_width < 2:
        return None

    face_height = rng.randint(*FACE_HEIGHT_RANGE)
    piece_width = rng.randint(2, max_piece_width)
    piece_height = 2 * face_height

    top_margin = rng.randint(1, 3)
    row_gap = rng.randint(1, 3) if row_total == 2 else 0
    bottom_gap = rng.randint(1, 3)
    used_height = row_total * piece_height + (row_total - 1) * row_gap
    grid_height = top_margin + used_height + bottom_gap + 1
    if grid_height > 30:
        return None

    row_layouts: list[tuple[int, list[int], int]] = []
    widest_row = 0
    for count in row_counts:
        gaps = [rng.randint(1, 3) for _ in range(count - 1)]
        used_width = count * piece_width + sum(gaps)
        row_layouts.append((count, gaps, used_width))
        widest_row = max(widest_row, used_width)

    strip_order_gaps = [rng.randint(0, 2) for _ in range(piece_count - 1)]
    strip_span = piece_count + sum(strip_order_gaps)
    min_grid_width = max(widest_row + 2, strip_span + 2)
    if min_grid_width > 30:
        return None
    grid_width = rng.randint(min_grid_width, min(30, min_grid_width + MAX_EXTRA_WIDTH))

    label_colors = rng.sample(list(range(1, 10)), piece_count)
    front_palette = [c for c in range(1, 10) if c not in label_colors]
    if len(front_palette) < 2:
        return None

    fronts: list[np.ndarray] = []
    seen_patterns: set[bytes] = set()
    for _ in range(piece_count):
        pattern = None
        for _ in range(50):
            candidate = _make_pattern(face_height, piece_width, front_palette, rng)
            sig = candidate.tobytes()
            if sig not in seen_patterns:
                seen_patterns.add(sig)
                pattern = candidate
                break
        if pattern is None:
            return None
        fronts.append(pattern)

    positions: list[tuple[int, int]] = []
    current_row = top_margin
    for count, gaps, used_width in row_layouts:
        start_col = rng.randint(1, grid_width - used_width - 1)
        col = start_col
        for i in range(count):
            positions.append((current_row, col))
            col += piece_width
            if i < len(gaps):
                col += gaps[i]
        current_row += piece_height + row_gap
    rng.shuffle(positions)

    grid = np.zeros((grid_height, grid_width), dtype=int)
    label_to_front: dict[int, np.ndarray] = {}
    for front, label, (r0, c0) in zip(fronts, label_colors, positions):
        piece = np.vstack((front, np.full((face_height, piece_width), label, dtype=int)))
        window = grid[r0 : r0 + piece_height, c0 : c0 + piece_width]
        if bool(window.any()):
            return None
        window[:] = piece
        label_to_front[label] = front

    strip_labels = label_colors[:]
    rng.shuffle(strip_labels)
    strip_start = rng.randint(1, grid_width - strip_span - 1)
    col = strip_start
    for i, label in enumerate(strip_labels):
        grid[-1, col] = label
        if i < len(strip_order_gaps):
            col += 1 + strip_order_gaps[i]

    expected = np.concatenate([label_to_front[label] for label in strip_labels], axis=1)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
