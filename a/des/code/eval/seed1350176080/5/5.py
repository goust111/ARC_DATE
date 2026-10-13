from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 300

FRAGMENT_TEMPLATES = (
    np.array(
        [
            [1, 2, 2],
            [2, 2, 0],
            [0, 2, 3],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 2, 0],
            [2, 2, 2],
            [3, 0, 2],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 3, 2],
            [2, 2, 2],
            [0, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 2, 3],
            [0, 2, 2],
            [2, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 2, 2],
            [0, 3, 2],
            [2, 2, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 2, 0],
            [3, 2, 2],
            [2, 2, 0],
        ],
        dtype=int,
    ),
)


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _connected_components8(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _find_separator_column(g: np.ndarray) -> int:
    candidates = [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]
    assert candidates, "expected a full-height grey separator"
    return int(candidates[0])


def _find_thumbnail(left: np.ndarray, thumb_h: int, thumb_w: int) -> np.ndarray:
    outer_h = thumb_h + 2
    outer_w = thumb_w + 2
    candidates: list[np.ndarray] = []
    for r0 in range(left.shape[0] - outer_h + 1):
        for c0 in range(left.shape[1] - outer_w + 1):
            patch = left[r0 : r0 + outer_h, c0 : c0 + outer_w]
            if not (
                np.all(patch[0, :] == GREY)
                and np.all(patch[-1, :] == GREY)
                and np.all(patch[:, 0] == GREY)
                and np.all(patch[:, -1] == GREY)
            ):
                continue
            inner = patch[1:-1, 1:-1]
            if int(np.count_nonzero(inner != BLACK)) < 4:
                continue
            candidates.append(inner.copy())
    assert len(candidates) == 1, "thumbnail frame should be unique"
    return candidates[0]


def _rotation_fix_k(piece: np.ndarray) -> int:
    marker_cells = np.argwhere(piece == MAROON)
    assert len(marker_cells) == 1, "each fragment must contain exactly one maroon marker"
    mr, mc = map(int, marker_cells[0])
    mapping = {
        (0, 0): 0,
        (0, 2): 1,
        (2, 2): 2,
        (2, 0): 3,
    }
    assert (mr, mc) in mapping, "maroon marker must occupy a fragment corner"
    return mapping[(mr, mc)]


def _main_color(piece: np.ndarray) -> int:
    vals = [int(v) for v in piece.ravel() if int(v) not in (BLACK, MAROON)]
    counts = Counter(vals)
    assert counts, "fragment must have a non-marker color"
    most_common = counts.most_common()
    best_color, best_count = most_common[0]
    if len(most_common) > 1:
        assert best_count > most_common[1][1], "fragment main color must be unique"
    return int(best_color)


def _extract_fragments(area: np.ndarray) -> list[np.ndarray]:
    fragments: list[np.ndarray] = []
    for comp in _connected_components8(area != BLACK):
        r0 = int(comp[:, 0].min())
        c0 = int(comp[:, 1].min())
        r1 = int(comp[:, 0].max())
        c1 = int(comp[:, 1].max())
        assert r1 - r0 == 2 and c1 - c0 == 2, "every fragment must occupy a 3x3 box"
        patch = area[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert int(np.count_nonzero(patch == MAROON)) == 1
        fragments.append(patch)
    assert fragments, "expected at least one fragment"
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble rotated 3x3 fragments onto a 3x-scaled canvas guided by a framed thumbnail.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_column(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert right.shape[0] % 3 == 0 and right.shape[1] % 3 == 0
    thumb_h = right.shape[0] // 3
    thumb_w = right.shape[1] // 3
    thumbnail = _find_thumbnail(left, thumb_h, thumb_w)

    output = np.zeros_like(right)
    fragments = _extract_fragments(right)
    for piece in fragments:
        fixed = np.rot90(piece, _rotation_fix_k(piece))
        color = _main_color(fixed)
        positions = np.argwhere(thumbnail == color)
        assert len(positions) == 1, "fragment main color must map to a unique thumbnail pixel"
        tr, tc = map(int, positions[0])
        fixed = fixed.copy()
        fixed[fixed == MAROON] = color
        r0 = tr * 3
        c0 = tc * 3
        output[r0 : r0 + 3, c0 : c0 + 3] = fixed

    return output


def _grow_connected_shape(height: int, width: int, count: int) -> list[tuple[int, int]]:
    start = (random.randrange(height), random.randrange(width))
    shape = {start}
    while len(shape) < count:
        frontier = []
        for r, c in shape:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in shape:
                    frontier.append((rr, cc))
        assert frontier
        shape.add(random.choice(frontier))
    cells = list(shape)
    random.shuffle(cells)
    return cells


def _instantiate_fragment(main_color: int) -> np.ndarray:
    template = random.choice(FRAGMENT_TEMPLATES)
    accent_choices = [c for c in USABLE_COLORS if c != main_color]
    accent_color = random.choice(accent_choices)
    piece = np.zeros((3, 3), dtype=int)
    piece[template == 1] = MAROON
    piece[template == 2] = main_color
    piece[template == 3] = accent_color
    return piece


def _scatter_positions(height: int, width: int) -> list[tuple[int, int]]:
    rows = list(range(0, height - 2, 4))
    cols = list(range(0, width - 2, 4))
    return [(r, c) for r in rows for c in cols]


def _build_expected_output(
    thumb_h: int,
    thumb_w: int,
    placements: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    out = np.zeros((thumb_h * 3, thumb_w * 3), dtype=int)
    for r, c, piece in placements:
        patch = piece.copy()
        color = _main_color(patch)
        patch[patch == MAROON] = color
        out[r * 3 : r * 3 + 3, c * 3 : c * 3 + 3] = patch
    return out


def _gen_attempt() -> np.ndarray | None:
    thumb_h = random.choice((4, 5))
    thumb_w = random.choice((4, 5))
    out_h = thumb_h * 3
    out_w = thumb_w * 3

    max_fragments = min(len(USABLE_COLORS), thumb_h * thumb_w)
    count = random.randint(4, max_fragments)
    shape_cells = _grow_connected_shape(thumb_h, thumb_w, count)
    colors = random.sample(list(USABLE_COLORS), count)

    thumbnail = np.zeros((thumb_h, thumb_w), dtype=int)
    placements: list[tuple[int, int, np.ndarray]] = []
    fragments: list[np.ndarray] = []
    for idx, ((r, c), color) in enumerate(zip(shape_cells, colors)):
        thumbnail[r, c] = color
        piece = _instantiate_fragment(color)
        placements.append((r, c, piece))
        rot_k = idx % 4
        fragments.append(np.rot90(piece, (-rot_k) % 4))
    random.shuffle(fragments)

    right = np.zeros((out_h, out_w), dtype=int)
    spots = _scatter_positions(out_h, out_w)
    if len(spots) < len(fragments):
        return None
    chosen_spots = random.sample(spots, len(fragments))
    for (r0, c0), piece in zip(chosen_spots, fragments):
        right[r0 : r0 + 3, c0 : c0 + 3] = piece

    left_pad = random.randint(1, 3)
    right_pad = random.randint(1, 3)
    left_w = left_pad + thumb_w + 2 + right_pad
    top_margin_max = out_h - (thumb_h + 2) - 1
    assert top_margin_max >= 1
    top = random.randint(1, top_margin_max)

    grid = np.zeros((out_h, left_w + 1 + out_w), dtype=int)
    grid[:, left_w] = GREY
    outer_r0 = top
    outer_c0 = left_pad
    grid[outer_r0 : outer_r0 + thumb_h + 2, outer_c0 : outer_c0 + thumb_w + 2] = GREY
    grid[outer_r0 + 1 : outer_r0 + 1 + thumb_h, outer_c0 + 1 : outer_c0 + 1 + thumb_w] = thumbnail
    grid[:, left_w + 1 :] = right

    expected = _build_expected_output(thumb_h, thumb_w, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
