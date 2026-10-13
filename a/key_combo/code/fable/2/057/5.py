from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists():
    utils_path = str(_UTILS_DIR)
    if utils_path not in sys.path:
        sys.path.append(utils_path)

try:
    from arc_utils import Color, bbox, crop_bbox, get_bg
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = box
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_ATTEMPTS = 500
LOCAL_FRAGMENT_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 300
CANVAS_CHOICES = tuple(range(18, 31))
MOTIF_SIZES = (7, 9)
MARKER_COUNT_RANGE = (2, 5)
AREA_MARGIN = 1


def _complete_by_reflection(observed: np.ndarray) -> np.ndarray:
    mask = np.asarray(observed, dtype=bool)
    return mask | np.flipud(mask) | np.fliplr(mask) | np.flipud(np.fliplr(mask))


def _stamp_mask(canvas: np.ndarray, motif: np.ndarray, center: tuple[int, int], color: int) -> None:
    half = motif.shape[0] // 2
    r0 = int(center[0]) - half
    c0 = int(center[1]) - half
    patch = canvas[r0 : r0 + motif.shape[0], c0 : c0 + motif.shape[1]]
    patch[motif] = int(color)


def _mask_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    box = bbox(mask)
    assert box is not None
    return box


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded(rect: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return rect[0] - pad, rect[1] - pad, rect[2] + pad, rect[3] + pad


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc]:
                nxt = (rr, cc)
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
    return len(seen) == int(mask.sum())


def _random_monotone_path(half: int) -> list[tuple[int, int]]:
    moves = [(1, 0)] * half + [(0, 1)] * half
    random.shuffle(moves)
    r = c = 0
    path = [(0, 0)]
    for dr, dc in moves:
        r += dr
        c += dc
        path.append((r, c))
    return path


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _boundary_counts(mask: np.ndarray) -> tuple[int, int, int, int]:
    last = mask.shape[0] - 1
    return (
        int(mask[0, :].sum()),
        int(mask[:, 0].sum()),
        int(mask[last, :].sum()),
        int(mask[:, last].sum()),
    )


def _local_fragment_ok(mask: np.ndarray) -> bool:
    if not _connected(mask):
        return False
    last = mask.shape[0] - 1
    top, left, bottom, right = _boundary_counts(mask)
    if min(top, left, bottom, right) < 2:
        return False
    if not mask[last, last]:
        return False
    if int(mask.sum()) >= mask.size:
        return False
    interior = mask[1:last, 1:last]
    if interior.size == 0 or int(interior.sum()) < 1:
        return False
    return True


def _generate_local_fragment(q: int) -> np.ndarray:
    half = q - 1
    for _ in range(LOCAL_FRAGMENT_ATTEMPTS):
        mask = np.zeros((q, q), dtype=bool)
        for r, c in _random_monotone_path(half):
            mask[r, c] = True
        top_c = random.randint(1, half)
        left_r = random.randint(1, half)
        bottom_c = random.randint(0, half - 1)
        right_r = random.randint(0, half - 1)
        mask[0, : top_c + 1] = True
        mask[: left_r + 1, 0] = True
        mask[half, bottom_c : half + 1] = True
        mask[right_r : half + 1, half] = True

        extra_budget = random.randint(0, max(1, q))
        for _extra in range(extra_budget):
            choices = _frontier(mask)
            if not choices:
                break
            r, c = random.choice(choices)
            mask[r, c] = True
            if int(mask.sum()) >= q * q - 1:
                break
        if _local_fragment_ok(mask):
            return mask
    raise RuntimeError("failed to generate a local fragment")


def _build_observed_box(fragment_a: np.ndarray, fragment_b: np.ndarray, orientation: str) -> np.ndarray:
    q = fragment_a.shape[0]
    size = 2 * q - 1
    half = q - 1
    observed = np.zeros((size, size), dtype=bool)
    if orientation == "nw_se":
        observed[:q, :q] |= fragment_a
        observed[half:, half:] |= np.flipud(np.fliplr(fragment_b))
    else:
        observed[:q, half:] |= np.fliplr(fragment_a)
        observed[half:, :q] |= np.flipud(fragment_b)
    return observed


def _build_output(shape: tuple[int, int], motif: np.ndarray, markers: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c, color in markers:
        _stamp_mask(out, motif, (r, c), color)
    return out


def _single_fragment_failures(fragment_a: np.ndarray, fragment_b: np.ndarray, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    empty = np.zeros_like(fragment_a, dtype=bool)
    only_a = _complete_by_reflection(_build_observed_box(fragment_a, empty, orientation))
    only_b = _complete_by_reflection(_build_observed_box(empty, fragment_b, orientation))
    return only_a, only_b


def _choose_fragments(q: int) -> tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    for _ in range(GEN_ATTEMPTS):
        fragment_a = _generate_local_fragment(q)
        fragment_b = _generate_local_fragment(q)
        if np.array_equal(fragment_a, fragment_b):
            continue
        if np.all(fragment_a <= fragment_b) or np.all(fragment_b <= fragment_a):
            continue
        orientation = random.choice(("nw_se", "ne_sw"))
        observed = _build_observed_box(fragment_a, fragment_b, orientation)
        motif = _complete_by_reflection(observed)
        if np.array_equal(observed, motif):
            continue
        only_a, only_b = _single_fragment_failures(fragment_a, fragment_b, orientation)
        if np.array_equal(only_a, motif) or np.array_equal(only_b, motif):
            continue
        if int(motif.sum()) <= motif.shape[0] + 3 or int(motif.sum()) == motif.size:
            continue
        off_axis = motif.copy()
        center = motif.shape[0] // 2
        off_axis[center, :] = False
        off_axis[:, center] = False
        if int(off_axis.sum()) < 4:
            continue
        return fragment_a, fragment_b, orientation, motif
    raise RuntimeError("failed to choose non-degenerate fragments")


def _place_markers(
    H: int,
    W: int,
    motif: np.ndarray,
    motif_rect: tuple[int, int, int, int],
    marker_colors: list[int],
) -> list[tuple[int, int, int]]:
    half = motif.shape[0] // 2
    reserved = [_expanded(motif_rect, AREA_MARGIN)]
    markers: list[tuple[int, int, int]] = []
    for color in marker_colors:
        candidates: list[tuple[int, int]] = []
        for r in range(half, H - half):
            for c in range(half, W - half):
                stamp_rect = (r - half, c - half, r + half, c + half)
                inflated = _expanded(stamp_rect, AREA_MARGIN)
                if any(_rects_overlap(inflated, rect) for rect in reserved):
                    continue
                candidates.append((r, c))
        if not candidates:
            raise RuntimeError("failed to place marker")
        r, c = random.choice(candidates)
        stamp_rect = (r - half, c - half, r + half, c + half)
        reserved.append(_expanded(stamp_rect, AREA_MARGIN))
        markers.append((r, c, int(color)))
    return markers


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    motif: np.ndarray,
    observed_box: np.ndarray,
    motif_rect: tuple[int, int, int, int],
    markers: list[tuple[int, int, int]],
    fragment_a: np.ndarray,
    fragment_b: np.ndarray,
    orientation: str,
) -> None:
    assert np.array_equal(main(grid), expected)

    marker_only = _build_output(grid.shape, observed_box, markers)
    assert not np.array_equal(marker_only, expected)

    only_a, only_b = _single_fragment_failures(fragment_a, fragment_b, orientation)
    out_a = _build_output(grid.shape, only_a, markers)
    out_b = _build_output(grid.shape, only_b, markers)
    assert not np.array_equal(out_a, expected)
    assert not np.array_equal(out_b, expected)

    box = _mask_rect(grid == _fragment_color(grid))
    assert box == motif_rect
    h = box[2] - box[0] + 1
    w = box[3] - box[1] + 1
    assert h == w and h % 2 == 1

    half = motif.shape[0] // 2
    stamp_boxes = [(r - half, c - half, r + half, c + half) for r, c, _ in markers]
    for i, rect_a in enumerate(stamp_boxes):
        for rect_b in stamp_boxes[i + 1 :]:
            assert not _rects_overlap(rect_a, rect_b)


def _fragment_color(grid: np.ndarray) -> int:
    bg = get_bg(grid)
    counts = Counter(map(int, grid.ravel()))
    candidates = [color for color, count in counts.items() if color != bg and count > 1]
    assert len(candidates) == 1
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing symmetric stamp from two diagonal corner fragments, then print it at every marker center.

    Input structure:
    - The grid has a black background with one repeated fragment color A and several isolated one-pixel markers of other colors.
    - Color A occupies two diagonal corner fragments of an odd square motif; together they already span that motif's true bounding square.
    - Each marker color appears exactly once and does not overlap the fragment square.

    Rule:
    - Find the odd square bounding box of color A.
    - Inside that square, complete the hidden motif by closing the observed A pixels under both horizontal and vertical reflection about the square center.
    - Erase the original A fragments.
    - For every non-A marker pixel, center the completed motif on that pixel and paint every motif cell with the marker's own color.

    Key insight:
    - The corner scraps are not an object to finish in place; they are generators of a reusable symmetric stamp that must be reconstructed before recolored stamping.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    fragment_color = _fragment_color(g)
    fragment_mask = g == fragment_color
    box = bbox(fragment_mask)
    assert box is not None
    r0, c0, r1, c1 = box
    assert (r1 - r0) == (c1 - c0)
    assert (r1 - r0 + 1) % 2 == 1

    observed = crop_bbox(fragment_mask, box)
    motif = _complete_by_reflection(observed)
    half = motif.shape[0] // 2

    out = np.full_like(g, bg)
    counts = Counter(map(int, g.ravel()))
    for color, count in sorted(counts.items()):
        if color == bg or color == fragment_color:
            continue
        assert count == 1
        marker = np.argwhere(g == color)
        assert len(marker) == 1
        r, c = map(int, marker[0])
        assert half <= r < g.shape[0] - half
        assert half <= c < g.shape[1] - half
        patch = out[r - half : r + half + 1, c - half : c + half + 1]
        assert patch.shape == motif.shape
        assert not np.any(patch[motif] != bg)
        patch[motif] = int(color)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        size = random.choice(MOTIF_SIZES)
        q = size // 2 + 1
        fragment_a, fragment_b, orientation, motif = _choose_fragments(q)
        observed_box = _build_observed_box(fragment_a, fragment_b, orientation)

        H = random.choice(CANVAS_CHOICES)
        W = random.choice(CANVAS_CHOICES)
        if H < size + 6 or W < size + 6:
            continue

        r0_max = H - size
        c0_max = W - size
        if r0_max < 0 or c0_max < 0:
            continue
        motif_r0 = random.randint(0, r0_max)
        motif_c0 = random.randint(0, c0_max)
        motif_rect = (motif_r0, motif_c0, motif_r0 + size - 1, motif_c0 + size - 1)

        marker_count = random.randint(*MARKER_COUNT_RANGE)
        colors = random.sample(list(Color.NOT_BLACK), marker_count + 1)
        fragment_color = int(colors[0])
        marker_colors = [int(color) for color in colors[1:]]

        try:
            markers = _place_markers(H, W, motif, motif_rect, marker_colors)
        except RuntimeError:
            continue

        grid = np.zeros((H, W), dtype=int)
        grid[motif_r0 : motif_r0 + size, motif_c0 : motif_c0 + size][observed_box] = fragment_color
        for r, c, color in markers:
            grid[r, c] = color

        expected = _build_output((H, W), motif, markers)
        try:
            _semantic_checks(
                grid,
                expected,
                motif,
                observed_box,
                motif_rect,
                markers,
                fragment_a,
                fragment_b,
                orientation,
            )
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
