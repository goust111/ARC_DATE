from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
SEPARATOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
STAMP_DIMS = [(3, 3), (3, 5), (5, 3), (5, 5)]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _solid_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color != bg and np.all(grid[r] == color):
            rows.append((r, color))
    assert len(rows) == 1, "expected exactly one solid separator row"
    return rows[0]


def _solid_separator_col(top: np.ndarray, sep_color: int) -> int:
    cols = [c for c in range(top.shape[1]) if np.all(top[:, c] == sep_color)]
    assert len(cols) == 1, "expected exactly one solid separator column in the top region"
    return int(cols[0])


def _extract_stamp(grid: np.ndarray) -> tuple[np.ndarray, int, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    sep_row, sep_color = _solid_separator_row(g, bg)
    top = g[:sep_row]
    canvas = g[sep_row + 1 :]
    assert top.shape[0] > 0 and canvas.shape[0] > 0

    sep_col = _solid_separator_col(top, sep_color)
    assert 2 * sep_col + 1 == g.shape[1], "separator must split equal-width top panels"

    left = top[:, :sep_col]
    right = np.fliplr(top[:, sep_col + 1 :])
    diff = left != right
    box = _bbox(diff)
    assert box is not None, "panels must differ somewhere"
    r0, c0, r1, c1 = box
    assert (r1 - r0) % 2 == 0 and (c1 - c0) % 2 == 0, "stamp bbox must be odd-sized"
    assert _is_connected(diff), "stamp shape must be connected"

    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    assert diff[cr, cc], "stamp center must be part of the difference shape"

    offsets = [(int(r - cr), int(c - cc)) for r, c in np.argwhere(diff)]
    return canvas, bg, offsets


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the mirror-difference shape from the upper panels onto the lower marker canvas.

    Input structure:
    - A full grey separator row splits the upper comparison zone from the lower canvas.
    - Above it, a grey separator column divides two equal-width panels; the right panel must be mirrored before comparison.
    - Below it, the canvas contains only background plus several single-pixel colored markers.

    Rule:
    - Mirror the right upper panel horizontally and compare it cell-by-cell with the left panel.
    - The set of mismatching positions is the stamp shape, with coordinates measured in the left panel and anchored at the center of its odd-sized bbox.
    - For each marker pixel on the lower canvas, paint that whole stamp in the marker's color with the anchor aligned to the marker, then output only the stamped canvas.

    Key insight:
    - The visible upper objects are not copied directly; they only serve as a fold-and-compare device whose mismatch positions define an invisible stamp.
    """
    canvas, bg, offsets = _extract_stamp(grid)
    out = np.full(canvas.shape, bg, dtype=int)

    claims: dict[tuple[int, int], int] = {}
    markers = np.argwhere(canvas != bg)
    assert len(markers) > 0, "expected at least one canvas marker"
    for r, c in markers:
        color = int(canvas[r, c])
        for dr, dc in offsets:
            rr = int(r + dr)
            cc = int(c + dc)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "stamp must fit on the canvas"
            key = (rr, cc)
            if key in claims:
                assert claims[key] == color, "different-colored stamps must not overlap"
            claims[key] = color

    for (r, c), color in claims.items():
        out[r, c] = color
    return out


def _pick_color(candidates: list[int], avoid: set[int]) -> int:
    choices = [color for color in candidates if color not in avoid]
    return int(random.choice(choices))


def _grow_region(h: int, w: int, blocked: set[tuple[int, int]], target: int) -> list[tuple[int, int]] | None:
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in blocked]
    if len(free) < target:
        return None

    seed = random.choice(free)
    region = {seed}
    frontier = {nb for nb in _neighbors4(*seed) if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in blocked}
    while len(region) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        region.add(cell)
        for nb in _neighbors4(*cell):
            if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in blocked and nb not in region:
                frontier.add(nb)
    return list(region)


def _manhattan_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    cells = [(r, c)]
    if random.random() < 0.5:
        while r != tr:
            r += 1 if tr > r else -1
            cells.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            cells.append((r, c))
    else:
        while c != tc:
            c += 1 if tc > c else -1
            cells.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            cells.append((r, c))
    return cells


def _make_stamp_mask() -> np.ndarray:
    for _ in range(200):
        h, w = random.choice(STAMP_DIMS)
        mask = np.zeros((h, w), dtype=bool)
        center = (h // 2, w // 2)

        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        for anchor in [center, *anchors]:
            for cell in _manhattan_path(anchor, center):
                mask[cell] = True

        max_extra = h * w - int(mask.sum()) - 1
        if max_extra > 0:
            target = int(mask.sum()) + random.randint(0, min(max_extra, max(1, (h * w) // 3)))
            while int(mask.sum()) < target:
                frontier = []
                for r, c in np.argwhere(mask):
                    for nr, nc in _neighbors4(int(r), int(c)):
                        if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                mask[random.choice(frontier)] = True

        if mask.all():
            continue
        if not mask[center]:
            continue
        if not _is_connected(mask):
            continue
        if _bbox(mask) != (0, 0, h - 1, w - 1):
            continue
        if int(mask.sum()) < 4:
            continue
        return mask
    raise RuntimeError("failed to build a non-rectangular connected stamp")


def _make_shared_panel(h: int, w: int, palette: list[int]) -> np.ndarray:
    panel = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()
    target = random.randint(max(6, (h * w) // 6), max(8, (h * w) // 3))
    tries = 0
    while len(occupied) < target and tries < 80:
        tries += 1
        remaining = target - len(occupied)
        if remaining <= 0:
            break
        size = random.randint(1, min(6, remaining))
        region = _grow_region(h, w, occupied, size)
        if region is None:
            continue
        color = random.choice(palette)
        for r, c in region:
            panel[r, c] = color
            occupied.add((r, c))
    return panel


def _place_stamp_on_panels(shared: np.ndarray, stamp_mask: np.ndarray, palette: list[int]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    h, w = shared.shape
    sh, sw = stamp_mask.shape
    r_choices = range(0, h - sh + 1)
    c_choices = range(0, w - sw + 1)
    r0 = random.choice(list(r_choices))
    c0 = random.choice(list(c_choices))

    left = shared.copy()
    right = shared.copy()
    coords = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(stamp_mask)]
    random.shuffle(coords)
    kinds = ["left_only", "right_only", "both"] + random.choices(["left_only", "right_only", "both"], k=len(coords) - 3)
    random.shuffle(kinds)

    counts = {"left_only": 0, "right_only": 0, "both": 0}
    for (r, c), kind in zip(coords, kinds):
        base = int(shared[r, c])
        if kind == "left_only":
            left[r, c] = _pick_color([BG, *palette], {base})
        elif kind == "right_only":
            right[r, c] = _pick_color([BG, *palette], {base})
        else:
            left[r, c] = _pick_color(palette, {base})
            right[r, c] = _pick_color(palette, {base, int(left[r, c])})
        counts[kind] += 1
        assert left[r, c] != right[r, c]

    return left, right, counts


def _stamp_offsets_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(mask)
    cr = (r0 + r1) // 2
    cc = (c0 + c1) // 2
    return [(int(r - cr), int(c - cc)) for r, c in np.argwhere(mask)]


def _pick_markers(canvas_shape: tuple[int, int], offsets: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    h, w = canvas_shape
    min_dr = min(dr for dr, _dc in offsets)
    max_dr = max(dr for dr, _dc in offsets)
    min_dc = min(dc for _dr, dc in offsets)
    max_dc = max(dc for _dr, dc in offsets)

    candidates = [
        (r, c)
        for r in range(-min_dr, h - max_dr)
        for c in range(-min_dc, w - max_dc)
    ]
    random.shuffle(candidates)

    chosen = []
    occupied: set[tuple[int, int]] = set()
    for r, c in candidates:
        cells = {(r + dr, c + dc) for dr, dc in offsets}
        if cells & occupied:
            continue
        chosen.append((r, c))
        occupied |= cells
        if len(chosen) == count:
            return chosen
    return []


def _render_expected(canvas_shape: tuple[int, int], markers: list[tuple[int, int, int]], offsets: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(canvas_shape, BG, dtype=int)
    for r, c, color in markers:
        for dr, dc in offsets:
            out[r + dr, c + dc] = color
    return out


def _rect_offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cr = h // 2
    cc = w // 2
    return [(r - cr, c - cc) for r in range(h) for c in range(w)]


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, shared: np.ndarray, left: np.ndarray, right: np.ndarray, stamp_mask: np.ndarray, offsets: list[tuple[int, int]], mismatch_counts: dict[str, int]) -> bool:
    if any(mismatch_counts[k] == 0 for k in ("left_only", "right_only", "both")):
        return False

    diff = left != right
    box = _bbox(diff)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if not np.array_equal(diff[r0 : r1 + 1, c0 : c1 + 1], stamp_mask):
        return False
    if np.array_equal(diff, (left != BG) ^ (right != BG)):
        return False

    rect_expected = _render_expected(expected.shape, _extract_markers(grid), _rect_offsets(stamp_mask))
    if np.array_equal(rect_expected, expected):
        return False

    _canvas, _bg, extracted_offsets = _extract_stamp(grid)
    if set(extracted_offsets) != set(offsets):
        return False

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    if (r1 - r0 + 1) not in {3, 5} or (c1 - c0 + 1) not in {3, 5}:
        return False

    if np.count_nonzero(shared) < 5:
        return False
    return True


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    canvas, bg, _offsets = _extract_stamp(grid)
    return [(int(r), int(c), int(canvas[r, c])) for r, c in np.argwhere(canvas != bg)]


def _gen_attempt() -> np.ndarray | None:
    stamp_mask = _make_stamp_mask()
    sh, sw = stamp_mask.shape

    panel_h = random.randint(sh + 2, 12)
    panel_w = random.randint(sw + 3, 11)
    total_w = 2 * panel_w + 1
    canvas_h = random.randint(max(6, sh + 1), 12)
    if max(panel_h + 1 + canvas_h, total_w) > 30:
        return None

    palette = random.sample(PALETTE, k=4)
    shared = _make_shared_panel(panel_h, panel_w, palette)
    left, right, mismatch_counts = _place_stamp_on_panels(shared, stamp_mask, palette)

    offsets = _stamp_offsets_from_mask(stamp_mask)
    marker_count = random.randint(2, 4)
    marker_positions = _pick_markers((canvas_h, total_w), offsets, marker_count)
    if len(marker_positions) != marker_count:
        return None

    marker_colors = random.sample(PALETTE, marker_count)
    markers = [(r, c, color) for (r, c), color in zip(marker_positions, marker_colors)]
    expected = _render_expected((canvas_h, total_w), markers, offsets)

    grid = np.full((panel_h + 1 + canvas_h, total_w), BG, dtype=int)
    grid[:panel_h, :panel_w] = left
    grid[:panel_h, panel_w] = SEPARATOR
    grid[:panel_h, panel_w + 1 :] = np.fliplr(right)
    grid[panel_h, :] = SEPARATOR
    for r, c, color in markers:
        grid[panel_h + 1 + r, c] = color

    if not _semantic_checks(grid, expected, shared, left, right, stamp_mask, offsets, mismatch_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-difference stamp puzzle")
