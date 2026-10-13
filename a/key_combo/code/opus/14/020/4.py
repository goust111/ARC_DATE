from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
MOTIF_COLOR = 5
MAX_ATTEMPTS = 400

MOTIF_TEMPLATES = [
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": int(len(cells)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _extract_motif_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _parse_input(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    comps = _components_by_color(g, bg)
    assert comps, "expected markers and one motif"

    motif_comps = [comp for comp in comps if int(comp["area"]) > 1]
    marker_comps = [comp for comp in comps if int(comp["area"]) == 1]
    assert len(motif_comps) == 1, "expected exactly one multi-pixel motif"
    assert 3 <= len(marker_comps) <= 4, "expected 3-4 marker pixels"

    motif_comp = motif_comps[0]
    assert 3 <= int(motif_comp["area"]) <= 5, "motif must contain 3-5 pixels"
    motif_mask = _extract_motif_mask(motif_comp)
    assert not np.array_equal(motif_mask, np.fliplr(motif_mask)), "motif must not be horizontally symmetric"

    markers: list[tuple[int, int, int]] = []
    for comp in marker_comps:
        r, c = map(int, comp["coords"][0])
        markers.append((r, c, int(comp["color"])))
    rows = [r for r, _c, _color in markers]
    cols = [c for _r, c, _color in markers]
    colors = [color for _r, _c, color in markers]
    assert len(rows) == len(set(rows)), "marker rows must be unique"
    assert len(colors) == len(set(colors)), "marker colors must be distinct"
    assert int(motif_comp["color"]) not in colors, "motif color must differ from marker colors"

    motif_rows = {int(r) for r, _c in motif_comp["coords"]}
    motif_cols = {int(c) for _r, c in motif_comp["coords"]}
    assert motif_rows.isdisjoint(rows), "motif may not share any row with a marker"
    assert motif_cols.isdisjoint(cols), "motif may not share any column with a marker"
    assert any(c < g.shape[1] / 2 for _r, c, _color in markers), "need a left-half marker"
    assert any(c >= g.shape[1] / 2 for _r, c, _color in markers), "need a right-half marker"
    return sorted(markers), motif_mask, bg


def _band_limits(markers: list[tuple[int, int, int]], h: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    for i, (row, _col, _color) in enumerate(markers):
        start = 0 if i == 0 else (markers[i - 1][0] + row) // 2 + 1
        end = h - 1 if i == len(markers) - 1 else (row + markers[i + 1][0]) // 2
        assert start <= row <= end
        bands.append((start, end))
    return bands


def _paint_motif(
    out: np.ndarray,
    motif_mask: np.ndarray,
    row: int,
    col: int,
    color: int,
    band_start: int,
    band_end: int,
) -> None:
    h, w = out.shape
    top = row + 1
    left = col - motif_mask.shape[1] // 2
    for dr, dc in np.argwhere(motif_mask):
        rr = top + int(dr)
        cc = left + int(dc)
        if 0 <= rr < h and 0 <= cc < w and band_start <= rr <= band_end:
            out[rr, cc] = color


def _render_output(
    shape: tuple[int, int],
    markers: list[tuple[int, int, int]],
    motif_mask: np.ndarray,
    bg: int,
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), bg, dtype=int)
    bands = _band_limits(markers, h)
    for (row, col, color), (start, end) in zip(markers, bands):
        out[start : end + 1, 0] = color
        out[start : end + 1, w - 1] = color
        out[row, :] = color
        if start == 0:
            out[0, :] = color
        if end == h - 1:
            out[h - 1, :] = color
        placed_mask = motif_mask if col < w / 2 else np.fliplr(motif_mask)
        _paint_motif(out, placed_mask, row, col, color, start, end)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split the canvas into marker-owned horizontal bands, frame each band, and copy a mirrored-or-not motif into it.

    Input structure:
    - The background is black and the foreground contains 3-4 singleton colored markers plus one small 4-connected motif.
    - Marker rows are all distinct; the motif is monochrome, has 3-5 pixels, and shares no row or column with any marker.
    - At least one marker lies in the left half of the grid and at least one lies in the right half.

    Rule:
    - Sort the markers by row and split the grid into horizontal bands at the midpoints between consecutive marker rows.
    - For each band, paint the marker row across the full width, paint the left and right border columns across the whole band, and also paint the top or bottom grid edge if that band touches it.
    - Erase the original motif, then place a recolored copy with its top edge on the row below the marker and its horizontal center aligned to the marker column; markers in the right half use the horizontally mirrored motif, and any copied pixels outside the grid or outside that band are discarded.

    Key insight:
    - The marker row determines vertical ownership of a band, while the marker column simultaneously determines both the motif's horizontal anchor and whether its orientation should be flipped.
    """
    markers, motif_mask, bg = _parse_input(np.asarray(grid, dtype=int))
    return _render_output(grid.shape, markers, motif_mask, bg)


def _sample_heights(n_markers: int, motif_h: int) -> list[int] | None:
    for _ in range(200):
        heights = [random.randint(4, 7) for _ in range(n_markers)]
        total = sum(heights)
        if not (14 <= total <= 22):
            continue
        if sum(h >= motif_h + 2 for h in heights) < 2:
            continue
        return heights
    return None


def _sample_rows(
    heights: list[int],
    motif_h: int,
    left_anchor: int,
    right_anchor: int,
) -> list[int] | None:
    rows: list[int] = []
    start = 0
    for idx, height in enumerate(heights):
        full_max = height - motif_h - 1
        clip_min = max(1, height - motif_h)
        if idx in (left_anchor, right_anchor):
            if full_max < 1:
                return None
            offset = random.randint(1, full_max)
        else:
            choose_clipped = clip_min <= height - 2 and random.random() < 0.45
            if choose_clipped:
                offset = random.randint(clip_min, height - 2)
            elif full_max >= 1:
                offset = random.randint(1, full_max)
            else:
                offset = random.randint(1, height - 2)
        rows.append(start + offset)
        start += height
    return rows


def _available_columns(width: int, used: set[int], *, left_half: bool, allow_edges: bool) -> list[int]:
    candidates: list[int] = []
    for col in range(width):
        if col in used:
            continue
        if not allow_edges and col in (0, width - 1):
            continue
        if left_half and not (col < width / 2):
            continue
        if not left_half and not (col >= width / 2):
            continue
        candidates.append(col)
    return candidates


def _sample_columns(width: int, n_markers: int, left_anchor: int, right_anchor: int) -> list[int] | None:
    cols = [None] * n_markers
    used: set[int] = set()

    left_choices = _available_columns(width, used, left_half=True, allow_edges=False)
    right_choices = _available_columns(width, used, left_half=False, allow_edges=False)
    if not left_choices or not right_choices:
        return None
    cols[left_anchor] = random.choice(left_choices)
    used.add(int(cols[left_anchor]))
    right_choices = _available_columns(width, used, left_half=False, allow_edges=False)
    if not right_choices:
        return None
    cols[right_anchor] = random.choice(right_choices)
    used.add(int(cols[right_anchor]))

    for idx in range(n_markers):
        if cols[idx] is not None:
            continue
        left_half = random.random() < 0.5
        allow_edges = random.random() < 0.35
        choices = _available_columns(width, used, left_half=left_half, allow_edges=allow_edges)
        if not choices:
            choices = _available_columns(width, used, left_half=not left_half, allow_edges=allow_edges)
        if not choices:
            choices = _available_columns(width, used, left_half=left_half, allow_edges=False)
        if not choices:
            choices = _available_columns(width, used, left_half=not left_half, allow_edges=False)
        if not choices:
            return None
        cols[idx] = random.choice(choices)
        used.add(int(cols[idx]))
    return [int(c) for c in cols]


def _corner_candidates(h: int, w: int, motif_h: int, motif_w: int) -> list[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()
    row_offsets = range(0, min(3, h - motif_h + 1))
    col_offsets = range(0, min(3, w - motif_w + 1))
    for r in row_offsets:
        for c in col_offsets:
            candidates.add((r, c))
            candidates.add((r, w - motif_w - c))
            candidates.add((h - motif_h - r, c))
            candidates.add((h - motif_h - r, w - motif_w - c))
    return [(int(r), int(c)) for r, c in candidates if 0 <= r <= h - motif_h and 0 <= c <= w - motif_w]


def _sample_motif_origin(
    h: int,
    w: int,
    motif_mask: np.ndarray,
    marker_rows: list[int],
    marker_cols: list[int],
) -> tuple[int, int] | None:
    motif_rows = np.where(motif_mask.any(axis=1))[0].astype(int).tolist()
    motif_cols = np.where(motif_mask.any(axis=0))[0].astype(int).tolist()
    candidates = _corner_candidates(h, w, *motif_mask.shape)
    random.shuffle(candidates)
    for r0, c0 in candidates:
        abs_rows = {r0 + r for r in motif_rows}
        abs_cols = {c0 + c for c in motif_cols}
        if abs_rows.isdisjoint(marker_rows) and abs_cols.isdisjoint(marker_cols):
            return (r0, c0)
    return None


def _build_input(
    h: int,
    w: int,
    markers: list[tuple[int, int, int]],
    motif_mask: np.ndarray,
    motif_origin: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for row, col, color in markers:
        grid[row, col] = color
    r0, c0 = motif_origin
    for dr, dc in np.argwhere(motif_mask):
        grid[r0 + int(dr), c0 + int(dc)] = MOTIF_COLOR
    return grid


def _anchor_patch_matches(
    out: np.ndarray,
    marker: tuple[int, int, int],
    motif_mask: np.ndarray,
) -> bool:
    row, col, color = marker
    top = row + 1
    left = col - motif_mask.shape[1] // 2
    if top + motif_mask.shape[0] > out.shape[0] or left < 1 or left + motif_mask.shape[1] > out.shape[1] - 1:
        return False
    expected = np.where(motif_mask, color, BG)
    patch = out[top : top + motif_mask.shape[0], left : left + motif_mask.shape[1]]
    return np.array_equal(patch, expected)


def _validate_example(
    inp: np.ndarray,
    expected: np.ndarray,
    markers: list[tuple[int, int, int]],
    motif_mask: np.ndarray,
    full_left_idx: int,
    full_right_idx: int,
) -> bool:
    try:
        parsed_markers, parsed_motif, bg = _parse_input(inp)
    except Exception:
        return False
    if bg != BG:
        return False
    if parsed_markers != sorted(markers):
        return False
    if not np.array_equal(parsed_motif, motif_mask):
        return False
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    bands = _band_limits(sorted(markers), inp.shape[0])
    for (row, _col, color), (start, end) in zip(sorted(markers), bands):
        if not np.all(expected[start : end + 1, 0] == color):
            return False
        if not np.all(expected[start : end + 1, -1] == color):
            return False
        if not np.all(expected[row, :] == color):
            return False
        if start == 0 and not np.all(expected[0, :] == color):
            return False
        if end == inp.shape[0] - 1 and not np.all(expected[-1, :] == color):
            return False

    left_marker = sorted(markers)[full_left_idx]
    right_marker = sorted(markers)[full_right_idx]
    if not _anchor_patch_matches(expected, left_marker, motif_mask):
        return False
    if not _anchor_patch_matches(expected, right_marker, np.fliplr(motif_mask)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    motif_mask = random.choice(MOTIF_TEMPLATES).copy()
    n_markers = random.randint(3, 4)
    heights = _sample_heights(n_markers, motif_mask.shape[0])
    if heights is None:
        return None

    left_anchor, right_anchor = random.sample(range(n_markers), 2)
    rows = _sample_rows(heights, motif_mask.shape[0], left_anchor, right_anchor)
    if rows is None:
        return None

    h = sum(heights)
    w = random.randint(14, 20)
    cols = _sample_columns(w, n_markers, left_anchor, right_anchor)
    if cols is None:
        return None

    marker_colors = random.sample([c for c in range(1, 10) if c != MOTIF_COLOR], n_markers)
    markers = [(int(r), int(c), int(color)) for r, c, color in zip(rows, cols, marker_colors)]
    markers = sorted(markers)

    left_sorted_idx = next(i for i, (_r, c, _color) in enumerate(markers) if c < w / 2)
    right_sorted_idx = next(i for i, (_r, c, _color) in enumerate(markers) if c >= w / 2)

    motif_origin = _sample_motif_origin(
        h,
        w,
        motif_mask,
        [r for r, _c, _color in markers],
        [c for _r, c, _color in markers],
    )
    if motif_origin is None:
        return None

    inp = _build_input(h, w, markers, motif_mask, motif_origin)
    expected = _render_output(inp.shape, markers, motif_mask, BG)

    if not _validate_example(inp, expected, markers, motif_mask, left_sorted_idx, right_sorted_idx):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
