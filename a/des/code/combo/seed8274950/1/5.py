from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


@dataclass(frozen=True)
class FrameSpec:
    color: int
    coords: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    direction: str
    gap_rows: tuple[int, int]
    marker_color: int | None
    marker_pos: tuple[int, int] | None


def _background_color(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(vals, minlength=10).argmax())


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _edge_gap(edge: np.ndarray) -> tuple[str, tuple[int, int] | None]:
    missing = np.flatnonzero(~edge)
    if missing.size == 0:
        return "closed", None
    if int(missing[0]) == 0 or int(missing[-1]) == edge.size - 1:
        return "invalid", None
    if int(missing.size) != int(missing[-1] - missing[0] + 1):
        return "invalid", None
    if not edge[: missing[0]].all():
        return "invalid", None
    if not edge[missing[-1] + 1 :].all():
        return "invalid", None
    return "open", (int(missing[0]), int(missing[-1]))


def _analyze_frame(grid: np.ndarray, color: int, coords: list[tuple[int, int]], bg: int) -> FrameSpec | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    local = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True

    if local[1:-1, 1:-1].any():
        return None
    if not local[0, :].all() or not local[-1, :].all():
        return None

    left_status, left_gap = _edge_gap(local[:, 0])
    right_status, right_gap = _edge_gap(local[:, -1])
    if left_status == "invalid" or right_status == "invalid":
        return None
    if left_status == right_status:
        return None

    direction = "left" if left_status == "open" else "right"
    gap_rows = left_gap if direction == "left" else right_gap
    assert gap_rows is not None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    marker_cells = np.argwhere(interior != bg)
    if marker_cells.shape[0] > 1:
        return None

    marker_color: int | None = None
    marker_pos: tuple[int, int] | None = None
    if marker_cells.shape[0] == 1:
        mr, mc = map(int, marker_cells[0])
        marker_color = int(interior[mr, mc])
        if marker_color == color:
            return None
        marker_pos = (r0 + 1 + mr, c0 + 1 + mc)

    return FrameSpec(
        color=int(color),
        coords=tuple(sorted((int(r), int(c)) for r, c in coords)),
        bbox=(int(r0), int(c0), int(r1), int(c1)),
        direction=direction,
        gap_rows=(int(gap_rows[0]), int(gap_rows[1])),
        marker_color=marker_color,
        marker_pos=marker_pos,
    )


def _extract_pairs(grid: np.ndarray, bg: int) -> list[tuple[FrameSpec, FrameSpec]]:
    pairs: list[tuple[FrameSpec, FrameSpec]] = []
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != bg)
    for color in colors:
        specs = []
        for comp in _components(grid == color):
            spec = _analyze_frame(grid, color, comp, bg)
            if spec is not None:
                specs.append(spec)
        if len(specs) != 2:
            continue
        right_open = [spec for spec in specs if spec.direction == "right"]
        left_open = [spec for spec in specs if spec.direction == "left"]
        if len(right_open) != 1 or len(left_open) != 1:
            continue
        pairs.append((right_open[0], left_open[0]))
    pairs.sort(key=lambda pair: (pair[0].bbox[0], pair[0].bbox[1], pair[0].color))
    return pairs


def _translate(coords: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple((r + dr, c + dc) for r, c in coords)


def _enclosed_background(frame_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(tuple(frame_cells))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    occ = np.zeros((h, w), dtype=bool)
    for r, c in frame_cells:
        occ[r - r0, c - c0] = True

    enclosed: list[tuple[int, int]] = []
    for comp in _components(~occ):
        if any(rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1 for rr, cc in comp):
            continue
        enclosed.extend((r0 + rr, c0 + cc) for rr, cc in comp)
    return enclosed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stitch opposite-opening frame pairs of the same color, then fill their merged cavity from the marker colors.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)

    out = np.full_like(g, bg)
    for left_piece, right_piece in _extract_pairs(g, bg):
        dr = (left_piece.bbox[0] + left_piece.gap_rows[0]) - (right_piece.bbox[0] + right_piece.gap_rows[0])
        dc = (left_piece.bbox[3] + 1) - right_piece.bbox[1]

        shifted_right = _translate(right_piece.coords, dr, dc)
        frame_cells = set(left_piece.coords) | set(shifted_right)

        if not all(0 <= r < h and 0 <= c < w for r, c in frame_cells):
            continue

        cavity = _enclosed_background(frame_cells)
        markers = []
        for spec in (left_piece, right_piece):
            if spec.marker_color is not None:
                ref_row = spec.marker_pos[0] if spec.marker_pos is not None else spec.bbox[0]
                ref_col = spec.marker_pos[1] if spec.marker_pos is not None else spec.bbox[1]
                markers.append((spec.marker_color, ref_row, ref_col))

        if markers and cavity:
            colors = {color for color, _, _ in markers}
            if len(markers) == 1 or len(colors) == 1:
                fill_color = markers[0][0]
                for r, c in cavity:
                    out[r, c] = fill_color
            else:
                markers.sort(key=lambda item: (item[1], item[2], item[0]))
                top_color = markers[0][0]
                bottom_color = markers[1][0]
                min_row = min(r for r, _ in cavity)
                max_row = max(r for r, _ in cavity)
                mid_row = (min_row + max_row) // 2
                for r, c in cavity:
                    out[r, c] = top_color if r <= mid_row else bottom_color

        for r, c in frame_cells:
            out[r, c] = left_piece.color

    return out


_MAX_ATTEMPTS = 400
_GRID_H_RANGE = (18, 30)
_GRID_W_RANGE = (20, 34)
_PAIR_COUNT_CHOICES = (1, 2, 3)
_FRAME_HEIGHT_RANGE = (5, 9)
_FRAME_WIDTH_RANGE = (4, 8)
_H_GAP_RANGE = (2, 6)
_PADDING = 1


def _frame_patch(height: int, width: int, direction: str) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = 1
    patch[-1, :] = 1
    patch[:, 0] = 1
    patch[:, -1] = 1
    if direction == "right":
        patch[1:-1, -1] = 0
    else:
        patch[1:-1, 0] = 0
    return patch


def _bbox_clear(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = _PADDING) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + height + pad)
    c1 = min(occupied.shape[1], left + width + pad)
    return not occupied[r0:r1, c0:c1].any()


def _stamp_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rows, cols = np.nonzero(mask)
    for rr, cc in zip(rows.tolist(), cols.tolist()):
        grid[top + rr, left + cc] = color


def _choose_marker_positions(
    rng: random.Random,
    top: int,
    left: int,
    height: int,
    width: int,
    count: int,
    distinct_rows: bool,
) -> list[tuple[int, int]]:
    rows = list(range(top + 1, top + height - 1))
    cols = list(range(left + 1, left + width - 1))
    positions = [(r, c) for r in rows for c in cols]
    if count == 1:
        return [rng.choice(positions)]
    if distinct_rows:
        row_a, row_b = rng.sample(rows, 2)
        return [
            (row_a, rng.choice(cols)),
            (row_b, rng.choice(cols)),
        ]
    return rng.sample(positions, 2)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*_GRID_H_RANGE)
    width = rng.randint(*_GRID_W_RANGE)
    bg = rng.choice((0, 1, 2))
    grid = np.full((height, width), bg, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    pair_count = rng.choice(_PAIR_COUNT_CHOICES)
    frame_colors = rng.sample([c for c in range(10) if c != bg], pair_count)
    marker_pool = [c for c in range(10) if c not in {bg, *frame_colors}]

    for pair_index in range(pair_count):
        frame_color = frame_colors[pair_index]
        branch = rng.randrange(4)

        placed = False
        for _ in range(120):
            fh = rng.randint(*_FRAME_HEIGHT_RANGE)
            fw_left = rng.randint(*_FRAME_WIDTH_RANGE)
            fw_right = rng.randint(*_FRAME_WIDTH_RANGE)
            gap = rng.randint(*_H_GAP_RANGE)
            span_h = fh
            span_w = fw_left + gap + fw_right
            if span_h > height - 2 or span_w > width - 2:
                continue

            top = rng.randint(1, height - span_h - 1)
            left = rng.randint(1, width - span_w - 1)
            if not _bbox_clear(occupied, top, left, span_h, span_w):
                continue

            left_patch = _frame_patch(fh, fw_left, "right")
            right_patch = _frame_patch(fh, fw_right, "left")
            _stamp_mask(grid, top, left, left_patch, frame_color)
            _stamp_mask(grid, top, left + fw_left + gap, right_patch, frame_color)

            if branch == 1:
                marker_color = rng.choice(marker_pool)
                target_is_left = bool(rng.randrange(2))
                if target_is_left:
                    pos = _choose_marker_positions(rng, top, left, fh, fw_left, count=1, distinct_rows=False)[0]
                else:
                    pos = _choose_marker_positions(
                        rng, top, left + fw_left + gap, fh, fw_right, count=1, distinct_rows=False
                    )[0]
                grid[pos] = marker_color
            elif branch == 2:
                marker_color = rng.choice(marker_pool)
                left_pos = _choose_marker_positions(rng, top, left, fh, fw_left, count=1, distinct_rows=False)[0]
                right_pos = _choose_marker_positions(
                    rng, top, left + fw_left + gap, fh, fw_right, count=1, distinct_rows=False
                )[0]
                grid[left_pos] = marker_color
                grid[right_pos] = marker_color
            elif branch == 3:
                marker_a, marker_b = rng.sample(marker_pool, 2)
                left_pos = _choose_marker_positions(rng, top, left, fh, fw_left, count=1, distinct_rows=False)[0]
                right_pos = _choose_marker_positions(
                    rng, top, left + fw_left + gap, fh, fw_right, count=1, distinct_rows=False
                )[0]
                for _ in range(20):
                    if left_pos[0] != right_pos[0]:
                        break
                    right_pos = _choose_marker_positions(
                        rng, top, left + fw_left + gap, fh, fw_right, count=1, distinct_rows=False
                    )[0]
                if left_pos[0] == right_pos[0]:
                    continue
                grid[left_pos] = marker_a
                grid[right_pos] = marker_b

            occupied[top : top + span_h, left : left + span_w] = True
            placed = True
            break

        if not placed:
            return None

    if _background_color(grid) != bg:
        return None
    if len(_extract_pairs(grid, bg)) != pair_count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retry budget")
