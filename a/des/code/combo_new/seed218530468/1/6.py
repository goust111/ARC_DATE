from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

FRAME_COUNT_RANGE = (2, 5)  # inclusive lower, exclusive upper
GRID_SIDE_MAX = 30
FRAME_OUTER_MIN = 6
EDGE_MARGIN = 2
FRAME_PADDING = 1
FRAME_PLACE_TRIES = 160
GENERATE_RETRIES = 400

QUADRANTS = (
    (True, True),    # top-left
    (True, False),   # top-right
    (False, True),   # bottom-left
    (False, False),  # bottom-right
)


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for start_r, start_c in np.argwhere(mask):
        r0 = int(start_r)
        c0 = int(start_c)
        if visited[r0, c0]:
            continue
        visited[r0, c0] = True
        queue = deque([(r0, c0)])
        comp: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            comp.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        components.append(comp)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_frame_component(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < FRAME_OUTER_MIN or outer_w < FRAME_OUTER_MIN:
        return False

    expected_cells = 2 * (outer_h + outer_w) - 4
    if len(coords) != expected_cells:
        return False

    pts = set(coords)
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != BG:
                return False
    return True


def _parse_instances(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    instances: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        comps = _connected_components(g == color)
        markers: list[tuple[int, int]] = []
        frame_bboxes: list[tuple[int, int, int, int]] = []
        for comp in comps:
            if len(comp) == 1:
                markers.extend(comp)
            elif _is_frame_component(g, color, comp):
                frame_bboxes.append(_bbox(comp))
            else:
                raise AssertionError(f"unexpected color-{color} component")

        assert len(markers) == 1, f"color {color} must have exactly one marker"
        assert len(frame_bboxes) == 1, f"color {color} must have exactly one frame"

        marker = markers[0]
        bbox = frame_bboxes[0]
        r0, c0, r1, c1 = bbox
        assert not (r0 <= marker[0] <= r1 and c0 <= marker[1] <= c1), "marker must stay outside its frame"
        instances.append({"color": color, "bbox": bbox, "marker": marker})
    return instances


def _paint_quadrant(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    marker: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    rows = np.arange(r0 + 1, r1, dtype=int)
    cols = np.arange(c0 + 1, c1, dtype=int)
    if rows.size == 0 or cols.size == 0:
        return

    top_half = marker[0] < center_r
    left_half = marker[1] < center_c
    row_mask = rows < center_r if top_half else rows >= center_r
    col_mask = cols < center_c if left_half else cols >= center_c
    fill_rows = rows[row_mask]
    fill_cols = cols[col_mask]
    if fill_rows.size == 0 or fill_cols.size == 0:
        return

    patch = out[np.ix_(fill_rows, fill_cols)]
    patch[patch == BG] = color
    out[np.ix_(fill_rows, fill_cols)] = patch


def _apply_rule_from_specs(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for spec in specs:
        _paint_quadrant(
            out,
            spec["bbox"],  # type: ignore[arg-type]
            int(spec["color"]),
            spec["marker"],  # type: ignore[arg-type]
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill one interior quadrant of each hollow frame using its matching external marker.

    Input structure:
    - Background is black.
    - Several non-touching hollow rectangular frames appear in distinct colors.
    - Each frame color also appears once as a single-pixel marker outside the frame.
    - Every frame interior is black and large enough to split into four quadrants.

    Rule:
    - Keep every frame border and every external marker unchanged.
    - Pair each marker with the frame of the same color.
    - Compare the marker with the frame center: above/below chooses the vertical half, left/right chooses the horizontal half.
    - Fill the black cells in the corresponding interior quadrant with the frame color.

    Key insight:
    - Color identifies which frame to control, while the marker's 2D position around that frame encodes which quadrant to paint.
    """
    g = np.asarray(grid, dtype=int)
    specs = _parse_instances(g)
    return _apply_rule_from_specs(g, specs)


def _mark_padded(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _is_isolated_singleton(occupied: np.ndarray, r: int, c: int) -> bool:
    if occupied[r, c]:
        return False
    h, w = occupied.shape
    rr0 = max(0, r - 1)
    cc0 = max(0, c - 1)
    rr1 = min(h, r + 2)
    cc1 = min(w, c + 2)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _half_columns(c0: int, c1: int, go_left: bool) -> list[int]:
    center_c = (c0 + c1) / 2.0
    return [c for c in range(c0, c1 + 1) if (c < center_c) == go_left]


def _half_rows(r0: int, r1: int, go_top: bool) -> list[int]:
    center_r = (r0 + r1) / 2.0
    return [r for r in range(r0, r1 + 1) if (r < center_r) == go_top]


def _marker_candidates(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    go_top: bool,
    go_left: bool,
    side_mode: str,
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if side_mode == "vertical":
        rows = list(range(0, r0 - 1)) if go_top else list(range(r1 + 2, h))
        cols = _half_columns(c0, c1, go_left)
    else:
        rows = _half_rows(r0, r1, go_top)
        cols = list(range(0, c0 - 1)) if go_left else list(range(c1 + 2, w))
    return [(r, c) for r in rows for c in cols]


def _sample_grid_size(rng: random.Random, frame_count: int) -> tuple[int, int]:
    side_min = 15 + 2 * frame_count
    h = rng.randint(side_min, GRID_SIDE_MAX)
    w = rng.randint(side_min, GRID_SIDE_MAX)
    return h, w


def _place_frames(
    rng: random.Random,
    shape: tuple[int, int],
    colors: list[int],
) -> list[dict[str, object]] | None:
    h, w = shape
    max_outer = min(11 - max(0, len(colors) - 2), h - 2 * EDGE_MARGIN, w - 2 * EDGE_MARGIN)
    if max_outer < FRAME_OUTER_MIN:
        return None

    dims: list[tuple[int, int, int]] = []
    for color in colors:
        outer_h = rng.randint(FRAME_OUTER_MIN, max_outer)
        outer_w = rng.randint(FRAME_OUTER_MIN, max_outer)
        dims.append((outer_h, outer_w, color))
    dims.sort(key=lambda item: item[0] * item[1], reverse=True)

    blocked = np.zeros((h, w), dtype=bool)
    frames: list[dict[str, object]] = []
    for outer_h, outer_w, color in dims:
        placed = False
        r_lo = EDGE_MARGIN
        c_lo = EDGE_MARGIN
        r_hi = h - EDGE_MARGIN - outer_h
        c_hi = w - EDGE_MARGIN - outer_w
        if r_lo > r_hi or c_lo > c_hi:
            return None
        for _ in range(FRAME_PLACE_TRIES):
            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            rr0 = max(0, r0 - FRAME_PADDING)
            cc0 = max(0, c0 - FRAME_PADDING)
            rr1 = min(h - 1, r1 + FRAME_PADDING)
            cc1 = min(w - 1, c1 + FRAME_PADDING)
            if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            _mark_padded(blocked, r0, c0, r1, c1, FRAME_PADDING)
            frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
            placed = True
            break
        if not placed:
            return None
    return frames


def _place_markers(
    rng: random.Random,
    grid: np.ndarray,
    frames: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    h, w = grid.shape
    occupied = grid != BG
    quadrants = list(QUADRANTS)
    rng.shuffle(quadrants)
    specs: list[dict[str, object]] = []

    for idx, frame in enumerate(frames):
        go_top, go_left = quadrants[idx % len(quadrants)]
        bbox = frame["bbox"]  # type: ignore[assignment]
        color = int(frame["color"])
        side_modes = ["vertical", "horizontal"]
        rng.shuffle(side_modes)
        marker: tuple[int, int] | None = None

        for side_mode in side_modes:
            candidates = _marker_candidates((h, w), bbox, go_top, go_left, side_mode)  # type: ignore[arg-type]
            rng.shuffle(candidates)
            for r, c in candidates:
                if _is_isolated_singleton(occupied, r, c):
                    marker = (r, c)
                    break
            if marker is not None:
                break

        if marker is None:
            return None

        grid[marker] = color
        occupied[marker] = True
        specs.append({"color": color, "bbox": bbox, "marker": marker})
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1] - 1)
    h, w = _sample_grid_size(rng, frame_count)
    colors = rng.sample(PALETTE, frame_count)

    frames = _place_frames(rng, (h, w), colors)
    if frames is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        _draw_frame(grid, frame["bbox"], int(frame["color"]))  # type: ignore[arg-type]

    specs = _place_markers(rng, grid, frames)
    if specs is None:
        return None

    expected = _apply_rule_from_specs(grid, sorted(specs, key=lambda spec: int(spec["color"])))
    try:
        parsed = _parse_instances(grid)
    except AssertionError:
        return None
    if len(parsed) != frame_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")
