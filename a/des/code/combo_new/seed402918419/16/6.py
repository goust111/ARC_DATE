from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 7
STRIPE_COLOR = 1
CHECKER_COLOR = 2

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _mode_color(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if np.array_equal(sub, expected):
        return r0, c0, r1, c1
    return None


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _components(grid == color)
        frame_bboxes: list[tuple[int, int, int, int]] = []
        covered = 0
        for coords in comps:
            comp_mask = np.zeros_like(grid, dtype=bool)
            comp_mask[coords[:, 0], coords[:, 1]] = True
            bbox = _rect_frame_bbox(comp_mask)
            if bbox is None:
                continue
            frame_bboxes.append(bbox)
            covered += int(coords.shape[0])
        if frame_bboxes and covered == int(np.count_nonzero(grid == color)):
            candidates.append((color, frame_bboxes))
    assert len(candidates) == 1, "expected a unique frame color"
    frame_color, frame_bboxes = candidates[0]
    return int(frame_color), frame_bboxes


def _is_stripe_pattern(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 2:
        return False
    for parity in (0, 1):
        expected = np.zeros((h, w), dtype=bool)
        expected[parity::2, :] = True
        if np.array_equal(mask, expected):
            return True
    return False


def _is_checker_pattern(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 2:
        return False
    rr, cc = np.indices((h, w))
    for parity in (0, 1):
        expected = (rr + cc) % 2 == parity
        if np.array_equal(mask, expected):
            return True
    return False


def _infer_reference_colors(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frame_bboxes: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    outside = np.ones(grid.shape, dtype=bool)
    for r0, c0, r1, c1 in frame_bboxes:
        outside[r0 : r1 + 1, c0 : c1 + 1] = False

    stripe_color = None
    checker_color = None
    for color in map(int, np.unique(grid[outside])):
        if color in {bg, frame_color}:
            continue
        coords = np.argwhere(outside & (grid == color))
        assert coords.size > 0, "reference colors must appear outside the frames"
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        sub = (outside & (grid == color))[r0 : r1 + 1, c0 : c1 + 1]
        if _is_stripe_pattern(sub):
            assert stripe_color is None
            stripe_color = color
        elif _is_checker_pattern(sub):
            assert checker_color is None
            checker_color = color
        else:
            raise AssertionError("unexpected outside pattern")
    assert stripe_color is not None and checker_color is not None
    return int(stripe_color), int(checker_color)


def _nearest_inner_corner(
    marker: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    corners = [
        (r0 + 1, c0 + 1),
        (r0 + 1, c1 - 1),
        (r1 - 1, c0 + 1),
        (r1 - 1, c1 - 1),
    ]
    dists = [abs(mr - rr) + abs(mc - cc) for rr, cc in corners]
    best = min(dists)
    winners = [corners[i] for i, dist in enumerate(dists) if dist == best]
    assert len(winners) == 1, "marker must have a unique nearest inner corner"
    return winners[0]


def _paint_interior(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
    fill_color: int,
    checker: bool,
) -> None:
    r0, c0, r1, c1 = bbox
    anchor_r, anchor_c = _nearest_inner_corner(marker, bbox)
    marker_r, marker_c = marker

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r, c) == (marker_r, marker_c):
                continue
            if checker:
                should_fill = (r - anchor_r + c - anchor_c) % 2 == 0
            else:
                should_fill = (r - anchor_r) % 2 == 0
            if should_fill and out[r, c] == BG:
                out[r, c] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed rectangle with corner-aligned stripes or checkerboard chosen from its shape.

    Input structure:
    - The background is black.
    - Several orange 1-pixel rectangular frames enclose empty interiors, and each interior contains exactly one blue or red marker pixel.
    - Two small unframed reference patches sit in one corner: a blue alternating-row stripe patch and a red checkerboard patch.

    Rule:
    - Detect the orange rectangular frames and the two outside reference patterns.
    - For each frame, use the checkerboard pattern when the interior is square and the stripe pattern when the interior is a non-square rectangle.
    - Find the marker's nearest inner corner and align the chosen pattern from that corner; keep the marker pixel and the orange frame unchanged.

    Key insight:
    - The marker color is a decoy for pattern selection: geometry chooses stripe versus checkerboard, while the marker position only chooses the alignment corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g.reshape(-1))
    frame_color, frame_bboxes = _find_frames(g, bg)
    stripe_color, checker_color = _infer_reference_colors(g, bg, frame_color, frame_bboxes)

    out = g.copy()
    for bbox in frame_bboxes:
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_coords = np.argwhere((interior != bg) & (interior != frame_color))
        assert marker_coords.shape[0] == 1, "each frame must contain exactly one marker"
        mr = int(marker_coords[0, 0] + r0 + 1)
        mc = int(marker_coords[0, 1] + c0 + 1)

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if inner_h == inner_w:
            _paint_interior(out, bbox, (mr, mc), checker_color, checker=True)
        else:
            _paint_interior(out, bbox, (mr, mc), stripe_color, checker=False)

    return out


def _stripe_patch(height: int, width: int, parity: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[parity::2, :] = STRIPE_COLOR
    return patch


def _checker_patch(height: int, width: int, parity: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    rr, cc = np.indices((height, width))
    patch[(rr + cc) % 2 == parity] = CHECKER_COLOR
    return patch


def _reserve(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0] - 1, r1 + pad)
    cc1 = min(blocked.shape[1] - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _fit_available(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0] - 1, r1 + pad)
    cc1 = min(blocked.shape[1] - 1, c1 + pad)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> tuple[int, int, int, int]:
    h, w = patch.shape
    mask = patch != BG
    grid[top : top + h, left : left + w][mask] = patch[mask]
    return top, left, top + h - 1, left + w - 1


def _reference_positions(
    grid_shape: tuple[int, int],
    stripe_shape: tuple[int, int],
    checker_shape: tuple[int, int],
    corner: str,
    layout: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    rows, cols = grid_shape
    sh, sw = stripe_shape
    ch, cw = checker_shape

    if layout == "horizontal":
        total_h = max(sh, ch)
        total_w = sw + 1 + cw
        if corner == "tl":
            return (0, 0), (0, sw + 1)
        if corner == "tr":
            left = cols - total_w
            return (0, left), (0, left + sw + 1)
        if corner == "bl":
            top = rows - total_h
            return (top, 0), (top, sw + 1)
        assert corner == "br"
        top = rows - total_h
        left = cols - total_w
        return (top, left), (top, left + sw + 1)

    total_h = sh + 1 + ch
    total_w = max(sw, cw)
    if corner == "tl":
        return (0, 0), (sh + 1, 0)
    if corner == "tr":
        left = cols - total_w
        return (0, left), (sh + 1, left)
    if corner == "bl":
        top = rows - total_h
        return (top, 0), (top + sh + 1, 0)
    assert corner == "br"
    top = rows - total_h
    left = cols - total_w
    return (top, left), (top + sh + 1, left)


def _top_left_unique_offsets(inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    corners = [(0, 0), (0, inner_w - 1), (inner_h - 1, 0), (inner_h - 1, inner_w - 1)]
    for dr in range(min(3, inner_h)):
        for dc in range(min(3, inner_w)):
            dists = [abs(dr - rr) + abs(dc - cc) for rr, cc in corners]
            if dists[0] < min(dists[1:]):
                offsets.append((dr, dc))
    assert offsets
    return offsets


def _marker_cell(
    bbox: tuple[int, int, int, int],
    corner: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    dr, dc = random.choice(_top_left_unique_offsets(inner_h, inner_w))

    if corner == "tl":
        return r0 + 1 + dr, c0 + 1 + dc
    if corner == "tr":
        return r0 + 1 + dr, c1 - 1 - dc
    if corner == "bl":
        return r1 - 1 - dr, c0 + 1 + dc
    assert corner == "br"
    return r1 - 1 - dr, c1 - 1 - dc


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _place_rectangles(
    grid: np.ndarray,
    blocked: np.ndarray,
    specs: list[tuple[int, int]],
) -> bool:
    rows, cols = grid.shape
    corners = ["tl", "tr", "bl", "br"]

    for inner_h, inner_w in sorted(specs, key=lambda hw: hw[0] * hw[1], reverse=True):
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        locations: list[tuple[int, int]] = []
        for r0 in range(1, rows - outer_h):
            for c0 in range(1, cols - outer_w):
                bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
                if _fit_available(blocked, bbox, pad=1):
                    locations.append((r0, c0))
        if not locations:
            return False

        r0, c0 = random.choice(locations)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        _draw_frame(grid, bbox)
        marker_r, marker_c = _marker_cell(bbox, random.choice(corners))
        grid[marker_r, marker_c] = random.choice([STRIPE_COLOR, CHECKER_COLOR])
        _reserve(blocked, bbox, pad=1)

    return True


def _sample_rectangle_specs(count: int) -> list[tuple[int, int]]:
    specs: list[tuple[int, int]] = []
    types = ["square", "rect"]
    for _ in range(count - 2):
        types.append(random.choice(["square", "rect"]))
    random.shuffle(types)

    for kind in types:
        if kind == "square":
            side = random.randint(3, 7)
            specs.append((side, side))
            continue

        h = random.randint(3, 7)
        w = random.randint(4, 9)
        if h == w:
            w += 1 if w < 9 else -1
        if random.random() < 0.5:
            h, w = w, h
        specs.append((h, w))
    return specs


def _make_case() -> np.ndarray:
    rows = random.randint(20, 30)
    cols = random.randint(20, 30)
    grid = np.full((rows, cols), BG, dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)

    stripe_patch = _stripe_patch(
        random.randint(3, 5),
        random.randint(3, 6),
        random.randint(0, 1),
    )
    checker_patch = _checker_patch(
        random.randint(3, 5),
        random.randint(3, 5),
        random.randint(0, 1),
    )
    corner = random.choice(["tl", "tr", "bl", "br"])
    layout = random.choice(["horizontal", "vertical"])
    stripe_pos, checker_pos = _reference_positions(
        (rows, cols),
        stripe_patch.shape,
        checker_patch.shape,
        corner,
        layout,
    )
    stripe_bbox = _stamp_patch(grid, stripe_patch, stripe_pos[0], stripe_pos[1])
    checker_bbox = _stamp_patch(grid, checker_patch, checker_pos[0], checker_pos[1])
    _reserve(blocked, stripe_bbox, pad=1)
    _reserve(blocked, checker_bbox, pad=1)

    n_rects = random.randint(2, 5)
    specs = _sample_rectangle_specs(n_rects)
    ok = _place_rectangles(grid, blocked, specs)
    assert ok
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _make_case()
            out = main(grid)
            assert grid.ndim == 2 and max(grid.shape) <= 30
            assert out.ndim == 2 and max(out.shape) <= 30
            assert np.any(out != grid), "transformation must change the grid"

            frame_color, frame_bboxes = _find_frames(grid, BG)
            assert frame_color == FRAME_COLOR
            shapes = {(r1 - r0 - 1, c1 - c0 - 1) for r0, c0, r1, c1 in frame_bboxes}
            has_square = any(h == w for h, w in shapes)
            has_rect = any(h != w for h, w in shapes)
            assert has_square and has_rect
            return grid
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid ARC sample")
