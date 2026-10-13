from __future__ import annotations

import math
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return tuple(out)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _is_rect_frame(comp: dict) -> bool:
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    pts = {tuple(map(int, rc)) for rc in coords}
    if len(pts) != 2 * h + 2 * w - 4:
        return False

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if ((r, c) in pts) != on_border:
                return False
    return True


def _sort_frames(frames: list[dict]) -> list[dict]:
    return sorted(frames, key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))


def _detect_frames(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    frames = [comp for comp in _color_components(grid, bg) if _is_rect_frame(comp)]
    assert frames, "expected at least one rectangular frame"

    shapes = {(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1]) for f in frames}
    assert len(shapes) == 1, "all frames must share a common size"

    colors = [int(f["color"]) for f in frames]
    assert len(colors) == len(set(colors)), "frame border colors must be unique"
    return _sort_frames(frames)


def _read_strip_order(grid: np.ndarray, coords: np.ndarray, frame_colors: set[int]) -> list[int] | None:
    comp_bbox = _bbox(coords)
    color_boxes: dict[int, tuple[int, int, int, int]] = {}

    for color in frame_colors:
        pts = coords[np.asarray([int(grid[r, c]) == color for r, c in coords], dtype=bool)]
        if pts.size == 0:
            return None
        bb = _bbox(pts)
        area = int((bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1))
        if int(pts.shape[0]) != area:
            return None
        color_boxes[int(color)] = bb

    if all(bb[0] == comp_bbox[0] and bb[2] == comp_bbox[2] for bb in color_boxes.values()):
        order = sorted(frame_colors, key=lambda color: color_boxes[int(color)][1])
        prev = comp_bbox[1]
        for color in order:
            bb = color_boxes[int(color)]
            if bb[1] != prev:
                return None
            prev = bb[3] + 1
        if prev == comp_bbox[3] + 1:
            return [int(color) for color in order]

    if all(bb[1] == comp_bbox[1] and bb[3] == comp_bbox[3] for bb in color_boxes.values()):
        order = sorted(frame_colors, key=lambda color: color_boxes[int(color)][0])
        prev = comp_bbox[0]
        for color in order:
            bb = color_boxes[int(color)]
            if bb[0] != prev:
                return None
            prev = bb[2] + 1
        if prev == comp_bbox[2] + 1:
            return [int(color) for color in order]

    return None


def _detect_strip_order(grid: np.ndarray, frames: list[dict]) -> list[int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    frame_colors = {int(frame["color"]) for frame in frames}
    frame_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        pts = frame["coords"]
        frame_mask[pts[:, 0], pts[:, 1]] = True

    remaining = (g != _get_bg(g)) & (~frame_mask)
    candidates: list[list[int]] = []
    for coords in _mask_components(remaining):
        r0, c0, r1, c1 = _bbox(coords)
        touches_corner = (r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)
        if not touches_corner:
            continue
        colors = {int(g[r, c]) for r, c in coords}
        if colors != frame_colors:
            continue
        order = _read_strip_order(g, coords, frame_colors)
        if order is not None:
            candidates.append(order)

    assert len(candidates) == 1, "expected a unique corner color strip"
    return candidates[0]


def _apply_reorder(grid: np.ndarray, frames: list[dict], strip_order: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    frame_by_color = {int(frame["color"]): frame for frame in frames}
    frame_colors = set(frame_by_color)
    assert set(strip_order) == frame_colors
    assert len(strip_order) == len(frames)

    interiors: dict[int, np.ndarray] = {}
    for color, frame in frame_by_color.items():
        r0, c0, r1, c1 = frame["bbox"]
        interiors[int(color)] = out[r0 + 1 : r1, c0 + 1 : c1].copy()

    for idx, color in enumerate(strip_order):
        r0, c0, r1, c1 = frames[idx]["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = interiors[int(color)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the interior motifs of equal-sized frames by the border-color order shown in a corner strip.

    Input structure:
    - The background is black, with several equal-sized one-pixel rectangular frames in distinct colors.
    - Each frame contains a small colored motif inside its border, and a corner of the grid holds a horizontal or vertical strip of color blocks.
    - The strip uses exactly the frame-border colors, once each, in a visible sequence.

    Rule:
    - Detect every rectangular frame and record its border color and interior bitmap.
    - Read the corner strip from left to right or top to bottom to obtain an order of border colors.
    - Number frame positions in reading order, then copy the interior of the frame whose border color is the k-th strip color into the k-th frame position.
    - Keep the borders and the strip unchanged; only the frame interiors are reassigned.

    Key insight:
    - The border colors are sortable keys, and the corner strip specifies the permutation that maps color identity to spatial frame position.
    """
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    strip_order = _detect_strip_order(g, frames)
    return _apply_reorder(g, frames, strip_order)


def _random_connected_shape(
    rng: np.random.Generator, max_h: int, max_w: int, area: int
) -> np.ndarray:
    dims = [
        (h, w)
        for h in range(1, max_h + 1)
        for w in range(1, max_w + 1)
        if h * w >= area
    ]
    h, w = dims[int(rng.integers(0, len(dims)))]
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    patch = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=bool)
    for r, c in cells:
        patch[r - r0, c - c0] = True
    return patch


def _random_pattern(
    rng: np.random.Generator,
    interior_h: int,
    interior_w: int,
    color: int,
    used: set[bytes],
) -> np.ndarray | None:
    for _ in range(100):
        max_h = max(1, min(interior_h - 1, 4))
        max_w = max(1, min(interior_w - 1, 4))
        area = int(rng.integers(1, min(6, max_h * max_w) + 1))
        shape = _random_connected_shape(rng, max_h, max_w, area)
        sh, sw = shape.shape

        canvas = np.full((interior_h, interior_w), BG, dtype=int)
        if interior_h - sh < 0 or interior_w - sw < 0:
            continue
        top = int(rng.integers(0, interior_h - sh + 1))
        left = int(rng.integers(0, interior_w - sw + 1))
        canvas[top : top + sh, left : left + sw][shape] = int(color)

        sig = canvas.tobytes()
        if sig not in used:
            used.add(sig)
            return canvas
    return None


def _can_place(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if r0 - gap <= pr1 and pr0 - gap <= r1 and c0 - gap <= pc1 and pc0 - gap <= c1:
            return False
    return True


def _sample_frame_positions(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_frames: int,
    frame_h: int,
    frame_w: int,
    reserved: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]] | None:
    placed = [reserved]
    frames: list[tuple[int, int, int, int]] = []
    for _ in range(n_frames):
        found = None
        for _ in range(400):
            r0 = int(rng.integers(1, h - frame_h))
            c0 = int(rng.integers(1, w - frame_w))
            rect = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            if _can_place(rect, placed, gap=1):
                found = rect
                break
        if found is None:
            return None
        frames.append(found)
        placed.append(found)
    return sorted(frames, key=lambda b: (b[0], b[1]))


def _strip_bbox(
    h: int,
    w: int,
    n_frames: int,
    orientation: str,
    corner: str,
    thickness: int,
    block_len: int,
) -> tuple[int, int, int, int]:
    strip_h = thickness if orientation == "h" else n_frames * block_len
    strip_w = n_frames * block_len if orientation == "h" else thickness

    if "t" in corner:
        r0 = 0
    else:
        r0 = h - strip_h
    if "l" in corner:
        c0 = 0
    else:
        c0 = w - strip_w
    return (r0, c0, r0 + strip_h - 1, c0 + strip_w - 1)


def _draw_frame(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_strip(
    grid: np.ndarray,
    order: list[int],
    orientation: str,
    corner: str,
    thickness: int,
    block_len: int,
) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = _strip_bbox(h, w, len(order), orientation, corner, thickness, block_len)

    if orientation == "h":
        for idx, color in enumerate(order):
            cc0 = c0 + idx * block_len
            grid[r0 : r1 + 1, cc0 : cc0 + block_len] = int(color)
    else:
        for idx, color in enumerate(order):
            rr0 = r0 + idx * block_len
            grid[rr0 : rr0 + block_len, c0 : c1 + 1] = int(color)


def _expected_output(
    inp: np.ndarray,
    frame_rects: list[tuple[int, int, int, int]],
    frame_colors_in_space_order: list[int],
    strip_order: list[int],
) -> np.ndarray:
    out = inp.copy()
    interiors = {
        color: inp[r0 + 1 : r1, c0 + 1 : c1].copy()
        for color, (r0, c0, r1, c1) in zip(frame_colors_in_space_order, frame_rects)
    }
    for idx, color in enumerate(strip_order):
        r0, c0, r1, c1 = frame_rects[idx]
        out[r0 + 1 : r1, c0 + 1 : c1] = interiors[int(color)]
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(3, 6))
    orientation = "h" if bool(rng.integers(0, 2)) else "v"
    corner = ("tl", "tr", "bl", "br")[int(rng.integers(0, 4))]
    thickness = int(rng.integers(1, 3))
    block_len = int(rng.integers(1, 4))
    frame_h = int(rng.integers(6, 9))
    frame_w = int(rng.integers(6, 9))

    strip_h = thickness if orientation == "h" else n_frames * block_len
    strip_w = n_frames * block_len if orientation == "h" else thickness
    min_side = max(
        18,
        frame_h + 3,
        frame_w + 3,
        strip_h + 2,
        strip_w + 2,
        int(math.ceil(math.sqrt(n_frames * (frame_h + 2) * (frame_w + 2) + strip_h * strip_w + 20))),
    )
    if min_side > 30:
        return None

    h = int(rng.integers(min_side, 31))
    w = int(rng.integers(min_side, 31))
    reserved = _strip_bbox(h, w, n_frames, orientation, corner, thickness, block_len)
    frame_rects = _sample_frame_positions(rng, h, w, n_frames, frame_h, frame_w, reserved)
    if frame_rects is None:
        return None

    frame_colors = [int(c) for c in rng.choice(PALETTE, size=n_frames, replace=False)]
    spatial_frame_colors = frame_colors.copy()
    strip_order = spatial_frame_colors.copy()
    while strip_order == spatial_frame_colors:
        strip_order = [int(c) for c in rng.permutation(spatial_frame_colors)]

    pattern_pool = [color for color in PALETTE if color not in spatial_frame_colors]
    interior_h = frame_h - 2
    interior_w = frame_w - 2
    used_patterns: set[bytes] = set()
    interiors: list[np.ndarray] = []
    for _ in range(n_frames):
        color = int(pattern_pool[int(rng.integers(0, len(pattern_pool)))])
        pattern = _random_pattern(rng, interior_h, interior_w, color, used_patterns)
        if pattern is None:
            return None
        interiors.append(pattern)

    inp = np.full((h, w), BG, dtype=int)
    for rect, frame_color, interior in zip(frame_rects, spatial_frame_colors, interiors):
        _draw_frame(inp, rect, int(frame_color))
        r0, c0, r1, c1 = rect
        inp[r0 + 1 : r1, c0 + 1 : c1] = interior
    _draw_strip(inp, strip_order, orientation, corner, thickness, block_len)

    out = _expected_output(inp, frame_rects, spatial_frame_colors, strip_order)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _gen_attempt()
        if built is None:
            continue
        inp, out = built
        try:
            if np.array_equal(main(inp), out):
                return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
