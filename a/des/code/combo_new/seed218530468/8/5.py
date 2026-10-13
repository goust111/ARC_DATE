from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
SIDES = ("top", "bottom", "left", "right")
INWARD = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

GRID_MIN = 18
GRID_MAX = 30
MARGIN = 4
FRAME_H_MIN = 10
FRAME_W_MIN = 10
FRAME_H_MAX = 18
FRAME_W_MAX = 18
N_FRAGMENTS_MIN = 2
N_FRAGMENTS_MAX = 6
FRAGMENT_TRIES = 80
MAX_ATTEMPTS = 400

CANONICAL_TOP_SHAPES = (
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (0, 1)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
)


def _normalize(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    out = arr.copy()
    out[:, 0] -= int(out[:, 0].min())
    out[:, 1] -= int(out[:, 1].min())
    order = np.lexsort((out[:, 1], out[:, 0]))
    return out[order]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _shape_dims(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _flip_h(coords: np.ndarray) -> np.ndarray:
    h, w = _shape_dims(coords)
    out = np.column_stack((coords[:, 0], (w - 1) - coords[:, 1]))
    return _normalize(out)


def _flip_v(coords: np.ndarray) -> np.ndarray:
    h, w = _shape_dims(coords)
    out = np.column_stack(((h - 1) - coords[:, 0], coords[:, 1]))
    return _normalize(out)


def _rot_cw(coords: np.ndarray) -> np.ndarray:
    h, w = _shape_dims(coords)
    out = np.column_stack((coords[:, 1], (h - 1) - coords[:, 0]))
    return _normalize(out)


def _rot_ccw(coords: np.ndarray) -> np.ndarray:
    h, w = _shape_dims(coords)
    out = np.column_stack(((w - 1) - coords[:, 1], coords[:, 0]))
    return _normalize(out)


def _orient_shape(coords: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return coords
    if side == "bottom":
        return _flip_v(coords)
    if side == "left":
        return _rot_ccw(coords)
    if side == "right":
        return _rot_cw(coords)
    raise ValueError(side)


def _local_anchor(coords: np.ndarray, side: str) -> tuple[int, int]:
    if side == "top":
        idx = np.flatnonzero(coords[:, 0] == int(coords[:, 0].max()))
    elif side == "bottom":
        idx = np.flatnonzero(coords[:, 0] == int(coords[:, 0].min()))
    elif side == "left":
        idx = np.flatnonzero(coords[:, 1] == int(coords[:, 1].max()))
    elif side == "right":
        idx = np.flatnonzero(coords[:, 1] == int(coords[:, 1].min()))
    else:
        raise ValueError(side)
    assert len(idx) == 1, "shape must have a unique anchor on the frame-facing extreme"
    cell = coords[int(idx[0])]
    return int(cell[0]), int(cell[1])


def _shape_library(side: str) -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[np.ndarray] = []
    for raw in CANONICAL_TOP_SHAPES:
        base = _normalize(np.asarray(raw, dtype=int))
        for flipped in (base, _flip_h(base)):
            shape = _orient_shape(flipped, side)
            key = tuple((int(r), int(c)) for r, c in shape)
            if key not in seen:
                seen.add(key)
                out.append(shape)
    return out


SIDE_SHAPES = {side: _shape_library(side) for side in SIDES}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in DIR4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(coords, dtype=int))
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(counts.argmax())])


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return False
    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(patch, border)


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray, tuple[int, int, int, int]]:
    frame: tuple[int, np.ndarray, tuple[int, int, int, int]] | None = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            if _is_rect_frame(comp):
                assert frame is None, "expected a unique hollow rectangular frame"
                frame = (color, comp, _bbox(comp))
    assert frame is not None, "expected one hollow rectangular frame"
    return frame


def _anchor_candidates(coords: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> list[tuple[int, str]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    candidates: list[tuple[int, str]] = []
    for i, (r, c) in enumerate(coords):
        r = int(r)
        c = int(c)
        if r == fr0 - 1 and fc0 <= c <= fc1:
            candidates.append((i, "top"))
        if r == fr1 + 1 and fc0 <= c <= fc1:
            candidates.append((i, "bottom"))
        if c == fc0 - 1 and fr0 <= r <= fr1:
            candidates.append((i, "left"))
        if c == fc1 + 1 and fr0 <= r <= fr1:
            candidates.append((i, "right"))
    return candidates


def _project_fragment(
    coords: np.ndarray,
    side: str,
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    fr0, fc0, fr1, fc1 = frame_bbox
    r0, c0, r1, c1 = _bbox(coords)
    depth = (r1 - r0 + 1) if side in {"top", "bottom"} else (c1 - c0 + 1)
    dr, dc = INWARD[side]
    shift = depth + 1
    projected = coords + np.asarray([dr * shift, dc * shift], dtype=int)
    assert np.all(projected[:, 0] > fr0) and np.all(projected[:, 0] < fr1)
    assert np.all(projected[:, 1] > fc0) and np.all(projected[:, 1] < fc1)
    return projected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each outside colored fragment into the hollow frame along the inward normal of its anchor side.

    Input structure:
    - The background is a dominant color.
    - A single one-pixel-thick hollow rectangular frame sits near the center.
    - Outside the frame are several small 4-connected monochrome fragments in distinct colors.
    - Each fragment has exactly one anchor pixel that is orthogonally adjacent to the outer side of one frame edge.

    Rule:
    - Detect which frame edge each fragment's anchor touches; that edge determines the inward projection direction.
    - Measure the fragment's thickness along that direction: vertical span for top/bottom anchors, horizontal span for left/right anchors.
    - Erase the original outside fragment, then translate the whole fragment inward by one border cell plus its measured thickness.
    - Keep the grey frame unchanged and draw the translated fragment inside the frame.

    Key insight:
    - The fragment shape encodes both what to draw and how far to move it: its span along the inward normal is reused as the projection distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    frame_color, frame_coords, frame_bbox = _find_frame(g, bg)

    fragments: list[tuple[int, np.ndarray]] = []
    for color in np.unique(g):
        color = int(color)
        if color in {bg, frame_color}:
            continue
        comps = _components(g == color)
        assert len(comps) == 1, "each fragment color must define one component"
        fragments.append((color, comps[0]))

    out = g.copy()
    projected_mask = np.zeros_like(g, dtype=bool)

    for color, coords in fragments:
        candidates = _anchor_candidates(coords, frame_bbox)
        assert len(candidates) == 1, "each fragment must have exactly one frame-adjacent anchor"
        _anchor_idx, side = candidates[0]
        projected = _project_fragment(coords, side, frame_bbox)

        out[coords[:, 0], coords[:, 1]] = bg
        assert not projected_mask[projected[:, 0], projected[:, 1]].any(), "projected fragments must not overlap"
        projected_mask[projected[:, 0], projected[:, 1]] = True
        out[projected[:, 0], projected[:, 1]] = color

    out[frame_coords[:, 0], frame_coords[:, 1]] = frame_color
    return out


def _draw_frame(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0:c1 + 1] = FRAME_COLOR
    grid[r1, c0:c1 + 1] = FRAME_COLOR
    grid[r0:r1 + 1, c0] = FRAME_COLOR
    grid[r0:r1 + 1, c1] = FRAME_COLOR


def _sample_frame(rng: np.random.Generator, h: int, w: int) -> tuple[int, int, int, int]:
    max_fh = min(FRAME_H_MAX, h - 2 * MARGIN)
    max_fw = min(FRAME_W_MAX, w - 2 * MARGIN)
    fh = int(rng.integers(FRAME_H_MIN, max_fh + 1))
    fw = int(rng.integers(FRAME_W_MIN, max_fw + 1))

    center_r = (h - fh) // 2
    center_c = (w - fw) // 2
    r_lo = max(MARGIN, center_r - 2)
    r_hi = min(h - fh - MARGIN, center_r + 2)
    c_lo = max(MARGIN, center_c - 2)
    c_hi = min(w - fw - MARGIN, center_c + 2)

    r0 = int(rng.integers(r_lo, r_hi + 1))
    c0 = int(rng.integers(c_lo, c_hi + 1))
    return r0, c0, r0 + fh - 1, c0 + fw - 1


def _sample_sides(rng: np.random.Generator, n_fragments: int) -> list[str]:
    side_ids = list(range(4))
    rng.shuffle(side_ids)
    chosen = side_ids[: min(4, n_fragments)]
    while len(chosen) < n_fragments:
        chosen.append(int(rng.integers(0, 4)))
    rng.shuffle(chosen)
    return [SIDES[i] for i in chosen]


def _touches_mask(mask: np.ndarray, coords: np.ndarray, dirs: tuple[tuple[int, int], ...]) -> bool:
    h, w = mask.shape
    for r, c in coords:
        r = int(r)
        c = int(c)
        if mask[r, c]:
            return True
        for dr, dc in dirs:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _candidate_anchors(
    shape: np.ndarray,
    side: str,
    frame_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    h, w = _shape_dims(shape)
    anchor_r, anchor_c = _local_anchor(shape, side)
    interior_h = fr1 - fr0 - 1
    interior_w = fc1 - fc0 - 1

    if side in {"top", "bottom"}:
        if h > interior_h or w > interior_w:
            return []
        lo = fc0 + 1 + anchor_c
        hi = fc1 - 1 - ((w - 1) - anchor_c)
        row = fr0 - 1 if side == "top" else fr1 + 1
        return [(row, c) for c in range(lo, hi + 1)]

    if w > interior_w or h > interior_h:
        return []
    lo = fr0 + 1 + anchor_r
    hi = fr1 - 1 - ((h - 1) - anchor_r)
    col = fc0 - 1 if side == "left" else fc1 + 1
    return [(r, col) for r in range(lo, hi + 1)]


def _place_shape(shape: np.ndarray, side: str, anchor: tuple[int, int]) -> np.ndarray:
    anchor_r, anchor_c = anchor
    local_r, local_c = _local_anchor(shape, side)
    offset = np.asarray([anchor_r - local_r, anchor_c - local_c], dtype=int)
    return shape + offset


def _valid_outside_coords(
    coords: np.ndarray,
    side: str,
    frame_bbox: tuple[int, int, int, int],
    outside_mask: np.ndarray,
) -> bool:
    h, w = outside_mask.shape
    if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= h):
        return False
    if np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= w):
        return False

    fr0, fc0, fr1, fc1 = frame_bbox
    inside_frame_box = (
        (coords[:, 0] >= fr0) & (coords[:, 0] <= fr1) &
        (coords[:, 1] >= fc0) & (coords[:, 1] <= fc1)
    )
    if np.any(inside_frame_box):
        return False
    if _touches_mask(outside_mask, coords, DIR8):
        return False

    anchors = _anchor_candidates(coords, frame_bbox)
    return len(anchors) == 1 and anchors[0][1] == side


def _valid_inside_coords(
    coords: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    inside_mask: np.ndarray,
) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    if not (
        np.all(coords[:, 0] > fr0) and np.all(coords[:, 0] < fr1) and
        np.all(coords[:, 1] > fc0) and np.all(coords[:, 1] < fc1)
    ):
        return False
    return not _touches_mask(inside_mask, coords, DIR4)


def _place_fragment(
    rng: np.random.Generator,
    side: str,
    frame_bbox: tuple[int, int, int, int],
    outside_mask: np.ndarray,
    inside_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    shapes = SIDE_SHAPES[side].copy()
    rng.shuffle(shapes)
    for shape in shapes:
        anchors = _candidate_anchors(shape, side, frame_bbox)
        if not anchors:
            continue
        rng.shuffle(anchors)
        for anchor in anchors:
            outside_coords = _place_shape(shape, side, anchor)
            if not _valid_outside_coords(outside_coords, side, frame_bbox, outside_mask):
                continue
            inside_coords = _project_fragment(outside_coords, side, frame_bbox)
            if not _valid_inside_coords(inside_coords, frame_bbox, inside_mask):
                continue
            return outside_coords, inside_coords
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    frame_bbox = _sample_frame(rng, h, w)

    n_fragments = int(rng.integers(N_FRAGMENTS_MIN, N_FRAGMENTS_MAX + 1))
    colors = [int(v) for v in rng.choice(PALETTE, size=n_fragments, replace=False)]
    sides = _sample_sides(rng, n_fragments)

    outside_mask = np.zeros((h, w), dtype=bool)
    inside_mask = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, np.ndarray, np.ndarray]] = []

    order = list(range(n_fragments))
    rng.shuffle(order)
    for idx in order:
        placed = None
        for _ in range(FRAGMENT_TRIES):
            placed = _place_fragment(rng, sides[idx], frame_bbox, outside_mask, inside_mask)
            if placed is not None:
                break
        if placed is None:
            return None
        outside_coords, inside_coords = placed
        outside_mask[outside_coords[:, 0], outside_coords[:, 1]] = True
        inside_mask[inside_coords[:, 0], inside_coords[:, 1]] = True
        placements.append((colors[idx], outside_coords, inside_coords))

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    _draw_frame(inp, frame_bbox)
    _draw_frame(out, frame_bbox)
    for color, outside_coords, inside_coords in placements:
        inp[outside_coords[:, 0], outside_coords[:, 1]] = color
        out[inside_coords[:, 0], inside_coords[:, 1]] = color

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
