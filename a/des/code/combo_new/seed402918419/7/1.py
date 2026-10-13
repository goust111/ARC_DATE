from __future__ import annotations

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDES = tuple(SIDE_TO_DIR)

SHAPE_LIBRARY = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
]

GEN_ATTEMPTS = 400
GEN_GRID_RANGE = (22, 31)
GEN_FRAME_COUNT_RANGE = (1, 4)
GEN_REPEAT_RANGE = (1, 5)
GEN_FRAME_SIDE_RANGE = (5, 9)
GEN_CHAIN_PADDING = 1


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack = [(r, c)]
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    r, c = gap
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _count_motif_components(interior: np.ndarray, bg: int, frame_color: int) -> int:
    total = 0
    for color in np.unique(interior):
        color = int(color)
        if color == bg or color == frame_color:
            continue
        total += len(_components_mask(interior == color))
    return total


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_mask(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            coord_set = {(int(r), int(c)) for r, c in coords}
            perimeter = _perimeter_cells(r0, c0, r1, c1)
            if not coord_set.issubset(perimeter):
                continue

            missing = list(perimeter - coord_set)
            if len(missing) != 1:
                continue

            gap = missing[0]
            side = _gap_side(gap, (r0, c0, r1, c1))
            if side is None or int(grid[gap]) != bg:
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            motif_mask = (interior != bg) & (interior != color)
            if not motif_mask.any():
                continue

            repeats = _count_motif_components(interior, bg, color)
            if repeats <= 0:
                continue

            motif_pixels = [
                (r0 + 1 + int(rr), c0 + 1 + int(cc), int(interior[rr, cc]))
                for rr, cc in np.argwhere(motif_mask)
            ]
            step = (r1 - r0 + 1) if side in ("top", "bottom") else (c1 - c0 + 1)
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": color,
                    "side": side,
                    "step": step,
                    "repeats": repeats,
                    "motif_pixels": motif_pixels,
                }
            )

    frames.sort(key=lambda item: item["bbox"])
    return frames


def _project_pattern(
    out: np.ndarray,
    motif_pixels: list[tuple[int, int, int]],
    side: str,
    step: int,
    repeats: int,
) -> None:
    h, w = out.shape
    dr, dc = SIDE_TO_DIR[side]
    for copy_idx in range(1, repeats + 1):
        shift = copy_idx * step
        for r, c, color in motif_pixels:
            rr = r + dr * shift
            cc = c + dc * shift
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each framed interior motif outward in the direction marked by the frame gap.

    Input structure:
    - A dominant black background contains several one-pixel-thick rectangular frames, each missing exactly one border pixel on one side.
    - Every frame is monochrome and encloses one or more same-colored interior blocks in another color.
    - The missing border pixel indicates one of the four cardinal projection directions.

    Rule:
    - For each frame, detect which side contains the single-pixel gap.
    - Count the number N of 4-connected interior blocks inside that frame.
    - Treat all non-frame colored cells inside the frame as one motif and copy that motif N times outside the frame along the gap direction.
    - Consecutive copies are separated by the full frame height for top/bottom gaps or the full frame width for left/right gaps, and the original frame stays unchanged.

    Key insight:
    - The gap chooses the projection axis, while the number of internal blocks chooses how many equally spaced copies to emit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        _project_pattern(
            out,
            frame["motif_pixels"],
            frame["side"],
            int(frame["step"]),
            int(frame["repeats"]),
        )

    return out


def _chain_bbox(
    r0: int,
    c0: int,
    height: int,
    width: int,
    side: str,
    repeats: int,
) -> tuple[int, int, int, int]:
    if side == "top":
        return (r0 - repeats * height, c0, r0 + height - 1, c0 + width - 1)
    if side == "bottom":
        return (r0, c0, r0 + (repeats + 1) * height - 1, c0 + width - 1)
    if side == "left":
        return (r0, c0 - repeats * width, r0 + height - 1, c0 + width - 1)
    return (r0, c0, r0 + height - 1, c0 + (repeats + 1) * width - 1)


def _bbox_is_free(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not bool(blocked[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _paint_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    color: int,
    side: str,
    gap_offset: int,
    bg: int,
) -> None:
    grid[r0, c0 : c0 + width] = color
    grid[r0 + height - 1, c0 : c0 + width] = color
    grid[r0 : r0 + height, c0] = color
    grid[r0 : r0 + height, c0 + width - 1] = color

    if side == "top":
        grid[r0, c0 + gap_offset] = bg
    elif side == "bottom":
        grid[r0 + height - 1, c0 + gap_offset] = bg
    elif side == "left":
        grid[r0 + gap_offset, c0] = bg
    else:
        grid[r0 + gap_offset, c0 + width - 1] = bg


def _place_motif(interior_h: int, interior_w: int, repeats: int, rng: np.random.Generator) -> np.ndarray | None:
    blocked = np.zeros((interior_h, interior_w), dtype=bool)
    motif = np.zeros((interior_h, interior_w), dtype=bool)

    max_area = 2 if repeats >= 3 else 3 if repeats == 2 else 4
    shape_ids = [
        idx
        for idx, shape in enumerate(SHAPE_LIBRARY)
        if shape.shape[0] <= interior_h and shape.shape[1] <= interior_w and int(shape.sum()) <= max_area
    ]
    if not shape_ids:
        return None

    for _ in range(repeats):
        placed = False
        for idx in rng.permutation(shape_ids):
            shape = SHAPE_LIBRARY[int(idx)]
            local = np.argwhere(shape)
            sh, sw = shape.shape
            positions: list[tuple[int, int]] = []
            for r0 in range(0, interior_h - sh + 1):
                for c0 in range(0, interior_w - sw + 1):
                    if any(blocked[r0 + int(rr), c0 + int(cc)] for rr, cc in local):
                        continue
                    positions.append((r0, c0))
            if not positions:
                continue

            r0, c0 = positions[int(rng.integers(0, len(positions)))]
            cells = [(r0 + int(rr), c0 + int(cc)) for rr, cc in local]
            for rr, cc in cells:
                motif[rr, cc] = True
            for rr, cc in cells:
                rr0 = max(0, rr - 1)
                cc0 = max(0, cc - 1)
                rr1 = min(interior_h, rr + 2)
                cc1 = min(interior_w, cc + 2)
                blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    if len(_components_mask(motif)) != repeats:
        return None
    return motif


def _dimension_choices(
    grid_h: int,
    grid_w: int,
    side: str,
    repeats: int,
) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for height in range(*GEN_FRAME_SIDE_RANGE):
        for width in range(*GEN_FRAME_SIDE_RANGE):
            if side in ("top", "bottom"):
                if (repeats + 1) * height <= grid_h and width <= grid_w:
                    dims.append((height, width))
            else:
                if height <= grid_h and (repeats + 1) * width <= grid_w:
                    dims.append((height, width))
    return dims


def _candidate_positions(
    blocked: np.ndarray,
    height: int,
    width: int,
    side: str,
    repeats: int,
) -> list[tuple[int, int]]:
    grid_h, grid_w = blocked.shape
    if side == "top":
        row_range = range(repeats * height, grid_h - height + 1)
        col_range = range(0, grid_w - width + 1)
    elif side == "bottom":
        row_range = range(0, grid_h - (repeats + 1) * height + 1)
        col_range = range(0, grid_w - width + 1)
    elif side == "left":
        row_range = range(0, grid_h - height + 1)
        col_range = range(repeats * width, grid_w - width + 1)
    else:
        row_range = range(0, grid_h - height + 1)
        col_range = range(0, grid_w - (repeats + 1) * width + 1)

    positions: list[tuple[int, int]] = []
    for r0 in row_range:
        for c0 in col_range:
            if _bbox_is_free(blocked, _chain_bbox(r0, c0, height, width, side, repeats)):
                positions.append((r0, c0))
    return positions


def _add_frame(
    inp: np.ndarray,
    expected: np.ndarray,
    blocked: np.ndarray,
    frame_color: int,
    motif_color: int,
    rng: np.random.Generator,
) -> bool:
    grid_h, grid_w = inp.shape
    for _ in range(200):
        repeats = int(rng.integers(*GEN_REPEAT_RANGE))
        side = SIDES[int(rng.integers(0, len(SIDES)))]
        dims = _dimension_choices(grid_h, grid_w, side, repeats)
        if not dims:
            continue

        height, width = dims[int(rng.integers(0, len(dims)))]
        motif = _place_motif(height - 2, width - 2, repeats, rng)
        if motif is None:
            continue

        positions = _candidate_positions(blocked, height, width, side, repeats)
        if not positions:
            continue

        r0, c0 = positions[int(rng.integers(0, len(positions)))]
        gap_span = width if side in ("top", "bottom") else height
        gap_offset = int(rng.integers(1, gap_span - 1))

        _paint_frame(inp, r0, c0, height, width, frame_color, side, gap_offset, 0)
        _paint_frame(expected, r0, c0, height, width, frame_color, side, gap_offset, 0)

        interior_in = inp[r0 + 1 : r0 + height - 1, c0 + 1 : c0 + width - 1]
        interior_out = expected[r0 + 1 : r0 + height - 1, c0 + 1 : c0 + width - 1]
        interior_in[motif] = motif_color
        interior_out[motif] = motif_color

        motif_pixels = [
            (r0 + 1 + int(rr), c0 + 1 + int(cc), motif_color)
            for rr, cc in np.argwhere(motif)
        ]
        step = height if side in ("top", "bottom") else width
        _project_pattern(expected, motif_pixels, side, step, repeats)

        _mark_blocked(blocked, _chain_bbox(r0, c0, height, width, side, repeats), GEN_CHAIN_PADDING)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    grid_h = int(rng.integers(*GEN_GRID_RANGE))
    grid_w = int(rng.integers(*GEN_GRID_RANGE))
    frame_count = int(rng.integers(*GEN_FRAME_COUNT_RANGE))

    inp = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    colors = [int(c) for c in rng.permutation(np.arange(1, 10))]
    for idx in range(frame_count):
        if 2 * idx + 1 >= len(colors):
            return None
        frame_color = colors[2 * idx]
        motif_color = colors[2 * idx + 1]
        if not _add_frame(inp, expected, blocked, frame_color, motif_color, rng):
            return None

    frames = _find_frames(inp, 0)
    if len(frames) != frame_count:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
