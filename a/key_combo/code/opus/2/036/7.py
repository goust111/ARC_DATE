from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
ORANGE = 7
MAROON = 9
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

FRAME_INTERIORS = tuple(
    (h, w)
    for h in range(1, 5)
    for w in range(1, 5)
    if 2 <= h * w <= 8
)
GRID_SIDE_RANGE = (15, 22)
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int)).astype(int)


def _frame_patch(inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = BLUE
    patch[-1, :] = BLUE
    patch[:, 0] = BLUE
    patch[:, -1] = BLUE
    return patch


def _template_patch(marker_corner: tuple[int, int]) -> np.ndarray:
    mr, mc = marker_corner
    r0 = mr
    c0 = mc
    patch = np.zeros((3, 3), dtype=int)
    patch[r0 : r0 + 2, c0 : c0 + 2] = ORANGE
    patch[r0 + mr, c0 + mc] = MAROON
    return patch


def _physical_corner_origin(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    corner: str,
) -> tuple[int, int]:
    h, w = grid_shape
    ph, pw = patch_shape
    max_r_slack = min(2, h - ph)
    max_c_slack = min(2, w - pw)
    if corner == "tl":
        return random.randint(0, max_r_slack), random.randint(0, max_c_slack)
    if corner == "tr":
        return random.randint(0, max_r_slack), w - pw - random.randint(0, max_c_slack)
    if corner == "bl":
        return h - ph - random.randint(0, max_r_slack), random.randint(0, max_c_slack)
    if corner == "br":
        return h - ph - random.randint(0, max_r_slack), w - pw - random.randint(0, max_c_slack)
    raise ValueError(f"unknown corner {corner}")


def _can_place(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    reserved[r0 : r0 + h, c0 : c0 + w] = True


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for coords in _components_mask(grid == BLUE):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        assert h >= 3 and w >= 3
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(patch[border] == BLUE)
        assert np.all(patch[~border] == BLACK)
        assert int(np.count_nonzero(patch == BLUE)) == int(border.sum())
        frames.append((r0, c0, r1, c1))
    return frames


def _extract_template_block(grid: np.ndarray) -> np.ndarray:
    mask = (grid == ORANGE) | (grid == MAROON)
    coords = np.argwhere(mask)
    assert coords.shape[0] == 4
    r0, c0, r1, c1 = _bbox_coords(coords)
    assert r1 - r0 + 1 == 2 and c1 - c0 + 1 == 2
    block = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert int(np.count_nonzero(block == MAROON)) == 1
    assert int(np.count_nonzero(block == ORANGE)) == 3
    assert np.all((block == ORANGE) | (block == MAROON))
    return block


def _build_output(scale: int, block: np.ndarray) -> np.ndarray:
    out = np.zeros((3 * scale, 3 * scale), dtype=int)
    scaled = _scale_sprite(block, scale)
    marker_r, marker_c = map(int, np.argwhere(block == MAROON)[0])
    r0 = 0 if marker_r == 0 else scale
    c0 = 0 if marker_c == 0 else scale
    out[r0 : r0 + 2 * scale, c0 : c0 + 2 * scale] = scaled
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Count hollow blue frames, then enlarge the marked 3x3 template by that count into the matching output corner.

    Input structure
    - The grid has black background, plus 2 to 4 separate one-cell-thick blue hollow rectangles.
    - A separate 3x3 template contains a 2x2 orange block with one maroon marker pixel at one corner; the remaining row and column are black.
    - The template sits away from the frames and no objects overlap or touch.

    Rule
    - Count the blue hollow rectangular frames to get the scale factor N.
    - Extract the template's colored 2x2 block and locate the maroon marker corner inside it.
    - Create a black (3N)x(3N) output and enlarge each template pixel into an NxN block.
    - Place the enlarged 2x2 block in the output corner opposite the expansion direction implied by the marker corner.

    Key insight
    - The frames do not appear in the output directly; they only encode the scaling factor, while the marker corner independently chooses which output corner receives the enlarged template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    scale = len(frames)
    assert scale in (2, 3, 4)

    block = _extract_template_block(g)
    return _build_output(scale, block)


def _attempt_input() -> np.ndarray | None:
    scale = random.choice((2, 3, 4))
    marker_corner = random.choice(((0, 0), (0, 1), (1, 0), (1, 1)))
    input_corner = random.choice(("tl", "tr", "bl", "br"))

    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    template = _template_patch(marker_corner)
    tr, tc = _physical_corner_origin(grid.shape, template.shape, input_corner)
    if not _can_place(reserved, tr, tc, 3, 3):
        return None
    grid[tr : tr + 3, tc : tc + 3] = template
    _reserve(reserved, tr, tc, 3, 3)

    frame_choices = [random.choice(FRAME_INTERIORS) for _ in range(scale)]
    frame_choices.sort(key=lambda dims: dims[0] * dims[1], reverse=True)
    for inner_h, inner_w in frame_choices:
        patch = _frame_patch(inner_h, inner_w)
        ph, pw = patch.shape
        spots = [
            (r0, c0)
            for r0 in range(height - ph + 1)
            for c0 in range(width - pw + 1)
            if _can_place(reserved, r0, c0, ph, pw)
        ]
        if not spots:
            return None
        r0, c0 = random.choice(spots)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        _reserve(reserved, r0, c0, ph, pw)

    expected = _build_output(scale, grid[tr + marker_corner[0] : tr + marker_corner[0] + 2, tc + marker_corner[1] : tc + marker_corner[1] + 2])
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
