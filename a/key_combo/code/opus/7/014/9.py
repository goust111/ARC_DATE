from __future__ import annotations

import random
import numpy as np


BACKGROUND = 0
FRAME_COLOR = 2
CONTROL_COLOR = 4
MAIN_COLORS = (1, 3, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(np.asarray(sprite, dtype=int), np.ones((factor, factor), dtype=int))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in _components_of_color(grid, FRAME_COLOR):
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 5:
            continue
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros(region.shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(region[border] == FRAME_COLOR):
            continue
        if np.any(region[~border] == FRAME_COLOR):
            continue
        if int(np.sum(region == FRAME_COLOR)) != 4 * side_h - 4:
            continue
        candidates.append(box)
    assert len(candidates) == 1, "expected exactly one hollow red square frame"
    return candidates[0]


def _find_pattern(grid: np.ndarray, frame_box: tuple[int, int, int, int], bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    mask = (grid != bg) & (grid != FRAME_COLOR)
    pattern_box = bbox(mask)
    assert pattern_box is not None, "expected a 3x3 pattern outside the frame"
    assert not _boxes_intersect(pattern_box, frame_box), "pattern must lie outside the frame"
    r0, c0, r1, c1 = pattern_box
    assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3, "pattern bbox must be 3x3"
    patch = crop_bbox(grid, pattern_box)
    assert np.all(patch != bg), "pattern must fully occupy its 3x3 bbox"
    colors = set(map(int, np.unique(patch)))
    assert FRAME_COLOR not in colors
    assert CONTROL_COLOR in colors
    assert len(colors) <= 2, "pattern must use one main color plus yellow control color"
    controls = np.argwhere(patch == CONTROL_COLOR)
    assert 1 <= len(controls) <= 2, "expected one or two control pixels"
    for pr, pc in controls:
        assert (int(pr), int(pc)) not in {(0, 0), (0, 2), (2, 0), (2, 2)}, "control pixels must not sit on corners"
    return patch, pattern_box


def _scaled_with_halo(pattern: np.ndarray, factor: int) -> np.ndarray:
    scaled = scale_sprite(pattern, factor)
    out = scaled.copy()
    side = int(out.shape[0])
    for pr, pc in np.argwhere(pattern == CONTROL_COLOR):
        r0 = int(pr) * factor
        c0 = int(pc) * factor
        r1 = r0 + factor - 1
        c1 = c0 + factor - 1
        er0 = max(0, r0 - 1)
        ec0 = max(0, c0 - 1)
        er1 = min(side - 1, r1 + 1)
        ec1 = min(side - 1, c1 + 1)
        out[er0 : er1 + 1, ec0 : ec1 + 1] = CONTROL_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Upscale the 3x3 pattern to exactly fill the red frame interior, then add a one-cell halo around each scaled yellow control block.

    Input structure:
    - The grid has a black background, exactly one red hollow square frame, and one separate 3x3 pattern outside the frame.
    - The frame has border width one and an interior side length that is an integer multiple of three.
    - The 3x3 pattern is fully filled by one main color plus one or two yellow control pixels.

    Rule:
    - Find the red frame and compute its interior side length S, then infer the integer scale factor F = S / 3.
    - Read the external 3x3 pattern and enlarge every pixel into an F x F same-color block to form an S x S image.
    - For each yellow control pixel, take its enlarged F x F block and expand that block's bounding box by exactly one cell in the four cardinal directions, clipped to the frame interior, repainting those cells yellow.
    - Return only the cropped frame together with its transformed interior.

    Key insight:
    - The yellow effect is applied after scaling, so the control region's base size depends on the frame while the extra halo thickness always stays exactly one pixel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    frame_box = _find_frame(g)
    pattern, _pattern_box = _find_pattern(g, frame_box, bg)

    side = frame_box[2] - frame_box[0] + 1
    inner = side - 2
    assert inner % 3 == 0, "frame interior must be divisible by pattern width"
    factor = inner // 3
    assert factor >= 1

    out = crop_bbox(g, frame_box)
    interior = _scaled_with_halo(pattern, factor)
    out[1:-1, 1:-1] = interior
    return out


CONTROL_SETS = (
    ((1, 1),),
    ((0, 1),),
    ((1, 0),),
    ((1, 2),),
    ((2, 1),),
    ((1, 1), (0, 1)),
    ((1, 1), (1, 0)),
    ((1, 1), (1, 2)),
    ((1, 1), (2, 1)),
    ((0, 1), (1, 0)),
    ((0, 1), (1, 2)),
    ((1, 0), (2, 1)),
    ((1, 2), (2, 1)),
)


def _render_expected(pattern: np.ndarray, factor: int) -> np.ndarray:
    side = 3 * factor + 2
    out = np.zeros((side, side), dtype=int)
    out[0, :] = FRAME_COLOR
    out[-1, :] = FRAME_COLOR
    out[:, 0] = FRAME_COLOR
    out[:, -1] = FRAME_COLOR
    out[1:-1, 1:-1] = _scaled_with_halo(pattern, factor)
    return out


def _render_no_halo(pattern: np.ndarray, factor: int) -> np.ndarray:
    side = 3 * factor + 2
    out = np.zeros((side, side), dtype=int)
    out[0, :] = FRAME_COLOR
    out[-1, :] = FRAME_COLOR
    out[:, 0] = FRAME_COLOR
    out[:, -1] = FRAME_COLOR
    out[1:-1, 1:-1] = scale_sprite(pattern, factor)
    return out


def _render_prescaled_expansion(pattern: np.ndarray, factor: int) -> np.ndarray:
    small = pattern.copy()
    for pr, pc in np.argwhere(pattern == CONTROL_COLOR):
        for nr in range(max(0, int(pr) - 1), min(2, int(pr) + 1) + 1):
            for nc in range(max(0, int(pc) - 1), min(2, int(pc) + 1) + 1):
                small[nr, nc] = CONTROL_COLOR
    side = 3 * factor + 2
    out = np.zeros((side, side), dtype=int)
    out[0, :] = FRAME_COLOR
    out[-1, :] = FRAME_COLOR
    out[:, 0] = FRAME_COLOR
    out[:, -1] = FRAME_COLOR
    out[1:-1, 1:-1] = scale_sprite(small, factor)
    return out


def _render_factor_halo(pattern: np.ndarray, factor: int) -> np.ndarray:
    scaled = scale_sprite(pattern, factor)
    out = scaled.copy()
    side = int(out.shape[0])
    for pr, pc in np.argwhere(pattern == CONTROL_COLOR):
        r0 = int(pr) * factor
        c0 = int(pc) * factor
        r1 = r0 + factor - 1
        c1 = c0 + factor - 1
        er0 = max(0, r0 - factor)
        ec0 = max(0, c0 - factor)
        er1 = min(side - 1, r1 + factor)
        ec1 = min(side - 1, c1 + factor)
        out[er0 : er1 + 1, ec0 : ec1 + 1] = CONTROL_COLOR
    wrapped = np.zeros((side + 2, side + 2), dtype=int)
    wrapped[0, :] = FRAME_COLOR
    wrapped[-1, :] = FRAME_COLOR
    wrapped[:, 0] = FRAME_COLOR
    wrapped[:, -1] = FRAME_COLOR
    wrapped[1:-1, 1:-1] = out
    return wrapped


def _sample_pattern() -> np.ndarray:
    main_color = random.choice(MAIN_COLORS)
    pattern = np.full((3, 3), main_color, dtype=int)
    if random.random() < 0.55:
        controls = random.choice(CONTROL_SETS[5:])
    else:
        controls = random.choice(CONTROL_SETS[:5])
    for r, c in controls:
        pattern[r, c] = CONTROL_COLOR
    return pattern


def _place_pattern(grid: np.ndarray, frame_box: tuple[int, int, int, int], pattern: np.ndarray) -> bool:
    h, w = grid.shape
    blocked = _expand_box(frame_box, 1, grid.shape)
    candidates: list[tuple[int, int]] = []
    for r in range(h - 2):
        for c in range(w - 2):
            box = (r, c, r + 2, c + 2)
            if _boxes_intersect(box, blocked):
                continue
            candidates.append((r, c))
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    grid[r0 : r0 + 3, c0 : c0 + 3] = pattern
    return True


def _semantic_checks(inp: np.ndarray, pattern: np.ndarray, factor: int) -> bool:
    expected = _render_expected(pattern, factor)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(expected, _render_no_halo(pattern, factor)):
        return False
    if np.array_equal(expected, _render_prescaled_expansion(pattern, factor)):
        return False
    if factor > 1 and np.array_equal(expected, _render_factor_halo(pattern, factor)):
        return False

    inner = expected[1:-1, 1:-1]
    if not np.any(inner == CONTROL_COLOR):
        return False
    if not np.any((inner != CONTROL_COLOR) & (inner != BACKGROUND)):
        return False
    if int(np.sum(inner == CONTROL_COLOR)) >= inner.size:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    factor = random.randint(2, 4)
    outer = 3 * factor + 2
    min_h = max(outer + 5, 14)
    min_w = max(outer + 5, 14)
    h = random.randint(min_h, 26)
    w = random.randint(min_w, 26)

    frame_r = random.randint(1, h - outer - 1)
    frame_c = random.randint(1, w - outer - 1)
    frame_box = (frame_r, frame_c, frame_r + outer - 1, frame_c + outer - 1)

    grid = np.zeros((h, w), dtype=int)
    grid[frame_r, frame_c : frame_c + outer] = FRAME_COLOR
    grid[frame_r + outer - 1, frame_c : frame_c + outer] = FRAME_COLOR
    grid[frame_r : frame_r + outer, frame_c] = FRAME_COLOR
    grid[frame_r : frame_r + outer, frame_c + outer - 1] = FRAME_COLOR

    pattern = _sample_pattern()
    if not _place_pattern(grid, frame_box, pattern):
        return None
    if not _semantic_checks(grid, pattern, factor):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
