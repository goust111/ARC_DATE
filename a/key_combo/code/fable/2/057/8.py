from __future__ import annotations

import random
from collections import Counter, deque
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")

try:
    _arc_utils = SourceFileLoader("_arc_utils_fable_2_057_8", str(_ARC_UTILS_PATH)).load_module()
    Color = _arc_utils.Color
    get_bg = _arc_utils.get_bg
    crop_bbox = _arc_utils.crop_bbox
    find_components = _arc_utils.find_components
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = map(int, bbox)
        return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()

    def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors=None,
        min_area: int = 1,
        return_format: str = "dict",
    ):
        if not monochromatic:
            raise ValueError("fallback find_components only supports monochromatic=True")
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        if colors is None:
            colors_iter = sorted(set(map(int, g.ravel())) - {int(bg)})
        else:
            colors_iter = [int(c) for c in colors if int(c) != int(bg)]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if int(connectivity) == 8:
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        out = []
        for color in colors_iter:
            mask = g == color
            seen = np.zeros_like(mask, dtype=bool)
            for r, c in np.argwhere(mask):
                rr, cc = int(r), int(c)
                if seen[rr, cc]:
                    continue
                q = deque([(rr, cc)])
                seen[rr, cc] = True
                coords = []
                while q:
                    r0, c0 = q.popleft()
                    coords.append((r0, c0))
                    for dr, dc in dirs:
                        r1, c1 = r0 + dr, c0 + dc
                        if 0 <= r1 < g.shape[0] and 0 <= c1 < g.shape[1] and mask[r1, c1] and not seen[r1, c1]:
                            seen[r1, c1] = True
                            q.append((r1, c1))
                if len(coords) < int(min_area):
                    continue
                arr = np.array(coords, dtype=int)
                if str(return_format).lower() == "dict":
                    out.append(
                        {
                            "color": int(color),
                            "coords": arr,
                            "bbox": _bbox_coords(arr),
                            "area": int(arr.shape[0]),
                        }
                    )
                else:
                    obj = np.full_like(g, int(bg))
                    obj[arr[:, 0], arr[:, 1]] = int(color)
                    out.append(obj)
        return out


CORNER_TO_ROTATE_TO_TL = {"tl": 0, "tr": 1, "br": 2, "bl": 3}
CORNER_FROM_TL_ROTATION = {"tl": 0, "tr": 3, "br": 2, "bl": 1}
CORNER_COORDS = {
    "tl": (0, 0),
    "tr": (0, -1),
    "br": (-1, -1),
    "bl": (-1, 0),
}
ALL_CORNERS = ("tl", "tr", "br", "bl")

GRID_MARGIN = 1
BLOCK_GAP = 2
GEN_RETRIES = 400
MASK_RETRIES = 200
PLACEMENT_RETRIES = 250


def _rotations(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k) for k in range(4)]


def _is_rotation_symmetric(mask: np.ndarray) -> bool:
    return any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _same_under_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    return any(np.array_equal(a, np.rot90(b, k)) for k in range(4))


def _detect_notch(block: np.ndarray, bg: int, fg: int) -> str:
    h, w = block.shape
    assert h == w and h >= 5
    assert np.all((block == bg) | (block == fg))

    missing = [name for name, (r, c) in CORNER_COORDS.items() if int(block[r, c]) == bg]
    assert len(missing) == 1

    border = np.zeros_like(block, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    mr, mc = CORNER_COORDS[missing[0]]
    border[mr, mc] = False
    assert np.all(block[border] == fg)
    return missing[0]


def _extract_normalized_masks(grid: np.ndarray) -> tuple[int, list[np.ndarray], list[np.ndarray], list[str]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(components) >= 3

    colors = {int(comp["color"]) for comp in components}
    assert len(colors) == 1
    fg = next(iter(colors))

    components = sorted(components, key=lambda comp: tuple(map(int, comp["bbox"])))
    normalized = []
    raw = []
    corners = []
    side_lengths = set()
    for comp in components:
        block = crop_bbox(g, comp["bbox"])
        side_lengths.add(int(block.shape[0]))
        assert block.shape[0] == block.shape[1]
        corner = _detect_notch(block, bg, fg)
        norm = np.rot90(block, CORNER_TO_ROTATE_TO_TL[corner])
        inner = norm[1:-1, 1:-1]
        assert np.any(inner == fg)
        assert np.any(inner == bg)
        normalized.append(inner == fg)
        raw.append(block[1:-1, 1:-1] == fg)
        corners.append(corner)

    assert len(side_lengths) == 1
    shapes = {mask.shape for mask in normalized}
    assert len(shapes) == 1
    return int(fg), normalized, raw, corners


def _parity_overlay(masks: list[np.ndarray]) -> np.ndarray:
    return np.logical_xor.reduce(np.stack(masks, axis=0), axis=0)


def _build_block(inner_mask: np.ndarray, fg: int) -> np.ndarray:
    inner = np.asarray(inner_mask, dtype=bool)
    side = inner.shape[0] + 2
    block = np.full((side, side), Color.BLACK, dtype=int)
    block[0, :] = fg
    block[-1, :] = fg
    block[:, 0] = fg
    block[:, -1] = fg
    block[0, 0] = Color.BLACK
    block[1:-1, 1:-1] = np.where(inner, fg, Color.BLACK)
    return block


def _neighbors4(r: int, c: int, side: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < side and 0 <= cc < side:
            yield rr, cc


def _random_connected_mask(side: int) -> np.ndarray:
    total = side * side
    area_lo = max(3, total // 3)
    area_hi = min(total - 2, (2 * total) // 3 + 2)
    if area_hi < area_lo:
        area_hi = total - 2
    target_area = random.randint(area_lo, area_hi)

    perimeter = [(0, c) for c in range(side)] + [(side - 1, c) for c in range(side)]
    perimeter += [(r, 0) for r in range(1, side - 1)] + [(r, side - 1) for r in range(1, side - 1)]

    mask = np.zeros((side, side), dtype=bool)
    start_r, start_c = random.choice(perimeter)
    mask[start_r, start_c] = True

    while int(mask.sum()) < target_area:
        frontier = []
        seen = set()
        for r, c in np.argwhere(mask):
            rr, cc = int(r), int(c)
            for nr, nc in _neighbors4(rr, cc, side):
                if mask[nr, nc] or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                frontier.append((nr, nc))
        if not frontier:
            break
        add_r, add_c = random.choice(frontier)
        mask[add_r, add_c] = True

    return mask


def _sample_distinct_masks(inner_side: int, count: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    for _ in range(MASK_RETRIES):
        if len(masks) == count:
            break
        mask = _random_connected_mask(inner_side)
        if _is_rotation_symmetric(mask):
            continue
        if any(_same_under_rotation(mask, other) for other in masks):
            continue
        masks.append(mask)
    if len(masks) != count:
        return None
    return masks


def _sample_corners(count: int) -> list[str]:
    while True:
        distinct = random.randint(2, min(4, count))
        corners = list(random.sample(ALL_CORNERS, distinct))
        while len(corners) < count:
            corners.append(random.choice(corners))
        random.shuffle(corners)
        if len(set(corners)) >= 2 and any(corner != "tl" for corner in corners):
            return corners


def _boxes_separated(r: int, c: int, side: int, placed: list[tuple[int, int]], gap: int) -> bool:
    for pr, pc in placed:
        if not (r + side + gap <= pr or pr + side + gap <= r or c + side + gap <= pc or pc + side + gap <= c):
            return False
    return True


def _place_blocks(canvas_shape: tuple[int, int], side: int, count: int) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    positions: list[tuple[int, int]] = []
    for _ in range(PLACEMENT_RETRIES):
        positions.clear()
        ok = True
        order = list(range(count))
        random.shuffle(order)
        for _index in order:
            placed = False
            for _ in range(PLACEMENT_RETRIES):
                r = random.randint(GRID_MARGIN, h - side - GRID_MARGIN)
                c = random.randint(GRID_MARGIN, w - side - GRID_MARGIN)
                if _boxes_separated(r, c, side, positions, BLOCK_GAP):
                    positions.append((r, c))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and len(positions) == count:
            random.shuffle(positions)
            return positions.copy()
    return None


def _semantic_output(masks: list[np.ndarray]) -> np.ndarray:
    return _parity_overlay(masks)


def _passes_semantic_checks(masks: list[np.ndarray], raw_masks: list[np.ndarray], corners: list[str]) -> bool:
    if len(set(corners)) < 2 or all(corner == "tl" for corner in corners):
        return False
    if any(_is_rotation_symmetric(mask) for mask in masks):
        return False
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if _same_under_rotation(masks[i], masks[j]):
                return False

    stack = np.stack(masks, axis=0)
    counts = stack.sum(axis=0)
    out = counts % 2 == 1
    raw_out = _parity_overlay(raw_masks)
    union = np.any(stack, axis=0)
    intersection = np.all(stack, axis=0)

    area = int(out.sum())
    if area < 2 or area > out.size - 2:
        return False
    if np.array_equal(out, raw_out) or int(np.sum(out != raw_out)) < 2:
        return False
    if np.array_equal(out, union) or np.array_equal(out, intersection):
        return False
    if any(np.array_equal(out, mask) for mask in masks):
        return False
    if any(np.array_equal(out, np.rot90(out, k)) for k in (1, 2, 3)):
        return False
    if not np.any(counts == 2):
        return False
    if not np.any((counts >= 3) & (counts % 2 == 1)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    block_count = random.choice([3, 3, 4, 5])
    side = random.randint(5, 7)
    inner_side = side - 2
    fg = random.choice([color for color in Color.NOT_BLACK if color != Color.BLACK])

    masks = _sample_distinct_masks(inner_side, block_count)
    if masks is None:
        return None

    corners = _sample_corners(block_count)
    raw_masks = [np.rot90(mask, CORNER_FROM_TL_ROTATION[corner]) for mask, corner in zip(masks, corners)]
    if not _passes_semantic_checks(masks, raw_masks, corners):
        return None

    cols = random.choice([c for c in (2, 3) if c <= block_count])
    rows = (block_count + cols - 1) // cols
    min_h = rows * side + (rows - 1) * BLOCK_GAP + 2 * GRID_MARGIN
    min_w = cols * side + (cols - 1) * BLOCK_GAP + 2 * GRID_MARGIN
    max_h = min(30, min_h + 6)
    max_w = min(30, min_w + 6)
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    positions = _place_blocks((h, w), side, block_count)
    if positions is None:
        return None

    grid = np.full((h, w), Color.BLACK, dtype=int)
    for mask, corner, (r, c) in zip(masks, corners, positions):
        block = _build_block(mask, fg)
        observed = np.rot90(block, CORNER_FROM_TL_ROTATION[corner])
        if np.any(grid[r : r + side, c : c + side] != Color.BLACK):
            return None
        grid[r : r + side, c : c + side] = observed

    expected_mask = _semantic_output(masks)
    expected = np.where(expected_mask, fg, Color.BLACK).astype(int)

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    solved_fg, solved_masks, solved_raw, solved_corners = _extract_normalized_masks(grid)
    if solved_fg != fg:
        return None
    if not _passes_semantic_checks(solved_masks, solved_raw, solved_corners):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each notched square to a common pose and keep the interior cells that appear an odd number of times.

    Input structure:
    - The background is black.
    - Several disconnected same-color square blocks are scattered on the canvas.
    - Each block has a solid one-pixel border with exactly one missing corner pixel, and a mixed foreground/background interior.

    Rule:
    - Identify every square block and use its missing border corner as the orientation marker.
    - Rotate each block so the missing corner moves to the top-left.
    - Remove the border, overlay all normalized interiors cell-by-cell, and keep a cell in the output iff it is foreground in an odd number of blocks.

    Key insight:
    - The canvas positions are irrelevant noise; the notch fixes the pose, and the final pattern is a parity overlay rather than a union or an intersection.
    """
    fg, normalized_masks, _raw_masks, _corners = _extract_normalized_masks(np.asarray(grid, dtype=int))
    out_mask = _parity_overlay(normalized_masks)
    return np.where(out_mask, fg, Color.BLACK).astype(int)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
