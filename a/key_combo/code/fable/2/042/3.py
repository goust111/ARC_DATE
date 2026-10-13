import importlib.util
import math
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("arc_utils_shared", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

Color = _ARC_UTILS.Color
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BACKGROUND = int(Color.BLACK)
HEAD_BORDER = int(Color.GREY)
OTHER_BORDER_COLORS = [int(c) for c in Color.NOT_BLACK if int(c) != HEAD_BORDER]

MIN_BOXES = 4
MAX_BOXES = 8
MAX_GENERATION_ATTEMPTS = 500


def _make_box(border_color: int, center_color: int) -> np.ndarray:
    patch = np.full((3, 3), int(border_color), dtype=int)
    patch[1, 1] = int(center_color)
    return patch


def _component_to_box(grid: np.ndarray, component: dict) -> dict:
    box = tuple(map(int, component["bbox"]))
    assert (bbox_h(box), bbox_w(box)) == (3, 3), "every object must be a 3x3 box"

    patch = crop_bbox(grid, box)
    border_mask = np.ones((3, 3), dtype=bool)
    border_mask[1, 1] = False

    border_colors = set(map(int, patch[border_mask].reshape(-1)))
    assert len(border_colors) == 1, "each box border must be monochrome"
    border_color = border_colors.pop()
    assert border_color != BACKGROUND, "box border cannot be black"

    center_color = int(patch[1, 1])
    assert center_color != border_color, "center color must differ from border color"

    expected_area = 8 if center_color == BACKGROUND else 9
    assert int(component["area"]) == expected_area, "box pixels must match the 3x3 bordered-sprite template"

    return {
        "bbox": box,
        "border": border_color,
        "center": center_color,
    }


def _extract_boxes(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, f"grid exceeds ARC size limit: {g.shape}"

    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected black to be the background"

    components = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    assert components, "expected at least one box"

    boxes = [_component_to_box(g, comp) for comp in components]
    border_colors = [int(box["border"]) for box in boxes]
    center_colors = [int(box["center"]) for box in boxes]
    assert len(border_colors) == len(set(border_colors)), "border colors must be unique"

    head_boxes = [box for box in boxes if int(box["border"]) == HEAD_BORDER]
    tail_boxes = [box for box in boxes if int(box["center"]) == BACKGROUND]
    assert len(head_boxes) == 1, "expected exactly one grey head box"
    assert len(tail_boxes) == 1, "expected exactly one black-centered tail box"

    targets = [color for color in center_colors if color != BACKGROUND]
    assert len(targets) == len(set(targets)), "every non-tail pointer target must be unique"
    assert set(targets) == set(border_colors) - {HEAD_BORDER}, "every non-head border color must be pointed to exactly once"

    by_border = {int(box["border"]): box for box in boxes}
    chain = []
    seen = set()
    current = head_boxes[0]
    while True:
        border = int(current["border"])
        assert border not in seen, "pointer relation must be acyclic"
        chain.append(current)
        seen.add(border)

        target = int(current["center"])
        if target == BACKGROUND:
            break
        assert target in by_border, "every non-black center must match a box border"
        current = by_border[target]

    assert len(chain) == len(boxes), "the grey-headed pointer chain must cover all boxes"
    return boxes, chain


def _render_concentric_rings(chain: list[dict]) -> np.ndarray:
    colors = [int(box["border"]) for box in chain]
    size = 2 * len(colors) - 1
    out = np.full((size, size), BACKGROUND, dtype=int)
    for layer, color in enumerate(colors):
        lo = int(layer)
        hi = int(size - 1 - layer)
        out[lo, lo : hi + 1] = color
        out[hi, lo : hi + 1] = color
        out[lo : hi + 1, lo] = color
        out[lo : hi + 1, hi] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the grey-headed color-pointer chain across 3x3 boxes and render its order as concentric square rings.

    Input structure:
    - The background is black, and the foreground consists of separated 3x3 boxes.
    - In each box, the 8 border pixels share one color and the center pixel has a different color.
    - Border colors are all distinct, exactly one border is grey, and exactly one center is black.

    Rule:
    - Treat the grey-bordered box as the chain head.
    - Repeatedly jump from the current box to the unique box whose border color equals the current center color, stopping when the center is black.
    - Output a (2n-1)x(2n-1) grid, where n is the number of boxes: paint the outer ring with the head border color, each inner ring with the next box's border color in chain order, and the center cell with the tail border color.

    Key insight:
    - The center pixel is a visible pointer to the next object, so spatial placement is irrelevant; only the color-equality chain determines the ring nesting order.
    """
    _boxes, chain = _extract_boxes(grid)
    return _render_concentric_rings(chain)


def _boxes_too_close(pos_a: tuple[int, int], pos_b: tuple[int, int]) -> bool:
    ar0, ac0 = map(int, pos_a)
    br0, bc0 = map(int, pos_b)
    ar1, ac1 = ar0 + 2, ac0 + 2
    br1, bc1 = br0 + 2, bc0 + 2
    return not (ar1 < br0 - 1 or br1 < ar0 - 1 or ac1 < bc0 - 1 or bc1 < ac0 - 1)


def _sample_positions(height: int, width: int, n_boxes: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, height - 3) for c in range(1, width - 3)]
    random.shuffle(candidates)
    positions: list[tuple[int, int]] = []
    for pos in candidates:
        if any(_boxes_too_close(pos, existing) for existing in positions):
            continue
        positions.append(pos)
        if len(positions) == n_boxes:
            return positions
    return None


def _row_major_order(boxes: list[dict]) -> list[int]:
    return [int(box["border"]) for box in sorted(boxes, key=lambda box: (box["bbox"][0], box["bbox"][1]))]


def _col_major_order(boxes: list[dict]) -> list[int]:
    return [int(box["border"]) for box in sorted(boxes, key=lambda box: (box["bbox"][1], box["bbox"][0]))]


def _semantic_accepts(grid: np.ndarray) -> bool:
    try:
        boxes, chain = _extract_boxes(grid)
    except AssertionError:
        return False

    chain_colors = [int(box["border"]) for box in chain]
    row_major = _row_major_order(boxes)
    col_major = _col_major_order(boxes)
    if chain_colors in (row_major, row_major[::-1], col_major, col_major[::-1]):
        return False

    non_head_colors = chain_colors[1:]
    if non_head_colors == sorted(non_head_colors) or non_head_colors == sorted(non_head_colors, reverse=True):
        return False

    rows = [int(box["bbox"][0]) for box in chain]
    cols = [int(box["bbox"][1]) for box in chain]
    if rows == sorted(rows) or rows == sorted(rows, reverse=True):
        return False
    if cols == sorted(cols) or cols == sorted(cols, reverse=True):
        return False

    out = _render_concentric_rings(chain)
    if max(out.shape) > 30:
        return False
    return np.array_equal(main(grid), out)


def _gen_attempt() -> np.ndarray | None:
    n_boxes = random.randint(MIN_BOXES, MAX_BOXES)
    chain_tail = random.sample(OTHER_BORDER_COLORS, n_boxes - 1)
    if chain_tail == sorted(chain_tail) or chain_tail == sorted(chain_tail, reverse=True):
        return None

    chain_colors = [HEAD_BORDER] + chain_tail
    centers = chain_colors[1:] + [BACKGROUND]
    specs = [{"border": border, "center": center} for border, center in zip(chain_colors, centers)]

    min_side = max(12, 5 * math.ceil(math.sqrt(n_boxes)) - 1)
    max_side = min(30, min_side + 8)
    height = random.randint(min_side, max_side)
    width = random.randint(min_side, max_side)

    positions = _sample_positions(height, width, n_boxes)
    if positions is None:
        return None
    random.shuffle(positions)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for spec, (r0, c0) in zip(specs, positions):
        grid[r0 : r0 + 3, c0 : c0 + 3] = _make_box(spec["border"], spec["center"])

    return grid if _semantic_accepts(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain ring input")
