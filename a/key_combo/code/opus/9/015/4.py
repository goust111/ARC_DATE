from __future__ import annotations

import random
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
if str(_ARC_UTILS_PATH.parent) not in sys.path:
    sys.path.append(str(_ARC_UTILS_PATH.parent))

try:
    from arc_utils import Color, crop_bbox, find_components, get_bg, paste
except Exception:
    _arc_utils = SourceFileLoader("arc_utils_task_015_4", str(_ARC_UTILS_PATH)).load_module()
    Color = _arc_utils.Color
    crop_bbox = _arc_utils.crop_bbox
    find_components = _arc_utils.find_components
    get_bg = _arc_utils.get_bg
    paste = _arc_utils.paste


TEMPLATE_MASKS: tuple[np.ndarray, ...] = (
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 0], [1, 1, 0]], dtype=bool),
)

COLOR_POOL: tuple[int, ...] = (
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
)

MAX_GENERATION_ATTEMPTS = 400


def _rotation_steps(r: int, c: int, shape: tuple[int, int]) -> int:
    h, w = map(int, shape)
    mid_r = h // 2
    mid_c = w // 2
    top = int(r) <= mid_r
    left = int(c) <= mid_c
    if top and left:
        return 0
    if top and not left:
        return 1
    if not top and left:
        return 3
    return 2


def _extract_template_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    grey_components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
        return_format="dict",
    )
    candidates = []
    for comp in grey_components:
        if int(comp["area"]) <= 1:
            continue
        patch = crop_bbox(grid, comp["bbox"])
        if patch.shape == (3, 3):
            candidates.append(patch == Color.GREY)
    assert len(candidates) == 1, "expected exactly one 3x3 grey template component"
    mask = np.asarray(candidates[0], dtype=bool)
    assert len({tuple(map(tuple, np.rot90(mask, -k))) for k in range(4)}) == 4
    return mask


def _parse_markers(grid: np.ndarray, bg: int) -> tuple[list[int], list[tuple[int, int]]]:
    h, w = grid.shape
    cols = [int(c) for c in np.where(grid[0] == Color.GREY)[0] if 0 < int(c) < w - 1]
    rows = []
    for r in range(1, h - 1):
        color = int(grid[r, w - 1])
        if color not in {bg, Color.GREY}:
            rows.append((int(r), color))
    assert cols, "expected top-row grey column markers"
    assert rows, "expected right-column colored row markers"
    return sorted(cols), sorted(rows)


def _render_scene(
    grid: np.ndarray,
    template_mask: np.ndarray,
    col_markers: list[int],
    row_markers: list[tuple[int, int]],
    *,
    reverse_order: bool = False,
) -> np.ndarray:
    bg = int(get_bg(grid))
    out = np.array(grid, dtype=int, copy=True)

    row_iter = list(reversed(row_markers)) if reverse_order else list(row_markers)
    col_iter = list(reversed(col_markers)) if reverse_order else list(col_markers)
    for r, color in row_iter:
        for c in col_iter:
            rot = _rotation_steps(r, c, grid.shape)
            rotated = np.rot90(template_mask, -rot)
            patch = np.where(rotated, int(color), bg).astype(int)
            paste(out, patch, r - 1, c - 1, transparent=bg, overwrite=True, clip=False)

    original = grid != bg
    out[original] = grid[original]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place colored rotated copies of a bottom-left template at the Cartesian-product row/column intersections.

    Input structure:
    - The background is black.
    - The top row contains 3-4 grey marker pixels selecting columns, and the rightmost
      column contains 3-4 non-grey colored marker pixels selecting rows.
    - The lower-left corner contains one grey 3x3 template sprite whose occupied cells
      form an asymmetric 4-connected pattern with four distinct quarter-turn rotations.

    Rule:
    - Take the Cartesian product of the selected rows and selected columns.
    - At each intersection, place a 3x3 copy of the template centered on that cell.
    - Recolor each copy with the color of its row marker, and rotate it according to the
      intersection quadrant: upper-left = 0 degrees, upper-right = 90 clockwise,
      lower-left = 270 clockwise, lower-right = 180 degrees; points on the center row or
      center column count as upper / left.
    - Process intersections in row-major order, then restore all original markers and the
      original template so they remain visible.

    Key insight:
    - Three independent cues meet at every copy: top markers choose where by Cartesian
      product, right-edge marker colors choose what color, and the intersection quadrant
      chooses how the template is rotated.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    template_mask = _extract_template_mask(g, bg)
    col_markers, row_markers = _parse_markers(g, bg)
    return _render_scene(g, template_mask, col_markers, row_markers)


def _template_is_connected(mask: np.ndarray) -> bool:
    patch = np.where(mask, 1, 0).astype(int)
    comps = find_components(
        patch,
        bg=0,
        connectivity=4,
        monochromatic=True,
        colors=[1],
        return_format="dict",
    )
    return len(comps) == 1


def _sample_spread(pool: list[int], count: int, min_gap: int, require: set[int] | None = None) -> list[int] | None:
    require = set() if require is None else {int(v) for v in require}
    if len(require) > count or any(v not in pool for v in require):
        return None
    pool_set = set(pool)
    for _ in range(200):
        remaining = [v for v in pool if v not in require]
        picks = list(require)
        picks.extend(random.sample(remaining, count - len(picks)))
        picks.sort()
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _fill_with_spacing(
    pool: list[int],
    initial: set[int],
    count: int,
    relaxed_pairs: set[frozenset[int]],
) -> list[int] | None:
    chosen = set(int(v) for v in initial)
    if len(chosen) > count or not chosen <= set(pool):
        return None
    for _ in range(200):
        work = set(chosen)
        while len(work) < count:
            candidates = []
            for v in pool:
                if v in work:
                    continue
                ok = True
                for u in work:
                    pair = frozenset({int(u), int(v)})
                    gap = abs(int(u) - int(v))
                    if pair in relaxed_pairs:
                        if gap != 2:
                            ok = False
                            break
                    elif gap < 3:
                        ok = False
                        break
                if ok:
                    candidates.append(int(v))
            if not candidates:
                break
            work.add(random.choice(candidates))
        if len(work) != count:
            continue
        picks = sorted(work)
        if all((b - a >= 3) or (frozenset({a, b}) in relaxed_pairs and b - a == 2) for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _sample_rows(height: int, count: int, overlap_mode: str | None) -> list[int] | None:
    mid = height // 2
    valid = list(range(2, height - 4))
    upper = [r for r in valid if r < mid]
    lower = [r for r in valid if r > mid]
    if overlap_mode == "row":
        pair = {mid, mid + 2}
        if not pair <= set(valid):
            return None
        initial = set(pair)
        initial.add(random.choice(upper))
        picks = _fill_with_spacing(valid, initial, count, {frozenset(pair)})
        if picks is not None and any(r < mid for r in picks) and any(r > mid for r in picks):
            return picks
        return None

    want_center = random.random() < 0.45
    require = {random.choice(upper), random.choice(lower)}
    if want_center and mid in valid and count >= 4:
        require.add(mid)
    picks = _sample_spread(valid, count, 3, require=require)
    if picks is None:
        return None
    if any(r < mid for r in picks) and any(r > mid for r in picks):
        return picks
    return None


def _sample_cols(width: int, count: int, overlap_mode: str | None) -> list[int] | None:
    mid = width // 2
    valid = list(range(4, width - 2))
    left = [c for c in valid if c <= mid]
    right = [c for c in valid if c > mid]
    if overlap_mode == "col":
        pair = {mid, mid + 2}
        if not pair <= set(valid):
            return None
        initial = set(pair)
        left_extras = [c for c in left if c < mid - 1]
        if not left_extras:
            return None
        initial.add(random.choice(left_extras))
        picks = _fill_with_spacing(valid, initial, count, {frozenset(pair)})
        if picks is not None and any(c <= mid for c in picks) and any(c > mid for c in picks):
            return picks
        return None

    want_center = random.random() < 0.45
    require = {random.choice(left), random.choice(right)}
    if want_center and mid in valid and count >= 4:
        require.add(mid)
    picks = _sample_spread(valid, count, 3, require=require)
    if picks is None:
        return None
    if any(c <= mid for c in picks) and any(c > mid for c in picks):
        return picks
    return None


def _build_input(
    height: int,
    width: int,
    template_mask: np.ndarray,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
) -> np.ndarray:
    grid = np.full((height, width), Color.BLACK, dtype=int)
    grid[0, np.array(col_positions, dtype=int)] = Color.GREY
    grid[np.array(row_positions, dtype=int), width - 1] = np.array(row_colors, dtype=int)
    template_patch = np.where(template_mask, Color.GREY, Color.BLACK).astype(int)
    paste(grid, template_patch, height - 3, 0, transparent=Color.BLACK, clip=False)
    return grid


def _validate_case(
    grid: np.ndarray,
    template_mask: np.ndarray,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
    overlap_mode: str | None,
) -> bool:
    if not _template_is_connected(template_mask):
        return False
    if len({tuple(map(tuple, np.rot90(template_mask, -k))) for k in range(4)}) != 4:
        return False
    if len(set(row_positions)) != len(row_positions):
        return False
    if len(set(col_positions)) != len(col_positions):
        return False
    if len(set(row_colors)) != len(row_colors):
        return False

    mid_r = grid.shape[0] // 2
    mid_c = grid.shape[1] // 2
    if not any(r <= mid_r for r in row_positions):
        return False
    if not any(r > mid_r for r in row_positions):
        return False
    if not any(c <= mid_c for c in col_positions):
        return False
    if not any(c > mid_c for c in col_positions):
        return False

    quadrants = {_rotation_steps(r, c, grid.shape) for r in row_positions for c in col_positions}
    if quadrants != {0, 1, 2, 3}:
        return False

    expected = _render_scene(
        grid,
        template_mask,
        sorted(col_positions),
        sorted(zip(row_positions, row_colors)),
    )
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    if overlap_mode is not None:
        reverse = _render_scene(
            grid,
            template_mask,
            sorted(col_positions),
            sorted(zip(row_positions, row_colors)),
            reverse_order=True,
        )
        if np.array_equal(expected, reverse):
            return False

    return True


def _generate_case() -> np.ndarray | None:
    height = random.randrange(19, 30, 2)
    width = random.randrange(19, 30, 2)
    row_count = random.choice((3, 4))
    col_count = random.choice((3, 4))
    overlap_mode = random.choices([None, "row", "col"], weights=[4, 3, 3], k=1)[0]

    template_mask = np.array(random.choice(TEMPLATE_MASKS), dtype=bool)
    row_positions = _sample_rows(height, row_count, overlap_mode)
    col_positions = _sample_cols(width, col_count, overlap_mode)
    if row_positions is None or col_positions is None:
        return None

    row_colors = random.sample(list(COLOR_POOL), row_count)
    grid = _build_input(height, width, template_mask, row_positions, row_colors, col_positions)
    if not _validate_case(grid, template_mask, row_positions, row_colors, col_positions, overlap_mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
