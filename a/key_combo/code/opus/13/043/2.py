from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask, find_components, get_bg


FRAME_COLOR = Color.GREY
MARK_COLOR = Color.RED
SPRITE_COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]
GEN_MAX_ATTEMPTS = 400
GEN_INTERIOR_SIDES = [5, 6, 7, 8]
GEN_SYMMETRIC_TEMPLATES = [
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (1, 0), (2, 0), (1, -1), (1, 1)),
    ((0, -1), (0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (1, -1), (1, 1)),
    ((0, -1), (0, 0), (0, 1), (1, 0), (2, -1), (2, 0), (2, 1)),
]


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_interior(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 + 1, c0 + 1, r1 - 1, c1 - 1


def _frame_mid_col(box: tuple[int, int, int, int]) -> int:
    _ir0, ic0, _ir1, ic1 = _frame_interior(box)
    interior_width = ic1 - ic0 + 1
    return ic0 + (interior_width - 1) // 2


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    grey_components = find_components(
        np.asarray(grid, dtype=int),
        bg=get_bg(grid),
        colors=[FRAME_COLOR],
        connectivity=4,
        monochromatic=True,
    )
    frames: list[tuple[int, int, int, int]] = []
    for comp in grey_components:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        rel = comp["coords"] - np.array([r0, c0], dtype=int)
        mask = np.zeros((height, width), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        if np.array_equal(mask, _border_mask(height, width)):
            frames.append((r0, c0, r1, c1))
    frames.sort(key=lambda box: (box[1], box[0]))
    return frames


def _extract_sprite(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    bg: int,
) -> tuple[int, list[tuple[int, int]]]:
    ir0, ic0, ir1, ic1 = _frame_interior(box)
    interior = np.asarray(grid[ir0 : ir1 + 1, ic0 : ic1 + 1], dtype=int)
    mask = (interior != int(bg)) & (interior != FRAME_COLOR)
    assert np.any(mask), "expected a sprite inside every frame"
    colors = sorted(set(map(int, np.unique(interior[mask]))))
    assert len(colors) == 1, "expected one monochrome sprite per frame"
    color = colors[0]
    comps = components_mask(mask, connectivity=4)
    assert len(comps) == 1, "expected one 4-connected sprite per frame"
    coords = [(ir0 + int(r), ic0 + int(c)) for r, c in comps[0]]
    return color, coords


def _is_symmetric_about_col(coords: list[tuple[int, int]], mid_col: int) -> bool:
    pts = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        if (r, 2 * int(mid_col) - c) not in pts:
            return False
    return True


def _sprite_self_mid_col(coords: list[tuple[int, int]]) -> int | None:
    cols = [int(c) for _r, c in coords]
    c0, c1 = min(cols), max(cols)
    if (c1 - c0 + 1) % 2 == 0:
        return None
    return (c0 + c1) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Gate each framed sprite by reflection across the frame's vertical center line, then either fill the frame or erase-and-mark it.

    Input structure:
    - The grid has a black background and 3-4 disjoint one-cell-thick grey rectangular frames.
    - Each frame encloses a single 4-connected monochrome sprite of size 3-8 pixels.
    - Sprite colors come from {blue, red, green, yellow}; frame borders stay grey.

    Rule:
    - For each frame, compute the interior vertical center column; if the interior width is even, use the left of the two middle columns.
    - Test whether the sprite is reflection-symmetric about that frame-defined column.
    - Symmetric sprites expand to fill the entire frame interior with their own color.
    - Non-symmetric sprites are erased, then the full interior center column is painted red.

    Key insight:
    - The reference axis comes from the containing frame, not from the sprite's own bbox or the whole grid, so a self-symmetric sprite can still fail if it is off-axis inside its frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = _find_frames(g)
    assert 3 <= len(frames) <= 4, "expected 3-4 grey rectangular frames"

    out = g.copy()
    for box in frames:
        color, coords = _extract_sprite(g, box, bg)
        mid_col = _frame_mid_col(box)
        ir0, ic0, ir1, ic1 = _frame_interior(box)
        if _is_symmetric_about_col(coords, mid_col):
            out[ir0 : ir1 + 1, ic0 : ic1 + 1] = int(color)
        else:
            for r, c in coords:
                out[int(r), int(c)] = int(bg)
            out[ir0 : ir1 + 1, mid_col] = MARK_COLOR
    return out


def _coords_from_template(
    template: tuple[tuple[int, int], ...],
    top_row: int,
    axis_col: int,
) -> list[tuple[int, int]]:
    return [(int(top_row + dr), int(axis_col + dc)) for dr, dc in template]


def _template_fits(
    template: tuple[tuple[int, int], ...],
    interior: tuple[int, int, int, int],
    axis_col: int,
) -> bool:
    ir0, ic0, ir1, ic1 = interior
    rows = [dr for dr, _dc in template]
    cols = [dc for _dr, dc in template]
    return (
        ic0 <= int(axis_col + min(cols))
        and int(axis_col + max(cols)) <= ic1
        and ir0 <= ir1 - max(rows)
    )


def _place_template(
    rng: np.random.Generator,
    template: tuple[tuple[int, int], ...],
    interior: tuple[int, int, int, int],
    axis_col: int,
) -> list[tuple[int, int]] | None:
    if not _template_fits(template, interior, axis_col):
        return None
    ir0, _ic0, ir1, _ic1 = interior
    max_row = max(dr for dr, _dc in template)
    top_row = int(rng.integers(ir0, ir1 - max_row + 1))
    return _coords_from_template(template, top_row, axis_col)


def _random_connected_shape(
    rng: np.random.Generator,
    interior: tuple[int, int, int, int],
    size: int,
) -> list[tuple[int, int]] | None:
    ir0, ic0, ir1, ic1 = interior
    for _ in range(50):
        start = (int(rng.integers(ir0, ir1 + 1)), int(rng.integers(ic0, ic1 + 1)))
        cells = {start}
        while len(cells) < int(size):
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    if ir0 <= rr <= ir1 and ic0 <= cc <= ic1 and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            frontier_list = list(frontier)
            cells.add(tuple(frontier_list[int(rng.integers(len(frontier_list)))]))
        if len(cells) == int(size):
            return sorted(cells)
    return None


def _render_expected(
    inp: np.ndarray,
    plan: list[dict[str, object]],
) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    for spec in plan:
        box = spec["box"]
        color = int(spec["color"])
        symmetric = bool(spec["symmetric"])
        ir0, ic0, ir1, ic1 = _frame_interior(box)
        if symmetric:
            out[ir0 : ir1 + 1, ic0 : ic1 + 1] = color
        else:
            for r, c in spec["coords"]:
                out[int(r), int(c)] = Color.BLACK
            out[ir0 : ir1 + 1, _frame_mid_col(box)] = MARK_COLOR
    return out


def _assign_colors(
    rng: np.random.Generator,
    symmetric_flags: list[bool],
) -> list[int]:
    colors = list(map(int, rng.choice(SPRITE_COLOR_POOL, size=len(symmetric_flags), replace=False)))
    assigned: list[int | None] = [None] * len(symmetric_flags)
    sym_indices = [idx for idx, flag in enumerate(symmetric_flags) if flag]
    if Color.RED in colors:
        red_target = sym_indices[0]
        assigned[red_target] = Color.RED
        colors.remove(Color.RED)
    rng.shuffle(colors)
    cursor = 0
    for idx in range(len(assigned)):
        if assigned[idx] is None:
            assigned[idx] = int(colors[cursor])
            cursor += 1
    return [int(color) for color in assigned]


def _build_slot_layout(
    rng: np.random.Generator,
    boxes_meta: list[dict[str, object]],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    margin = int(rng.integers(1, 3))
    row_gap = int(rng.integers(2, 4))
    col_gap = int(rng.integers(2, 4))

    row_max = [0, 0]
    col_max = [0, 0]
    for spec in boxes_meta:
        slot_r, slot_c = spec["slot"]
        outer_h = int(spec["interior_h"]) + 2
        outer_w = int(spec["interior_w"]) + 2
        row_max[slot_r] = max(row_max[slot_r], outer_h)
        col_max[slot_c] = max(col_max[slot_c], outer_w)

    height = margin + row_max[0] + row_gap + row_max[1] + margin
    width = margin + col_max[0] + col_gap + col_max[1] + margin
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    row_starts = [margin, margin + row_max[0] + row_gap]
    col_starts = [margin, margin + col_max[0] + col_gap]

    for spec in boxes_meta:
        slot_r, slot_c = spec["slot"]
        outer_h = int(spec["interior_h"]) + 2
        outer_w = int(spec["interior_w"]) + 2
        r_base = row_starts[slot_r]
        c_base = col_starts[slot_c]
        r_slack = row_max[slot_r] - outer_h
        c_slack = col_max[slot_c] - outer_w
        r0 = int(r_base + rng.integers(r_slack + 1))
        c0 = int(c_base + rng.integers(c_slack + 1))
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        spec["box"] = (r0, c0, r1, c1)
    return grid, boxes_meta


def _choose_symmetric_sprite(
    rng: np.random.Generator,
    interior: tuple[int, int, int, int],
    axis_col: int,
) -> list[tuple[int, int]] | None:
    templates = list(GEN_SYMMETRIC_TEMPLATES)
    rng.shuffle(templates)
    for template in templates:
        coords = _place_template(rng, template, interior, axis_col)
        if coords is not None:
            return coords
    return None


def _choose_offset_symmetric_sprite(
    rng: np.random.Generator,
    interior: tuple[int, int, int, int],
    frame_mid: int,
) -> list[tuple[int, int]] | None:
    ir0, ic0, _ir1, ic1 = interior
    candidate_axes = [c for c in range(ic0, ic1 + 1) if c != int(frame_mid)]
    rng.shuffle(candidate_axes)
    templates = list(GEN_SYMMETRIC_TEMPLATES)
    for axis_col in candidate_axes:
        rng.shuffle(templates)
        for template in templates:
            coords = _place_template(rng, template, interior, axis_col)
            if coords is not None and not _is_symmetric_about_col(coords, frame_mid):
                return coords
    return None


def _choose_irregular_asymmetric_sprite(
    rng: np.random.Generator,
    interior: tuple[int, int, int, int],
    frame_mid: int,
) -> list[tuple[int, int]] | None:
    for _ in range(80):
        size = int(rng.integers(3, 9))
        coords = _random_connected_shape(rng, interior, size)
        if coords is None:
            continue
        if not _is_symmetric_about_col(coords, frame_mid):
            return coords
    return None


def _semantic_checks(inp: np.ndarray, plan: list[dict[str, object]]) -> bool:
    frames = _find_frames(inp)
    if len(frames) != len(plan):
        return False

    symmetric_count = 0
    asymmetric_count = 0
    saw_offset_self_symmetric = False
    saw_even_width = False
    saw_odd_width = False

    bg = get_bg(inp)
    by_box = {tuple(spec["box"]): spec for spec in plan}
    for box in frames:
        if box not in by_box:
            return False
        color, coords = _extract_sprite(inp, box, bg)
        spec = by_box[box]
        if color != int(spec["color"]):
            return False
        if set(coords) != {tuple(cell) for cell in spec["coords"]}:
            return False
        actual = _is_symmetric_about_col(coords, _frame_mid_col(box))
        if actual != bool(spec["symmetric"]):
            return False
        width = _frame_interior(box)[3] - _frame_interior(box)[1] + 1
        if width % 2 == 0:
            saw_even_width = True
        else:
            saw_odd_width = True
        if actual:
            symmetric_count += 1
        else:
            asymmetric_count += 1
            own_mid = _sprite_self_mid_col(coords)
            if own_mid is not None and own_mid != _frame_mid_col(box):
                if _is_symmetric_about_col(coords, own_mid):
                    saw_offset_self_symmetric = True

    if symmetric_count == 0 or asymmetric_count == 0:
        return False
    if not saw_offset_self_symmetric:
        return False
    if not (saw_even_width and saw_odd_width):
        return False

    expected = _render_expected(inp, plan)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_frames = int(rng.integers(3, 5))
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rng.shuffle(slots)
    slots = slots[:n_frames]

    widths = list(map(int, rng.choice(GEN_INTERIOR_SIDES, size=n_frames, replace=False)))
    heights = list(map(int, rng.choice(GEN_INTERIOR_SIDES, size=n_frames, replace=True)))

    sym_count = int(rng.integers(1, n_frames))
    symmetric_flags = [True] * sym_count + [False] * (n_frames - sym_count)
    rng.shuffle(symmetric_flags)
    colors = _assign_colors(rng, symmetric_flags)

    asym_indices = [idx for idx, flag in enumerate(symmetric_flags) if not flag]
    asym_modes = ["offset"]
    for _ in range(len(asym_indices) - 1):
        asym_modes.append("irregular" if float(rng.random()) < 0.5 else "offset")
    rng.shuffle(asym_modes)

    specs: list[dict[str, object]] = []
    asym_cursor = 0
    for idx in range(n_frames):
        mode = "symmetric"
        if not symmetric_flags[idx]:
            mode = asym_modes[asym_cursor]
            asym_cursor += 1
        specs.append(
            {
                "slot": slots[idx],
                "interior_h": heights[idx],
                "interior_w": widths[idx],
                "color": colors[idx],
                "symmetric": symmetric_flags[idx],
                "mode": mode,
            }
        )

    built = _build_slot_layout(rng, specs)
    if built is None:
        return None
    grid, specs = built

    for spec in specs:
        box = spec["box"]
        interior = _frame_interior(box)
        frame_mid = _frame_mid_col(box)
        if bool(spec["symmetric"]):
            coords = _choose_symmetric_sprite(rng, interior, frame_mid)
        elif spec["mode"] == "offset":
            coords = _choose_offset_symmetric_sprite(rng, interior, frame_mid)
        else:
            coords = _choose_irregular_asymmetric_sprite(rng, interior, frame_mid)
        if coords is None or not (3 <= len(coords) <= 8):
            return None
        color = int(spec["color"])
        for r, c in coords:
            if grid[r, c] != Color.BLACK:
                return None
            grid[r, c] = color
        spec["coords"] = tuple(coords)

    if np.count_nonzero(grid == Color.BLACK) <= grid.size // 2:
        return None
    if not _semantic_checks(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for framed symmetry gating")
