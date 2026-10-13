from __future__ import annotations

from importlib.machinery import SourceFileLoader

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_1_024_2",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
Color = _ARC_UTILS.Color
colors_present = _ARC_UTILS.colors_present
crop_bbox = _ARC_UTILS.crop_bbox
get_bg = _ARC_UTILS.get_bg
paint_coords = _ARC_UTILS.paint_coords


PANEL_SIZE_RANGE = (6, 14)
SYMMETRIC_MAX_DENSITY = 0.38
DEFECTIVE_MAX_DENSITY = 0.28
MIN_MASK_AREA = 4
MAX_ATTEMPTS_PER_MASK = 200
MAX_GENERATION_ATTEMPTS = 400
RNG = np.random.default_rng()


def _other_axis(axis: str) -> str:
    return "v" if axis == "h" else "h"


def _reflect_mask(mask: np.ndarray, axis: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if axis == "h":
        return mask[::-1, :].copy()
    if axis == "v":
        return mask[:, ::-1].copy()
    raise ValueError(f"unknown axis: {axis}")


def _is_symmetric(mask: np.ndarray, axis: str) -> bool:
    return np.array_equal(np.asarray(mask, dtype=bool), _reflect_mask(mask, axis))


def _uniform_color(line: np.ndarray) -> int | None:
    return int(line[0]) if np.all(line == line[0]) else None


def _panel_bboxes(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    candidates = []

    for split_r in range(1, h - 1):
        row_color = _uniform_color(g[split_r, :])
        if row_color is None or row_color == bg:
            continue
        top_h = split_r
        bottom_h = h - split_r - 1
        if top_h <= 0 or top_h != bottom_h:
            continue

        for split_c in range(1, w - 1):
            col_color = _uniform_color(g[:, split_c])
            if col_color is None or col_color != row_color:
                continue
            left_w = split_c
            right_w = w - split_c - 1
            if left_w <= 0 or left_w != right_w or top_h != left_w:
                continue

            bboxes = [
                (0, 0, split_r - 1, split_c - 1),
                (0, split_c + 1, split_r - 1, w - 1),
                (split_r + 1, 0, h - 1, split_c - 1),
                (split_r + 1, split_c + 1, h - 1, w - 1),
            ]
            panels = [crop_bbox(g, bbox) for bbox in bboxes]
            if not all(panel.shape == (top_h, left_w) for panel in panels):
                continue

            fg_colors = []
            ok = True
            for panel in panels:
                if int(np.count_nonzero(panel == bg)) == 0:
                    ok = False
                    break
                panel_colors = colors_present(panel, exclude={bg})
                if len(panel_colors) != 1:
                    ok = False
                    break
                fg_colors.append(int(panel_colors[0]))
            if not ok or len(set(fg_colors)) != 4:
                continue

            candidates.append((bg, int(row_color), top_h, bboxes))

    assert len(candidates) == 1, "expected one separator row/column split into four square panels"
    return candidates[0]


def _extract_panels(grid: np.ndarray) -> tuple[int, int, list[np.ndarray]]:
    bg, sep, _panel_size, bboxes = _panel_bboxes(grid)
    panels = [crop_bbox(np.asarray(grid, dtype=int), bbox) for bbox in bboxes]
    return bg, sep, panels


def _panel_info(panel: np.ndarray, bg: int) -> tuple[np.ndarray, int, bool, bool]:
    fg_colors = colors_present(panel, exclude={bg})
    assert len(fg_colors) == 1, "each panel must contain exactly one foreground color"
    color = int(fg_colors[0])
    mask = np.asarray(panel == color, dtype=bool)
    return mask, color, _is_symmetric(mask, "h"), _is_symmetric(mask, "v")


def _majority_axis_and_broken(
    panels: list[np.ndarray], bg: int
) -> tuple[str, int, list[tuple[np.ndarray, int, bool, bool]]]:
    infos = [_panel_info(panel, bg) for panel in panels]
    h_votes = sum(int(sym_h) for _mask, _color, sym_h, _sym_v in infos)
    v_votes = sum(int(sym_v) for _mask, _color, _sym_h, sym_v in infos)
    assert sorted((h_votes, v_votes)) == [1, 2], "expected a 2-to-1 axis vote"

    broken = [idx for idx, (_mask, _color, sym_h, sym_v) in enumerate(infos) if not sym_h and not sym_v]
    assert len(broken) == 1, "expected exactly one panel to be asymmetric on both axes"

    axis = "h" if h_votes > v_votes else "v"
    return axis, broken[0], infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Output only the missing mirror patch for the uniquely broken panel.

    Input structure:
    - A single separator row and a single separator column divide the grid into
      four equal square panels.
    - Each panel contains one foreground mask color on a shared background, and
      the four panel colors are all different.

    Rule:
    - Test each panel mask for horizontal-midline and vertical-midline symmetry.
    - The axis with two votes is the majority axis; the remaining symmetric panel
      is the decoy, and the only panel symmetric on neither axis is the broken one.
    - Reflect the broken panel's mask across the majority axis and keep only the
      reflected cells that land on background in the original panel.
    - Return a single panel-sized grid showing just those newly added cells in the
      broken panel's own color, with background elsewhere.

    Key insight:
    - The separator only partitions the evidence; the true axis is decided by a
      2-to-1 panel vote, and the answer is the missing reflected patch rather than
      the fully repaired panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, _sep, panels = _extract_panels(g)
    axis, broken_idx, infos = _majority_axis_and_broken(panels, bg)

    broken_mask, broken_color, _sym_h, _sym_v = infos[broken_idx]
    additions = _reflect_mask(broken_mask, axis) & ~broken_mask

    out = np.full(broken_mask.shape, bg, dtype=int)
    paint_coords(out, np.argwhere(additions), broken_color)
    return out


def _paint_random_rectangles(
    mask: np.ndarray,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    rng: np.random.Generator,
) -> None:
    if row_hi <= row_lo or col_hi <= col_lo:
        return

    parts = int(rng.integers(2, 5))
    for _ in range(parts):
        rect_h = int(rng.integers(1, min(3, row_hi - row_lo) + 1))
        rect_w = int(rng.integers(1, min(4, col_hi - col_lo) + 1))
        r0 = int(rng.integers(row_lo, row_hi - rect_h + 1))
        c0 = int(rng.integers(col_lo, col_hi - rect_w + 1))
        mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True

    if rng.random() < 0.5:
        if rng.random() < 0.5:
            r = int(rng.integers(row_lo, row_hi))
            w = int(rng.integers(1, min(5, col_hi - col_lo) + 1))
            c0 = int(rng.integers(col_lo, col_hi - w + 1))
            mask[r, c0 : c0 + w] = True
        else:
            c = int(rng.integers(col_lo, col_hi))
            h = int(rng.integers(1, min(5, row_hi - row_lo) + 1))
            r0 = int(rng.integers(row_lo, row_hi - h + 1))
            mask[r0 : r0 + h, c] = True


def _inject_axis_pair(mask: np.ndarray, axis: str, rng: np.random.Generator) -> bool:
    n = mask.shape[0]
    for _ in range(80):
        if axis == "h":
            if n // 2 == 0:
                return False
            r = int(rng.integers(0, n // 2))
            c = int(rng.integers(0, n))
            mc = n - 1 - c
            rr = n - 1 - r
            if c == mc:
                continue
            if mask[r, c] or mask[rr, c] or mask[r, mc] or mask[rr, mc]:
                continue
            mask[r, c] = True
            mask[rr, c] = True
            return True

        if axis == "v":
            if n // 2 == 0:
                return False
            c = int(rng.integers(0, n // 2))
            r = int(rng.integers(0, n))
            mr = n - 1 - r
            cc = n - 1 - c
            if r == mr:
                continue
            if mask[r, c] or mask[r, cc] or mask[mr, c] or mask[mr, cc]:
                continue
            mask[r, c] = True
            mask[r, cc] = True
            return True

    return False


def _break_one_sided_symmetry(
    mask: np.ndarray,
    axis_to_break: str,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    rng: np.random.Generator,
) -> bool:
    for _ in range(80):
        r = int(rng.integers(row_lo, row_hi))
        c = int(rng.integers(col_lo, col_hi))
        if axis_to_break == "v":
            mc = mask.shape[1] - 1 - c
            if c == mc or mask[r, c] or mask[r, mc]:
                continue
            mask[r, c] = True
            return True
        if axis_to_break == "h":
            mr = mask.shape[0] - 1 - r
            if r == mr or mask[r, c] or mask[mr, c]:
                continue
            mask[r, c] = True
            return True
    return False


def _make_axis_only_symmetric_mask(n: int, axis: str, rng: np.random.Generator) -> np.ndarray:
    other = _other_axis(axis)
    for _ in range(MAX_ATTEMPTS_PER_MASK):
        seed = np.zeros((n, n), dtype=bool)
        if axis == "h":
            _paint_random_rectangles(seed, 0, (n + 1) // 2, 0, n, rng)
        else:
            _paint_random_rectangles(seed, 0, n, 0, (n + 1) // 2, rng)

        mask = seed | _reflect_mask(seed, axis)
        if _is_symmetric(mask, other):
            _inject_axis_pair(mask, axis, rng)

        area = int(mask.sum())
        if not (MIN_MASK_AREA <= area <= int(n * n * SYMMETRIC_MAX_DENSITY)):
            continue
        if _is_symmetric(mask, axis) and not _is_symmetric(mask, other):
            return mask

    raise RuntimeError(f"failed to generate {axis}-only symmetric mask")


def _make_defective_mask(n: int, axis: str, rng: np.random.Generator) -> np.ndarray:
    for _ in range(MAX_ATTEMPTS_PER_MASK):
        mask = np.zeros((n, n), dtype=bool)
        if axis == "h":
            side = str(rng.choice(["top", "bottom"]))
            row_lo, row_hi = (0, n // 2) if side == "top" else (n - n // 2, n)
            _paint_random_rectangles(mask, row_lo, row_hi, 0, n, rng)
            if _is_symmetric(mask, "v"):
                _break_one_sided_symmetry(mask, "v", row_lo, row_hi, 0, n, rng)
        else:
            side = str(rng.choice(["left", "right"]))
            col_lo, col_hi = (0, n // 2) if side == "left" else (n - n // 2, n)
            _paint_random_rectangles(mask, 0, n, col_lo, col_hi, rng)
            if _is_symmetric(mask, "h"):
                _break_one_sided_symmetry(mask, "h", 0, n, col_lo, col_hi, rng)

        patch = _reflect_mask(mask, axis) & ~mask
        area = int(mask.sum())
        if not (MIN_MASK_AREA <= area <= int(n * n * DEFECTIVE_MAX_DENSITY)):
            continue
        if int(patch.sum()) == 0:
            continue
        if not _is_symmetric(mask, "h") and not _is_symmetric(mask, "v"):
            return mask

    raise RuntimeError(f"failed to generate defective mask for axis {axis}")


def _render_panel(mask: np.ndarray, bg: int, color: int) -> np.ndarray:
    panel = np.full(mask.shape, bg, dtype=int)
    panel[mask] = int(color)
    return panel


def _assemble_grid(panel_masks: list[np.ndarray], panel_colors: list[int], bg: int, sep: int) -> np.ndarray:
    n = panel_masks[0].shape[0]
    grid = np.full((2 * n + 1, 2 * n + 1), sep, dtype=int)
    starts = [(0, 0), (0, n + 1), (n + 1, 0), (n + 1, n + 1)]

    for (r0, c0), mask, color in zip(starts, panel_masks, panel_colors):
        panel = _render_panel(mask, bg, color)
        grid[r0 : r0 + n, c0 : c0 + n] = panel

    return grid


def _expected_patch(mask: np.ndarray, axis: str, bg: int, color: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    out[_reflect_mask(mask, axis) & ~mask] = int(color)
    return out


def _generate_case() -> np.ndarray | None:
    n = int(RNG.integers(PANEL_SIZE_RANGE[0], PANEL_SIZE_RANGE[1] + 1))
    majority_axis = str(RNG.choice(["h", "v"]))

    palette = RNG.choice(np.array(Color.ALL_COLORS, dtype=int), size=6, replace=False)
    bg = int(palette[0])
    sep = int(palette[1])
    panel_colors = [int(x) for x in palette[2:]]

    roles = ["majority", "majority", "decoy", "broken"]
    RNG.shuffle(roles)
    broken_idx = roles.index("broken")

    panel_masks: list[np.ndarray] = []
    for role in roles:
        if role == "majority":
            panel_masks.append(_make_axis_only_symmetric_mask(n, majority_axis, RNG))
        elif role == "decoy":
            panel_masks.append(_make_axis_only_symmetric_mask(n, _other_axis(majority_axis), RNG))
        else:
            panel_masks.append(_make_defective_mask(n, majority_axis, RNG))

    if len({mask.tobytes() for mask in panel_masks}) < 3:
        return None

    grid = _assemble_grid(panel_masks, panel_colors, bg, sep)
    expected = _expected_patch(panel_masks[broken_idx], majority_axis, bg, panel_colors[broken_idx])

    found_bg, found_sep, panels = _extract_panels(grid)
    if found_bg != bg or found_sep != sep:
        return None

    axis, found_broken_idx, infos = _majority_axis_and_broken(panels, found_bg)
    if axis != majority_axis or found_broken_idx != broken_idx:
        return None

    if not np.array_equal(main(grid), expected):
        return None

    h_votes = sum(int(sym_h) for _mask, _color, sym_h, _sym_v in infos)
    v_votes = sum(int(sym_v) for _mask, _color, _sym_h, sym_v in infos)
    if sorted((h_votes, v_votes)) != [1, 2]:
        return None

    if int(np.count_nonzero(expected != bg)) == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
