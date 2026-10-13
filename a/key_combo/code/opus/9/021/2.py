from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RIGHT_ONLY = 1
LEFT_ONLY = 2
DIFF_BOTH = 4
SEPARATOR = 5
PANEL_COLORS = (3, 6, 7, 8, 9)

PANEL_H_RANGE = (10, 12)
PANEL_W_RANGE = (10, 12)
LEGEND_ROW_OPTIONS = (2, 3)
ROLE_COUNT = 6
AREA_RANGE = (3, 6)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_sizes(mask: np.ndarray) -> list[int]:
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        area = 0
        while q:
            rr, cc = q.popleft()
            area += 1
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        sizes.append(area)
    return sizes


def _find_separator_column(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates: list[int] = []
    for c in range(1, w - 1):
        if np.all(grid[:, c] == SEPARATOR):
            left_w = c
            right_w = w - c - 1
            if left_w == right_w and left_w > 0:
                if not np.any(grid[:, :c] == SEPARATOR) and not np.any(grid[:, c + 1 :] == SEPARATOR):
                    candidates.append(c)
    assert len(candidates) == 1, "expected one separator column"
    return candidates[0]


def _find_legend_bbox(panel: np.ndarray) -> tuple[int, int, int, int]:
    h, w = panel.shape
    candidates: list[tuple[int, int, int, int]] = []
    for legend_h in LEGEND_ROW_OPTIONS:
        for r0 in range(h - legend_h + 1):
            for c0 in range(w - 1):
                sub = panel[r0 : r0 + legend_h, c0 : c0 + 2]
                if np.any(sub == BG):
                    continue
                if any(int(sub[rr, 0]) == int(sub[rr, 1]) for rr in range(legend_h)):
                    continue
                if len(set(map(int, sub.ravel()))) < 3:
                    continue
                rr0 = max(0, r0 - 1)
                rr1 = min(h, r0 + legend_h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w, c0 + 3)
                ring = panel[rr0:rr1, cc0:cc1].copy()
                ring[r0 - rr0 : r0 - rr0 + legend_h, c0 - cc0 : c0 - cc0 + 2] = BG
                if np.any(ring != BG):
                    continue
                candidates.append((r0, c0, r0 + legend_h - 1, c0 + 1))
    assert len(candidates) == 1, "expected one isolated 2-column legend"
    return candidates[0]


def _apply_chain(color: int, pairs: list[tuple[int, int]]) -> tuple[int, int]:
    current = int(color)
    steps = 0
    for src, dst in pairs:
        if current == src:
            current = dst
            steps += 1
    return current, steps


def _transform_left_panel(panel: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int, int, int]]:
    legend = _find_legend_bbox(panel)
    r0, c0, r1, c1 = legend
    pairs = [(int(panel[r, c0]), int(panel[r, c1])) for r in range(r0, r1 + 1)]
    out = panel.copy()
    for r in range(panel.shape[0]):
        for c in range(panel.shape[1]):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            color = int(panel[r, c])
            if color == BG:
                continue
            out[r, c] = _apply_chain(color, pairs)[0]
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    return out, pairs, legend


def _compare_panels(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.full(left.shape, BG, dtype=int)
    both_bg = (left == BG) & (right == BG)
    same_fg = (left != BG) & (left == right)
    left_only = (left != BG) & (right == BG)
    right_only = (left == BG) & (right != BG)
    both_diff = (left != BG) & (right != BG) & (left != right)

    out[both_bg] = BG
    out[same_fg] = left[same_fg]
    out[left_only] = LEFT_ONLY
    out[right_only] = RIGHT_ONLY
    out[both_diff] = DIFF_BOTH
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Apply the left legend only, erase it, and compare the transformed left panel against the untouched right panel.

    Input structure:
    - The grid contains two equal-width black-background panels separated by one gray divider column.
    - The left panel has an isolated 2-column legend near its upper-left area; each legend row gives a source color in the left column and a destination color in the right column.
    - Outside the legend, both panels contain scattered small foreground regions made of non-gray colors.

    Rule:
    - Read the legend from top to bottom and perform ordered chain substitution on every non-background pixel of the left panel outside the legend.
    - After substitution, erase the legend to black and remove the divider so the transformed left panel aligns with the original right panel.
    - Compare aligned cells: background/background stays black, equal foreground keeps that color, left-only becomes red, right-only becomes blue, and unequal foreground/foreground becomes yellow.

    Key insight:
    - The legend changes only the left panel, and its row order matters because earlier replacements can trigger later ones before the final difference map is colored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep = _find_separator_column(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape

    transformed_left, _pairs, _legend = _transform_left_panel(left)
    return _compare_panels(transformed_left, right)


def _normalize_shape(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    coords = np.array(sorted((r - min_r, c - min_c) for r, c in cells), dtype=int)
    return coords


def _random_connected_shape(area: int) -> np.ndarray:
    for _ in range(200):
        box_h = random.randint(2, 4)
        box_w = random.randint(2, 4)
        if box_h * box_w < area:
            continue
        seed = (random.randrange(box_h), random.randrange(box_w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        coords = _normalize_shape(cells)
        h = int(coords[:, 0].max()) + 1
        w = int(coords[:, 1].max()) + 1
        if h == 1 or w == 1:
            continue
        if w == 2 and area == 2 * h:
            continue
        return coords
    raise RuntimeError("failed to sample a connected shape")


def _shape_mask(coords: np.ndarray) -> np.ndarray:
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _reserve_region(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    rr1 = min(mask.shape[0], r0 + h + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(mask.shape[1], c0 + w + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _place_shape(
    coords: np.ndarray,
    panel_shape: tuple[int, int],
    blocked: np.ndarray,
) -> np.ndarray | None:
    h = int(coords[:, 0].max()) + 1
    w = int(coords[:, 1].max()) + 1
    candidates = []
    for r0 in range(panel_shape[0] - h + 1):
        for c0 in range(panel_shape[1] - w + 1):
            abs_cells = coords + np.array([r0, c0], dtype=int)
            if np.any(blocked[abs_cells[:, 0], abs_cells[:, 1]]):
                continue
            candidates.append(abs_cells)
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _paint_cells(panel: np.ndarray, cells: np.ndarray, color: int) -> None:
    if color == BG:
        return
    panel[cells[:, 0], cells[:, 1]] = int(color)


def _legend_pairs_and_roles():
    if random.random() < 0.5:
        a, b, c, e, f = random.sample(PANEL_COLORS, 5)
        pairs = [(a, b), (b, c)]
        roles = [
            ("shared_unchanged", e, e),
            ("shared_multi", a, c),
            ("shared_one", b, c),
            ("left_only", random.choice((a, b)), BG),
            ("right_only", BG, random.choice((a, c, e, f))),
            ("both_diff_chain", b, b),
        ]
    else:
        d, a, b, c, e = random.sample(PANEL_COLORS, 5)
        both_diff_color = random.choice((a, b))
        pairs = [(d, a), (a, b), (b, c)]
        roles = [
            ("shared_unchanged", e, e),
            ("shared_multi", d, c),
            ("shared_one", b, c),
            ("left_only", random.choice((d, a, b)), BG),
            ("right_only", BG, a),
            ("both_diff_chain", both_diff_color, both_diff_color),
        ]
    return pairs, roles


def _build_case() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)

    left = np.full((panel_h, panel_w), BG, dtype=int)
    right = np.full((panel_h, panel_w), BG, dtype=int)

    pairs, roles = _legend_pairs_and_roles()
    legend_h = len(pairs)
    legend_r0 = random.choice((0, 1))
    legend_c0 = random.choice((0, 1))
    if legend_r0 + legend_h >= panel_h - 2 or legend_c0 + 2 >= panel_w - 4:
        return None

    for i, (src, dst) in enumerate(pairs):
        left[legend_r0 + i, legend_c0] = src
        left[legend_r0 + i, legend_c0 + 1] = dst

    blocked = np.zeros((panel_h, panel_w), dtype=bool)
    _reserve_region(blocked, legend_r0, legend_c0, legend_h, 2)
    _reserve_region(blocked, 0, 0, legend_h + 1, legend_c0 + 3)

    shapes = []
    for role_name, left_color, right_color in roles:
        area = random.randint(*AREA_RANGE)
        shape = _random_connected_shape(area)
        shapes.append((role_name, left_color, right_color, shape))

    shapes.sort(key=lambda item: -len(item[3]))
    placements: dict[str, tuple[np.ndarray, int, int]] = {}

    for role_name, left_color, right_color, shape in shapes:
        placed = _place_shape(shape, (panel_h, panel_w), blocked)
        if placed is None:
            return None
        mask = _shape_mask(shape)
        h, w = mask.shape
        r0 = int(placed[:, 0].min())
        c0 = int(placed[:, 1].min())
        _reserve_region(blocked, r0, c0, h, w)
        placements[role_name] = (placed, left_color, right_color)

    for cells, left_color, right_color in placements.values():
        _paint_cells(left, cells, left_color)
        _paint_cells(right, cells, right_color)

    inp = np.concatenate(
        [left, np.full((panel_h, 1), SEPARATOR, dtype=int), right],
        axis=1,
    )
    if not _validate_generated_case(inp):
        return None
    return inp


def _validate_generated_case(inp: np.ndarray) -> bool:
    try:
        sep = _find_separator_column(inp)
        left = inp[:, :sep]
        right = inp[:, sep + 1 :]
        transformed_left, pairs, legend = _transform_left_panel(left)
        out = _compare_panels(transformed_left, right)
        if not np.array_equal(main(inp), out):
            return False

        r0, c0, r1, c1 = legend
        left_without_legend = left.copy()
        left_without_legend[r0 : r1 + 1, c0 : c1 + 1] = BG

        legend_colors = {color for pair in pairs for color in pair}
        visible_colors = set(map(int, left_without_legend[left_without_legend != BG])) | set(
            map(int, right[right != BG])
        )
        if not legend_colors <= visible_colors:
            return False

        left_comp_sizes = _component_sizes(left_without_legend != BG)
        right_comp_sizes = _component_sizes(right != BG)
        if len(left_comp_sizes) < ROLE_COUNT - 1 or len(right_comp_sizes) < ROLE_COUNT - 1:
            return False
        if not all(3 <= size <= 8 for size in left_comp_sizes + right_comp_sizes):
            return False

        saw_multi_step = False
        saw_match_created = False
        saw_diff_created = False
        category_same = False
        category_left_only = False
        category_right_only = False
        category_both_diff = False

        for r in range(left.shape[0]):
            for c in range(left.shape[1]):
                if r0 <= r <= r1 and c0 <= c <= c1:
                    continue
                left_raw = int(left[r, c])
                right_raw = int(right[r, c])
                left_final = int(transformed_left[r, c])
                final_color, steps = _apply_chain(left_raw, pairs) if left_raw != BG else (BG, 0)
                if left_raw != BG and left_final != final_color:
                    return False
                if steps >= 2:
                    saw_multi_step = True
                if left_raw != right_raw and left_final == right_raw and right_raw != BG:
                    saw_match_created = True
                if left_raw == right_raw and left_raw != BG and left_final != right_raw:
                    saw_diff_created = True

                out_val = int(out[r, c])
                if out_val not in (BG, LEFT_ONLY, RIGHT_ONLY, DIFF_BOTH):
                    category_same = True
                elif out_val == LEFT_ONLY:
                    category_left_only = True
                elif out_val == RIGHT_ONLY:
                    category_right_only = True
                elif out_val == DIFF_BOTH:
                    category_both_diff = True

        if not (saw_multi_step and saw_match_created and saw_diff_created):
            return False
        if not (category_same and category_left_only and category_right_only and category_both_diff):
            return False
        if out.shape != left.shape:
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        candidate = _build_case()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid panel-difference instance")
