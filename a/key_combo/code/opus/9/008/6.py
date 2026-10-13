from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, is_contiguous


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

_MASK_BANK: list[np.ndarray] | None = None


def _offsets(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _extract_template(grid: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    green_comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[GREEN],
        return_format="dict",
    )
    assert len(green_comps) == 1, "expected exactly one green template"

    box = tuple(map(int, green_comps[0]["bbox"]))
    sprite = crop_bbox(grid, box)
    green_mask = sprite == GREEN
    support_mask = sprite == GRAY

    assert 3 <= int(green_mask.sum()) <= 5, "template must contain 3 to 5 green cells"
    assert int(support_mask.sum()) >= 1, "template bbox must contain gray support cells"
    assert is_contiguous(green_mask.astype(int), background=0, connectivity=4), "template must be 4-connected"
    return box, green_mask, support_mask


def _stamp_matches(
    grid: np.ndarray,
    green_mask: np.ndarray,
    support_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    bh, bw = green_mask.shape
    work = grid.copy()
    matches: list[tuple[int, int]] = []

    for r0 in range(grid.shape[0] - bh + 1):
        for c0 in range(grid.shape[1] - bw + 1):
            window = work[r0 : r0 + bh, c0 : c0 + bw]
            if not np.all(window[green_mask] == BG):
                continue
            if not np.all(window[support_mask] == GRAY):
                continue
            matches.append((r0, c0))
            window[green_mask] = GREEN

    return matches, work


def _recolor_by_edges(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    width = out.shape[1]
    for r, c in np.argwhere(out == GREEN):
        out[int(r), int(c)] = BLUE if int(c) <= width - 1 - int(c) else RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the green template into every matching gray-supported slot, then recolor each stamped pixel by its nearest border line.

    Input structure:
    - The grid has a black background, a full blue edge line in the leftmost column, and a full red edge line in the rightmost column.
    - One 4-connected green template sits inside a small bounding box whose other marked cells are gray support pixels.
    - Elsewhere, matching empty slots repeat the same gray support signature while the template-mask cells are still black.

    Rule:
    - Find the unique green template and use its bounding box to read a green stamp mask plus the gray support signature inside that box.
    - Scan candidate box positions from top-left to bottom-right on the current working grid; a slot matches when every green-mask cell is black and every support-signature cell is gray.
    - Stamp green into every matched slot, then recolor every green pixel, including the original template, by the nearer full-height border line: blue from the left line, red from the right line, with ties going to blue.

    Key insight:
    - The stamp does not keep a fixed color: support matching decides where copies appear, and each copied pixel independently inherits its color from horizontal distance to the two border lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert np.all(g[:, 0] == BLUE), "left border must be blue"
    assert np.all(g[:, -1] == RED), "right border must be red"

    _box, green_mask, support_mask = _extract_template(g)
    matches, stamped = _stamp_matches(g, green_mask, support_mask)
    assert matches, "expected at least one empty supported slot"
    return _recolor_by_edges(stamped)


def _mask_bank() -> list[np.ndarray]:
    global _MASK_BANK
    if _MASK_BANK is not None:
        return _MASK_BANK

    bank: list[np.ndarray] = []
    for h in (2, 3):
        for w in (2, 3):
            for bits in range(1, 1 << (h * w)):
                mask = np.array([(bits >> i) & 1 for i in range(h * w)], dtype=int).reshape(h, w).astype(bool)
                area = int(mask.sum())
                if not (3 <= area <= 5):
                    continue
                if area == h * w:
                    continue
                if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                if not is_contiguous(mask.astype(int), background=0, connectivity=4):
                    continue

                open_cells = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]
                adjacent_open = [
                    (r, c)
                    for r, c in open_cells
                    if any(
                        0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                        for dr, dc in DIRS4
                    )
                ]
                if not adjacent_open:
                    continue
                bank.append(mask)

    _MASK_BANK = bank
    return _MASK_BANK


def _choose_support(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    open_cells = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]]
    adjacent_open = [
        (r, c)
        for r, c in open_cells
        if any(0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc] for dr, dc in DIRS4)
    ]
    assert adjacent_open

    support = {random.choice(adjacent_open)}
    if len(open_cells) > 1:
        max_extra = min(2, len(open_cells)) - 1
        extra_n = random.randint(0, max_extra)
        remaining = [cell for cell in open_cells if cell not in support]
        if extra_n > 0:
            for cell in random.sample(remaining, extra_n):
                support.add(cell)

    out = np.zeros_like(mask, dtype=bool)
    for r, c in support:
        out[r, c] = True
    return out


def _windows_overlap(a: tuple[int, int], b: tuple[int, int], h: int, w: int) -> bool:
    ar, ac = a
    br, bc = b
    return not (ar + h <= br or br + h <= ar or ac + w <= bc or bc + w <= ac)


def _pick_width(want_tie: bool) -> int:
    if want_tie:
        return random.choice([19, 21, 23, 25])
    return random.choice([18, 20, 22, 24, 26])


def _center_col(width: int, bbox_w: int, want_tie: bool) -> int:
    threshold = (width - 1) // 2
    if want_tie:
        return threshold
    return threshold - (bbox_w - 2)


def _instance_colors(anchor: tuple[int, int], mask: np.ndarray, width: int) -> set[int]:
    colors: set[int] = set()
    for _r, c in _offsets(mask):
        abs_c = anchor[1] + c
        colors.add(BLUE if abs_c <= width - 1 - abs_c else RED)
    return colors


def _paint_window(grid: np.ndarray, anchor: tuple[int, int], green_mask: np.ndarray, support_mask: np.ndarray, template: bool) -> None:
    r0, c0 = anchor
    window = grid[r0 : r0 + green_mask.shape[0], c0 : c0 + green_mask.shape[1]]
    window[support_mask] = GRAY
    if template:
        window[green_mask] = GREEN


def _expanded_window_mask(shape: tuple[int, int], anchors: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0 in anchors:
        rr0 = max(0, r0 - 1)
        cc0 = max(1, c0 - 1)
        rr1 = min(shape[0], r0 + h + 1)
        cc1 = min(shape[1] - 1, c0 + w + 1)
        mask[rr0:rr1, cc0:cc1] = True
    return mask


def _add_gray_clutter(grid: np.ndarray, protected: np.ndarray) -> None:
    pieces = [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
    ]
    for _ in range(random.randint(0, 2)):
        cells = random.choice(pieces)
        for _attempt in range(30):
            h = max(r for r, _c in cells) + 1
            w = max(c for _r, c in cells) + 1
            r0 = random.randint(1, grid.shape[0] - h - 1)
            c0 = random.randint(1, grid.shape[1] - w - 2)
            if any(protected[r0 + dr, c0 + dc] or grid[r0 + dr, c0 + dc] != BG for dr, dc in cells):
                continue
            for dr, dc in cells:
                grid[r0 + dr, c0 + dc] = GRAY
            break


def _spec_output(grid: np.ndarray, green_mask: np.ndarray, support_mask: np.ndarray) -> tuple[list[tuple[int, int]], np.ndarray]:
    matches, stamped = _stamp_matches(grid, green_mask, support_mask)
    return matches, _recolor_by_edges(stamped)


def _validate_semantics(
    grid: np.ndarray,
    template_anchor: tuple[int, int],
    planned_slots: list[tuple[int, int]],
    green_mask: np.ndarray,
    support_mask: np.ndarray,
    want_tie: bool,
) -> np.ndarray | None:
    green_comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[GREEN],
        return_format="dict",
    )
    if len(green_comps) != 1 or int(green_comps[0]["area"]) != int(green_mask.sum()):
        return None

    box = bbox(grid == GREEN)
    if box is None or tuple(map(int, box[:2])) != template_anchor:
        return None

    matches, expected = _spec_output(grid, green_mask, support_mask)
    if matches != planned_slots:
        return None

    all_instances = [template_anchor, *planned_slots]
    instance_colors = [_instance_colors(anchor, green_mask, grid.shape[1]) for anchor in all_instances]
    if {BLUE} not in instance_colors or {RED} not in instance_colors:
        return None
    if not any(colors == {BLUE, RED} for colors in instance_colors):
        return None

    if want_tie:
        midpoint = (grid.shape[1] - 1) // 2
        center_anchor = planned_slots[[colors == {BLUE, RED} for colors in instance_colors[1:]].index(True)]
        center_cols = [center_anchor[1] + c for _r, c in _offsets(green_mask)]
        if midpoint not in center_cols:
            return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return expected


def _attempt_generate() -> np.ndarray | None:
    want_tie = random.random() < 0.5
    height = random.randint(13, 19)
    width = _pick_width(want_tie)

    green_mask = random.choice(_mask_bank()).copy()
    support_mask = _choose_support(green_mask)
    bh, bw = green_mask.shape

    center_c = _center_col(width, bw, want_tie)
    threshold = (width - 1) // 2
    left_choices = list(range(1, threshold - bw + 1))
    right_start = max(threshold + 1, center_c + bw)
    right_choices = list(range(right_start, width - bw))
    if not left_choices or not right_choices:
        return None

    template_side = random.choice(["left", "right"])
    slot_count = random.choice([2, 3])

    columns: dict[str, int] = {"center": center_c}
    columns["left"] = random.choice(left_choices)
    columns["right"] = random.choice(right_choices)

    roles = [("template", template_side), ("slot", "center"), ("slot", "right" if template_side == "left" else "left")]
    if slot_count == 3:
        roles.append(("slot", random.choice(["left", "right"])))

    anchors: list[tuple[str, str, tuple[int, int]]] = []
    used_rects: list[tuple[int, int]] = []
    for role_kind, zone in roles:
        col = columns[zone]
        row_options = list(range(1, height - bh))
        random.shuffle(row_options)
        placed = None
        for row in row_options:
            anchor = (row, col)
            if all(not _windows_overlap(anchor, old, bh, bw) for old in used_rects):
                placed = anchor
                break
        if placed is None:
            return None
        used_rects.append(placed)
        anchors.append((role_kind, zone, placed))

    grid = np.full((height, width), BG, dtype=int)
    grid[:, 0] = BLUE
    grid[:, -1] = RED

    template_anchor = (-1, -1)
    planned_slots: list[tuple[int, int]] = []
    for role_kind, _zone, anchor in anchors:
        _paint_window(grid, anchor, green_mask, support_mask, template=(role_kind == "template"))
        if role_kind == "template":
            template_anchor = anchor
        else:
            planned_slots.append(anchor)

    protected = _expanded_window_mask(grid.shape, [anchor for _kind, _zone, anchor in anchors], bh, bw)
    _add_gray_clutter(grid, protected)

    if _validate_semantics(grid, template_anchor, planned_slots, green_mask, support_mask, want_tie) is None:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
