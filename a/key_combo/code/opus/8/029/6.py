from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("shared_arc_utils", arc_utils_path)
    assert spec is not None and spec.loader is not None, "failed to load arc_utils.py"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
find_components = _ARC.find_components
paste = _ARC.paste
random_free_location = _ARC.random_free_location


BG = Color.BLACK
SEPARATOR = Color.GREY
BAR = Color.ORANGE
ANCHOR = Color.TEAL
LEFT_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK)
GEN_MAX_ATTEMPTS = 400


def _is_contiguous(cols: np.ndarray) -> bool:
    return cols.size > 0 and np.array_equal(cols, np.arange(int(cols[0]), int(cols[-1]) + 1))


def _separator_col(grid: np.ndarray) -> int:
    grey_cols = np.flatnonzero(np.any(grid == SEPARATOR, axis=0))
    assert len(grey_cols) == 1, "expected exactly one grey separator column"
    sep = int(grey_cols[0])
    assert np.all(grid[:, sep] == SEPARATOR), "separator column must be solid grey"
    return sep


def _largest_left_component(left: np.ndarray) -> tuple[dict, np.ndarray]:
    components = find_components(
        left,
        bg=BG,
        colors=LEFT_COLORS,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert 3 <= len(components) <= 5, "expected 3 to 5 left-side objects"

    max_area = max(int(comp["area"]) for comp in components)
    largest = [comp for comp in components if int(comp["area"]) == max_area]
    assert len(largest) == 1, "largest left object must be unique"
    comp = largest[0]

    r0, c0, r1, c1 = map(int, comp["bbox"])
    proto = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    color = int(comp["color"])
    coords = comp["coords"]
    proto[coords[:, 0] - r0, coords[:, 1] - c0] = color
    return comp, proto


def _parse_right_panel(right: np.ndarray, proto_h: int, proto_w: int) -> tuple[list[int], list[int]]:
    assert set(map(int, np.unique(right))) <= {BG, BAR, ANCHOR}, "unexpected right-panel colors"

    bottom = right[-1, :]
    bar_cols = np.flatnonzero(bottom == BAR)
    assert len(bar_cols) == proto_h, "bar length must equal prototype height"
    assert _is_contiguous(bar_cols), "orange bar must be contiguous"
    assert np.all(bottom[np.setdiff1d(np.arange(right.shape[1]), bar_cols)] == BG), "only the bar may occupy the bottom row"

    anchor_rows: list[int] = []
    for index, col_raw in enumerate(bar_cols):
        col = int(col_raw)
        rows = np.flatnonzero(right[:, col] == ANCHOR)
        assert len(rows) == 1, "each bar column must have exactly one teal anchor"
        row = int(rows[0])
        expected_row = right.shape[0] - 2 - index * proto_w
        assert row == expected_row, "anchor rows must follow the staircase width rule"
        anchor_rows.append(row)

    teal_coords = np.argwhere(right == ANCHOR)
    assert len(teal_coords) == proto_h, "anchors must appear only above the bar columns"
    teal_cols = sorted(int(c) for _r, c in teal_coords)
    assert teal_cols == [int(c) for c in bar_cols], "anchor columns must match the bar columns"

    return [int(c) for c in bar_cols], anchor_rows


def _render_from_scene(grid: np.ndarray, proto: np.ndarray, sep: int, bar_cols: list[int], anchor_rows: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    h, w = proto.shape

    for row_index in range(h):
        repeat = row_index
        if repeat == 0:
            continue
        global_col = sep + 1 + int(bar_cols[row_index])
        start_row = int(anchor_rows[row_index]) + 1
        for proto_col in range(w):
            color = int(proto[row_index, proto_col])
            if color == BG:
                continue
            r0 = start_row + proto_col * repeat
            r1 = r0 + repeat
            out[r0:r1, global_col] = color

    return out


def _extract_input_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    sep = _separator_col(g)
    assert 1 <= sep <= g.shape[1] - 3, "separator must leave space on both sides"

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert right.shape[1] >= 3, "right panel must be non-trivial"

    largest, proto = _largest_left_component(left)
    h, w = proto.shape
    assert h >= 3 and w >= 2, "prototype must be non-degenerate"

    row_masks = [tuple(int(v != BG) for v in row) for row in proto]
    assert len(set(row_masks)) >= 2, "prototype rows must not all be identical"
    assert any(0 < np.count_nonzero(row) < w for row in proto), "at least one prototype row must contain a gap"

    bar_cols, anchor_rows = _parse_right_panel(right, h, w)
    return {
        "grid": g,
        "sep": sep,
        "largest": largest,
        "proto": proto,
        "bar_cols": bar_cols,
        "anchor_rows": anchor_rows,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch the tight crop of the uniquely largest left object onto the right staircase anchors.

    Input structure:
    - A solid grey separator column divides the grid into a left object panel and a right scaffold panel on black background.
    - The left panel contains 3 to 5 separated monochrome 4-connected objects in blue, red, green, yellow, or pink, with exactly one unique largest object by area.
    - The right panel contains only a contiguous orange bottom bar and one teal anchor above each bar column, with anchor heights stepping upward by the largest object's cropped width.

    Rule:
    - Find the uniquely largest left-side object, crop it to its tight bounding box, and treat non-object cells inside that box as black.
    - Let the crop have height h and width w; the orange bar length must be h, and the teal anchors define h target columns in left-to-right order.
    - For crop row r, use right-panel column r and fill the cells strictly below its teal anchor and above the orange bar by repeating each of that row's w pixels vertically r times in order, leaving black crop pixels black.
    - Preserve the original left panel, the grey separator, the orange bar, and the teal anchors.

    Key insight:
    - The largest object's crop simultaneously determines what vertical pattern is stamped and how far apart the staircase anchors must be, so the left prototype and the right scaffold constrain each other.
    """
    scene = _extract_input_scene(np.asarray(grid, dtype=int))
    return _render_from_scene(scene["grid"], scene["proto"], scene["sep"], scene["bar_cols"], scene["anchor_rows"])


def _path_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    r = c = 0
    mask[r, c] = True
    while r < height - 1 or c < width - 1:
        moves = []
        if r < height - 1:
            moves.append((r + 1, c))
        if c < width - 1:
            moves.append((r, c + 1))
        r, c = random.choice(moves)
        mask[r, c] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = int(rr) + dr, int(cc) + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier, "frontier unexpectedly empty while growing a connected mask"
        rr, cc = random.choice(tuple(frontier))
        mask[rr, cc] = True

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _largest_mask_is_good(mask: np.ndarray) -> bool:
    width = mask.shape[1]
    row_patterns = [tuple(int(v) for v in row) for row in mask]
    return len(set(row_patterns)) >= 2 and any(0 < int(row.sum()) < width for row in mask)


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _choose_largest_spec() -> tuple[int, int, int]:
    specs: list[tuple[int, int, int, int]] = []
    for h in range(3, 6):
        for w in range(2, 5):
            min_area = max(6, h + w - 1)
            max_area = min(10, h * w - 1)
            if min_area <= max_area:
                specs.append((h, w, min_area, max_area))
    h, w, area_lo, area_hi = random.choice(specs)
    return h, w, random.randint(area_lo, area_hi)


def _gen_largest_patch(color: int) -> tuple[np.ndarray, int, int]:
    for _ in range(120):
        h, w, area = _choose_largest_spec()
        mask = _path_mask(h, w, area)
        if _largest_mask_is_good(mask):
            return _mask_to_patch(mask, color), h, w
    raise RuntimeError("failed to generate a non-degenerate largest prototype")


def _gen_distractor_patch(color: int, max_area: int) -> np.ndarray:
    for _ in range(120):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        min_area = max(3, h + w - 2)
        area_hi = min(max_area, h * w)
        if min_area > area_hi:
            continue
        mask = _path_mask(h, w, random.randint(min_area, area_hi))
        return _mask_to_patch(mask, color)
    raise RuntimeError("failed to generate a distractor patch")


def _place_left_objects(height: int, width: int, patches: list[tuple[np.ndarray, int]]) -> np.ndarray | None:
    canvas = np.zeros((height, width), dtype=int)
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: patches[idx][1], reverse=True)
    for idx in order:
        patch, _area = patches[idx]
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, r0, c0, transparent=BG)
    return canvas


def _validate_generated_example(grid: np.ndarray) -> None:
    scene = _extract_input_scene(grid)
    out = main(grid)
    expected = _render_from_scene(scene["grid"], scene["proto"], scene["sep"], scene["bar_cols"], scene["anchor_rows"])
    assert np.array_equal(out, expected), "main() disagrees with the explicit scene rendering"
    assert not np.array_equal(out, grid), "the task should produce a visible change"

    proto = scene["proto"]
    h, w = proto.shape
    assert len(scene["bar_cols"]) == h
    assert w >= 2 and h >= 3
    assert all(scene["anchor_rows"][i - 1] - scene["anchor_rows"][i] == w for i in range(1, h))

    changed = np.argwhere(out != grid)
    assert len(changed) > 0
    allowed_cols = {scene["sep"] + 1 + bar_col for bar_col in scene["bar_cols"]}
    for r_raw, c_raw in changed:
        r, c = int(r_raw), int(c_raw)
        assert c in allowed_cols, "output changes must stay on the staircase columns"
        assert r < grid.shape[0] - 1
        assert int(grid[r, c]) == BG, "stretched content may only overwrite black scaffold cells"


def _gen_attempt() -> np.ndarray | None:
    largest_color = random.choice(LEFT_COLORS)
    largest_patch, proto_h, proto_w = _gen_largest_patch(largest_color)
    largest_area = int(np.count_nonzero(largest_patch))

    min_height = 2 + (proto_h - 1) * proto_w + 1
    grid_h = random.randint(max(18, min_height), 28)

    right_left_margin = random.randint(1, 3)
    right_right_margin = random.randint(1, 3)
    right_w = right_left_margin + proto_h + right_right_margin

    min_left_w = max(10, largest_patch.shape[1] + 4)
    max_left_w = 29 - right_w
    if min_left_w > max_left_w:
        return None
    left_w = random.randint(min_left_w, min(max_left_w, 16))

    object_count = random.randint(3, 5)
    colors = random.sample(LEFT_COLORS, object_count)
    if largest_color not in colors:
        colors[0] = largest_color
        random.shuffle(colors)
    largest_index = colors.index(largest_color)

    distractor_area_cap = largest_area - 1
    if distractor_area_cap < 3:
        return None

    patches: list[tuple[np.ndarray, int]] = []
    for idx, color in enumerate(colors):
        if idx == largest_index:
            patch = largest_patch
        else:
            patch = _gen_distractor_patch(color, distractor_area_cap)
        patches.append((patch, int(np.count_nonzero(patch))))

    left = _place_left_objects(grid_h, left_w, patches)
    if left is None:
        return None

    total_w = left_w + 1 + right_w
    grid = np.zeros((grid_h, total_w), dtype=int)
    grid[:, :left_w] = left
    grid[:, left_w] = SEPARATOR

    right = grid[:, left_w + 1 :]
    bar_start = right_left_margin
    right[-1, bar_start : bar_start + proto_h] = BAR
    for index in range(proto_h):
        anchor_row = grid_h - 2 - index * proto_w
        if anchor_row < 0:
            return None
        right[anchor_row, bar_start + index] = ANCHOR

    try:
        _validate_generated_example(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
