import random
from collections import Counter

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
FOREGROUND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
RECT_COUNT_RANGE = (2, 4)
RECT_H_RANGE = (5, 8)
RECT_W_RANGE = (5, 8)
INNER_AREA_MIN = 2
LAYOUT_MARGIN = 1
LAYOUT_GAP = 2
LEFT_MAX_W = 20
RIGHT_MAX_W = 9
LOCAL_MAX_H = 24


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    tight = _tight_crop(mask)
    return tuple(map(int, tight.shape)), tight.tobytes()


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, _w = grid.shape
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        coords = np.argwhere(grid == color)
        if coords.shape[0] == h and np.all(coords[:, 1] == coords[0, 1]):
            candidates.append((color, int(coords[0, 1])))
    assert len(candidates) == 1, f"expected one full-height separator column, found {candidates}"
    return candidates[0]


def _template_records(left: np.ndarray, separator_color: int) -> list[dict]:
    records = []
    for color in np.unique(left):
        color = int(color)
        if color in (BG, separator_color):
            continue
        for coords in _components(left == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            patch = left[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros_like(patch, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(patch[border] == color):
                continue
            hole_mask = patch == BG
            if not np.any(hole_mask):
                continue
            records.append(
                {
                    "color": color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "hole_mask": hole_mask.copy(),
                    "key": _mask_key(hole_mask),
                }
            )
    assert records, "no left-side templates found"
    return records


def _piece_counters(right: np.ndarray) -> dict[int, Counter]:
    counters: dict[int, Counter] = {}
    for color in np.unique(right):
        color = int(color)
        if color == BG:
            continue
        counter = Counter()
        for coords in _components(right == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            counter[_mask_key(mask)] += 1
        counters[color] = counter
    assert counters, "no right-side pieces found"
    return counters


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each damaged rectangle by matching its hole shape to the same-colored loose piece.

    Input structure:
    - Black background with a single full-height gray separator column.
    - Left of the separator are several non-overlapping monochrome rectangles whose borders are intact but whose interiors contain black missing cells.
    - Right of the separator are scattered 4-connected monochrome pieces; each piece shares its color with one left-side rectangle.

    Rule:
    - Detect every left-side rectangle from its solid rectangular border and extract the full shape of its black hole cells.
    - On the right side, find a piece of the same color whose shape matches that hole exactly under translation only.
    - Fill the rectangle's hole with its own color, and clear every right-side piece back to black.
    - Keep the separator column and all existing rectangle pixels unchanged.

    Key insight:
    - Color alone is insufficient when multiple rectangles share one color; the missing-cell shape is the unique template that identifies which loose piece belongs to which rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC-sized grid"

    separator_color, separator_col = _find_separator(g)
    left = g[:, :separator_col]
    right = g[:, separator_col + 1 :]

    templates = _template_records(left, separator_color)
    pieces = _piece_counters(right)

    out = g.copy()
    out[:, separator_col + 1 :] = BG

    for record in templates:
        color = int(record["color"])
        counter = pieces.get(color)
        assert counter is not None and counter[record["key"]] > 0, "missing matching piece"
        counter[record["key"]] -= 1
        if counter[record["key"]] == 0:
            del counter[record["key"]]
        hole_mask = record["hole_mask"]
        r0 = int(record["r0"])
        c0 = int(record["c0"])
        patch = out[r0 : r0 + hole_mask.shape[0], c0 : c0 + hole_mask.shape[1]]
        patch[hole_mask] = color

    for color, counter in pieces.items():
        assert not counter, f"unused piece(s) remain for color {color}"

    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    area_max = h * w
    target_area = random.randint(INNER_AREA_MIN, area_max)
    cells = {(random.randrange(h), random.randrange(w))}
    frontier = set()
    while len(cells) < target_area:
        if not frontier:
            frontier = {
                (nr, nc)
                for r, c in cells
                for dr, dc in NEIGHBORS4
                for nr, nc in [(r + dr, c + dc)]
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
            }
        chosen = random.choice(tuple(frontier))
        frontier.remove(chosen)
        cells.add(chosen)
        for dr, dc in NEIGHBORS4:
            nr, nc = chosen[0] + dr, chosen[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_rectangle_and_piece(color: int, used_keys: set[tuple[tuple[int, int], bytes]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(200):
        rect_h = random.randint(*RECT_H_RANGE)
        rect_w = random.randint(*RECT_W_RANGE)
        inner_h = rect_h - 2
        inner_w = rect_w - 2
        hole_inner = _random_connected_mask(inner_h, inner_w)
        hole_mask = np.zeros((rect_h, rect_w), dtype=bool)
        hole_mask[1:-1, 1:-1] = hole_inner
        key = _mask_key(hole_mask)
        if key in used_keys:
            continue
        used_keys.add(key)

        rect_patch = np.full((rect_h, rect_w), color, dtype=int)
        rect_patch[hole_mask] = BG
        filled_patch = np.full((rect_h, rect_w), color, dtype=int)
        piece_mask = _tight_crop(hole_mask)
        piece_patch = np.zeros(piece_mask.shape, dtype=int)
        piece_patch[piece_mask] = color
        return rect_patch, filled_patch, piece_patch
    raise RuntimeError("failed to generate a unique hole shape")


def _arrange_patches(patches: list[np.ndarray], max_w: int, max_h: int) -> tuple[int, int, dict[int, tuple[int, int]]] | None:
    count = len(patches)
    column_options = [1] if count == 1 else [1, 2]
    random.shuffle(column_options)
    for cols in column_options:
        for _ in range(80):
            order = list(range(count))
            random.shuffle(order)
            if cols == 1:
                groups = [order]
            else:
                split = random.randint(1, count - 1)
                groups = [order[:split], order[split:]]
                random.shuffle(groups)
            if any(not group for group in groups):
                continue

            col_widths = [max(int(patches[i].shape[1]) for i in group) for group in groups]
            col_heights = [
                sum(int(patches[i].shape[0]) for i in group) + LAYOUT_GAP * (len(group) - 1)
                for group in groups
            ]
            width = 2 * LAYOUT_MARGIN + sum(col_widths) + LAYOUT_GAP * (cols - 1)
            height = 2 * LAYOUT_MARGIN + max(col_heights)
            if width > max_w or height > max_h:
                continue

            positions: dict[int, tuple[int, int]] = {}
            x = LAYOUT_MARGIN
            for group, col_w, col_h in zip(groups, col_widths, col_heights):
                y = LAYOUT_MARGIN + random.randint(0, max(0, height - 2 * LAYOUT_MARGIN - col_h))
                for idx in group:
                    patch = patches[idx]
                    ph, pw = map(int, patch.shape)
                    px = x + random.randint(0, col_w - pw)
                    positions[idx] = (y, px)
                    y += ph + LAYOUT_GAP
                x += col_w + LAYOUT_GAP
            return height, width, positions
    return None


def _render_layout(height: int, width: int, patches: list[np.ndarray], positions: dict[int, tuple[int, int]]) -> np.ndarray:
    canvas = np.zeros((height, width), dtype=int)
    for idx, patch in enumerate(patches):
        r0, c0 = positions[idx]
        ph, pw = map(int, patch.shape)
        sub = canvas[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != BG
        assert not np.any(sub[mask] != BG), "patches must not overlap"
        sub[mask] = patch[mask]
    return canvas


def _sample_colors(count: int) -> list[int]:
    if count >= 2 and random.random() < 0.5:
        repeat_count = 2 if count == 2 else random.choice([2, min(3, count)])
        picked = random.sample(FOREGROUND_COLORS, count - repeat_count + 1)
        colors = [picked[0]] * repeat_count + picked[1:]
        random.shuffle(colors)
        return colors
    return random.sample(FOREGROUND_COLORS, count)


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.randint(*RECT_COUNT_RANGE)
    colors = _sample_colors(rect_count)

    rect_patches: list[np.ndarray] = []
    filled_patches: list[np.ndarray] = []
    piece_patches: list[np.ndarray] = []
    used_keys_by_color: dict[int, set[tuple[tuple[int, int], bytes]]] = {}

    for color in colors:
        used = used_keys_by_color.setdefault(int(color), set())
        rect_patch, filled_patch, piece_patch = _make_rectangle_and_piece(int(color), used)
        rect_patches.append(rect_patch)
        filled_patches.append(filled_patch)
        piece_patches.append(piece_patch)

    left_layout = _arrange_patches(rect_patches, max_w=LEFT_MAX_W, max_h=LOCAL_MAX_H)
    if left_layout is None:
        return None
    left_h, left_w, left_pos = left_layout

    right_layout = _arrange_patches(piece_patches, max_w=RIGHT_MAX_W, max_h=LOCAL_MAX_H)
    if right_layout is None:
        return None
    right_h, right_w, right_pos = right_layout

    total_w = left_w + 1 + right_w
    if total_w > 30:
        return None

    base_h = max(left_h, right_h)
    total_h = base_h + random.randint(0, min(5, 30 - base_h))
    if total_h > 30:
        return None

    left_row = random.randint(0, total_h - left_h)
    right_row = random.randint(0, total_h - right_h)

    left_input = _render_layout(left_h, left_w, rect_patches, left_pos)
    left_output = _render_layout(left_h, left_w, filled_patches, left_pos)
    right_input = _render_layout(right_h, right_w, piece_patches, right_pos)

    grid = np.zeros((total_h, total_w), dtype=int)
    expected = np.zeros((total_h, total_w), dtype=int)
    grid[:, left_w] = SEPARATOR_COLOR
    expected[:, left_w] = SEPARATOR_COLOR

    grid[left_row : left_row + left_h, :left_w] = left_input
    expected[left_row : left_row + left_h, :left_w] = left_output
    grid[right_row : right_row + right_h, left_w + 1 :] = right_input

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
