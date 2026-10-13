from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
ORANGE = 7
COLOR_POOL = [1, 2, 3, 4, 5, 6, 8, 9]
MODES = ("ab", "bc", "both")
MODE_WEIGHTS = (3, 3, 2)
GEN_TRIES = 500


@dataclass(frozen=True)
class Box:
    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height - 1

    @property
    def right(self) -> int:
        return self.left + self.width - 1

    @property
    def rows(self) -> range:
        return range(self.top, self.bottom + 1)

    @property
    def cols(self) -> range:
        return range(self.left, self.right + 1)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _flip_color_masks_vertical(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = np.full_like(g, bg)
    for color in _colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        r0, _c0, r1, _c1 = _bbox_coords(coords)
        rr = r0 + r1 - coords[:, 0]
        cc = coords[:, 1]
        out[rr, cc] = int(color)
    return out


def _adjacent_pairs(grid: np.ndarray, bg: int = BG) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    pairs: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            a = int(g[r, c])
            if a == bg:
                continue
            for rr, cc in ((r + 1, c), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w:
                    b = int(g[rr, cc])
                    if b != bg and b != a:
                        pairs.add(tuple(sorted((a, b))))
    return pairs


def _shared_black_cells_for_pair(
    grid: np.ndarray, pair: tuple[int, int], bg: int = BG
) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    a, b = map(int, pair)
    cells: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != bg:
                continue
            neigh = set()
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w:
                    color = int(g[rr, cc])
                    if color != bg:
                        neigh.add(color)
            if a in neigh and b in neigh:
                cells.add((r, c))
    return cells


def _orange_targets(grid: np.ndarray, bg: int = BG) -> set[tuple[int, int]]:
    targets: set[tuple[int, int]] = set()
    for pair in _adjacent_pairs(grid, bg=bg):
        targets |= _shared_black_cells_for_pair(grid, pair, bg=bg)
    return targets


def _has_cross_color_adjacency(grid: np.ndarray, bg: int = BG) -> bool:
    return bool(_adjacent_pairs(grid, bg=bg))


def _box_mask(cells: set[tuple[int, int]], box: Box) -> np.ndarray:
    mask = np.zeros((box.height, box.width), dtype=bool)
    for r, c in cells:
        if box.top <= r <= box.bottom and box.left <= c <= box.right:
            mask[r - box.top, c - box.left] = True
    return mask


def _is_box_vertically_symmetric(cells: set[tuple[int, int]], box: Box) -> bool:
    mask = _box_mask(cells, box)
    return np.array_equal(mask, np.flipud(mask))


def _active_pairs_for_mode(mode: str) -> set[tuple[str, str]]:
    if mode == "ab":
        return {("a", "b")}
    if mode == "bc":
        return {("b", "c")}
    return {("a", "b"), ("b", "c")}


def _gap_for_pair(mode: str, pair_name: str) -> int:
    if pair_name == "ab":
        return 0 if mode in {"ab", "both"} else 1
    return 0 if mode in {"bc", "both"} else 1


def _make_boxes(mode: str) -> tuple[dict[str, Box], int, int]:
    widths = {name: random.randint(3, 4) for name in ("a", "b", "c")}
    heights = {name: random.randint(5, 6) for name in ("a", "b", "c")}

    gap_ab = _gap_for_pair(mode, "ab")
    gap_bc = _gap_for_pair(mode, "bc")

    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    x_a = left_margin
    x_b = x_a + widths["a"] + gap_ab
    x_c = x_b + widths["b"] + gap_bc
    width = x_c + widths["c"] + right_margin

    pattern_options = {
        "ab": [(0, 2, 1), (2, 0, 1)],
        "bc": [(1, 0, 2), (1, 2, 0)],
        "both": [(0, 2, 0), (2, 0, 2)],
    }
    da, db, dc = random.choice(pattern_options[mode])
    max_delta = max(da, db, dc)
    height = random.randint(max(13, max(heights.values()) + max_delta + 3), 18)
    row0_max = height - max(
        heights["a"] + da,
        heights["b"] + db,
        heights["c"] + dc,
    ) - 1
    row0 = random.randint(1, row0_max)

    boxes = {
        "a": Box(row0 + da, x_a, heights["a"], widths["a"]),
        "b": Box(row0 + db, x_b, heights["b"], widths["b"]),
        "c": Box(row0 + dc, x_c, heights["c"], widths["c"]),
    }
    return boxes, height, width


def _overlap_rows(box1: Box, box2: Box, *, need_two_rows: bool) -> list[int]:
    start = max(box1.top, box2.top)
    end = min(box1.bottom, box2.bottom)
    if need_two_rows:
        end -= 1
    if start > end:
        return []
    return list(range(start, end + 1))


def _anchor_cols(name: str, width: int) -> tuple[int, int]:
    if name == "a":
        options = [0, min(1, width - 1)]
    elif name == "b":
        mid = width // 2
        options = [mid]
        if width >= 4:
            options.append(mid - 1)
    else:
        options = [max(0, width - 2), width - 1]
    top_col = random.choice(options)
    bottom_col = random.choice(options)
    if width >= 3 and top_col == bottom_col and random.random() < 0.6:
        alt = [c for c in options if c != top_col]
        if alt:
            bottom_col = random.choice(alt)
    return int(top_col), int(bottom_col)


def _add_anchors(
    seeds: dict[str, set[tuple[int, int]]], boxes: dict[str, Box]
) -> None:
    for name, box in boxes.items():
        top_col, bottom_col = _anchor_cols(name, box.width)
        seeds[name].add((box.top, box.left + top_col))
        seeds[name].add((box.bottom, box.left + bottom_col))


def _add_active_motif(
    left_name: str,
    right_name: str,
    seeds: dict[str, set[tuple[int, int]]],
    reserved_black: set[tuple[int, int]],
    boxes: dict[str, Box],
) -> None:
    left_box = boxes[left_name]
    right_box = boxes[right_name]
    row = random.choice(_overlap_rows(left_box, right_box, need_two_rows=True))
    if random.random() < 0.5:
        seeds[left_name].add((row, left_box.right))
        seeds[right_name].add((row, right_box.left))
        seeds[right_name].add((row + 1, right_box.left))
        reserved_black.add((row + 1, left_box.right))
    else:
        seeds[left_name].add((row, left_box.right))
        seeds[right_name].add((row, right_box.left))
        seeds[left_name].add((row + 1, left_box.right))
        reserved_black.add((row + 1, right_box.left))


def _add_decoy_gap_motif(
    left_name: str,
    right_name: str,
    seeds: dict[str, set[tuple[int, int]]],
    decoy_black: set[tuple[int, int]],
    boxes: dict[str, Box],
    avoid_rows: set[int],
) -> None:
    left_box = boxes[left_name]
    right_box = boxes[right_name]
    gap_col = left_box.right + 1
    rows = _overlap_rows(left_box, right_box, need_two_rows=False)
    options = [r for r in rows if r not in avoid_rows] or rows
    row = random.choice(options)
    seeds[left_name].add((row, left_box.right))
    seeds[right_name].add((row, right_box.left))
    decoy_black.add((row, gap_col))


def _protected_boundary_columns(mode: str, boxes: dict[str, Box]) -> dict[str, set[int]]:
    protected = {name: set() for name in boxes}
    protected["a"].add(boxes["a"].right)
    protected["b"].add(boxes["b"].left)
    protected["b"].add(boxes["b"].right)
    protected["c"].add(boxes["c"].left)
    return protected


def _grow_color_cells(
    name: str,
    box: Box,
    seeds: set[tuple[int, int]],
    reserved_black: set[tuple[int, int]],
    protected_cols: set[int],
) -> set[tuple[int, int]] | None:
    allowed: set[tuple[int, int]] = set()
    for r in box.rows:
        for c in box.cols:
            if (r, c) in reserved_black:
                continue
            if c in protected_cols and (r, c) not in seeds:
                continue
            allowed.add((r, c))
    if not seeds <= allowed:
        return None

    target_low = max(6, len(seeds))
    target_high = min(10, len(allowed))
    if target_low > target_high:
        return None
    target = random.randint(target_low, target_high)

    cells = set(seeds)
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for rr, cc in _neighbors4(r, c):
                if (rr, cc) in allowed and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            return None

        def _weight(cell: tuple[int, int]) -> float:
            rr, cc = cell
            center_r = box.top + (box.height - 1) / 2
            center_c = box.left + (box.width - 1) / 2
            dist = abs(rr - center_r) + abs(cc - center_c)
            return 1.0 / (1.0 + dist)

        candidates = list(frontier)
        weights = [_weight(cell) for cell in candidates]
        chosen = random.choices(candidates, weights=weights, k=1)[0]
        cells.add(chosen)

    if _is_box_vertically_symmetric(cells, box):
        return None
    return cells


def _build_foreground_scene() -> tuple[np.ndarray, dict[str, int], str]:
    mode = random.choices(MODES, weights=MODE_WEIGHTS, k=1)[0]
    boxes, height, width = _make_boxes(mode)
    colors = dict(zip(("a", "b", "c"), random.sample(COLOR_POOL, 3)))

    seeds = {name: set() for name in boxes}
    reserved_black: set[tuple[int, int]] = set()
    decoy_black: set[tuple[int, int]] = set()
    used_rows: set[int] = set()

    active_pairs = _active_pairs_for_mode(mode)
    if ("a", "b") in active_pairs:
        _add_active_motif("a", "b", seeds, reserved_black, boxes)
    if ("b", "c") in active_pairs:
        _add_active_motif("b", "c", seeds, reserved_black, boxes)

    for name in ("a", "b", "c"):
        for r, _c in seeds[name]:
            used_rows.add(r)

    if mode == "ab":
        _add_decoy_gap_motif("b", "c", seeds, decoy_black, boxes, used_rows)
    elif mode == "bc":
        _add_decoy_gap_motif("a", "b", seeds, decoy_black, boxes, used_rows)

    _add_anchors(seeds, boxes)
    protected_cols = _protected_boundary_columns(mode, boxes)

    color_cells: dict[str, set[tuple[int, int]]] = {}
    for name, box in boxes.items():
        own_reserved = {
            cell
            for cell in reserved_black
            if box.top <= cell[0] <= box.bottom and box.left <= cell[1] <= box.right
        }
        cells = _grow_color_cells(name, box, seeds[name], own_reserved, protected_cols[name])
        if cells is None:
            raise ValueError("shape growth failed")
        color_cells[name] = cells

    scene = np.zeros((height, width), dtype=int)
    for name, cells in color_cells.items():
        color = colors[name]
        for r, c in cells:
            if scene[r, c] != BG:
                raise ValueError("overlapping colors")
            scene[r, c] = color
    return scene, colors, mode


def _validate_scene(scene: np.ndarray, colors: dict[str, int], mode: str) -> bool:
    boxes_ok = True
    for color in colors.values():
        area = int(np.count_nonzero(scene == color))
        if not (6 <= area <= 12):
            return False
        coords = np.argwhere(scene == color)
        bbox = _bbox_coords(coords)
        box = Box(bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)
        mask_cells = {(int(r), int(c)) for r, c in coords.tolist()}
        boxes_ok &= not _is_box_vertically_symmetric(mask_cells, box)
    if not boxes_ok:
        return False

    color_to_name = {v: k for k, v in colors.items()}
    triggered_names = {
        tuple(sorted((color_to_name[a], color_to_name[b])))
        for a, b in _adjacent_pairs(scene)
    }
    expected_active = _active_pairs_for_mode(mode)
    if triggered_names != expected_active:
        return False

    orange_targets = _orange_targets(scene)
    if not (1 <= len(orange_targets) <= 6):
        return False

    if mode == "ab":
        pair = tuple(sorted((colors["b"], colors["c"])))
        decoy = _shared_black_cells_for_pair(scene, pair)
        if not decoy or decoy & orange_targets:
            return False
    if mode == "bc":
        pair = tuple(sorted((colors["a"], colors["b"])))
        decoy = _shared_black_cells_for_pair(scene, pair)
        if not decoy or decoy & orange_targets:
            return False

    input_grid = _flip_color_masks_vertical(scene)
    if np.array_equal(input_grid, scene):
        return False
    if _has_cross_color_adjacency(input_grid):
        return False

    output = main(input_grid)
    expected = scene.copy()
    for r, c in orange_targets:
        expected[r, c] = ORANGE
    return np.array_equal(output, expected)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each foreground color vertically inside its own bbox, then mark the black cells that become jointly adjacent to newly adjacent colors.

    Input structure:
    - The grid has a black background and exactly three non-black foreground colors.
    - Each color forms its own monochrome mask and different colors are separated in the input.
    - A color may be split into multiple 4-connected pieces, but its whole color layer shares one bounding box.

    Rule:
    - For each non-background color independently, find that color's full bbox and vertically reflect every pixel within that bbox.
    - After all colors have been reflected, collect every unordered pair of different colors that now has at least one orthogonally adjacent pixel pair.
    - For each such color pair, paint orange every black cell that is orthogonally adjacent to at least one pixel of each color; never overwrite existing non-black pixels.

    Key insight:
    - The orange marks are not triggered by a dedicated marker color: they emerge only after the per-color mirror step changes which color layers touch.
    """
    g = np.asarray(grid, dtype=int)
    mirrored = _flip_color_masks_vertical(g)
    out = mirrored.copy()
    for r, c in _orange_targets(mirrored):
        if int(out[r, c]) == BG:
            out[r, c] = ORANGE
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        try:
            scene, colors, mode = _build_foreground_scene()
        except ValueError:
            continue
        if _validate_scene(scene, colors, mode):
            return _flip_color_masks_vertical(scene)
    raise RuntimeError("failed to generate a valid puzzle instance")
