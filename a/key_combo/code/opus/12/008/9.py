from __future__ import annotations

import random

import numpy as np


BLACK = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
CORE_ATTEMPTS = 120
WALK_ATTEMPTS = 160
PARTITION_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_dicts(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BLACK):
        color_mask = g == color
        for r in range(h):
            for c in range(w):
                if not color_mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while stack:
                    cr, cc = stack.pop()
                    coords.append((cr, cc))
                    for nr, nc in _neighbors4(cr, cc):
                        if 0 <= nr < h and 0 <= nc < w and color_mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                components.append(
                    {
                        "color": color,
                        "coords": coords,
                        "bbox": _bbox(coords),
                        "area": len(coords),
                    }
                )
    return components


def _filled_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for comp in _component_dicts(grid):
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] == (r1 - r0 + 1) * (c1 - c0 + 1):
            rectangles.append(comp)
    return rectangles


def _clean_grid(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    rectangles = _filled_rectangles(grid)
    assert rectangles, "expected at least one filled rectangle"

    min_area = min(rect["area"] for rect in rectangles)
    removed = [rect for rect in rectangles if rect["area"] == min_area]

    out = np.asarray(grid, dtype=int).copy()
    for rect in removed:
        for r, c in rect["coords"]:
            out[r, c] = BLACK
    return out, removed


def _mirror_tile(grid: np.ndarray, repeat_count: int) -> np.ndarray:
    rows = []
    for block_r in range(int(repeat_count)):
        row_tiles = []
        row_base = np.flipud(grid) if block_r % 2 else grid
        for block_c in range(int(repeat_count)):
            tile = np.fliplr(row_base) if block_c % 2 else row_base
            row_tiles.append(tile)
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the smallest solid rectangles, then mirror-tile the cleaned grid by its remaining color count.

    Input structure:
    - The background is black (0).
    - A large irregular foreground pattern uses two or three non-black colors and stays 4-connected as a whole.
    - One or two separate monochrome filled rectangles act as distractors.

    Rule:
    - Find every monochrome non-background 4-connected component whose pixels exactly fill its bounding box.
    - Remove all filled rectangles with the smallest area by turning their cells black.
    - Count the distinct colors left in the cleaned grid, including black, and call that count N.
    - Tile the cleaned grid into an N x N block array where odd block rows are vertically flipped,
      odd block columns are horizontally flipped, and both flips apply together on odd/odd blocks.

    Key insight:
    - The removed rectangles may or may not contribute a unique color, so rectangle filtering changes the
      palette size before the mirror tiling scale is determined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    cleaned, _removed = _clean_grid(g)
    repeat_count = len(np.unique(cleaned))
    assert repeat_count > 0
    return _mirror_tile(cleaned, repeat_count)


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for a, b, c in zip(path, path[1:], path[2:]):
        d1 = (b[0] - a[0], b[1] - a[1])
        d2 = (c[0] - b[0], c[1] - b[1])
        if d1 != d2:
            turns += 1
    return turns


def _walk_with_length(h: int, w: int, length: int) -> list[tuple[int, int]] | None:
    for _ in range(WALK_ATTEMPTS):
        start = (random.randrange(h), random.randrange(w))
        path = [start]
        seen = {start}

        def dfs() -> bool:
            if len(path) == length:
                return True

            r, c = path[-1]
            candidates = [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen]
            random.shuffle(candidates)
            candidates.sort(
                key=lambda pos: sum(
                    0 <= rr < h and 0 <= cc < w and (rr, cc) not in seen
                    for rr, cc in _neighbors4(pos[0], pos[1])
                ),
                reverse=True,
            )
            for nxt in candidates:
                seen.add(nxt)
                path.append(nxt)
                if dfs():
                    return True
                path.pop()
                seen.remove(nxt)
            return False

        if dfs() and _path_turns(path) >= 2:
            return path
    return None


def _coords_fill_bbox(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _partition_path(path: list[tuple[int, int]], color_count: int) -> list[list[tuple[int, int]]] | None:
    total = len(path)
    for _ in range(PARTITION_ATTEMPTS):
        if color_count == 2:
            a = random.randint(3, total - 3)
            lengths = [a, total - a]
        else:
            a = random.randint(3, total - 6)
            b = random.randint(3, total - a - 3)
            lengths = [a, b, total - a - b]

        segments = []
        start = 0
        ok = True
        for seg_len in lengths:
            seg = path[start : start + seg_len]
            start += seg_len
            if _coords_fill_bbox(seg):
                ok = False
                break
            segments.append(seg)
        if ok:
            return segments
    return None


def _make_core_patch(max_h: int, max_w: int, color_count: int) -> tuple[np.ndarray, list[int]] | None:
    colors = random.sample(range(1, 10), color_count)
    for _ in range(CORE_ATTEMPTS):
        core_h = random.randint(4, max_h)
        core_w = random.randint(4, max_w)
        min_len = max(3 * color_count, core_h + core_w + 1)
        max_len = min(core_h * core_w - 1, 14)
        if min_len > max_len:
            continue
        length = random.randint(min_len, max_len)
        path = _walk_with_length(core_h, core_w, length)
        if path is None:
            continue
        segments = _partition_path(path, color_count)
        if segments is None:
            continue

        patch = np.zeros((core_h, core_w), dtype=int)
        for color, seg in zip(colors, segments):
            for r, c in seg:
                patch[r, c] = int(color)

        if len(_filled_rectangles(patch)) != 0:
            continue
        if np.array_equal(patch, np.fliplr(patch)) or np.array_equal(patch, np.flipud(patch)):
            continue
        return patch, colors
    return None


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = patch.shape
    H, W = canvas.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    fg = np.argwhere(patch != BLACK)
    if fg.size == 0:
        return False

    for dr, dc in fg:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if canvas[rr, cc] != BLACK:
            return False
        rr0 = max(0, rr - padding)
        rr1 = min(H, rr + padding + 1)
        cc0 = max(0, cc - padding)
        cc1 = min(W, cc + padding + 1)
        if np.any(canvas[rr0:rr1, cc0:cc1] != BLACK):
            return False
    return True


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _free_positions(canvas: np.ndarray, patch: np.ndarray, padding: int = 1) -> list[tuple[int, int]]:
    H, W = canvas.shape
    h, w = patch.shape
    positions = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if _can_place(canvas, patch, r0, c0, padding=padding):
                positions.append((r0, c0))
    return positions


def _rect_patch(shape: tuple[int, int], color: int) -> np.ndarray:
    return np.full(shape, int(color), dtype=int)


def _intended_removed_mask(rect_specs: list[tuple[np.ndarray, tuple[int, int]]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for patch, (r0, c0) in rect_specs:
        h, w = patch.shape
        mask[r0 : r0 + h, c0 : c0 + w] |= patch != BLACK
    return mask


def _validate_case(grid: np.ndarray, intended_removed: np.ndarray, unique_branch: bool, rect_count: int, core_color_count: int) -> bool:
    rectangles = _filled_rectangles(grid)
    if len(rectangles) != rect_count:
        return False
    min_area = min(rect["area"] for rect in rectangles)
    removed = [rect for rect in rectangles if rect["area"] == min_area]
    if len(removed) != rect_count:
        return False

    removed_mask = np.zeros(grid.shape, dtype=bool)
    for rect in removed:
        for r, c in rect["coords"]:
            removed_mask[r, c] = True
    if not np.array_equal(removed_mask, intended_removed):
        return False

    cleaned, _ = _clean_grid(grid)
    if len(_filled_rectangles(cleaned)) != 0:
        return False

    expected_colors = core_color_count + 1
    if len(np.unique(cleaned)) != expected_colors:
        return False

    input_palette = len(np.unique(grid))
    if unique_branch and input_palette != expected_colors + 1:
        return False
    if not unique_branch and input_palette != expected_colors:
        return False

    if np.array_equal(cleaned, np.fliplr(cleaned)) or np.array_equal(cleaned, np.flipud(cleaned)):
        return False

    mirrored = _mirror_tile(cleaned, expected_colors)
    if mirrored.shape[0] > 30 or mirrored.shape[1] > 30:
        return False
    if np.array_equal(mirrored, np.tile(cleaned, (expected_colors, expected_colors))):
        return False
    if not np.array_equal(main(grid), mirrored):
        return False
    return True


def _build_case() -> np.ndarray | None:
    rect_count = random.choice([1, 2])
    unique_branch = bool(random.getrandbits(1))
    core_color_count = random.choice([2, 3])

    if rect_count == 2:
        H = random.choice([6, 7])
        W = random.choice([6, 7])
        core_max_h = H - 2
        core_max_w = W - 2
        rect_shapes = [random.choice([(1, 2), (2, 1)]), random.choice([(1, 2), (2, 1)])]
    else:
        H = random.choice([5, 6, 7])
        W = random.choice([5, 6, 7])
        core_max_h = max(4, H - 1)
        core_max_w = max(4, W - 1)
        rect_shapes = [random.choice([(1, 2), (2, 1), (2, 2)])]

    if core_max_h < 4 or core_max_w < 4:
        return None

    core_data = _make_core_patch(min(core_max_h, 5), min(core_max_w, 5), core_color_count)
    if core_data is None:
        return None
    core_patch, core_colors = core_data

    canvas = np.zeros((H, W), dtype=int)
    core_positions = _free_positions(canvas, core_patch, padding=0)
    random.shuffle(core_positions)
    if not core_positions:
        return None

    placed = False
    rect_specs: list[tuple[np.ndarray, tuple[int, int]]] = []
    for core_r0, core_c0 in core_positions:
        canvas.fill(BLACK)
        _paste(canvas, core_patch, core_r0, core_c0)
        rect_specs.clear()

        unique_color = random.choice([c for c in range(1, 10) if c not in core_colors])
        rect_colors: list[int]
        if rect_count == 1:
            rect_colors = [unique_color if unique_branch else random.choice(core_colors)]
        else:
            shared_color = random.choice(core_colors)
            if unique_branch:
                rect_colors = [unique_color, shared_color]
            else:
                rect_colors = random.sample(core_colors, 2) if len(core_colors) >= 2 else [shared_color, shared_color]
            random.shuffle(rect_colors)

        ok = True
        for shape, color in zip(rect_shapes, rect_colors):
            patch = _rect_patch(shape, color)
            options = _free_positions(canvas, patch, padding=1)
            random.shuffle(options)
            if not options:
                ok = False
                break
            pos = options[0]
            _paste(canvas, patch, pos[0], pos[1])
            rect_specs.append((patch, pos))

        if ok:
            placed = True
            break

    if not placed:
        return None

    intended_removed = _intended_removed_mask(rect_specs, canvas.shape)
    if not _validate_case(canvas, intended_removed, unique_branch, rect_count, core_color_count):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
