from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sprites: list[dict[str, object]] = []
    for coords in _components(g != bg):
        rows = np.array([r for r, _c in coords], dtype=int)
        cols = np.array([c for _r, c in coords], dtype=int)
        values = g[rows, cols]
        uniq, counts = np.unique(values, return_counts=True)
        label_colors = uniq[counts == 1]
        assert len(label_colors) == 1, "each sprite must contain exactly one label pixel"
        assert len(uniq) == 2, "each sprite must be monochrome except for one label pixel"
        label_color = int(label_colors[0])
        label_index = int(np.flatnonzero(values == label_color)[0])
        label_pos = (int(rows[label_index]), int(cols[label_index]))

        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[rows - r0, cols - c0] = True
        sprites.append(
            {
                "label_color": label_color,
                "label_pos": label_pos,
                "mask": mask,
            }
        )
    assert sprites, "expected at least one sprite"
    return bg, sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered sprites by their embedded label-pixel color and merge masks inside each group.
    """
    bg, sprites = _extract_sprites(np.asarray(grid, dtype=int))

    box_h = max(int(sprite["mask"].shape[0]) for sprite in sprites)
    box_w = max(int(sprite["mask"].shape[1]) for sprite in sprites)

    merged: dict[int, np.ndarray] = {}
    first_seen: dict[int, tuple[int, int]] = {}

    for sprite in sprites:
        label_color = int(sprite["label_color"])
        label_pos = tuple(int(v) for v in sprite["label_pos"])
        mask = np.asarray(sprite["mask"], dtype=bool)
        padded = np.zeros((box_h, box_w), dtype=bool)
        padded[: mask.shape[0], : mask.shape[1]] = mask
        if label_color not in merged:
            merged[label_color] = padded
            first_seen[label_color] = label_pos
        else:
            merged[label_color] |= padded
            if label_pos < first_seen[label_color]:
                first_seen[label_color] = label_pos

    ordered_colors = sorted(merged, key=lambda color: first_seen[color])
    out_w = len(ordered_colors) * box_w + max(0, len(ordered_colors) - 1)
    out = np.full((box_h, out_w), bg, dtype=int)

    col = 0
    for color in ordered_colors:
        group_mask = merged[color]
        out[:box_h, col : col + box_w][group_mask] = int(color)
        col += box_w + 1

    return out


# --------------- generator ---------------

COLORS = tuple(range(10))
GRID_LIMIT = 30
BOX_H_RANGE = (4, 6)
BOX_W_RANGE = (4, 6)
GROUP_COUNT_CHOICES = (2, 3)
MAX_EXTRA_SPRITES = 3
MAX_ATTEMPTS = 400


def _touches_all_sides(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    return (
        any(r == 0 for r, _c in cells)
        and any(r == h - 1 for r, _c in cells)
        and any(c == 0 for _r, c in cells)
        and any(c == w - 1 for _r, c in cells)
    )


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    path = [(r, c)]
    if random.random() < 0.5:
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
    else:
        while c != tc:
            c += 1 if tc > c else -1
            path.append((r, c))
        while r != tr:
            r += 1 if tr > r else -1
            path.append((r, c))
    return path


def _shape_from_cells(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _grow_shape(h: int, w: int) -> set[tuple[int, int]]:
    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    order = anchors[:]
    random.shuffle(order)
    cells: set[tuple[int, int]] = {order[0]}
    current = order[0]
    for nxt in order[1:]:
        cells.update(_manhattan_path(current, nxt))
        current = nxt

    area = h * w
    min_target = max(len(cells), max(5, (area + 2) // 3))
    max_target = max(min_target, max(6, (4 * area) // 5))
    target = random.randint(min_target, min(area - 1, max_target))

    while len(cells) < target:
        frontier = {
            (nr, nc)
            for r, c in cells
            for nr, nc in _neighbors4(r, c)
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells
        }
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    return cells


def _carve_shape(h: int, w: int) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(h) for c in range(w)}
    area = h * w
    target = random.randint(max(6, (3 * area) // 5), area - 1)
    attempts = 0
    while len(cells) > target and attempts < 600:
        attempts += 1
        cell = random.choice(tuple(cells))
        reduced = set(cells)
        reduced.remove(cell)
        if _touches_all_sides(reduced, h, w) and _is_connected(reduced):
            cells = reduced
    return cells


def _random_shape(h: int, w: int) -> np.ndarray:
    for _ in range(200):
        cells = _grow_shape(h, w) if random.random() < 0.5 else _carve_shape(h, w)
        if len(cells) < 3:
            continue
        if not _touches_all_sides(cells, h, w):
            continue
        if not _is_connected(cells):
            continue
        return _shape_from_cells(cells, h, w)
    raise RuntimeError("failed to generate a valid sprite mask")


def _mask_key(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8).tobytes()


def _choose_label_cell(mask: np.ndarray) -> tuple[int, int]:
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    cell_set = set(cells)
    interior = [
        (r, c)
        for r, c in cells
        if sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c)) >= 2
    ]
    edge = [cell for cell in cells if cell not in interior]
    if random.random() < 0.5 and interior:
        return random.choice(interior)
    if edge:
        return random.choice(edge)
    return random.choice(cells)


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _group_sizes(group_count: int) -> list[int]:
    sizes = [1] * group_count
    for _ in range(random.randint(1, MAX_EXTRA_SPRITES)):
        sizes[random.randrange(group_count)] += 1
    random.shuffle(sizes)
    return sizes


def _expected_output(specs: list[dict[str, object]], box_h: int, box_w: int, bg: int) -> np.ndarray:
    merged: dict[int, np.ndarray] = {}
    first_seen: dict[int, tuple[int, int]] = {}
    for spec in specs:
        color = int(spec["label_color"])
        pos = tuple(int(v) for v in spec["label_abs"])
        mask = np.asarray(spec["mask"], dtype=bool)
        if color not in merged:
            merged[color] = mask.copy()
            first_seen[color] = pos
        else:
            merged[color] |= mask
            if pos < first_seen[color]:
                first_seen[color] = pos

    ordered = sorted(merged, key=lambda color: first_seen[color])
    out = np.full((box_h, len(ordered) * box_w + len(ordered) - 1), bg, dtype=int)
    col = 0
    for color in ordered:
        out[:, col : col + box_w][merged[color]] = color
        col += box_w + 1
    return out


def _generate_attempt() -> np.ndarray | None:
    box_h = random.randint(*BOX_H_RANGE)
    box_w = random.randint(*BOX_W_RANGE)
    group_count = random.choice(GROUP_COUNT_CHOICES)
    group_sizes = _group_sizes(group_count)
    total_sprites = sum(group_sizes)

    bg = random.choice(COLORS)
    palette = [color for color in COLORS if color != bg]
    label_colors = random.sample(palette, group_count)
    body_choices = [color for color in palette if color not in label_colors]
    if not body_choices:
        return None

    sprite_specs: list[dict[str, object]] = []
    for label_color, count in zip(label_colors, group_sizes, strict=False):
        seen_masks: set[bytes] = set()
        for _ in range(count):
            for _attempt in range(200):
                mask = _random_shape(box_h, box_w)
                key = _mask_key(mask)
                if key not in seen_masks or len(seen_masks) + 1 >= count:
                    seen_masks.add(key)
                    break
            else:
                return None
            lr, lc = _choose_label_cell(mask)
            sprite_specs.append(
                {
                    "label_color": int(label_color),
                    "body_color": int(random.choice(body_choices)),
                    "mask": mask,
                    "label_rc": (int(lr), int(lc)),
                }
            )

    random.shuffle(sprite_specs)

    rows_hint = random.randint(1, min(4, total_sprites))
    cols_hint = (total_sprites + rows_hint - 1) // rows_hint
    min_h = rows_hint * box_h + (rows_hint + 1)
    min_w = cols_hint * box_w + (cols_hint + 1)
    if min_h > GRID_LIMIT or min_w > GRID_LIMIT:
        return None
    grid_h = random.randint(min_h, min(GRID_LIMIT, min_h + 6))
    grid_w = random.randint(min_w, min(GRID_LIMIT, min_w + 8))

    placed: list[tuple[int, int, int, int]] = []
    for spec in sprite_specs:
        placed_ok = False
        for _ in range(500):
            r0 = random.randint(1, grid_h - box_h - 1)
            c0 = random.randint(1, grid_w - box_w - 1)
            expanded = (r0 - 1, c0 - 1, r0 + box_h, c0 + box_w)
            if any(_rectangles_overlap(expanded, other) for other in placed):
                continue
            spec["top_left"] = (r0, c0)
            spec["label_abs"] = (r0 + int(spec["label_rc"][0]), c0 + int(spec["label_rc"][1]))
            placed.append(expanded)
            placed_ok = True
            break
        if not placed_ok:
            return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for spec in sprite_specs:
        mask = np.asarray(spec["mask"], dtype=bool)
        r0, c0 = spec["top_left"]
        body_color = int(spec["body_color"])
        grid[r0 : r0 + box_h, c0 : c0 + box_w][mask] = body_color
        lr, lc = spec["label_rc"]
        grid[r0 + lr, c0 + lc] = int(spec["label_color"])

    if int(np.sum(grid == bg)) <= int(np.sum(grid != bg)):
        return None

    components = _components(grid != bg)
    if len(components) != total_sprites:
        return None
    for coords in components:
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        if max(rows) - min(rows) + 1 != box_h or max(cols) - min(cols) + 1 != box_w:
            return None
        values = np.array([grid[r, c] for r, c in coords], dtype=int)
        uniq, counts = np.unique(values, return_counts=True)
        if len(uniq) != 2 or np.sum(counts == 1) != 1:
            return None

    expected = _expected_output(sprite_specs, box_h, box_w, bg)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
