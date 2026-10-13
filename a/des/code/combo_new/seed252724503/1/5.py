import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
OBJECT_RETRIES = 120
PLACEMENT_RETRIES = 120


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            colors = set()
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                colors.add(int(grid[rr, cc]))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(
                {
                    "coords": np.array(coords, dtype=int),
                    "bbox": (r0, c0, r1, c1),
                    "colors": colors,
                }
            )
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == int(mask.sum())


def _priority_bar(last_row: np.ndarray) -> list[int]:
    colors = [int(v) for v in last_row if int(v) != BG]
    assert len(colors) >= 2, "priority bar must contain at least two colors"
    assert len(colors) == len(set(colors)), "priority bar colors must be distinct"
    return colors


def _has_extra_diagonal_axis(mask: np.ndarray) -> bool:
    if mask.shape[0] != mask.shape[1]:
        return False
    return np.array_equal(mask, mask.T) or np.array_equal(mask, np.fliplr(mask).T)


def _winner(a: int, b: int, priority_rank: dict[int, int]) -> int:
    if a == BG:
        return b
    if b == BG:
        return a
    if a == b:
        return a
    assert a in priority_rank and b in priority_rank
    return a if priority_rank[a] < priority_rank[b] else b


def _fold_component(grid: np.ndarray, component: dict, priority_rank: dict[int, int]) -> dict[tuple[int, int], int]:
    r0, c0, r1, c1 = map(int, component["bbox"])
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = patch != BG
    vertical = np.array_equal(mask, mask[:, ::-1])
    horizontal = np.array_equal(mask, mask[::-1, :])
    assert vertical ^ horizontal, "each object must have exactly one axis"

    placed: dict[tuple[int, int], int] = {}
    h, w = patch.shape
    if vertical:
        for rr in range(h):
            for cc in range((w + 1) // 2):
                left = int(patch[rr, cc])
                right = int(patch[rr, w - 1 - cc])
                if left == BG and right == BG:
                    continue
                assert left != BG and right != BG
                placed[(r0 + rr, c0 + cc)] = _winner(left, right, priority_rank)
    else:
        for rr in range((h + 1) // 2):
            for cc in range(w):
                top = int(patch[rr, cc])
                bottom = int(patch[h - 1 - rr, cc])
                if top == BG and bottom == BG:
                    continue
                assert top != BG and bottom != BG
                placed[(r0 + rr, c0 + cc)] = _winner(top, bottom, priority_rank)
    return placed


def main(grid: np.ndarray) -> np.ndarray:
    """Fold every multicolor object across its unique axis and keep the winning overlap colors.

    Input structure:
    - The background is black.
    - Above the bottom row are several separated 4-connected multicolor objects.
    - Each object has exactly one horizontal or vertical reflection axis when colors are ignored, and the bottom row contains a distinct-color priority bar.

    Rule:
    - Read the bottom row from left to right as a high-to-low color priority order.
    - For each object, detect whether its binary shape is symmetric left-right or up-down inside its tight bounding box.
    - Fold the right half onto the left half for vertical symmetry, or the bottom half onto the top half for horizontal symmetry; when mirrored colors disagree, keep the color that appears earlier in the priority bar.
    - Remove the priority bar and crop the final canvas tightly around all folded objects while preserving their relative placement.

    Key insight:
    - The axis is found from geometry alone, while the surviving color at each overlapped cell is determined by an external priority code on the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    priority = _priority_bar(g[-1])
    priority_rank = {color: idx for idx, color in enumerate(priority)}
    body = g[:-1]
    components = _find_components(body, bg=BG)
    assert components, "expected at least one foldable object"

    folded: dict[tuple[int, int], int] = {}
    for component in components:
        assert len(component["colors"]) >= 2, "objects must be multicolor"
        for coord, color in _fold_component(body, component, priority_rank).items():
            assert coord not in folded, "folded objects should stay disjoint"
            folded[coord] = color

    rows = [coord[0] for coord in folded]
    cols = [coord[1] for coord in folded]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for (r, c), color in folded.items():
        out[r - r0, c - c0] = color
    return out


def _grow_connected_half(height: int, width: int, target_cells: int, seam: str) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((height, width), dtype=bool)
        if seam == "right":
            start = (random.randrange(height), width - 1)
        else:
            start = (height - 1, random.randrange(width))
        mask[start] = True
        cells = [start]
        while len(cells) < target_cells:
            frontier = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True
            cells.append((nr, nc))
        if int(mask.sum()) != target_cells or mask.all():
            continue
        return mask
    return None


def _sample_priority_pair(priority_colors: list[int]) -> tuple[int, int]:
    hi_idx, lo_idx = sorted(random.sample(range(len(priority_colors)), 2))
    return priority_colors[hi_idx], priority_colors[lo_idx]


def _colorize_vertical(half_mask: np.ndarray, priority_colors: list[int]) -> np.ndarray | None:
    h, half_w = half_mask.shape
    pair_cells = [(r, c) for r in range(h) for c in range(half_w) if half_mask[r, c]]
    if len(pair_cells) < 3:
        return None

    modes = ["same", "fixed_wins", "moving_wins"]
    while len(modes) < len(pair_cells):
        modes.append(random.choice(("same", "fixed_wins", "moving_wins")))
    random.shuffle(modes)

    patch = np.zeros((h, half_w * 2), dtype=int)
    for (r, c), mode in zip(pair_cells, modes):
        if mode == "same":
            left = right = random.choice(priority_colors)
        elif mode == "fixed_wins":
            high, low = _sample_priority_pair(priority_colors)
            left, right = high, low
        else:
            high, low = _sample_priority_pair(priority_colors)
            left, right = low, high
        patch[r, c] = left
        patch[r, patch.shape[1] - 1 - c] = right

    if len(set(int(v) for v in patch[patch != BG])) < 2:
        return None
    if not any(int(patch[r, c]) != int(patch[r, patch.shape[1] - 1 - c]) for r, c in pair_cells):
        return None
    return patch


def _colorize_horizontal(half_mask: np.ndarray, priority_colors: list[int]) -> np.ndarray | None:
    half_h, w = half_mask.shape
    pair_cells = [(r, c) for r in range(half_h) for c in range(w) if half_mask[r, c]]
    if len(pair_cells) < 3:
        return None

    modes = ["same", "fixed_wins", "moving_wins"]
    while len(modes) < len(pair_cells):
        modes.append(random.choice(("same", "fixed_wins", "moving_wins")))
    random.shuffle(modes)

    patch = np.zeros((half_h * 2, w), dtype=int)
    for (r, c), mode in zip(pair_cells, modes):
        if mode == "same":
            top = bottom = random.choice(priority_colors)
        elif mode == "fixed_wins":
            high, low = _sample_priority_pair(priority_colors)
            top, bottom = high, low
        else:
            high, low = _sample_priority_pair(priority_colors)
            top, bottom = low, high
        patch[r, c] = top
        patch[patch.shape[0] - 1 - r, c] = bottom

    if len(set(int(v) for v in patch[patch != BG])) < 2:
        return None
    if not any(int(patch[r, c]) != int(patch[patch.shape[0] - 1 - r, c]) for r, c in pair_cells):
        return None
    return patch


def _sample_object_patch(priority_colors: list[int], orientation: str) -> np.ndarray | None:
    for _ in range(OBJECT_RETRIES):
        if orientation == "vertical":
            height = random.randint(4, 8)
            half_w = random.randint(2, 4)
            area = height * half_w
            target = random.randint(max(3, area // 3), max(3, area - 1))
            half_mask = _grow_connected_half(height, half_w, target, seam="right")
            if half_mask is None:
                continue
            full_mask = np.concatenate([half_mask, half_mask[:, ::-1]], axis=1)
            if (
                np.array_equal(full_mask, full_mask[::-1, :])
                or _has_extra_diagonal_axis(full_mask)
                or not _is_connected(full_mask)
            ):
                continue
            patch = _colorize_vertical(half_mask, priority_colors)
        else:
            half_h = random.randint(2, 4)
            width = random.randint(4, 8)
            area = half_h * width
            target = random.randint(max(3, area // 3), max(3, area - 1))
            half_mask = _grow_connected_half(half_h, width, target, seam="bottom")
            if half_mask is None:
                continue
            full_mask = np.concatenate([half_mask, half_mask[::-1, :]], axis=0)
            if (
                np.array_equal(full_mask, full_mask[:, ::-1])
                or _has_extra_diagonal_axis(full_mask)
                or not _is_connected(full_mask)
            ):
                continue
            patch = _colorize_horizontal(half_mask, priority_colors)
        if patch is None:
            continue
        return patch
    return None


def _can_place(blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(blocked.shape[0], top + h + 1)
    c1 = min(blocked.shape[1], left + w + 1)
    return not blocked[r0:r1, c0:c1].any()


def _mark_blocked(blocked: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(blocked.shape[0], top + h + 1)
    c1 = min(blocked.shape[1], left + w + 1)
    blocked[r0:r1, c0:c1] = True


def _generate_attempt() -> np.ndarray | None:
    priority_len = random.randint(4, 7)
    priority_colors = random.sample(PALETTE, priority_len)

    height = random.randint(12, 24)
    width = random.randint(max(12, priority_len + 4), 28)
    grid = np.zeros((height, width), dtype=int)

    bar_start = random.randint(1, width - priority_len - 1)
    grid[-1, bar_start : bar_start + priority_len] = np.array(priority_colors, dtype=int)
    body_h = height - 2
    blocked = np.zeros((body_h, width), dtype=bool)

    object_count = random.randint(2, 4)
    orientations = [random.choice(("vertical", "horizontal")) for _ in range(object_count)]
    if object_count >= 3 and len(set(orientations)) == 1:
        orientations[-1] = "horizontal" if orientations[-1] == "vertical" else "vertical"

    for orientation in orientations:
        patch = _sample_object_patch(priority_colors, orientation)
        if patch is None or patch.shape[0] > body_h or patch.shape[1] > width:
            return None
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(0, body_h - patch.shape[0])
            left = random.randint(0, width - patch.shape[1])
            if not _can_place(blocked, patch, top, left):
                continue
            region = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
            if np.any(region != BG):
                continue
            region[patch != BG] = patch[patch != BG]
            _mark_blocked(blocked, patch, top, left)
            placed = True
            break
        if not placed:
            return None

    if not np.all(grid[-2] == BG):
        return None
    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-and-priority ARC input")
