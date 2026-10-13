from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


BG = 8
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = tuple(color for color in range(10) if color != BG)

# Generator bounds.
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (20, 32)
SPRITES_PER_GROUP_RANGE = (1, 2)
EXTRA_SPRITE_PROB = 0.45
MAX_SHAPE_ATTEMPTS = 120
MAX_PLACEMENT_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    out[r, c] = True
                    break
    return out


def _extract_sprite(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    mask, bbox = _shape_mask(coords)
    r0, c0, r1, c1 = bbox
    patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    rel_colors = [int(patch[r - r0, c - c0]) for r, c in coords]
    counts = Counter(rel_colors)
    max_count = max(counts.values())
    body_colors = [color for color, count in counts.items() if count == max_count]
    assert len(body_colors) == 1, "each sprite must have a unique majority body color"
    body_color = int(body_colors[0])

    boundary = _boundary_mask(mask)
    marker_count = 0
    for rr, cc in zip(*np.where(mask)):
        color = int(patch[rr, cc])
        if color != body_color:
            assert boundary[rr, cc], "non-body pixels must lie on the sprite boundary"
            marker_count += 1

    assert 0 <= marker_count <= 3
    return {
        "bbox": bbox,
        "mask": mask,
        "h": int(mask.shape[0]),
        "w": int(mask.shape[1]),
        "body_color": body_color,
        "marker_count": marker_count,
    }


def _overlay_group(sprites: list[dict], bg: int) -> np.ndarray:
    height = max(sprite["h"] for sprite in sprites)
    width = max(sprite["w"] for sprite in sprites)
    out = np.full((height, width), bg, dtype=int)

    for sprite in sprites:
        top = (height - sprite["h"]) // 2
        left = (width - sprite["w"]) // 2
        view = out[top : top + sprite["h"], left : left + sprite["w"]]
        view[sprite["mask"]] = int(sprite["body_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group sprites by the number of boundary marker pixels, then center-overlay each group.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g != bg)
    assert components, "expected at least one foreground sprite"

    sprites = [_extract_sprite(g, coords) for coords in components]
    sprites.sort(key=lambda sprite: (sprite["bbox"][1], sprite["bbox"][0]))

    groups: dict[int, list[dict]] = defaultdict(list)
    for sprite in sprites:
        groups[int(sprite["marker_count"])].append(sprite)

    patterns = [_overlay_group(groups[count], bg) for count in sorted(groups)]
    out_h = max(pattern.shape[0] for pattern in patterns)
    out_w = sum(pattern.shape[1] for pattern in patterns) + len(patterns) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for pattern in patterns:
        h, w = pattern.shape
        view = out[:h, col : col + w]
        view[pattern != bg] = pattern[pattern != bg]
        col += w + 1

    return out


def _cells_to_mask(cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _has_hole(cells: set[tuple[int, int]], h: int, w: int) -> bool:
    solid = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        solid[r, c] = True

    padded = np.pad(solid, 1, constant_values=False)
    ph, pw = padded.shape
    seen = np.zeros((ph, pw), dtype=bool)
    queue = deque([(0, 0)])
    seen[0, 0] = True

    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ph and 0 <= nc < pw and not seen[nr, nc] and not padded[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    interior_empty = ~padded[1:-1, 1:-1]
    exterior_reached = seen[1:-1, 1:-1]
    return bool(np.any(interior_empty & ~exterior_reached))


def _boundary_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in cells:
        for dr, dc in NEIGHBORS4:
            if (r + dr, c + dc) not in cells:
                out.append((r, c))
                break
    return out


def _generate_shape(h: int, w: int, target_area: int) -> set[tuple[int, int]] | None:
    full = {(r, c) for r in range(h) for c in range(w)}
    cells = set(full)

    while len(cells) > target_area:
        removable: list[tuple[int, int]] = []
        for cell in list(cells):
            trial = cells - {cell}
            rows = [r for r, _c in trial]
            cols = [c for _r, c in trial]
            if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
                continue
            if not _is_connected(trial):
                continue
            if _has_hole(trial, h, w):
                continue
            removable.append(cell)
        if not removable:
            return None
        cells.remove(random.choice(removable))

    if len(cells) != target_area or not _is_connected(cells) or _has_hole(cells, h, w):
        return None
    return cells


def _shape_patch(marker_count: int, h: int, w: int) -> np.ndarray | None:
    min_area = max(5, h + w - 1)
    max_area = min(15, h * w)
    if min_area > max_area:
        return None

    for _ in range(MAX_SHAPE_ATTEMPTS):
        area = random.randint(min_area, max_area)
        cells = _generate_shape(h, w, area)
        if cells is None:
            continue
        boundary = _boundary_cells(cells)
        if len(boundary) < marker_count:
            continue

        body_color = random.choice(PALETTE)
        patch = np.full((h, w), BG, dtype=int)
        for r, c in cells:
            patch[r, c] = body_color

        marker_colors = random.sample([color for color in PALETTE if color != body_color], marker_count)
        for (r, c), color in zip(random.sample(boundary, marker_count), marker_colors):
            patch[r, c] = color
        return patch
    return None


def _group_counts() -> dict[int, int]:
    counts = {count: random.randint(*SPRITES_PER_GROUP_RANGE) for count in range(4)}
    if all(value == 1 for value in counts.values()) or random.random() < EXTRA_SPRITE_PROB:
        counts[random.randrange(4)] += 1
    return counts


def _dimension_choices(parity_r: int, parity_c: int) -> list[tuple[int, int]]:
    choices: list[tuple[int, int]] = []
    for h in range(2, 6):
        for w in range(2, 6):
            if h % 2 == parity_r and w % 2 == parity_c and min(15, h * w) >= max(5, h + w - 1):
                choices.append((h, w))
    return choices


def _build_sprite_specs() -> list[dict] | None:
    counts = _group_counts()
    specs: list[dict] = []

    for marker_count in range(4):
        parity_r = random.randint(0, 1)
        parity_c = random.randint(0, 1)
        dims = _dimension_choices(parity_r, parity_c)
        if not dims:
            return None

        for _ in range(counts[marker_count]):
            for _attempt in range(MAX_SHAPE_ATTEMPTS):
                h, w = random.choice(dims)
                patch = _shape_patch(marker_count, h, w)
                if patch is None:
                    continue
                specs.append({"marker_count": marker_count, "patch": patch})
                break
            else:
                return None

    random.shuffle(specs)
    return specs


def _can_place(forbidden: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(not forbidden[r, c] for r, c in cells)


def _mark_forbidden(forbidden: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    forbidden[nr, nc] = True


def _place_sprites(specs: list[dict], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)

    for spec in specs:
        patch = spec["patch"]
        ph, pw = patch.shape
        coords = [(int(r), int(c)) for r, c in np.argwhere(patch != BG)]
        positions = [
            (top, left)
            for top in range(height - ph + 1)
            for left in range(width - pw + 1)
        ]
        random.shuffle(positions)

        placed = False
        for top, left in positions[:MAX_PLACEMENT_ATTEMPTS]:
            abs_cells = [(top + r, left + c) for r, c in coords]
            if not _can_place(forbidden, abs_cells):
                continue
            for r, c in coords:
                grid[top + r, left + c] = int(patch[r, c])
            _mark_forbidden(forbidden, abs_cells)
            placed = True
            break

        if not placed:
            return None

    return grid


def _generate_attempt() -> np.ndarray | None:
    specs = _build_sprite_specs()
    if specs is None:
        return None

    min_h = max(GRID_HEIGHT_RANGE[0], max(spec["patch"].shape[0] for spec in specs) + 4)
    min_w = max(
        GRID_WIDTH_RANGE[0],
        sum(spec["patch"].shape[1] for spec in specs) // 2 + 6,
    )
    if min_h > GRID_HEIGHT_RANGE[1] or min_w > GRID_WIDTH_RANGE[1]:
        return None

    height = random.randint(min_h, GRID_HEIGHT_RANGE[1])
    width = random.randint(min_w, GRID_WIDTH_RANGE[1])
    grid = _place_sprites(specs, height, width)
    if grid is None:
        return None

    out = main(grid)
    if out.size == 0 or _get_bg(grid) != BG or _find_components(grid != BG) is None:
        return None
    if np.count_nonzero(grid != BG) < 20:
        return None
    if np.count_nonzero(out != BG) < 10:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
