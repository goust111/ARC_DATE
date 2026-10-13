from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9

TRANSFORM_SYMBOLS = {
    "hflip": {(0, 0), (1, 0), (2, 0), (1, 1)},
    "vflip": {(0, 0), (0, 1), (0, 2), (1, 1)},
    "rot90": {(0, 0), (0, 1), (1, 1)},
}

MARKER_PATTERNS = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
]

INSTRUCTION_TRANSFORMS = ("hflip", "vflip", "rot90")
INSTRUCTION_COLORS = [1, 2, 3, 4, 6, 7, 8]
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _normalize_points(points: list[tuple[int, int]] | np.ndarray) -> set[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in points]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return {(r - r0, c - c0) for r, c in pts}


def _consecutive_groups(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    groups = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        groups.append((start, prev))
        start = prev = v
    groups.append((start, prev))
    return groups


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(grid == MAROON)
    if pts.size == 0:
        raise ValueError("legend border not found")
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise ValueError("legend border too small")
    if not np.all(grid[r0, c0 : c1 + 1] == MAROON):
        raise ValueError("legend top border invalid")
    if not np.all(grid[r1, c0 : c1 + 1] == MAROON):
        raise ValueError("legend bottom border invalid")
    if not np.all(grid[r0 : r1 + 1, c0] == MAROON):
        raise ValueError("legend left border invalid")
    if not np.all(grid[r0 : r1 + 1, c1] == MAROON):
        raise ValueError("legend right border invalid")
    return r0, c0, r1, c1


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, str], np.ndarray]:
    r0, c0, r1, c1 = _find_legend_bbox(grid)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    row_ids = np.where(np.any(inner != BLACK, axis=1))[0].tolist()
    groups = _consecutive_groups(row_ids)
    mapping: dict[int, str] = {}
    for a, b in groups:
        band = inner[a : b + 1, :]
        colors = {int(v) for v in np.unique(band)}
        colors.discard(BLACK)
        colors.discard(GREY)
        if len(colors) != 1:
            continue
        grey_pts = np.argwhere(band == GREY)
        if grey_pts.size == 0:
            continue
        symbol = _normalize_points(grey_pts)
        transform = None
        for name, shape in TRANSFORM_SYMBOLS.items():
            if symbol == shape:
                transform = name
                break
        if transform is None:
            continue
        mapping[next(iter(colors))] = transform
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    if not mapping:
        raise ValueError("legend mapping not found")
    return mapping, legend_mask


def _enclosed_cells(loop_cells: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    rs = [r for r, _ in loop_cells]
    cs = [c for _, c in loop_cells]
    r0 = max(min(rs) - 1, 0)
    c0 = max(min(cs) - 1, 0)
    r1 = min(max(rs) + 1, h - 1)
    c1 = min(max(cs) + 1, w - 1)
    local_h = r1 - r0 + 1
    local_w = c1 - c0 + 1
    barrier = np.zeros((local_h, local_w), dtype=bool)
    for r, c in loop_cells:
        barrier[r - r0, c - c0] = True
    seen = np.zeros((local_h, local_w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for rr in range(local_h):
        for cc in range(local_w):
            if rr not in (0, local_h - 1) and cc not in (0, local_w - 1):
                continue
            if barrier[rr, cc] or seen[rr, cc]:
                continue
            seen[rr, cc] = True
            q.append((rr, cc))
    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < local_h and 0 <= nc < local_w and not barrier[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    enclosed = np.argwhere((~barrier) & (~seen))
    return [(int(rr + r0), int(cc + c0)) for rr, cc in enclosed]


def _apply_transform(arr: np.ndarray, transform: str) -> np.ndarray:
    if transform == "hflip":
        return np.fliplr(arr)
    if transform == "vflip":
        return np.flipud(arr)
    if transform == "rot90":
        return np.rot90(arr, -1)
    raise ValueError(f"unknown transform: {transform}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each loop's interior sprite according to the legend entry keyed by
    the loop's colored wall marker. The marker itself is restored to grey.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _parse_legend(g)
    out = g.copy()

    instruction_mask = np.isin(g, np.array(sorted(mapping), dtype=int))
    loop_mask = ((g == GREY) | instruction_mask) & (~legend_mask)
    for comp in _components(loop_mask):
        if not any(g[r, c] == GREY for r, c in comp):
            continue
        markers = [(r, c) for r, c in comp if int(g[r, c]) in mapping]
        if len(markers) != 1:
            continue
        interior = _enclosed_cells(comp, g.shape)
        if not interior:
            continue
        sprite_cells = [(r, c) for r, c in interior if g[r, c] != BLACK]
        if not sprite_cells:
            out[markers[0]] = GREY
            continue
        r0 = min(r for r, _ in sprite_cells)
        c0 = min(c for _, c in sprite_cells)
        r1 = max(r for r, _ in sprite_cells)
        c1 = max(c for _, c in sprite_cells)
        sprite = g[r0 : r1 + 1, c0 : c1 + 1]
        out[r0 : r1 + 1, c0 : c1 + 1] = _apply_transform(sprite, mapping[int(g[markers[0]])])
        out[markers[0]] = GREY
    return out


def _perimeter(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    border = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    border[r, c] = True
                    break
    return border


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shifted = {(r - min_r, c - min_c) for r, c in cells}
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _square_positions(mask: np.ndarray, side: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    for r in range(h - side + 1):
        for c in range(w - side + 1):
            if mask[r : r + side, c : c + side].all():
                out.append((r, c))
    return out


def _build_loop_shape(rng: random.Random) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    for _ in range(120):
        base_h = rng.randint(7, 10)
        base_w = rng.randint(7, 10)
        cells = {(r, c) for r in range(base_h) for c in range(base_w)}

        if rng.random() < 0.6:
            trim_h = rng.randint(1, min(2, base_h - 3))
            trim_w = rng.randint(1, min(2, base_w - 3))
            corner = rng.choice(("tl", "tr", "bl", "br"))
            if corner == "tl":
                remove = {(r, c) for r in range(trim_h) for c in range(trim_w)}
            elif corner == "tr":
                remove = {(r, c) for r in range(trim_h) for c in range(base_w - trim_w, base_w)}
            elif corner == "bl":
                remove = {(r, c) for r in range(base_h - trim_h, base_h) for c in range(trim_w)}
            else:
                remove = {
                    (r, c)
                    for r in range(base_h - trim_h, base_h)
                    for c in range(base_w - trim_w, base_w)
                }
            cells.difference_update(remove)

        if rng.random() < 0.7:
            side = rng.choice(("top", "bottom", "left", "right"))
            depth = rng.randint(1, 2)
            if side in ("top", "bottom"):
                span = rng.randint(2, max(2, base_w - 2))
                start = rng.randint(1, base_w - span - 1)
                rows = range(-depth, 0) if side == "top" else range(base_h, base_h + depth)
                add = {(r, c) for r in rows for c in range(start, start + span)}
            else:
                span = rng.randint(2, max(2, base_h - 2))
                start = rng.randint(1, base_h - span - 1)
                cols = range(-depth, 0) if side == "left" else range(base_w, base_w + depth)
                add = {(r, c) for r in range(start, start + span) for c in cols}
            cells.update(add)

        mask = _cells_to_mask(cells)
        border = _perimeter(mask)
        interior = mask & (~border)
        placements: list[tuple[int, int, int]] = []
        for side in (5, 4, 3):
            for r, c in _square_positions(interior, side):
                placements.append((side, r, c))
        if interior.sum() >= 9 and placements:
            return mask, border, placements
    raise RuntimeError("failed to build loop shape")


def _connected_sprite_mask(rng: random.Random, side: int) -> np.ndarray:
    target = rng.randint(max(6, side * side // 2), side * side)
    start = (side // 2, side // 2)
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    edge_targets = [
        (0, rng.randint(0, side - 1)),
        (side - 1, rng.randint(0, side - 1)),
        (rng.randint(0, side - 1), 0),
        (rng.randint(0, side - 1), side - 1),
    ]
    for tr, tc in edge_targets:
        rr, cc = tr, tc
        cells.add((rr, cc))
        while rr != start[0]:
            rr += 1 if start[0] > rr else -1
            cells.add((rr, cc))
        while cc != start[1]:
            cc += 1 if start[1] > cc else -1
            cells.add((rr, cc))

    mask = np.zeros((side, side), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _build_sprite(rng: random.Random, side: int, palette: list[int], transform: str) -> np.ndarray:
    for _ in range(100):
        mask = _connected_sprite_mask(rng, side)
        arr = np.zeros((side, side), dtype=int)
        num_colors = min(len(palette), rng.randint(2, min(4, len(palette))))
        colors = rng.sample(palette, num_colors)
        order = list(np.argwhere(mask))
        rng.shuffle(order)
        for i, (r, c) in enumerate(order):
            arr[int(r), int(c)] = colors[i % num_colors]
        if all(
            [
                np.any(arr[0, :] != BLACK),
                np.any(arr[-1, :] != BLACK),
                np.any(arr[:, 0] != BLACK),
                np.any(arr[:, -1] != BLACK),
            ]
        ):
            if len({int(v) for v in np.unique(arr)} - {BLACK}) >= 2 and not np.array_equal(arr, _apply_transform(arr, transform)):
                return arr
    raise RuntimeError("failed to build non-degenerate sprite")


def _make_legend(
    rng: random.Random, mapping: dict[int, str]
) -> np.ndarray:
    inner_h = len(mapping) * 4
    inner_w = 7
    inner = np.zeros((inner_h, inner_w), dtype=int)
    entries = list(mapping.items())
    rng.shuffle(entries)
    for idx, (color, transform) in enumerate(entries):
        slot_r = idx * 4
        marker = rng.choice(MARKER_PATTERNS)
        marker_r_shift = rng.randint(0, 1)
        for dr, dc in marker:
            inner[slot_r + marker_r_shift + dr, dc] = color
        symbol = TRANSFORM_SYMBOLS[transform]
        symbol_r_shift = 0 if transform == "hflip" else rng.randint(0, 1)
        for dr, dc in symbol:
            inner[slot_r + symbol_r_shift + dr, 4 + dc] = GREY
    legend = np.full((inner_h + 2, inner_w + 2), MAROON, dtype=int)
    legend[1:-1, 1:-1] = inner
    return legend


def _legend_origin(corner: str, grid_h: int, grid_w: int, legend_h: int, legend_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, grid_w - legend_w
    if corner == "bl":
        return grid_h - legend_h, 0
    return grid_h - legend_h, grid_w - legend_w


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    return not blocked[rr0:rr1, cc0:cc1].any()


def _reserve(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(blocked.shape[0], r0 + h + pad)
    cc1 = min(blocked.shape[1], c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(26, 30)
    grid_w = rng.randint(26, 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    chosen_instruction_colors = rng.sample(INSTRUCTION_COLORS, 3)
    shuffled_transforms = list(INSTRUCTION_TRANSFORMS)
    rng.shuffle(shuffled_transforms)
    color_to_transform = {
        chosen_instruction_colors[i]: shuffled_transforms[i]
        for i in range(3)
    }

    legend = _make_legend(rng, color_to_transform)
    legend_corner = rng.choice(("tl", "tr", "bl", "br"))
    lr, lc = _legend_origin(legend_corner, grid_h, grid_w, legend.shape[0], legend.shape[1])
    _paste_patch(grid, legend, lr, lc)
    _reserve(blocked, lr, lc, legend.shape[0], legend.shape[1], pad=1)

    loop_count = rng.choice((3, 4))
    loop_colors = chosen_instruction_colors.copy()
    if loop_count == 4:
        loop_colors.append(rng.choice(chosen_instruction_colors))
    rng.shuffle(loop_colors)

    sprite_palette = [c for c in INSTRUCTION_COLORS if c not in chosen_instruction_colors]
    for loop_color in loop_colors:
        transform = color_to_transform[loop_color]
        shape_mask, border_mask, placements = _build_loop_shape(rng)
        side_choices = sorted({side for side, _, _ in placements}, reverse=True)
        sprite_side = rng.choice(side_choices[: min(2, len(side_choices))])
        candidates = [(r, c) for side, r, c in placements if side == sprite_side]
        sprite_r, sprite_c = rng.choice(candidates)
        sprite = _build_sprite(rng, sprite_side, sprite_palette, transform)

        patch_h, patch_w = shape_mask.shape
        for _ in range(240):
            pr = rng.randint(0, grid_h - patch_h)
            pc = rng.randint(0, grid_w - patch_w)
            if not _can_place(blocked, pr, pc, patch_h, patch_w, pad=1):
                continue
            border_pts = [(int(r), int(c)) for r, c in np.argwhere(border_mask)]
            marker_r, marker_c = rng.choice(border_pts)
            loop_patch = np.zeros((patch_h, patch_w), dtype=int)
            loop_patch[border_mask] = GREY
            loop_patch[marker_r, marker_c] = loop_color
            sr0 = sprite_r
            sc0 = sprite_c
            sprite_region = loop_patch[sr0 : sr0 + sprite_side, sc0 : sc0 + sprite_side]
            sprite_region[sprite != BLACK] = sprite[sprite != BLACK]
            _paste_patch(grid, loop_patch, pr, pc)
            _reserve(blocked, pr, pc, patch_h, patch_w, pad=1)
            break
        else:
            return None

    solved = main(grid)
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
