from __future__ import annotations

import random

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    for dr, dc in _NEIGHBORS4:
        yield int(r + dr), int(c + dc)


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(coords)

    return components


def _component_color_info(
    grid: np.ndarray, coords: list[tuple[int, int]]
) -> tuple[dict[int, int], list[int]]:
    counts: dict[int, int] = {}
    order: list[int] = []
    for r, c in coords:
        color = int(grid[r, c])
        if color not in counts:
            counts[color] = 0
            order.append(color)
        counts[color] += 1
    return counts, order


def _dominant_color(grid: np.ndarray, coords: list[tuple[int, int]]) -> int:
    counts, order = _component_color_info(grid, coords)
    best = int(order[0])
    best_count = int(counts[best])
    for color in order[1:]:
        if int(counts[color]) > best_count:
            best = int(color)
            best_count = int(counts[color])
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the largest same-color region inside each cut object, and erase the cut lines.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = np.full_like(g, bg)

    for coords in components_mask(g != bg):
        counts, order = _component_color_info(g, coords)
        if len(order) == 1:
            color = int(order[0])
            for r, c in coords:
                out[r, c] = color
            continue

        base_color = _dominant_color(g, coords)
        base_mask = np.zeros_like(g, dtype=bool)
        for r, c in coords:
            if int(g[r, c]) == int(base_color):
                base_mask[r, c] = True

        regions = components_mask(base_mask)
        if not regions:
            continue

        sizes = [len(region) for region in regions]
        keep_size = max(sizes)
        for region, size in zip(regions, sizes):
            if int(size) != int(keep_size):
                continue
            for r, c in region:
                out[r, c] = int(base_color)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 26)
OBJECT_COUNT_RANGE = (2, 4)
PALETTE = tuple(range(10))
BORDER_MARGIN = 1
OBJECT_CLEARANCE = 1
GEN_ATTEMPTS = 400


def _region_sizes(local: np.ndarray) -> list[int]:
    return sorted((len(comp) for comp in components_mask(local == 1)), reverse=True)


def _valid_local_object(local: np.ndarray) -> bool:
    sizes = _region_sizes(local)
    return len(sizes) >= 2 and sizes[0] > sizes[1]


def _single_mask() -> np.ndarray | None:
    for _ in range(80):
        orientation = random.choice(("vertical", "horizontal"))
        if orientation == "vertical":
            h = random.randint(5, 9)
            w = random.randint(6, 10)
            candidates = [c for c in range(2, w - 2) if c != w - c - 1]
            if not candidates:
                continue
            cut = random.choice(candidates)
            local = np.ones((h, w), dtype=int)
            local[:, cut] = 2
        else:
            h = random.randint(6, 10)
            w = random.randint(5, 9)
            candidates = [r for r in range(2, h - 2) if r != h - r - 1]
            if not candidates:
                continue
            cut = random.choice(candidates)
            local = np.ones((h, w), dtype=int)
            local[cut, :] = 2
        if _valid_local_object(local):
            return local
    return None


def _parallel_triples(span: int) -> list[tuple[int, int, int]]:
    total = int(span) - 2
    triples: list[tuple[int, int, int]] = []
    for a in range(2, total - 3):
        for b in range(2, total - a - 1):
            c = total - a - b
            widths = [int(a), int(b), int(c)]
            if c < 2 or max(widths) == sorted(widths)[-2]:
                continue
            triples.append((int(a), int(b), int(c)))
    return triples


def _parallel_mask() -> np.ndarray | None:
    for _ in range(80):
        orientation = random.choice(("vertical", "horizontal"))
        if orientation == "vertical":
            h = random.randint(5, 9)
            w = random.randint(8, 11)
            triples = _parallel_triples(w)
            if not triples:
                continue
            a, b, _c = random.choice(triples)
            cut1 = a
            cut2 = a + 1 + b
            local = np.ones((h, w), dtype=int)
            local[:, cut1] = 2
            local[:, cut2] = 2
        else:
            h = random.randint(8, 11)
            w = random.randint(5, 9)
            triples = _parallel_triples(h)
            if not triples:
                continue
            a, b, _c = random.choice(triples)
            cut1 = a
            cut2 = a + 1 + b
            local = np.ones((h, w), dtype=int)
            local[cut1, :] = 2
            local[cut2, :] = 2
        if _valid_local_object(local):
            return local
    return None


def _cross_mask() -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(6, 10)
        w = random.randint(6, 10)
        candidates: list[tuple[int, int]] = []
        for cut_r in range(2, h - 2):
            top = cut_r
            bottom = h - cut_r - 1
            for cut_c in range(2, w - 2):
                left = cut_c
                right = w - cut_c - 1
                areas = [
                    int(top * left),
                    int(top * right),
                    int(bottom * left),
                    int(bottom * right),
                ]
                if max(areas) == sorted(areas)[-2]:
                    continue
                candidates.append((int(cut_r), int(cut_c)))
        if not candidates:
            continue
        cut_r, cut_c = random.choice(candidates)
        local = np.ones((h, w), dtype=int)
        local[cut_r, :] = 2
        local[:, cut_c] = 2
        if _valid_local_object(local):
            return local
    return None


def _build_object(pattern: str) -> np.ndarray | None:
    if pattern == "single":
        return _single_mask()
    if pattern == "parallel":
        return _parallel_mask()
    if pattern == "cross":
        return _cross_mask()
    return None


def _pattern_schedule(n_objects: int) -> list[str]:
    n_multi = random.randint(1, n_objects - 1)
    families = ["single"] * (n_objects - n_multi) + ["multi"] * n_multi
    random.shuffle(families)
    patterns: list[str] = []
    for family in families:
        if family == "single":
            patterns.append("single")
        else:
            patterns.append(random.choice(("parallel", "cross")))
    return patterns


def _candidate_positions(
    blocked: np.ndarray, obj_h: int, obj_w: int
) -> list[tuple[int, int]]:
    h, w = map(int, blocked.shape)
    positions: list[tuple[int, int]] = []
    for r0 in range(BORDER_MARGIN, h - obj_h - BORDER_MARGIN + 1):
        for c0 in range(BORDER_MARGIN, w - obj_w - BORDER_MARGIN + 1):
            r1 = r0 + obj_h - 1
            c1 = c0 + obj_w - 1
            rr0 = max(0, r0 - OBJECT_CLEARANCE)
            rr1 = min(h, r1 + OBJECT_CLEARANCE + 1)
            cc0 = max(0, c0 - OBJECT_CLEARANCE)
            cc1 = min(w, c1 + OBJECT_CLEARANCE + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((int(r0), int(c0)))
    return positions


def _place_local(
    grid: np.ndarray,
    blocked: np.ndarray,
    local: np.ndarray,
    *,
    color: int,
    cut: int,
    top: int,
    left: int,
) -> None:
    obj_h, obj_w = map(int, local.shape)
    window = grid[top : top + obj_h, left : left + obj_w]
    window[local == 1] = int(color)
    window[local == 2] = int(cut)

    r1 = top + obj_h - 1
    c1 = left + obj_w - 1
    rr0 = max(0, top - OBJECT_CLEARANCE)
    rr1 = min(grid.shape[0], r1 + OBJECT_CLEARANCE + 1)
    cc0 = max(0, left - OBJECT_CLEARANCE)
    cc1 = min(grid.shape[1], c1 + OBJECT_CLEARANCE + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _validate_grid(grid: np.ndarray, *, bg: int, cut: int, n_objects: int) -> bool:
    if int(get_bg(grid)) != int(bg):
        return False

    values, counts = np.unique(grid, return_counts=True)
    bg_count = 0
    for value, count in zip(values, counts):
        if int(value) == int(bg):
            bg_count = int(count)
            break
    if bg_count == 0:
        return False
    for value, count in zip(values, counts):
        if int(value) != int(bg) and int(count) >= int(bg_count):
            return False

    components = components_mask(grid != bg)
    if len(components) != int(n_objects):
        return False

    for coords in components:
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 2 or int(cut) not in colors:
            return False

        object_color = next(color for color in colors if int(color) != int(cut))
        object_mask = np.zeros_like(grid, dtype=bool)
        object_pixels = 0
        cut_pixels = 0
        for r, c in coords:
            if int(grid[r, c]) == int(object_color):
                object_mask[r, c] = True
                object_pixels += 1
            else:
                cut_pixels += 1
        if object_pixels <= cut_pixels:
            return False

        sizes = sorted((len(comp) for comp in components_mask(object_mask)), reverse=True)
        if len(sizes) < 2 or sizes[0] == sizes[1]:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)

    colors = list(PALETTE)
    random.shuffle(colors)
    bg = int(colors.pop())
    cut = int(colors.pop())
    object_colors = [int(c) for c in colors[:n_objects]]

    patterns = _pattern_schedule(n_objects)
    specs: list[tuple[np.ndarray, int]] = []
    for pattern, color in zip(patterns, object_colors):
        local = _build_object(pattern)
        if local is None:
            return None
        specs.append((local, int(color)))

    specs.sort(key=lambda item: int(item[0].shape[0] * item[0].shape[1]), reverse=True)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    for local, color in specs:
        positions = _candidate_positions(blocked, int(local.shape[0]), int(local.shape[1]))
        if not positions:
            return None
        top, left = random.choice(positions)
        _place_local(grid, blocked, local, color=color, cut=cut, top=top, left=left)

    if not _validate_grid(grid, bg=bg, cut=cut, n_objects=n_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
