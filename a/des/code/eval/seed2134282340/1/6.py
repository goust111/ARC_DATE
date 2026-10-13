from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
SIDE_ORDER = ("top", "right", "bottom", "left")
SIDE_TO_IDX = {side: idx for idx, side in enumerate(SIDE_ORDER)}
USABLE_COLORS = tuple(color for color in range(10) if color not in (BG, YELLOW))
ODD_SIZES = (3, 5, 7)
GEN_RETRIES = 400
PORTAL_PADDING = 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(cells)
    return parts


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _marker_side(marker: tuple[int, int], box: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    candidates: list[str] = []
    if marker == (r0, c0 + w // 2):
        candidates.append("top")
    if marker == (r0 + h // 2, c1):
        candidates.append("right")
    if marker == (r1, c0 + w // 2):
        candidates.append("bottom")
    if marker == (r0 + h // 2, c0):
        candidates.append("left")
    assert len(candidates) == 1, "yellow marker must identify exactly one portal side"
    return candidates[0]


def _portal_from_box(
    g: np.ndarray, bg: int, box: tuple[int, int, int, int], marker: tuple[int, int]
) -> dict | None:
    r0, c0, r1, c1 = box
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    try:
        side = _marker_side(marker, box)
    except AssertionError:
        return None

    border_color: int | None = None
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r not in (r0, r1) and c not in (c0, c1):
                continue
            value = int(g[r, c])
            if (r, c) == marker:
                if value != YELLOW:
                    return None
                continue
            if value in (bg, YELLOW):
                return None
            if border_color is None:
                border_color = value
            elif value != border_color:
                return None

    if border_color is None:
        return None
    return {
        "color": border_color,
        "bbox": box,
        "side": side,
        "inner": g[r0 + 1 : r1, c0 + 1 : c1].copy(),
    }


def _portal_components(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    portals: list[dict] = []
    yellow_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)]
    assert yellow_cells, "expected yellow portal markers"

    for r, c in yellow_cells:
        max_half_w = min(c, w - 1 - c)
        max_half_h = min(r, h - 1 - r)
        best: dict | None = None
        best_area = -1

        for half_w in range(1, max_half_w + 1):
            c0 = c - half_w
            c1 = c + half_w
            for r1 in range(r + 2, h):
                portal = _portal_from_box(g, bg, (r, c0, r1, c1), (r, c))
                if portal is not None:
                    area = (r1 - r + 1) * (c1 - c0 + 1)
                    if area > best_area:
                        best = portal
                        best_area = area
            for r0 in range(0, r - 1):
                portal = _portal_from_box(g, bg, (r0, c0, r, c1), (r, c))
                if portal is not None:
                    area = (r - r0 + 1) * (c1 - c0 + 1)
                    if area > best_area:
                        best = portal
                        best_area = area

        for half_h in range(1, max_half_h + 1):
            r0 = r - half_h
            r1 = r + half_h
            for c1 in range(c + 2, w):
                portal = _portal_from_box(g, bg, (r0, c, r1, c1), (r, c))
                if portal is not None:
                    area = (r1 - r0 + 1) * (c1 - c + 1)
                    if area > best_area:
                        best = portal
                        best_area = area
            for c0 in range(0, c - 1):
                portal = _portal_from_box(g, bg, (r0, c0, r1, c), (r, c))
                if portal is not None:
                    area = (r1 - r0 + 1) * (c - c0 + 1)
                    if area > best_area:
                        best = portal
                        best_area = area

        assert best is not None, "every yellow marker must belong to one portal frame"
        portals.append(best)

    assert portals, "expected at least one portal pair"
    return portals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored portal frames and copy the filled portal interior into the empty partner,
    rotating the copied pattern by the clockwise difference between the yellow marker sides.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = int(np.bincount(g.ravel()).argmax())
    portals = _portal_components(g, bg)

    by_color: dict[int, list[dict]] = {}
    for portal in portals:
        by_color.setdefault(int(portal["color"]), []).append(portal)

    out = g.copy()
    for color, pair in by_color.items():
        assert len(pair) == 2, f"portal color {color} must appear exactly twice"
        filled = [portal for portal in pair if np.any(portal["inner"] != bg)]
        empty = [portal for portal in pair if not np.any(portal["inner"] != bg)]
        assert len(filled) == 1 and len(empty) == 1, "each portal pair needs one entrance and one exit"

        src = filled[0]
        dst = empty[0]
        steps = (SIDE_TO_IDX[dst["side"]] - SIDE_TO_IDX[src["side"]]) % 4
        rotated = np.rot90(src["inner"], -steps)
        assert rotated.shape == dst["inner"].shape, "rotated payload must fit the exit portal"
        r0, c0, r1, c1 = dst["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated

    return out


def _portal_patch(inner: np.ndarray | None, border_color: int, side: str) -> np.ndarray:
    inner_h = 3 if inner is None else int(inner.shape[0])
    inner_w = 3 if inner is None else int(inner.shape[1])
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color

    mid_r = patch.shape[0] // 2
    mid_c = patch.shape[1] // 2
    if side == "top":
        patch[0, mid_c] = YELLOW
    elif side == "right":
        patch[mid_r, -1] = YELLOW
    elif side == "bottom":
        patch[-1, mid_c] = YELLOW
    elif side == "left":
        patch[mid_r, 0] = YELLOW
    else:
        raise AssertionError(side)

    if inner is not None:
        patch[1:-1, 1:-1] = inner
    return patch


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - PORTAL_PADDING)
    c0 = max(0, left - PORTAL_PADDING)
    r1 = min(h, top + height + PORTAL_PADDING)
    c1 = min(w, left + width + PORTAL_PADDING)
    return not bool(occupied[r0:r1, c0:c1].any())


def _stamp(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    canvas[top : top + h, left : left + w] = patch


def _mark(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occupied[top : top + height, left : left + width] = True


def _grow_connected_mask(height: int, width: int, target: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        choice = random.choice(frontier)
        mask[choice] = True
        cells.add(choice)
    return mask


def _random_pattern(height: int, width: int, steps: int, palette: tuple[int, ...]) -> np.ndarray:
    area = height * width
    min_fill = max(4, area // 3)
    max_fill = min(area - 1, (2 * area) // 3 + 1)
    assert min_fill <= max_fill

    for _ in range(200):
        fill = random.randint(min_fill, max_fill)
        mask = _grow_connected_mask(height, width, fill)
        if int(mask.sum()) != fill:
            continue

        patch = np.full((height, width), BG, dtype=int)
        color_count = 2 if fill >= 4 else 1
        colors = random.sample(palette, color_count)
        cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(cells)
        split = random.randint(1, fill - 1) if color_count == 2 else fill
        for idx, (r, c) in enumerate(cells):
            patch[r, c] = colors[0] if idx < split else colors[-1]

        if len({int(v) for v in patch.ravel() if int(v) != BG}) < 2:
            continue
        if steps != 0 and np.array_equal(patch, np.rot90(patch, -steps)):
            continue
        if np.all(patch[0, :] == BG) or np.all(patch[-1, :] == BG):
            continue
        if np.all(patch[:, 0] == BG) or np.all(patch[:, -1] == BG):
            continue
        return patch

    raise RuntimeError("failed to sample a non-degenerate portal pattern")


def _place_patch(occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = patch.shape
    candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(candidates)
    for top, left in candidates:
        if _can_place(occupied, top, left, ph, pw):
            return top, left
    return None


def _sample_inner_shape(steps: int) -> tuple[int, int]:
    if steps % 2 == 1 and random.random() < 0.7:
        height = random.choice(ODD_SIZES)
        width_choices = [size for size in ODD_SIZES if size != height]
        width = random.choice(width_choices)
        return height, width
    return random.choice(ODD_SIZES), random.choice(ODD_SIZES)


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    pair_count = random.randint(2, 4)
    rows = random.randint(20, 30)
    cols = random.randint(20, 30)
    grid = np.full((rows, cols), BG, dtype=int)
    expected = grid.copy()
    occupied = np.zeros((rows, cols), dtype=bool)

    border_colors = random.sample(USABLE_COLORS, pair_count)
    steps_list = random.sample((0, 1, 2, 3), pair_count)
    pattern_palette = tuple(color for color in USABLE_COLORS if color not in border_colors)
    assert len(pattern_palette) >= 2

    portal_specs = []
    for border_color, steps in zip(border_colors, steps_list):
        src_side_idx = random.randrange(4)
        dst_side_idx = (src_side_idx + steps) % 4
        src_side = SIDE_ORDER[src_side_idx]
        dst_side = SIDE_ORDER[dst_side_idx]

        inner_h, inner_w = _sample_inner_shape(steps)
        entry_inner = _random_pattern(inner_h, inner_w, steps, pattern_palette)
        exit_shape = entry_inner.shape if steps % 2 == 0 else (entry_inner.shape[1], entry_inner.shape[0])
        exit_inner = np.full(exit_shape, BG, dtype=int)

        entry_patch = _portal_patch(entry_inner, border_color, src_side)
        exit_patch = _portal_patch(exit_inner, border_color, dst_side)
        portal_specs.append((entry_patch, exit_patch, entry_inner, steps))

    portal_specs.sort(key=lambda spec: spec[0].size + spec[1].size, reverse=True)
    for entry_patch, exit_patch, entry_inner, steps in portal_specs:
        entry_pos = _place_patch(occupied, entry_patch)
        if entry_pos is None:
            raise RuntimeError("failed to place entry portal")
        _mark(occupied, entry_pos[0], entry_pos[1], entry_patch.shape[0], entry_patch.shape[1])

        exit_pos = _place_patch(occupied, exit_patch)
        if exit_pos is None:
            raise RuntimeError("failed to place exit portal")
        _mark(occupied, exit_pos[0], exit_pos[1], exit_patch.shape[0], exit_patch.shape[1])

        _stamp(grid, entry_patch, entry_pos[0], entry_pos[1])
        _stamp(grid, exit_patch, exit_pos[0], exit_pos[1])
        _stamp(expected, entry_patch, entry_pos[0], entry_pos[1])
        _stamp(expected, exit_patch, exit_pos[0], exit_pos[1])

        rotated = np.rot90(entry_inner, -steps)
        er, ec = exit_pos
        expected[
            er + 1 : er + 1 + rotated.shape[0],
            ec + 1 : ec + 1 + rotated.shape[1],
        ] = rotated

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid, expected = _build_instance()
        except RuntimeError:
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid portal-rotation ARC input")
