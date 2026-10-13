from __future__ import annotations

import math
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    """Return 4-connected monochrome components cropped to their bboxes."""
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[dict] = []

    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            for rr, cc in coords:
                sprite[rr - r0, cc - c0] = color

            components.append(
                {
                    "color": color,
                    "area": len(coords),
                    "bbox": (r0, c0, r1, c1),
                    "sprite": sprite,
                }
            )

    return components


def _pedestals_from_bottom_row(grid: np.ndarray, bg: int = BG) -> list[dict]:
    """Read equal-width colored base runs from the last row."""
    row = np.asarray(grid[-1], dtype=int)
    W = row.shape[0]
    pedestals: list[dict] = []
    c = 0
    while c < W:
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        start = c
        while c < W and int(row[c]) == color:
            c += 1
        end = c - 1
        pedestals.append({"color": color, "c0": start, "c1": end, "width": end - start + 1})

    assert pedestals, "expected at least one pedestal on the bottom row"
    widths = {ped["width"] for ped in pedestals}
    assert len(widths) == 1, "all pedestals must have the same width"
    colors = [ped["color"] for ped in pedestals]
    assert len(colors) == len(set(colors)), "pedestal colors must be distinct"
    return pedestals


def _paste(canvas: np.ndarray, sprite: np.ndarray, row: int, col: int, bg: int = BG) -> None:
    h, w = sprite.shape
    assert 0 <= row and 0 <= col and row + h <= canvas.shape[0] and col + w <= canvas.shape[1]
    mask = sprite != bg
    canvas[row : row + h, col : col + w][mask] = sprite[mask]


def _layer_appearance(sprite: np.ndarray, layer_idx: int, color: int) -> np.ndarray:
    out = np.array(sprite, copy=True)
    if layer_idx == 0:
        return out
    if layer_idx == 1:
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                if int(out[r, c]) == BG:
                    continue
                if (r + c) % 2 == 1:
                    out[r, c] = GREY
                else:
                    out[r, c] = color
        return out
    out[out != BG] = GREY
    return out


def _render_stacks(shape: tuple[int, int], pedestals: list[dict], groups: dict[int, list[dict]]) -> np.ndarray:
    H, W = shape
    out = np.full((H, W), BG, dtype=int)
    for ped in pedestals:
        out[H - 1, ped["c0"] : ped["c1"] + 1] = int(ped["color"])

    for ped in pedestals:
        color = int(ped["color"])
        components = sorted(
            groups.get(color, []),
            key=lambda comp: (-int(comp["area"]), int(comp["bbox"][0]), int(comp["bbox"][1])),
        )
        top = H - 1
        for layer_idx, comp in enumerate(components[:3]):
            sprite = _layer_appearance(comp["sprite"], layer_idx, color)
            h, w = sprite.shape
            row = top - h
            col = (int(ped["c0"]) + int(ped["c1"]) - (w - 1)) // 2
            assert 0 <= row and 0 <= col and col + w <= W, "stacked sprite must fit inside the canvas"
            _paste(out, sprite, row, col, bg=BG)
            top = row

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-colored fragments by pedestal color, sort them by area, and stack them.

    The bottom row contains same-width colored pedestals separated by black gaps.
    Above them are scattered 4-connected monochrome fragments whose colors match
    pedestal colors. For each pedestal color, fragments are sorted by area from
    largest to smallest and stacked upward, horizontally centered on the pedestal.
    Layer 1 keeps its original color, layer 2 becomes a grey/original checkerboard
    anchored at the fragment's top-left corner, layer 3 becomes fully grey, and
    layer 4+ is discarded.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 3

    pedestals = _pedestals_from_bottom_row(g, bg=BG)
    components = _find_components(g[:-1, :], bg=BG)
    groups: dict[int, list[dict]] = {}
    for comp in components:
        groups.setdefault(int(comp["color"]), []).append(comp)

    return _render_stacks(g.shape, pedestals, groups)


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    H, W = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _minimum_edge_spanning_area(h: int, w: int) -> int:
    if h == 1:
        return w
    if w == 1:
        return h
    return h + w - 1


def _make_exact_mask(h: int, w: int, area: int) -> np.ndarray:
    assert 1 <= area <= h * w
    if area == h * w:
        return np.ones((h, w), dtype=bool)

    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)
        while int(mask.sum()) > area:
            candidates = [(r, c) for r in range(h) for c in range(w) if bool(mask[r, c])]
            random.shuffle(candidates)
            removed = False
            for r, c in candidates:
                mask[r, c] = False
                if int(mask.sum()) >= area and _touches_all_edges(mask) and _mask_connected(mask):
                    removed = True
                    break
                mask[r, c] = True
            if not removed:
                break
        if int(mask.sum()) == area and _touches_all_edges(mask) and _mask_connected(mask):
            return mask

    raise RuntimeError(f"could not build a connected {h}x{w} mask with area {area}")


def _make_sprite(color: int, h: int, w: int, area: int) -> np.ndarray:
    mask = _make_exact_mask(h, w, area)
    sprite = np.full((h, w), BG, dtype=int)
    sprite[mask] = int(color)
    if random.random() < 0.5:
        sprite = np.fliplr(sprite)
    if random.random() < 0.5:
        sprite = np.flipud(sprite)
    return sprite


def _sample_group_sprites(base_w: int, count: int, color: int) -> list[np.ndarray] | None:
    for _ in range(120):
        dims: list[tuple[int, int]] = []
        width = random.randint(max(3, base_w - 1), base_w)
        height = random.randint(2, 4)
        dims.append((height, width))
        for idx in range(1, count):
            if idx < 3:
                width = max(1, width - random.randint(1, 2))
                height = max(1, height - random.randint(0, 1))
            else:
                width = max(1, width - random.randint(0, 1))
                height = max(1, height - random.randint(0, 1))
            dims.append((height, width))

        sprites: list[np.ndarray] = []
        prev_area = 10**9
        ok = True
        for h, w in dims:
            bbox_area = h * w
            low = max(_minimum_edge_spanning_area(h, w), math.ceil(0.6 * bbox_area))
            high = min(bbox_area, prev_area - 1)
            if low > high:
                ok = False
                break
            dense_low = max(low, high - max(1, bbox_area // 3))
            area = random.randint(dense_low, high)
            try:
                sprite = _make_sprite(color, h, w, area)
            except RuntimeError:
                ok = False
                break
            sprites.append(sprite)
            prev_area = area

        if ok:
            return sprites

    return None


def _component_record_from_sprite(sprite: np.ndarray) -> dict:
    rows, cols = np.where(sprite != BG)
    return {
        "color": int(sprite[rows[0], cols[0]]),
        "area": int(len(rows)),
        "bbox": (0, 0, sprite.shape[0] - 1, sprite.shape[1] - 1),
        "sprite": sprite,
    }


def _can_place_scattered(grid: np.ndarray, sprite: np.ndarray, row: int, col: int) -> bool:
    h, w = sprite.shape
    if row < 0 or col < 0 or row + h > grid.shape[0] - 1 or col + w > grid.shape[1]:
        return False
    if row + h > grid.shape[0] - 2:
        return False

    rr0 = max(0, row - 1)
    rr1 = min(grid.shape[0], row + h + 1)
    cc0 = max(0, col - 1)
    cc1 = min(grid.shape[1], col + w + 1)
    return not bool(np.any(grid[rr0:rr1, cc0:cc1] != BG))


def _place_scattered(grid: np.ndarray, sprites: list[np.ndarray]) -> bool:
    ordered = sorted(
        sprites,
        key=lambda sprite: (int(np.count_nonzero(sprite != BG)), sprite.shape[0] * sprite.shape[1]),
        reverse=True,
    )
    H, W = grid.shape
    for sprite in ordered:
        h, w = sprite.shape
        max_row = H - 2 - h
        if max_row < 0:
            return False
        positions = [(r, c) for r in range(max_row + 1) for c in range(W - w + 1)]
        random.shuffle(positions)
        placed = False
        for row, col in positions:
            if _can_place_scattered(grid, sprite, row, col):
                _paste(grid, sprite, row, col, bg=BG)
                placed = True
                break
        if not placed:
            return False
    return True


def _sample_layout() -> tuple[int, int, int, int, int] | None:
    H = random.randint(18, 25)
    W = random.randint(22, 30)
    pedestal_count = random.randint(2, 4)
    gap = random.randint(1, 2)
    max_base_w = (W - 2 - gap * (pedestal_count - 1)) // pedestal_count
    if max_base_w < 4:
        return None
    base_w = random.randint(4, min(6, max_base_w))
    return H, W, pedestal_count, base_w, gap


def _build_pedestals(W: int, pedestal_count: int, base_w: int, gap: int, colors: list[int]) -> list[dict]:
    total_span = pedestal_count * base_w + (pedestal_count - 1) * gap
    left_margin = random.randint(1, W - total_span - 1)
    pedestals: list[dict] = []
    c = left_margin
    for color in colors:
        pedestals.append({"color": int(color), "c0": c, "c1": c + base_w - 1, "width": base_w})
        c += base_w + gap
    return pedestals


def _choose_counts(pedestal_count: int) -> list[int]:
    max_layers = random.choice([1, 2, 3, 4])
    counts = [random.randint(1, max_layers) for _ in range(pedestal_count)]
    counts[random.randrange(pedestal_count)] = max_layers
    return counts


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    H, W, pedestal_count, base_w, gap = layout
    colors = random.sample(PALETTE, pedestal_count)
    pedestals = _build_pedestals(W, pedestal_count, base_w, gap, colors)
    counts = _choose_counts(pedestal_count)

    groups: dict[int, list[dict]] = {}
    sprites_to_place: list[np.ndarray] = []
    max_visible_height = 0

    for color, count in zip(colors, counts):
        sprites = _sample_group_sprites(base_w, count, color)
        if sprites is None:
            return None
        visible_height = sum(sprite.shape[0] for sprite in sprites[:3])
        max_visible_height = max(max_visible_height, visible_height)
        groups[int(color)] = [_component_record_from_sprite(sprite) for sprite in sprites]
        sprites_to_place.extend(sprites)

    if max_visible_height > H - 2:
        return None

    expected = _render_stacks((H, W), pedestals, groups)
    grid = np.full((H, W), BG, dtype=int)
    for ped in pedestals:
        grid[H - 1, ped["c0"] : ped["c1"] + 1] = int(ped["color"])

    if not _place_scattered(grid, sprites_to_place):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
