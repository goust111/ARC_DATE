from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _scale_mask(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """Reorder the large layered pattern using the right-side template order."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    parts = _components(g != bg)
    assert len(parts) >= 3, "expected one large pattern and multiple stamp templates"

    infos = []
    for coords in parts:
        r0, c0, r1, c1 = _bbox(coords)
        infos.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": int(coords.shape[0]),
            }
        )

    left = max(infos, key=lambda info: info["area"])
    lr0, lc0, lr1, lc1 = left["bbox"]
    left_sub = g[lr0 : lr1 + 1, lc0 : lc1 + 1]

    stamps = []
    for info in infos:
        if info is left:
            continue
        r0, c0, r1, c1 = info["bbox"]
        assert c0 > lc1, "all small templates must sit to the right of the large pattern"
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, sub[sub != bg].ravel()))
        assert len(colors) == 1, "each stamp must be monochrome"
        color = next(iter(colors))
        stamps.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "mask": sub == color,
            }
        )

    stamps.sort(key=lambda info: (info["bbox"][0], info["bbox"][1]))
    assert len(stamps) >= 2

    small_sizes = {(info["mask"].shape[0], info["mask"].shape[1]) for info in stamps}
    assert len(small_sizes) == 1, "all stamps must share the same scaled canvas"
    small_h, small_w = next(iter(small_sizes))

    big_h, big_w = left_sub.shape
    assert big_h % small_h == 0 and big_w % small_w == 0
    scale_h = big_h // small_h
    scale_w = big_w // small_w
    assert scale_h == scale_w and scale_h >= 1
    scale = scale_h

    left_colors = set(map(int, left_sub[left_sub != bg].ravel()))
    stamp_colors = {info["color"] for info in stamps}
    assert left_colors == stamp_colors, "template colors must match the visible large-pattern colors"

    full_layers = []
    for info in stamps:
        full_mask = _scale_mask(info["mask"], scale)
        assert full_mask.shape == left_sub.shape
        visible = left_sub == info["color"]
        assert np.all(~visible | full_mask), "visible pixels must fit inside the reconstructed full layer"
        full_layers.append((info["color"], full_mask))

    out = g.copy()
    canvas = out[lr0 : lr1 + 1, lc0 : lc1 + 1]
    canvas[:] = bg
    for color, full_mask in full_layers:
        canvas[full_mask] = color
    return out


def _random_path(
    start: tuple[int, int], end: tuple[int, int], forbidden: set[tuple[int, int]]
) -> set[tuple[int, int]]:
    r, c = start
    cells = {(r, c)}
    while (r, c) != end:
        moves = []
        if r < end[0]:
            moves.append((r + 1, c))
        elif r > end[0]:
            moves.append((r - 1, c))
        if c < end[1]:
            moves.append((r, c + 1))
        elif c > end[1]:
            moves.append((r, c - 1))
        random.shuffle(moves)
        chosen = None
        for nr, nc in moves:
            if (nr, nc) == end or (nr, nc) not in forbidden:
                chosen = (nr, nc)
                break
        assert chosen is not None
        r, c = chosen
        cells.add((r, c))
    return cells


def _side_anchor(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(1, w - 2)
    if side == "bottom":
        return h - 1, random.randint(1, w - 2)
    if side == "left":
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _generate_mask(
    h: int,
    w: int,
    hub: tuple[int, int],
    own_corner: tuple[int, int],
    all_corners: list[tuple[int, int]],
) -> np.ndarray:
    forbidden = set(all_corners) - {own_corner}
    side_map = {
        (0, 0): ("bottom", "right"),
        (0, w - 1): ("bottom", "left"),
        (h - 1, 0): ("top", "right"),
        (h - 1, w - 1): ("top", "left"),
    }
    cells = {hub}
    anchors = [own_corner] + [_side_anchor(side, h, w) for side in side_map[own_corner]]
    for anchor in anchors:
        cells |= _random_path(hub, anchor, forbidden)

    max_area = h * w - len(forbidden)
    target_area = random.randint(len(cells), min(max_area, len(cells) + random.randint(2, h + w - 2)))
    attempts = 0
    while len(cells) < target_area and attempts < 200:
        attempts += 1
        candidates = []
        for r in range(h):
            for c in range(w):
                if (r, c) in cells or (r, c) in forbidden:
                    continue
                if any(
                    0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) in cells
                    for dr, dc in NEIGHBORS4
                ):
                    candidates.append((r, c))
        if not candidates:
            break
        cells.add(random.choice(candidates))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _compose_layers(
    masks: list[np.ndarray], colors: list[int], order: tuple[int, ...], scale: int
) -> np.ndarray:
    h, w = masks[0].shape
    canvas = np.full((h * scale, w * scale), BG, dtype=int)
    for idx in order:
        full_mask = _scale_mask(masks[idx], scale)
        canvas[full_mask] = colors[idx]
    return canvas


def _visible_regions(masks: list[np.ndarray], order: tuple[int, ...]) -> dict[int, np.ndarray]:
    covered = np.zeros_like(masks[0], dtype=bool)
    visible: dict[int, np.ndarray] = {}
    for idx in reversed(order):
        visible[idx] = masks[idx] & ~covered
        covered |= masks[idx]
    return visible


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    layer_count = random.randint(2, 4)
    stamp_h = random.randint(4, 6)
    stamp_w = random.randint(max(4, layer_count), 6)
    hub = (random.randint(1, stamp_h - 2), random.randint(1, stamp_w - 2))

    corners = random.sample(
        [(0, 0), (0, stamp_w - 1), (stamp_h - 1, 0), (stamp_h - 1, stamp_w - 1)],
        layer_count,
    )
    masks = [_generate_mask(stamp_h, stamp_w, hub, corner, corners) for corner in corners]
    mask_keys = {tuple("".join("1" if cell else "0" for cell in row) for row in mask) for mask in masks}
    if len(mask_keys) != layer_count:
        return None

    union = np.zeros((stamp_h, stamp_w), dtype=bool)
    for mask in masks:
        union |= mask
    if len(_components(union)) != 1:
        return None

    perms = list(itertools.permutations(range(layer_count)))
    stamp_order = random.choice(perms)
    input_choices = [perm for perm in perms if perm != stamp_order and perm[-1] != stamp_order[-1]]
    input_order = random.choice(input_choices)

    input_visible = _visible_regions(masks, input_order)
    output_visible = _visible_regions(masks, stamp_order)
    if any(not input_visible[idx].any() for idx in range(layer_count)):
        return None
    if any(not output_visible[idx].any() for idx in range(layer_count)):
        return None

    scale = random.randint(2, 4)
    colors = random.sample(PALETTE, layer_count)
    left_input = _compose_layers(masks, colors, input_order, scale)
    left_output = _compose_layers(masks, colors, stamp_order, scale)
    if np.array_equal(left_input, left_output):
        return None

    big_h, big_w = left_input.shape
    left_top = random.randint(1, 3)
    left_left = random.randint(1, 3)
    stamp_top = random.randint(1, 3)
    gap_cols = random.randint(3, 6)
    gap_rows = random.randint(2, 4)
    stamp_left = left_left + big_w + gap_cols
    stamp_block_h = layer_count * stamp_h + (layer_count - 1) * gap_rows
    grid_h = max(left_top + big_h, stamp_top + stamp_block_h) + random.randint(1, 3)
    grid_w = stamp_left + stamp_w + random.randint(1, 3)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[left_top : left_top + big_h, left_left : left_left + big_w] = left_input

    cur_row = stamp_top
    for idx in stamp_order:
        stamp = inp[cur_row : cur_row + stamp_h, stamp_left : stamp_left + stamp_w]
        stamp[masks[idx]] = colors[idx]
        cur_row += stamp_h + gap_rows

    out = inp.copy()
    out[left_top : left_top + big_h, left_left : left_left + big_w] = left_output

    parts = _components(inp != BG)
    if len(parts) != layer_count + 1:
        return None
    areas = sorted((int(part.shape[0]) for part in parts), reverse=True)
    if not (areas[0] > areas[1]):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input")
