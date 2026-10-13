from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

OBSERVED_FRAMES = 4
TOTAL_FRAMES = 5
MAX_ATTEMPTS = 1000

OFFSET_CHOICES = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, 1),
    (1, -1),
    (0, 0),
]


def _shift_mask(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    dr = int(dr)
    dc = int(dc)

    src_r0 = max(0, -dr)
    src_r1 = min(h, h - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(w, w - dc)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out

    dst_r0 = max(0, dr)
    dst_r1 = min(h, h + dr)
    dst_c0 = max(0, dc)
    dst_c1 = min(w, w + dc)
    out[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _shift_array(grid: np.ndarray, dr: int, dc: int, fill: int = BG) -> np.ndarray:
    out = np.full(grid.shape, int(fill), dtype=int)
    h, w = grid.shape
    dr = int(dr)
    dc = int(dc)

    src_r0 = max(0, -dr)
    src_r1 = min(h, h - dr)
    src_c0 = max(0, -dc)
    src_c1 = min(w, w - dc)
    if src_r0 >= src_r1 or src_c0 >= src_c1:
        return out

    dst_r0 = max(0, dr)
    dst_r1 = min(h, h + dr)
    dst_c0 = max(0, dc)
    dst_c1 = min(w, w + dc)
    out[dst_r0:dst_r1, dst_c0:dst_c1] = grid[src_r0:src_r1, src_c0:src_c1]
    return out


def _compose_panel(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    out = np.zeros(mask_a.shape, dtype=int)
    overlap = mask_a & mask_b
    out[mask_a & ~mask_b] = BLUE
    out[mask_b & ~mask_a] = RED
    out[overlap] = YELLOW
    return out


def _decode_layer(panel: np.ndarray, layer_color: int) -> np.ndarray:
    if layer_color == BLUE:
        return (panel == BLUE) | (panel == YELLOW)
    if layer_color == RED:
        return (panel == RED) | (panel == YELLOW)
    raise ValueError("unknown layer color")


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    separator_rows = [r for r in range(g.shape[0]) if np.all(g[r] == GREY)]
    assert len(separator_rows) == 3, "expected exactly three grey separator rows"

    panel_h = separator_rows[0]
    assert panel_h > 0, "panels must be non-empty"
    expected = [panel_h + i * (panel_h + 1) for i in range(3)]
    assert separator_rows == expected, "separator rows must be evenly spaced"
    assert g.shape[0] == 4 * panel_h + 3, "unexpected stacked-panel height"

    panels = []
    start = 0
    for sep in separator_rows:
        panels.append(g[start:sep, :].copy())
        start = sep + 1
    panels.append(g[start:, :].copy())
    assert len({panel.shape for panel in panels}) == 1, "panel sizes must match"
    return panels


def _find_unique_translation(src: np.ndarray, dst: np.ndarray) -> tuple[int, int]:
    src = np.asarray(src, dtype=bool)
    dst = np.asarray(dst, dtype=bool)
    assert src.shape == dst.shape
    assert int(src.sum()) == int(dst.sum()) > 0, "layer area must be preserved"

    h, w = src.shape
    candidates: list[tuple[int, int]] = []
    for dr in range(-h, h + 1):
        for dc in range(-w, w + 1):
            if np.array_equal(_shift_mask(src, dr, dc), dst):
                candidates.append((dr, dc))
    assert len(candidates) == 1, "expected a unique layer translation"
    return candidates[0]


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    h, w = mask.shape
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _safe_window(h: int, w: int, dr: int, dc: int) -> np.ndarray:
    r0 = max(0, -4 * dr)
    r1 = h - 1 - max(0, 4 * dr)
    c0 = max(0, -4 * dc)
    c1 = w - 1 - max(0, 4 * dc)
    out = np.zeros((h, w), dtype=bool)
    if r0 <= r1 and c0 <= c1:
        out[r0 : r1 + 1, c0 : c1 + 1] = True
    return out


def _manhattan_path(a: tuple[int, int], b: tuple[int, int], row_first: bool) -> list[tuple[int, int]]:
    r, c = map(int, a)
    tr, tc = map(int, b)
    path = [(r, c)]
    if row_first:
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


def _build_connected_seed(
    h: int,
    w: int,
    required: list[tuple[int, int]],
    allowed: np.ndarray,
) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in required:
        if not (0 <= r < h and 0 <= c < w and allowed[r, c]):
            return None
        mask[r, c] = True
    for p, q in zip(required, required[1:]):
        path = _manhattan_path(p, q, row_first=bool(np.random.randint(2)))
        for r, c in path:
            if not allowed[r, c]:
                return None
            mask[r, c] = True
    return mask


def _grow_connected(mask: np.ndarray, allowed: np.ndarray, target: int) -> np.ndarray | None:
    mask = mask.copy()
    if int(mask.sum()) > target:
        return None
    while int(mask.sum()) < target:
        frontier = np.zeros_like(mask, dtype=bool)
        for r, c in np.argwhere(mask):
            r = int(r)
            c = int(c)
            if r > 0:
                frontier[r - 1, c] = True
            if r + 1 < mask.shape[0]:
                frontier[r + 1, c] = True
            if c > 0:
                frontier[r, c - 1] = True
            if c + 1 < mask.shape[1]:
                frontier[r, c + 1] = True
        frontier &= allowed & ~mask
        choices = np.argwhere(frontier)
        if len(choices) == 0:
            return None
        rr, cc = choices[np.random.randint(len(choices))]
        mask[int(rr), int(cc)] = True
    return mask


def _colors_ok(panel: np.ndarray) -> bool:
    return (
        np.any(panel == BLUE)
        and np.any(panel == RED)
        and np.any(panel == YELLOW)
    )


def _composite_has_constant_translation(frames: list[np.ndarray]) -> bool:
    h, w = frames[0].shape
    for dr in range(-h, h + 1):
        for dc in range(-w, w + 1):
            if all(np.array_equal(_shift_array(frames[i], dr, dc), frames[i + 1]) for i in range(len(frames) - 1)):
                return True
    return False


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    offset_a = OFFSET_CHOICES[np.random.randint(len(OFFSET_CHOICES))]
    offset_b = OFFSET_CHOICES[np.random.randint(len(OFFSET_CHOICES))]
    if offset_a == offset_b or (offset_a == (0, 0) and offset_b == (0, 0)):
        return None

    delta = (offset_a[0] - offset_b[0], offset_a[1] - offset_b[1])
    if delta == (0, 0):
        return None

    panel_h = 6 if (offset_a[0] != 0 or offset_b[0] != 0) else int(np.random.randint(5, 7))
    panel_w = int(np.random.randint(6, 9))

    safe_a = _safe_window(panel_h, panel_w, offset_a[0], offset_a[1])
    safe_b = _safe_window(panel_h, panel_w, offset_b[0], offset_b[1])
    if not safe_a.any() or not safe_b.any():
        return None

    include_overlap_in_output = bool(np.random.randint(2))
    needed_steps = 5 if include_overlap_in_output else 4
    anchor_ok = np.zeros((panel_h, panel_w), dtype=bool)
    for r in range(panel_h):
        for c in range(panel_w):
            if not safe_a[r, c]:
                continue
            good = True
            for t in range(needed_steps):
                br = r + t * delta[0]
                bc = c + t * delta[1]
                if not (0 <= br < panel_h and 0 <= bc < panel_w and safe_b[br, bc]):
                    good = False
                    break
            if good:
                anchor_ok[r, c] = True
    anchors = np.argwhere(anchor_ok)
    if len(anchors) == 0:
        return None
    anchor = tuple(map(int, anchors[np.random.randint(len(anchors))]))

    required_b = [
        (anchor[0] + t * delta[0], anchor[1] + t * delta[1])
        for t in range(needed_steps)
    ]

    seed_a = _build_connected_seed(panel_h, panel_w, [anchor], safe_a)
    seed_b = _build_connected_seed(panel_h, panel_w, required_b, safe_b)
    if seed_a is None or seed_b is None:
        return None

    max_a = min(6, int(safe_a.sum()))
    max_b = min(7, int(safe_b.sum()))
    if max_a < max(3, int(seed_a.sum())) or max_b < max(4, int(seed_b.sum())):
        return None
    target_a = int(np.random.randint(max(3, int(seed_a.sum())), max_a + 1))
    target_b = int(np.random.randint(max(4, int(seed_b.sum())), max_b + 1))

    base_a = _grow_connected(seed_a, safe_a, target_a)
    base_b = _grow_connected(seed_b, safe_b, target_b)
    if base_a is None or base_b is None:
        return None
    if not _is_connected(base_a) or not _is_connected(base_b):
        return None

    frames_a = [_shift_mask(base_a, t * offset_a[0], t * offset_a[1]) for t in range(TOTAL_FRAMES)]
    frames_b = [_shift_mask(base_b, t * offset_b[0], t * offset_b[1]) for t in range(TOTAL_FRAMES)]
    if any(int(frame.sum()) != int(base_a.sum()) for frame in frames_a):
        return None
    if any(int(frame.sum()) != int(base_b.sum()) for frame in frames_b):
        return None

    panels = [_compose_panel(frames_a[t], frames_b[t]) for t in range(TOTAL_FRAMES)]
    observed = panels[:OBSERVED_FRAMES]

    if any(not _colors_ok(panel) for panel in observed):
        return None

    overlap_counts = [int(np.sum(frames_a[t] & frames_b[t])) for t in range(OBSERVED_FRAMES)]
    if any(count == 0 for count in overlap_counts):
        return None
    if max(overlap_counts) > max(2, min(target_a, target_b) // 2):
        return None

    out = panels[-1]
    if any(np.array_equal(out, panel) for panel in observed):
        return None

    dec_a = [_decode_layer(panel, BLUE) for panel in observed]
    dec_b = [_decode_layer(panel, RED) for panel in observed]
    try:
        found_a = [_find_unique_translation(dec_a[i], dec_a[i + 1]) for i in range(OBSERVED_FRAMES - 1)]
        found_b = [_find_unique_translation(dec_b[i], dec_b[i + 1]) for i in range(OBSERVED_FRAMES - 1)]
    except AssertionError:
        return None
    if len(set(found_a)) != 1 or len(set(found_b)) != 1:
        return None
    if found_a[0] != offset_a or found_b[0] != offset_b:
        return None

    if _composite_has_constant_translation(observed):
        return None

    stacked = np.full((4 * panel_h + 3, panel_w), BG, dtype=int)
    row = 0
    for idx, panel in enumerate(observed):
        stacked[row : row + panel_h] = panel
        row += panel_h
        if idx < OBSERVED_FRAMES - 1:
            stacked[row] = GREY
            row += 1

    if not np.array_equal(main(stacked), out):
        return None
    return stacked, out


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate two hidden translated layers from four composite panels and recompose the fifth panel.

    Input structure:
    - The input is a vertical stack of four equal-size black panels separated by full-width grey rows.
    - Each panel contains blue cells, red cells, and yellow cells where the blue and red layers overlap.
    - Blue plus yellow belong to one hidden 4-connected layer, and red plus yellow belong to the other.

    Rule:
    - Split the grid at the grey separator rows into four time-ordered panels.
    - Decode layer A from blue-or-yellow cells and layer B from red-or-yellow cells in each panel.
    - For each layer, recover the unique constant translation between consecutive panels and verify it stays the same.
    - Extrapolate both layers one more step, then overlay them: blue for only A, red for only B, yellow for overlap, black otherwise.

    Key insight:
    - Yellow is not an independent object; it is the visible evidence that both moving layers occupy the same cell, so the sequence must be solved by separating the composite panels before extrapolating motion.
    """
    panels = _split_panels(np.asarray(grid, dtype=int))
    assert len(panels) == OBSERVED_FRAMES, "expected four observed panels"

    layer_a = [_decode_layer(panel, BLUE) for panel in panels]
    layer_b = [_decode_layer(panel, RED) for panel in panels]

    offsets_a = [_find_unique_translation(layer_a[i], layer_a[i + 1]) for i in range(OBSERVED_FRAMES - 1)]
    offsets_b = [_find_unique_translation(layer_b[i], layer_b[i + 1]) for i in range(OBSERVED_FRAMES - 1)]
    assert len(set(offsets_a)) == 1, "layer A must use one constant translation"
    assert len(set(offsets_b)) == 1, "layer B must use one constant translation"

    next_a = _shift_mask(layer_a[-1], offsets_a[0][0], offsets_a[0][1])
    next_b = _shift_mask(layer_b[-1], offsets_b[0][0], offsets_b[0][1])
    return _compose_panel(next_a, next_b)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        built = _build_instance()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid layered-panel instance")
