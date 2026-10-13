from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GEN_ATTEMPTS = 400
PALETTE_WIDTHS = (4, 5, 6)
GRID_HEIGHT_RANGE = (20, 29)
GRID_WIDTH_RANGE = (22, 31)
GROUP_COUNT_OPTIONS = (1, 2, 3)
LAYER_COUNT_OPTIONS = (2, 3, 4)
SHAPE_KINDS = ("rectangle", "octagon")
PALETTE_MODES = ("rotate", "reverse_rotate")
PLACEMENT_PADDING = 2
SLOT_GAP = 2
MIN_SLOT_SIDE = 9


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _erode4(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    return (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )


def _inside_mask(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    reachable = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if not blocked[r, c] and not reachable[r, c]:
            reachable[r, c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not reachable[nr, nc]:
                reachable[nr, nc] = True
                q.append((nr, nc))

    return (~blocked) & (~reachable)


def _find_palette(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = g.shape
    assert h >= 2 and w >= 4, "grid too small for palette"
    for start in range(0, w - 2):
        width = w - start
        if width < 4:
            continue
        if not np.all(g[: h - 2, start:w] == BG):
            continue
        top = g[h - 2, start:w]
        bottom = g[h - 1, start:w]
        if not np.all(top != BG) or not np.all(bottom != BG):
            continue
        top_list = [int(v) for v in top.tolist()]
        if len(set(top_list)) != width:
            continue
        return start, top_list, [int(v) for v in bottom.tolist()]
    raise AssertionError("failed to find bottom-right palette")


def _mix_color(a: int, b: int, top_index: dict[int, int], bottom: list[int]) -> int:
    assert a in top_index and b in top_index, "mixed colors must be present in palette keys"
    return int(bottom[(top_index[a] + top_index[b]) % len(bottom)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the lower-right palette, recursively mix each nested contour group's
    colors from outer to inner, fill the innermost hole, and erase the palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette_start, top_colors, bottom_colors = _find_palette(g)
    top_index = {color: idx for idx, color in enumerate(top_colors)}

    palette_mask = np.zeros_like(g, dtype=bool)
    palette_mask[g.shape[0] - 2 :, palette_start :] = True

    contour_mask = (g != BG) & (~palette_mask)
    raw_components = _components(contour_mask)
    assert raw_components, "expected at least one contour component"

    components: list[dict[str, object]] = []
    for coords in raw_components:
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert colors.size == 1, "each contour must be monochrome"
        blocked = np.zeros_like(contour_mask, dtype=bool)
        blocked[coords[:, 0], coords[:, 1]] = True
        inside = _inside_mask(blocked)
        hole = inside & (g == BG) & (~palette_mask)
        assert np.any(hole), "every contour must enclose background"
        components.append(
            {
                "coords": coords,
                "color": int(colors[0]),
                "inside": inside,
                "inside_area": int(inside.sum()),
            }
        )

    n = len(components)
    parent = [-1] * n
    children: list[list[int]] = [[] for _ in range(n)]
    for child_id in range(n):
        coords = components[child_id]["coords"]
        assert isinstance(coords, np.ndarray)
        containers: list[int] = []
        for parent_id in range(n):
            if parent_id == child_id:
                continue
            inside = components[parent_id]["inside"]
            assert isinstance(inside, np.ndarray)
            if np.all(inside[coords[:, 0], coords[:, 1]]):
                containers.append(parent_id)
        if containers:
            best_parent = min(containers, key=lambda idx: int(components[idx]["inside_area"]))
            parent[child_id] = int(best_parent)
            children[best_parent].append(child_id)

    out = g.copy()
    for root in range(n):
        if parent[root] != -1:
            continue
        chain = [root]
        cur = root
        while children[cur]:
            assert len(children[cur]) == 1, "each nested group must form a single chain"
            cur = children[cur][0]
            chain.append(cur)
        assert len(chain) >= 2, "each group must have at least two contours"

        layer_colors = [int(components[idx]["color"]) for idx in chain]
        mixed = _mix_color(layer_colors[0], layer_colors[1], top_index, bottom_colors)
        for color in layer_colors[2:]:
            mixed = _mix_color(mixed, color, top_index, bottom_colors)

        inner_inside = components[chain[-1]]["inside"]
        assert isinstance(inner_inside, np.ndarray)
        hole = inner_inside & (g == BG) & (~palette_mask)
        out[hole] = mixed

    out[palette_mask] = BG
    return out


def _rectangle_mask(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _octagon_mask(h: int, w: int, cut: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    keep = (
        (rows + cols >= cut)
        & (rows + (w - 1 - cols) >= cut)
        & ((h - 1 - rows) + cols >= cut)
        & ((h - 1 - rows) + (w - 1 - cols) >= cut)
    )
    return keep.astype(bool)


def _contours_from_base(base: np.ndarray, layers: int) -> tuple[list[np.ndarray], np.ndarray] | None:
    current = base.copy()
    contours: list[np.ndarray] = []
    for layer_idx in range(layers):
        inner = _erode4(current)
        contour = current & (~inner)
        if not np.any(contour):
            return None
        if len(_components(contour)) != 1:
            return None
        contours.append(contour)
        if layer_idx == layers - 1:
            fill = inner
        else:
            current = _erode4(inner)
            if not np.any(current):
                return None
    if not np.any(fill):
        return None
    if len(_components(fill)) != 1:
        return None
    return contours, fill


def _sample_base_shape(layers: int, kind: str, max_h: int, max_w: int) -> np.ndarray | None:
    min_dim = 4 * layers - 1
    if kind == "rectangle":
        if max_h < min_dim or max_w < min_dim:
            return None
        h = random.randint(min_dim, min(max_h, min_dim + 6))
        w = random.randint(min_dim, min(max_w, min_dim + 6))
        return _rectangle_mask(h, w)
    if kind == "octagon":
        min_oct = min_dim + 2
        if max_h < min_oct or max_w < min_oct:
            return None
        h = random.randint(min_oct, min(max_h, min_dim + 8))
        w = random.randint(min_oct, min(max_w, min_dim + 8))
        cut_hi = min(h, w) // 4
        if cut_hi < 1:
            return None
        for _ in range(20):
            cut = random.randint(1, cut_hi)
            if w - 2 * cut >= 3 and h - 2 * cut >= 3:
                return _octagon_mask(h, w, cut)
        return None
    raise AssertionError("unknown shape kind")


def _build_palette(width: int) -> tuple[list[int], list[int]]:
    top = random.sample(list(range(1, 10)), width)
    mode = random.choice(PALETTE_MODES)
    shift = random.randrange(width)
    if mode == "rotate":
        bottom = top[shift:] + top[:shift]
    else:
        rev = list(reversed(top))
        bottom = rev[shift:] + rev[:shift]
    return top, bottom


def _mix_sequence(colors: list[int], top_index: dict[int, int], bottom: list[int]) -> int:
    mixed = _mix_color(colors[0], colors[1], top_index, bottom)
    for color in colors[2:]:
        mixed = _mix_color(mixed, color, top_index, bottom)
    return mixed


def _make_group_patch(
    top: list[int],
    bottom: list[int],
    layers: int,
    kind: str,
    max_h: int,
    max_w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    top_index = {color: idx for idx, color in enumerate(top)}
    for _ in range(80):
        base = _sample_base_shape(layers, kind, max_h, max_w)
        if base is None:
            continue
        derived = _contours_from_base(base, layers)
        if derived is None:
            continue
        contour_masks, fill = derived
        colors = random.sample(top, layers)
        patch_in = np.zeros(base.shape, dtype=int)
        for mask, color in zip(contour_masks, colors, strict=False):
            patch_in[mask] = color
        patch_out = patch_in.copy()
        patch_out[fill] = _mix_sequence(colors, top_index, bottom)
        return patch_in, patch_out
    return None


def _paste_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[patch != BG] = patch[patch != BG]


def _rand_split(total: int, left_min: int, right_min: int, gap: int) -> int | None:
    lo = left_min
    hi = total - right_min - gap
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _slot_options(draw_h: int, draw_w: int) -> dict[int, list[list[tuple[int, int, int, int]]]]:
    options: dict[int, list[list[tuple[int, int, int, int]]]] = {1: [[(0, draw_h, 0, draw_w)]], 2: [], 3: []}

    split_w = _rand_split(draw_w, MIN_SLOT_SIDE, MIN_SLOT_SIDE, SLOT_GAP)
    if split_w is not None:
        options[2].append([(0, draw_h, 0, split_w), (0, draw_h, split_w + SLOT_GAP, draw_w)])

    split_h = _rand_split(draw_h, MIN_SLOT_SIDE, MIN_SLOT_SIDE, SLOT_GAP)
    if split_h is not None:
        options[2].append([(0, split_h, 0, draw_w), (split_h + SLOT_GAP, draw_h, 0, draw_w)])

    left_w = _rand_split(draw_w, MIN_SLOT_SIDE, MIN_SLOT_SIDE, SLOT_GAP)
    if left_w is not None:
        right_w0 = left_w + SLOT_GAP
        right_h = _rand_split(draw_h, MIN_SLOT_SIDE, MIN_SLOT_SIDE, SLOT_GAP)
        if right_h is not None:
            options[3].append(
                [
                    (0, draw_h, 0, left_w),
                    (0, right_h, right_w0, draw_w),
                    (right_h + SLOT_GAP, draw_h, right_w0, draw_w),
                ]
            )

    top_h = _rand_split(draw_h, MIN_SLOT_SIDE, MIN_SLOT_SIDE, SLOT_GAP)
    if top_h is not None:
        bottom_h0 = top_h + SLOT_GAP
        bottom_w = _rand_split(draw_w, MIN_SLOT_SIDE, MIN_SLOT_SIDE, SLOT_GAP)
        if bottom_w is not None:
            options[3].append(
                [
                    (0, top_h, 0, draw_w),
                    (bottom_h0, draw_h, 0, bottom_w),
                    (bottom_h0, draw_h, bottom_w + SLOT_GAP, draw_w),
                ]
            )

    return options


def _max_layers_for_slot(slot_h: int, slot_w: int) -> int:
    bound = min(slot_h, slot_w)
    if bound >= 15:
        return 4
    if bound >= 11:
        return 3
    if bound >= 7:
        return 2
    return 0


def _place_patch_in_slot(
    grid_in: np.ndarray,
    grid_out: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    slot: tuple[int, int, int, int],
) -> bool:
    r0, r1, c0, c1 = slot
    slot_h = r1 - r0
    slot_w = c1 - c0
    ph, pw = patch_in.shape
    margin_r = 1 if slot_h - ph >= 2 else 0
    margin_c = 1 if slot_w - pw >= 2 else 0
    min_r = r0 + margin_r
    max_r = r1 - ph - margin_r
    min_c = c0 + margin_c
    max_c = c1 - pw - margin_c
    if min_r > max_r or min_c > max_c:
        return False
    pr = random.randint(min_r, max_r)
    pc = random.randint(min_c, max_c)
    _paste_patch(grid_in, patch_in, pr, pc)
    _paste_patch(grid_out, patch_out, pr, pc)
    return True


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    palette_w = random.choice(PALETTE_WIDTHS)
    draw_h = h - 3
    draw_w = w - palette_w - 2
    if draw_h < 8 or draw_w < 8:
        return None

    top, bottom = _build_palette(palette_w)
    grid_in = np.zeros((h, w), dtype=int)
    grid_out = np.zeros((h, w), dtype=int)
    slot_options = _slot_options(draw_h, draw_w)
    group_choices = [count for count in GROUP_COUNT_OPTIONS if slot_options[count]]
    if not group_choices:
        return None
    group_count = random.choice(group_choices)
    slots = random.choice(slot_options[group_count])
    random.shuffle(slots)

    for slot in slots:
        r0, r1, c0, c1 = slot
        slot_h = r1 - r0
        slot_w = c1 - c0
        usable_h = slot_h - 2 if slot_h >= 9 else slot_h
        usable_w = slot_w - 2 if slot_w >= 9 else slot_w
        max_layers = _max_layers_for_slot(usable_h, usable_w)
        if max_layers < 2:
            return None
        layer_choices = [layers for layers in LAYER_COUNT_OPTIONS if layers <= max_layers]
        if not layer_choices:
            return None
        placed = False
        for _ in range(80):
            layers = random.choice(layer_choices)
            kind = random.choice(SHAPE_KINDS)
            made = _make_group_patch(top, bottom, layers, kind, usable_h, usable_w)
            if made is None:
                continue
            patch_in, patch_out = made
            if _place_patch_in_slot(grid_in, grid_out, patch_in, patch_out, slot):
                placed = True
                break
        if not placed:
            return None

    palette_start = w - palette_w
    grid_in[h - 2, palette_start:w] = np.array(top, dtype=int)
    grid_in[h - 1, palette_start:w] = np.array(bottom, dtype=int)

    if np.array_equal(main(grid_in), grid_out):
        return grid_in
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
