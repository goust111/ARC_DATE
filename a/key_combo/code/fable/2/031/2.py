from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(_ARC_UTILS_PATH))

from arc_utils import get_bg, paste


QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
COMMON_POSITION_ORDERS = (
    (0, 1, 2, 3),
    (0, 3, 2, 1),
    (3, 2, 1, 0),
    (1, 0, 3, 2),
)
GEN_ATTEMPTS = 400


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _find_separator(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for color in np.unique(grid):
        rows = [r for r in range(1, h - 1) if np.all(grid[r, :] == color)]
        cols = [c for c in range(1, w - 1) if np.all(grid[:, c] == color)]
        for r in rows:
            if h != 2 * r + 1:
                continue
            for c in cols:
                if w != 2 * c + 1:
                    continue
                candidates.append((int(color), int(r), int(c)))
    assert len(candidates) == 1, f"expected one separator cross, got {candidates}"
    return candidates[0]


def _split_panels(grid: np.ndarray) -> tuple[tuple[np.ndarray, ...], int]:
    sep_color, sep_r, sep_c = _find_separator(grid)
    ul = grid[:sep_r, :sep_c]
    ur = grid[:sep_r, sep_c + 1 :]
    ll = grid[sep_r + 1 :, :sep_c]
    lr = grid[sep_r + 1 :, sep_c + 1 :]
    panels = (ul, ur, ll, lr)
    assert len({panel.shape for panel in panels}) == 1, "quadrant panels must be equal-sized"
    return panels, sep_color


def _infer_panel_colors(
    panels: tuple[np.ndarray, ...], bg: int
) -> tuple[int, list[int]]:
    panel_colors = [set(map(int, np.unique(panel))) - {bg} for panel in panels]
    shared = set.intersection(*panel_colors)
    assert len(shared) == 1, f"expected one shared marker color, got {shared}"
    marker = shared.pop()

    body_colors: list[int] = []
    for colors in panel_colors:
        colors = set(colors)
        colors.discard(marker)
        assert len(colors) == 1, f"each panel must have one body color, got {colors}"
        body_colors.append(colors.pop())

    assert len(set(body_colors)) == 4, "panel body colors must be distinct"
    return int(marker), body_colors


def _compose_masks(
    masks: list[np.ndarray], body_colors: list[int], order: tuple[int, ...], bg: int
) -> np.ndarray:
    out = np.full(masks[0].shape, bg, dtype=int)
    for idx in order:
        patch = np.full_like(out, bg)
        patch[masks[idx]] = body_colors[idx]
        paste(out, patch, 0, 0, transparent=bg)
    return out


def _compose_panels(
    panels: tuple[np.ndarray, ...], body_colors: list[int], order: tuple[int, ...], bg: int
) -> np.ndarray:
    masks = [panel == color for panel, color in zip(panels, body_colors)]
    return _compose_masks(masks, body_colors, order, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Composite the four quadrant panels into one panel by using marker counts as the hidden depth order.

    Input structure:
    - One solid horizontal separator row and one solid vertical separator column split the canvas into four equal panels.
    - Each panel contains one unique body-color shape plus a sparse set of shared-color marker pixels on the panel background.
    - Corresponding coordinates across the four panels refer to the same output cell.

    Rule:
    - Remove the separator cross and work on the four equal panels only.
    - Infer the shared marker color from the non-background color that appears in all four panels, and count its pixels inside each panel.
    - Overlay the four body-color shapes onto one panel from largest marker count to smallest marker count; later non-background body pixels overwrite earlier ones, while marker pixels are never drawn.

    Key insight:
    - The visible overlap order is not tied to quadrant position or color identity; it is encoded only by how many sparse marker pixels each panel carries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    panels, _sep_color = _split_panels(g)
    bg = get_bg(np.stack(panels))
    marker, body_colors = _infer_panel_colors(panels, bg)

    counts = [int(np.count_nonzero(panel == marker)) for panel in panels]
    assert len(set(counts)) == 4, f"marker counts must be pairwise distinct, got {counts}"
    order = tuple(sorted(range(4), key=lambda idx: (-counts[idx], idx)))
    return _compose_panels(panels, body_colors, order, bg)


def _sample_self_avoiding_path(
    coarse_h: int, coarse_w: int, length: int
) -> list[tuple[int, int]] | None:
    def turn_count(path: list[tuple[int, int]]) -> int:
        turns = 0
        for a, b, c in zip(path, path[1:], path[2:]):
            if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
                turns += 1
        return turns

    for _ in range(200):
        start = (random.randrange(coarse_h), random.randrange(coarse_w))
        path = [start]
        used = {start}
        while len(path) < length:
            r, c = path[-1]
            options = [
                nbr
                for nbr in _neighbors4((coarse_h, coarse_w), r, c)
                if nbr not in used
            ]
            if not options:
                break
            options.sort(key=lambda rc: abs(rc[0] - coarse_h / 2) + abs(rc[1] - coarse_w / 2))
            pick_pool = options[: max(1, min(3, len(options)))]
            nxt = random.choice(pick_pool)
            path.append(nxt)
            used.add(nxt)
        if len(path) == length and turn_count(path) >= 2:
            return path
    return None


def _path_piece_masks(
    panel_shape: tuple[int, int], path: list[tuple[int, int]]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    h, w = panel_shape
    blocks: list[np.ndarray] = []
    for r, c in path:
        block = np.zeros((h, w), dtype=bool)
        r0 = 3 * r
        c0 = 3 * c
        block[r0 : r0 + 2, c0 : c0 + 2] = True
        blocks.append(block)

    corridors: list[np.ndarray] = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        corridor = np.zeros((h, w), dtype=bool)
        if r0 == r1:
            col = 3 * min(c0, c1) + 2
            corridor[3 * r0 : 3 * r0 + 2, col] = True
        else:
            row = 3 * min(r0, r1) + 2
            corridor[row, 3 * c0 : 3 * c0 + 2] = True
        corridors.append(corridor)
    return blocks, corridors


def _role_masks(panel_shape: tuple[int, int], path: list[tuple[int, int]]) -> list[np.ndarray]:
    blocks, corridors = _path_piece_masks(panel_shape, path)
    role_specs = (
        ((0, 1), (0,)),
        ((1, 2, 3), (1, 2)),
        ((3, 4, 5), (3, 4)),
        ((5, 6), (5,)),
    )
    masks: list[np.ndarray] = []
    for block_ids, corridor_ids in role_specs:
        mask = np.zeros(panel_shape, dtype=bool)
        for idx in block_ids:
            mask |= blocks[idx]
        for idx in corridor_ids:
            mask |= corridors[idx]
        masks.append(mask)
    return masks


def _grow_exclusive(mask: np.ndarray, allowed: np.ndarray, target: int) -> tuple[np.ndarray, np.ndarray]:
    out = mask.copy()
    free = allowed.copy()
    for _ in range(target):
        boundary = []
        coords = np.argwhere(out)
        for r, c in coords:
            for nr, nc in _neighbors4(out.shape, int(r), int(c)):
                if free[nr, nc] and not out[nr, nc]:
                    boundary.append((nr, nc))
        if not boundary:
            break
        boundary = sorted(set(boundary), key=lambda rc: (random.random(), rc[0], rc[1]))
        nr, nc = boundary[0]
        out[nr, nc] = True
        free[nr, nc] = False
    return out, free


def _sample_sparse_cells(free_mask: np.ndarray, count: int) -> list[tuple[int, int]] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(free_mask)]
    if len(coords) < count:
        return None
    for _ in range(40):
        random.shuffle(coords)
        chosen: list[tuple[int, int]] = []
        for r, c in coords:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return random.sample(coords, count)


def _visible_owner_map(order: tuple[int, ...], masks: list[np.ndarray]) -> np.ndarray:
    owners = np.full(masks[0].shape, -1, dtype=int)
    for idx in order:
        owners[masks[idx]] = idx
    return owners


def _render_input(
    masks: list[np.ndarray],
    body_colors: list[int],
    marker_color: int,
    marker_counts: list[int],
    bg: int,
    sep_color: int,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    panel_h, panel_w = masks[0].shape
    grid = np.full((2 * panel_h + 1, 2 * panel_w + 1), bg, dtype=int)
    grid[panel_h, :] = sep_color
    grid[:, panel_w] = sep_color

    panels: list[np.ndarray] = []
    for idx, (qr, qc) in enumerate(QUADRANTS):
        panel = np.full((panel_h, panel_w), bg, dtype=int)
        panel[masks[idx]] = body_colors[idx]
        marker_cells = _sample_sparse_cells(~masks[idx], marker_counts[idx])
        assert marker_cells is not None
        for r, c in marker_cells:
            panel[r, c] = marker_color
        r0 = qr * (panel_h + 1)
        c0 = qc * (panel_w + 1)
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
        panels.append(panel)
    return grid, tuple(panels)


def _shortcut_orders(
    areas: list[int], totals: list[int], body_colors: list[int]
) -> list[tuple[int, ...]]:
    orders = list(COMMON_POSITION_ORDERS)
    orders.append(tuple(sorted(range(4), key=lambda idx: (-areas[idx], idx))))
    orders.append(tuple(sorted(range(4), key=lambda idx: (areas[idx], idx))))
    orders.append(tuple(sorted(range(4), key=lambda idx: (-totals[idx], idx))))
    orders.append(tuple(sorted(range(4), key=lambda idx: (totals[idx], idx))))
    orders.append(tuple(sorted(range(4), key=lambda idx: (body_colors[idx], idx))))
    orders.append(tuple(sorted(range(4), key=lambda idx: (-body_colors[idx], idx))))
    deduped: list[tuple[int, ...]] = []
    for order in orders:
        if order not in deduped:
            deduped.append(order)
    return deduped


def _valid_semantics(
    inp: np.ndarray,
    panels: tuple[np.ndarray, ...],
    masks: list[np.ndarray],
    body_colors: list[int],
    marker_counts: list[int],
    order: tuple[int, ...],
    bg: int,
) -> bool:
    expected = _compose_masks(masks, body_colors, order, bg)
    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    owners = _visible_owner_map(order, masks)
    if any(not np.any(owners == idx) for idx in range(4)):
        return False
    if any(not np.any(masks[idx] & np.logical_or.reduce([masks[j] for j in range(4) if j != idx])) for idx in range(4)):
        return False

    target_sig = expected.tobytes()
    for perm in itertools.permutations(range(4)):
        if perm == order:
            continue
        if _compose_masks(masks, body_colors, perm, bg).tobytes() == target_sig:
            return False

    areas = [int(mask.sum()) for mask in masks]
    totals = [area + marker_counts[idx] for idx, area in enumerate(areas)]
    for shortcut in _shortcut_orders(areas, totals, body_colors):
        if shortcut == order:
            return False
    if any(np.array_equal(panel, expected) for panel in panels):
        return False
    return True


def _attempt() -> np.ndarray | None:
    coarse_h = random.choice((4, 5))
    coarse_w = random.choice((4, 5))
    panel_shape = (3 * coarse_h - 1, 3 * coarse_w - 1)
    path = _sample_self_avoiding_path(coarse_h, coarse_w, 7)
    if path is None:
        return None

    order = tuple(random.sample(range(4), 4))
    role_masks = _role_masks(panel_shape, path)
    masks = [np.zeros(panel_shape, dtype=bool) for _ in range(4)]
    for depth, panel_idx in enumerate(order):
        masks[panel_idx] = role_masks[depth].copy()

    union_base = np.logical_or.reduce(masks)
    reserved = union_base.copy()
    for panel_idx in random.sample(range(4), 4):
        target_extra = random.randint(1, 8)
        allowed = ~reserved
        grown, _unused = _grow_exclusive(masks[panel_idx], allowed, target_extra)
        new_cells = grown & ~masks[panel_idx]
        masks[panel_idx] = grown
        reserved |= new_cells

    areas = [int(mask.sum()) for mask in masks]
    if min(areas) < 10 or max(areas) > 26:
        return None

    palette = random.sample(range(10), 7)
    bg, sep_color, marker_color = palette[:3]
    body_colors = palette[3:]

    marker_values = sorted(random.sample(range(1, 8), 4), reverse=True)
    marker_counts = [0] * 4
    for depth, panel_idx in enumerate(order):
        marker_counts[panel_idx] = marker_values[depth]

    if any(panel_shape[0] * panel_shape[1] - area <= marker_counts[idx] + 4 for idx, area in enumerate(areas)):
        return None

    inp, panels = _render_input(
        masks,
        body_colors,
        marker_color,
        marker_counts,
        bg,
        sep_color,
    )
    if not _valid_semantics(inp, panels, masks, body_colors, marker_counts, order, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
