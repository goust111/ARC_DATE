from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
OUTLINE_COLOR = 1
PANEL_COLOR = 8
LEGEND_COLORS = [2, 3, 4, 5, 6, 7, 9]
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray | list[tuple[int, int]] | set[tuple[int, int]]):
    pts = np.asarray(list(coords), dtype=int)
    assert pts.ndim == 2 and pts.shape[0] > 0 and pts.shape[1] == 2
    r0, c0, r1, c1 = _bbox_coords(pts)
    shifted = pts - np.array([r0, c0])
    shape = frozenset((int(r), int(c)) for r, c in shifted)
    return shape, (int(r1 - r0 + 1), int(c1 - c0 + 1))


def _d4_variants(shape: frozenset[tuple[int, int]], dims: tuple[int, int]):
    h, w = dims
    transforms = (
        lambda r, c: (r, c),
        lambda r, c: (c, h - 1 - r),
        lambda r, c: (h - 1 - r, w - 1 - c),
        lambda r, c: (w - 1 - c, r),
        lambda r, c: (r, w - 1 - c),
        lambda r, c: (h - 1 - r, c),
        lambda r, c: (c, r),
        lambda r, c: (w - 1 - c, h - 1 - r),
    )
    return {_normalize([transform(r, c) for r, c in shape]) for transform in transforms}


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_from_lines(lines: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in lines], dtype=bool)


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for candidate in (rot, np.fliplr(rot)):
            trimmed = _trim_mask(candidate)
            key = (trimmed.shape, trimmed.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(trimmed)
    return variants


RAW_SHAPES = [
    ["###", "#.#", "###"],
    ["####", "#..#", "####"],
    ["#####", "#...#", "#####"],
    ["####", "#..#", "#.##", "###."],
    [".###", "##.#", "#.##", "###."],
    ["####", "#..#", "#..#", "####"],
]

SHAPE_BANK = []
for raw in RAW_SHAPES:
    mask = _mask_from_lines(raw)
    shape, dims = _normalize(np.argwhere(mask))
    orbit = _d4_variants(shape, dims)
    SHAPE_BANK.append(
        {
            "mask": mask,
            "variants": _mask_variants(mask),
            "orbit": orbit,
            "asymmetric": len(orbit) >= 4,
        }
    )

ASYMMETRIC_IDS = [i for i, info in enumerate(SHAPE_BANK) if info["asymmetric"]]


def _panel_box(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    h, w = grid.shape
    assert h > 0 and w > 0
    panel_color = int(grid[0, 0])
    assert panel_color != BG

    r1 = 0
    while r1 + 1 < h and int(grid[r1 + 1, 0]) != BG:
        r1 += 1
    c1 = 0
    while c1 + 1 < w and int(grid[0, c1 + 1]) != BG:
        c1 += 1

    assert np.all(grid[: r1 + 1, : c1 + 1] != BG)
    return panel_color, (0, 0, r1, c1)


def _legend_shapes(grid: np.ndarray, panel_color: int, panel_box: tuple[int, int, int, int]):
    r0, c0, r1, c1 = panel_box
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    comps = _find_components(region != panel_color)
    legend = {}
    for comp in comps:
        values = {int(region[r, c]) for r, c in comp}
        assert len(values) == 1
        color = next(iter(values))
        shape, dims = _normalize(comp)
        legend[color] = _d4_variants(shape, dims)
    assert legend
    return legend


def _outer_adjacent_coords(coords: np.ndarray, grid_shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = grid_shape
    r0, c0, r1, c1 = _bbox_coords(coords)
    br0 = max(0, r0 - 1)
    bc0 = max(0, c0 - 1)
    br1 = min(h - 1, r1 + 1)
    bc1 = min(w - 1, c1 + 1)

    local = np.zeros((br1 - br0 + 1, bc1 - bc0 + 1), dtype=bool)
    for r, c in coords:
        local[int(r) - br0, int(c) - bc0] = True

    exterior = np.zeros_like(local, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    lh, lw = local.shape

    for rr in range(lh):
        for cc in (0, lw - 1):
            if not local[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                q.append((rr, cc))
    for cc in range(lw):
        for rr in (0, lh - 1):
            if not local[rr, cc] and not exterior[rr, cc]:
                exterior[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < lh and 0 <= nc < lw and not local[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    outer = set()
    for rr, cc in np.argwhere(local):
        for nr, nc in _neighbors4(int(rr), int(cc)):
            if 0 <= nr < lh and 0 <= nc < lw and exterior[nr, nc]:
                outer.add((nr + br0, nc + bc0))
    return outer


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each blue outline to a legend shape up to rotation/reflection, then repaint only its outside contour in the legend color.

    Input structure:
    - Black background surrounds a teal panel in the top-left corner.
    - The teal panel contains several separated monochrome legend shapes, one color per shape.
    - Outside the panel are several blue one-cell-wide closed outline objects on the background.

    Rule:
    - Extract every non-teal shape from the panel as a color/shape dictionary entry.
    - For each blue outline object, crop its pixel mask and compare it against all legend shapes under the 8 D4 rotation/reflection variants.
    - Erase the blue outline itself to black, then color only the black cells that are 4-neighbor to the outline and reachable from the exterior with the matched legend color.
    - Leave the teal panel and every other background cell unchanged.

    Key insight:
    - The panel supplies the replacement color, but only after shape matching under rotation/reflection; the painted cells come from tracing the outline's outside neighborhood rather than filling the outline or its interior.
    """
    g = np.asarray(grid, dtype=int)
    panel_color, panel_box = _panel_box(g)
    legend = _legend_shapes(g, panel_color, panel_box)

    out = g.copy()
    for comp in _find_components(g == OUTLINE_COLOR):
        shape, dims = _normalize(comp)
        matches = [color for color, orbit in legend.items() if (shape, dims) in orbit]
        assert len(matches) == 1
        color = matches[0]

        out[comp[:, 0], comp[:, 1]] = BG
        for r, c in _outer_adjacent_coords(comp, g.shape):
            if int(g[r, c]) == BG:
                out[r, c] = color
    return out


def _expand_box(r: int, c: int, h: int, w: int, pad: int, shape: tuple[int, int]):
    gh, gw = shape
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(gh, r + h + pad)
    c1 = min(gw, c + w + pad)
    return r0, c0, r1, c1


def _can_place(occupied: np.ndarray, h: int, w: int, r: int, c: int, pad: int) -> bool:
    r0, c0, r1, c1 = _expand_box(r, c, h, w, pad, occupied.shape)
    return not bool(occupied[r0:r1, c0:c1].any())


def _reserve(occupied: np.ndarray, h: int, w: int, r: int, c: int, pad: int) -> None:
    r0, c0, r1, c1 = _expand_box(r, c, h, w, pad, occupied.shape)
    occupied[r0:r1, c0:c1] = True


def _distribute(extra: int, n_slots: int) -> list[int]:
    slots = [1] * n_slots
    for _ in range(extra):
        slots[random.randrange(n_slots)] += 1
    return slots


def _pick_shape_infos() -> list[dict]:
    n_shapes = random.choice([3, 4])
    asym_id = random.choice(ASYMMETRIC_IDS)
    other_ids = [i for i in range(len(SHAPE_BANK)) if i != asym_id]
    chosen_ids = [asym_id] + random.sample(other_ids, n_shapes - 1)
    random.shuffle(chosen_ids)
    return [SHAPE_BANK[i] for i in chosen_ids]


def _gen_attempt() -> np.ndarray | None:
    infos = _pick_shape_infos()
    colors = random.sample(LEGEND_COLORS, len(infos))
    panel_masks = [random.choice(info["variants"]) for info in infos]
    panel_widths = [mask.shape[1] for mask in panel_masks]
    panel_heights = [mask.shape[0] for mask in panel_masks]

    base_panel_w = sum(panel_widths) + len(infos) + 1
    extra_panel_w = random.randint(0, max(0, min(3, 24 - base_panel_w)))
    gaps = _distribute(extra_panel_w, len(infos) + 1)
    panel_w = sum(panel_widths) + sum(gaps)
    panel_h = max(panel_heights) + 2

    if panel_h > 7 or panel_w > 24:
        return None

    grid_h = random.randint(max(18, panel_h + 8), 30)
    grid_w = random.randint(max(20, panel_w + 2), 30)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    grid[:panel_h, :panel_w] = PANEL_COLOR
    _reserve(occupied, panel_h, panel_w, 0, 0, 2)

    cursor = gaps[0]
    entries = []
    for info, color, mask in zip(infos, colors, panel_masks):
        h, w = mask.shape
        r = random.randint(1, panel_h - h - 1)
        grid[r : r + h, cursor : cursor + w][mask] = color
        entries.append((info, color))
        cursor += w + gaps[len(entries)]

    placements = []
    order = list(entries)
    random.shuffle(order)
    for info, color in order:
        mask = random.choice(info["variants"])
        h, w = mask.shape
        positions = [
            (r, c)
            for r in range(1, grid_h - h)
            for c in range(1, grid_w - w)
            if _can_place(occupied, h, w, r, c, 2)
        ]
        if not positions:
            return None
        r, c = random.choice(positions)
        grid[r : r + h, c : c + w][mask] = OUTLINE_COLOR
        _reserve(occupied, h, w, r, c, 2)
        placements.append((np.argwhere(mask) + np.array([r, c]), color))

    expected = grid.copy()
    for coords, color in placements:
        expected[coords[:, 0], coords[:, 1]] = BG
        for r, c in _outer_adjacent_coords(coords, grid.shape):
            expected[r, c] = color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
