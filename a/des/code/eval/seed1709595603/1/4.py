from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


TEMPLATE_MASKS = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
]


GRID_H_RANGE = (24, 35)
GRID_W_RANGE = (24, 35)
TEMPLATE_COUNT_RANGE = (2, 4)
GROUP_COUNT_RANGE = (3, 4)
HOLE_SIDE_OPTIONS = (1, 3)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_PAD = 1
PLACEMENT_TRIES = 200


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_square_ring(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    side = r1 - r0 + 1
    if side < 3 or side != c1 - c0 + 1:
        return False
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    coords = set(comp["coords"])
    return coords == expected


def _detect_groups(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[int, list[tuple[int, int]]]]:
    rings_by_center: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for comp in _monochrome_components(grid, bg):
        if not _is_square_ring(comp):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        rings_by_center[center].append(comp)

    groups: list[dict] = []
    marker_positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for center, rings in rings_by_center.items():
        rings.sort(key=lambda comp: int(comp["bbox"][2]) - int(comp["bbox"][0]), reverse=True)
        sizes = [int(comp["bbox"][2]) - int(comp["bbox"][0]) + 1 for comp in rings]
        if any(sizes[i] - sizes[i + 1] != 4 for i in range(len(sizes) - 1)):
            continue
        cr, cc = center
        marker_color = int(grid[cr, cc])
        ring_colors = {int(comp["color"]) for comp in rings}
        if marker_color == bg or marker_color in ring_colors:
            continue
        outer = tuple(map(int, rings[0]["bbox"]))
        groups.append(
            {
                "center": (int(cr), int(cc)),
                "layers": len(rings),
                "marker_color": marker_color,
                "outer_bbox": outer,
            }
        )
        marker_positions[marker_color].append((int(cr), int(cc)))

    groups.sort(key=lambda grp: (grp["outer_bbox"][0], grp["outer_bbox"][1]))
    assert groups, "expected at least one nested-ring group"
    return groups, marker_positions


def _extract_template_mask(
    grid: np.ndarray, color: int, marker_positions: list[tuple[int, int]]
) -> np.ndarray:
    marker_set = set(marker_positions)
    template_points = [(int(r), int(c)) for r, c in np.argwhere(grid == color) if (int(r), int(c)) not in marker_set]
    assert template_points, "template pixels missing for marker color"
    rows = [p[0] for p in template_points]
    cols = [p[1] for p in template_points]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    assert r1 - r0 + 1 == 3 and c1 - c0 + 1 == 3, "template bbox must be 3x3"
    mask = np.zeros((3, 3), dtype=bool)
    for r, c in template_points:
        mask[r - r0, c - c0] = True
    return mask


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    groups, marker_positions = _detect_groups(g, bg)
    templates: dict[int, np.ndarray] = {}
    for color, positions in marker_positions.items():
        templates[int(color)] = _extract_template_mask(g, int(color), positions)
    return bg, groups, templates


def _paint_mask(out: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = out.shape
    mh, mw = mask.shape
    for r in range(mh):
        rr = r0 + r
        if not (0 <= rr < h):
            continue
        for c in range(mw):
            cc = c0 + c
            if 0 <= cc < w and bool(mask[r, c]):
                out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, groups, templates = _parse_scene(g)

    out = g.copy()
    for group in groups:
        r0, c0, r1, c1 = map(int, group["outer_bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    for group in groups:
        r0, c0, _, _ = map(int, group["outer_bbox"])
        scale = int(group["layers"])
        color = int(group["marker_color"])
        mask = templates[color]
        scaled = np.kron(mask, np.ones((scale, scale), dtype=bool))
        _paint_mask(out, scaled, r0, c0, color)

    return out


def _ring_patch(side: int, color: int, bg: int) -> np.ndarray:
    patch = np.full((side, side), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _build_group_patch(layers: int, hole_side: int, marker_color: int, ring_colors: list[int], bg: int) -> np.ndarray:
    outer_side = hole_side + 4 * layers - 2
    patch = np.full((outer_side, outer_side), bg, dtype=int)
    for layer in range(layers):
        ring_side = hole_side + 2 + 4 * layer
        offset = (outer_side - ring_side) // 2
        ring = _ring_patch(ring_side, ring_colors[layer], bg)
        patch[offset : offset + ring_side, offset : offset + ring_side] = np.where(
            ring != bg, ring, patch[offset : offset + ring_side, offset : offset + ring_side]
        )
    center = outer_side // 2
    patch[center, center] = marker_color
    return patch


def _build_bank_patch(colors: list[int], masks: list[np.ndarray], bg: int, side: str) -> np.ndarray:
    if side in ("top", "bottom"):
        patch = np.full((3, 4 * len(colors) - 1), bg, dtype=int)
        for idx, (color, mask) in enumerate(zip(colors, masks)):
            c0 = 4 * idx
            tile = patch[:, c0 : c0 + 3]
            tile[mask] = int(color)
        return patch

    patch = np.full((4 * len(colors) - 1, 3), bg, dtype=int)
    for idx, (color, mask) in enumerate(zip(colors, masks)):
        r0 = 4 * idx
        tile = patch[r0 : r0 + 3, :]
        tile[mask] = int(color)
    return patch


def _can_place(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    bh, bw = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(bh, r0 + h + pad)
    cc1 = min(bw, c0 + w + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    bh, bw = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(bh, r0 + h + pad)
    cc1 = min(bw, c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _free_positions(blocked: np.ndarray, h: int, w: int, pad: int) -> list[tuple[int, int]]:
    bh, bw = blocked.shape
    choices: list[tuple[int, int]] = []
    for r0 in range(bh - h + 1):
        for c0 in range(bw - w + 1):
            if _can_place(blocked, r0, c0, h, w, pad):
                choices.append((int(r0), int(c0)))
    return choices


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    region = grid[r0 : r0 + h, c0 : c0 + w]
    region[patch != bg] = patch[patch != bg]


def _place_bank(grid: np.ndarray, blocked: np.ndarray, patch: np.ndarray, side: str, bg: int) -> tuple[int, int]:
    gh, gw = grid.shape
    ph, pw = patch.shape
    if side == "top":
        r0 = 0
        c0 = random.randint(0, gw - pw)
    elif side == "bottom":
        r0 = gh - ph
        c0 = random.randint(0, gw - pw)
    elif side == "left":
        r0 = random.randint(0, gh - ph)
        c0 = 0
    else:
        r0 = random.randint(0, gh - ph)
        c0 = gw - pw
    _paste_patch(grid, patch, r0, c0, bg)
    _mark_blocked(blocked, r0, c0, ph, pw, PLACEMENT_PAD)
    return r0, c0


def _sample_specs() -> tuple[int, list[int], list[int], list[dict], str]:
    bg = random.randint(0, 9)
    template_count = random.randint(*TEMPLATE_COUNT_RANGE)
    usable_colors = [color for color in range(10) if color != bg]
    random.shuffle(usable_colors)
    marker_colors = usable_colors[:template_count]
    ring_palette = usable_colors[template_count:]

    group_count = random.randint(*GROUP_COUNT_RANGE)
    layers_list = [1, 2, 3]
    while len(layers_list) < group_count:
        layers_list.append(random.randint(1, 3))
    random.shuffle(layers_list)

    assigned_markers = marker_colors[:]
    while len(assigned_markers) < group_count:
        assigned_markers.append(random.choice(marker_colors))
    random.shuffle(assigned_markers)

    specs: list[dict] = []
    for layers, marker_color in zip(layers_list, assigned_markers):
        hole_side = random.choice(HOLE_SIDE_OPTIONS)
        ring_colors = random.sample(ring_palette, layers)
        outer_side = hole_side + 4 * layers - 2
        specs.append(
            {
                "layers": layers,
                "hole_side": hole_side,
                "marker_color": marker_color,
                "ring_colors": ring_colors,
                "outer_side": outer_side,
            }
        )

    side = random.choice(["top", "bottom", "left", "right"])
    return bg, marker_colors, ring_palette, specs, side


def _gen_attempt() -> np.ndarray | None:
    bg, marker_colors, _, specs, side = _sample_specs()
    template_masks = random.sample(TEMPLATE_MASKS, len(marker_colors))
    bank_patch = _build_bank_patch(marker_colors, template_masks, bg, side)
    max_side = max(spec["outer_side"] for spec in specs)
    min_h = max(GRID_H_RANGE[0], bank_patch.shape[0] + max_side + 2)
    min_w = max(GRID_W_RANGE[0], bank_patch.shape[1] + max_side + 2)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    grid_h = random.randint(min_h, GRID_H_RANGE[1])
    grid_w = random.randint(min_w, GRID_W_RANGE[1])
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    _place_bank(grid, blocked, bank_patch, side, bg)

    placed_specs = sorted(specs, key=lambda spec: spec["outer_side"], reverse=True)
    for spec in placed_specs:
        patch = _build_group_patch(
            spec["layers"], spec["hole_side"], spec["marker_color"], spec["ring_colors"], bg
        )
        ph, pw = patch.shape
        choices = _free_positions(blocked, ph, pw, PLACEMENT_PAD)
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        _paste_patch(grid, patch, r0, c0, bg)
        _mark_blocked(blocked, r0, c0, ph, pw, PLACEMENT_PAD)

    try:
        _, groups, templates = _parse_scene(grid)
        if len(groups) != len(specs):
            return None
        if set(templates) != set(marker_colors):
            return None
        out = main(grid)
        if np.array_equal(out, grid):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
