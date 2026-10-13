import random

import numpy as np


BG = 0


ARROW_SPECS = {
    "right": {
        "cells": frozenset({(0, 0), (1, 0), (1, 1), (2, 0)}),
        "tip": (1, 1),
        "dir": (0, 1),
    },
    "left": {
        "cells": frozenset({(0, 1), (1, 0), (1, 1), (2, 1)}),
        "tip": (1, 0),
        "dir": (0, -1),
    },
    "down": {
        "cells": frozenset({(0, 0), (0, 1), (0, 2), (1, 1)}),
        "tip": (1, 1),
        "dir": (1, 0),
    },
    "up": {
        "cells": frozenset({(0, 1), (1, 0), (1, 1), (1, 2)}),
        "tip": (0, 1),
        "dir": (-1, 0),
    },
}

ARROW_FROM_TIP = {
    "right": [(-1, -1), (0, -1), (0, 0), (1, -1)],
    "left": [(-1, 1), (0, 0), (0, 1), (1, 1)],
    "down": [(-1, -1), (-1, 0), (-1, 1), (0, 0)],
    "up": [(0, 0), (1, -1), (1, 0), (1, 1)],
}

SHAPE_TO_ARROW = {spec["cells"]: name for name, spec in ARROW_SPECS.items()}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comp_ids = np.full((h, w), -1, dtype=int)
    components = []

    for r0 in range(h):
        for c0 in range(w):
            if visited[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            stack = [(r0, c0)]
            coords = []
            visited[r0, c0] = True
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            rmin = int(arr[:, 0].min())
            cmin = int(arr[:, 1].min())
            rmax = int(arr[:, 0].max())
            cmax = int(arr[:, 1].max())
            comp_id = len(components)
            comp_ids[arr[:, 0], arr[:, 1]] = comp_id
            components.append(
                {
                    "id": comp_id,
                    "color": color,
                    "coords": arr,
                    "bbox": (rmin, cmin, rmax, cmax),
                    "area": int(arr.shape[0]),
                }
            )

    return components, comp_ids


def _classify(comp: dict) -> tuple[str, str | None]:
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    area = int(comp["area"])
    normalized = frozenset((int(r) - r0, int(c) - c0) for r, c in comp["coords"])

    arrow_name = SHAPE_TO_ARROW.get(normalized)
    if arrow_name is not None:
        return "arrow", arrow_name
    if area == height * width and height >= 2 and width >= 2:
        return "rectangle", None
    raise AssertionError("unrecognized foreground component")


def _scan_target(
    grid: np.ndarray,
    bg: int,
    comp_ids: np.ndarray,
    arrow: dict,
    direction_name: str,
    rectangles: set[int],
) -> int:
    spec = ARROW_SPECS[direction_name]
    r0, c0, _, _ = arrow["bbox"]
    tip_r = r0 + spec["tip"][0]
    tip_c = c0 + spec["tip"][1]
    dr, dc = spec["dir"]

    r = tip_r + dr
    c = tip_c + dc
    while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
        if int(grid[r, c]) != bg:
            target_id = int(comp_ids[r, c])
            assert target_id in rectangles
            return target_id
        r += dr
        c += dc
    raise AssertionError("arrow does not reach any rectangle")


def main(grid: np.ndarray) -> np.ndarray:
    """Swap each triangle arrow's color with the rectangle it points to."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    components, comp_ids = _find_components(g, bg)

    arrows = []
    rectangles = []
    arrow_dirs = {}
    for comp in components:
        kind, direction = _classify(comp)
        if kind == "arrow":
            arrows.append(comp)
            arrow_dirs[comp["id"]] = direction
        else:
            rectangles.append(comp)

    assert len(arrows) >= 2
    assert len(arrows) == len(rectangles)
    rectangle_ids = {comp["id"] for comp in rectangles}

    target_to_arrow_color: dict[int, int] = {}
    arrow_to_rect_color: dict[int, int] = {}
    rect_color_by_id = {comp["id"]: int(comp["color"]) for comp in rectangles}

    for arrow in arrows:
        direction = arrow_dirs[arrow["id"]]
        target_id = _scan_target(g, bg, comp_ids, arrow, direction, rectangle_ids)
        assert target_id not in target_to_arrow_color
        target_color = rect_color_by_id[target_id]
        assert target_color != int(arrow["color"])
        target_to_arrow_color[target_id] = int(arrow["color"])
        arrow_to_rect_color[arrow["id"]] = target_color

    assert len(target_to_arrow_color) == len(rectangles)

    out = g.copy()
    for rect in rectangles:
        out[rect["coords"][:, 0], rect["coords"][:, 1]] = target_to_arrow_color[rect["id"]]
    for arrow in arrows:
        out[arrow["coords"][:, 0], arrow["coords"][:, 1]] = arrow_to_rect_color[arrow["id"]]
    return out


RECT_COUNT_RANGE = (2, 5)
RECT_SIDE_RANGE = (2, 5)
GAP_RANGE = (1, 4)
PALETTE = tuple(range(1, 10))
PADDING_CHOICES = (1, 2)
GRID_SIDE_MAX = 30
GEN_RETRIES = 500


def _derangement(n: int) -> list[int]:
    perm = list(range(n))
    for _ in range(200):
        random.shuffle(perm)
        if all(i != p for i, p in enumerate(perm)):
            return perm
    raise RuntimeError("failed to sample a derangement")


def _build_package(
    rect_h: int,
    rect_w: int,
    rect_color: int,
    arrow_color: int,
    direction: str,
    gap: int,
    align: int,
) -> tuple[np.ndarray, np.ndarray]:
    rect_cells = [(r, c) for r in range(rect_h) for c in range(rect_w)]
    if direction == "right":
        tip = (align, -gap - 1)
    elif direction == "left":
        tip = (align, rect_w + gap)
    elif direction == "down":
        tip = (-gap - 1, align)
    else:
        tip = (rect_h + gap, align)

    arrow_cells = [(tip[0] + dr, tip[1] + dc) for dr, dc in ARROW_FROM_TIP[direction]]
    all_cells = rect_cells + arrow_cells
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)

    h = max_r - min_r + 1
    w = max_c - min_c + 1
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for r, c in rect_cells:
        rr = r - min_r
        cc = c - min_c
        inp[rr, cc] = rect_color
        out[rr, cc] = arrow_color
    for r, c in arrow_cells:
        rr = r - min_r
        cc = c - min_c
        inp[rr, cc] = arrow_color
        out[rr, cc] = rect_color

    return inp, out


def _sample_canvas(patches: list[np.ndarray], padding: int) -> tuple[int, int] | None:
    reserve_area = sum((patch.shape[0] + 2 * padding) * (patch.shape[1] + 2 * padding) for patch in patches)
    max_h = max(patch.shape[0] + 2 * padding for patch in patches)
    max_w = max(patch.shape[1] + 2 * padding for patch in patches)
    need = int(np.ceil(reserve_area * 1.15))

    for _ in range(80):
        if random.random() < 0.5:
            h = random.randint(max_h, GRID_SIDE_MAX)
            w_min = max(max_w, (need + h - 1) // h)
            if w_min > GRID_SIDE_MAX:
                continue
            w = random.randint(w_min, GRID_SIDE_MAX)
        else:
            w = random.randint(max_w, GRID_SIDE_MAX)
            h_min = max(max_h, (need + w - 1) // w)
            if h_min > GRID_SIDE_MAX:
                continue
            h = random.randint(h_min, GRID_SIDE_MAX)
        return h, w
    return None


def _stamp(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BG
    view[mask] = patch[mask]


def _scatter_patches(
    input_patches: list[np.ndarray],
    output_patches: list[np.ndarray],
    canvas_shape: tuple[int, int],
    padding: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = canvas_shape
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    order = list(range(len(input_patches)))
    random.shuffle(order)
    for idx in order:
        patch_in = input_patches[idx]
        patch_out = output_patches[idx]
        ph, pw = patch_in.shape
        candidates = []
        for r0 in range(padding, h - ph - padding + 1):
            for c0 in range(padding, w - pw - padding + 1):
                rr0 = r0 - padding
                cc0 = c0 - padding
                rr1 = r0 + ph + padding
                cc1 = c0 + pw + padding
                if not occupied[rr0:rr1, cc0:cc1].any():
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occupied[r0 - padding : r0 + ph + padding, c0 - padding : c0 + pw + padding] = True
        _stamp(inp, patch_in, r0, c0)
        _stamp(out, patch_out, r0, c0)

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*RECT_COUNT_RANGE)
    rect_colors = random.sample(PALETTE, count)
    color_perm = _derangement(count)
    directions = list(ARROW_SPECS)
    padding = random.choice(PADDING_CHOICES)

    input_patches = []
    output_patches = []
    for i in range(count):
        rect_h = random.randint(*RECT_SIDE_RANGE)
        rect_w = random.randint(*RECT_SIDE_RANGE)
        direction = random.choice(directions)
        gap = random.randint(*GAP_RANGE)
        align = random.randrange(rect_h if direction in ("left", "right") else rect_w)
        patch_in, patch_out = _build_package(
            rect_h=rect_h,
            rect_w=rect_w,
            rect_color=rect_colors[i],
            arrow_color=rect_colors[color_perm[i]],
            direction=direction,
            gap=gap,
            align=align,
        )
        input_patches.append(patch_in)
        output_patches.append(patch_out)

    canvas = _sample_canvas(input_patches, padding)
    if canvas is None:
        return None
    scattered = _scatter_patches(input_patches, output_patches, canvas, padding)
    if scattered is None:
        return None
    inp, expected = scattered
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
