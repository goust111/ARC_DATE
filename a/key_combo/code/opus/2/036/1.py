from __future__ import annotations

import random
from collections import deque

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    ORANGE = 7


Coord = tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    comps: list[dict] = []
    h, w = grid.shape
    for color in colors:
        seen = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(grid[r, c]) != color:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                cells: list[Coord] = []
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for nr, nc in _neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                coords = np.asarray(cells, dtype=int)
                comps.append(
                    {
                        "color": color,
                        "coords": coords,
                        "bbox": _bbox_coords(coords),
                        "area": int(len(cells)),
                    }
                )
    return comps


def _detect_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for comp in _find_components(grid, (Color.BLUE,)):
        r0, c0, r1, c1 = comp["bbox"]
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        ph, pw = patch.shape
        if ph < 3 or pw < 4:
            continue
        if not np.all(patch[:, 0] == Color.BLUE):
            continue
        if not np.all(patch[:, -1] == Color.BLUE):
            continue
        if not np.all(patch[-1, :] == Color.BLUE):
            continue

        top_closed = bool(np.all(patch[0, :] == Color.BLUE))
        if top_closed:
            cavity_r0 = r0 + 1
            cavity_r1 = r1 - 1
            required = np.zeros_like(patch, dtype=bool)
            required[0, :] = True
        else:
            if not (int(patch[0, 0]) == Color.BLUE and int(patch[0, -1]) == Color.BLUE):
                continue
            if np.any(patch[0, 1:-1] == Color.BLUE):
                continue
            cavity_r0 = r0
            cavity_r1 = r1 - 1
            required = np.zeros_like(patch, dtype=bool)

        cavity_c0 = c0 + 1
        cavity_c1 = c1 - 1
        if cavity_r0 > cavity_r1 or cavity_c0 > cavity_c1:
            continue

        required[:, 0] = True
        required[:, -1] = True
        required[-1, :] = True
        if not np.array_equal(required, patch == Color.BLUE):
            continue
        cavity = grid[cavity_r0 : cavity_r1 + 1, cavity_c0 : cavity_c1 + 1]
        if cavity.size == 0 or np.any(cavity == Color.BLUE):
            continue

        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "cavity_bbox": (cavity_r0, cavity_c0, cavity_r1, cavity_c1),
                "cavity_h": int(cavity_r1 - cavity_r0 + 1),
                "cavity_w": int(cavity_c1 - cavity_c0 + 1),
                "top_closed": top_closed,
            }
        )

    containers.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return containers


def _gravity_order(comp: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    return (-int(r1), -int(r0), int(c0), int(c1))


def _settle_blocks(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    moving_mask = np.isin(out, (Color.RED, Color.GREEN, Color.YELLOW))
    out[moving_mask] = Color.BLACK

    occupied = grid == Color.BLUE
    blocks = _find_components(grid, (Color.RED, Color.GREEN, Color.YELLOW))
    for comp in sorted(blocks, key=_gravity_order):
        coords = np.asarray(comp["coords"], dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        shift = 0
        while True:
            next_rows = rows + shift + 1
            if np.any(next_rows >= grid.shape[0]):
                break
            if np.any(occupied[next_rows, cols]):
                break
            shift += 1

        dest_rows = rows + shift
        assert not np.any(occupied[dest_rows, cols])
        out[dest_rows, cols] = int(comp["color"])
        occupied[dest_rows, cols] = True

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Drop small colored blocks into blue rectangular containers, then parity-fill the remaining cavity cells.

    Input structure
    - A black background contains several blue one-cell-thick rectangular containers near the bottom.
    - Each container has full left and right walls plus a full bottom edge; generated examples use an open top so falling blocks can enter.
    - Above the containers are separated 4-connected red, green, or yellow mini-objects aligned to container columns.

    Rule
    - Remove the mini-objects from their starting cells and let them fall straight downward, processing lower-starting objects first.
    - A falling object keeps its shape and color and stops just before the grid edge, any already settled object, or any blue border pixel.
    - For each blue container, count how many cavity cells are still black after settling; fill those black cells red when the count is even, or orange when it is odd.

    Key insight
    - The fill color is not fixed by the container size alone: gravity changes how much empty cavity area remains, and that remaining-area parity chooses the final fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(map(int, np.unique(g))).issubset({Color.BLACK, Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW})

    containers = _detect_containers(g)
    assert containers, "expected at least one blue container"

    out = _settle_blocks(g)
    for container in containers:
        r0, c0, r1, c1 = container["cavity_bbox"]
        cavity = out[r0 : r1 + 1, c0 : c1 + 1]
        black_mask = cavity == Color.BLACK
        fill_color = Color.RED if int(np.sum(black_mask)) % 2 == 0 else Color.ORANGE
        cavity[black_mask] = fill_color

    return out


_SHAPES: tuple[tuple[Coord, ...], ...] = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
)


def _shape_dims(shape: tuple[Coord, ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _split_extra(extra: int, slots: int) -> list[int]:
    cuts = sorted(random.randint(0, extra) for _ in range(slots - 1))
    parts: list[int] = []
    prev = 0
    for cut in cuts + [extra]:
        parts.append(cut - prev)
        prev = cut
    random.shuffle(parts)
    return parts


def _placement_catalog(inner_h: int, inner_w: int) -> list[dict]:
    placements: list[dict] = []
    for shape in _SHAPES:
        sh, sw = _shape_dims(shape)
        if sh > inner_h or sw > inner_w:
            continue
        for left in range(inner_w - sw + 1):
            placements.append(
                {
                    "shape": shape,
                    "shape_h": sh,
                    "shape_w": sw,
                    "left": left,
                    "area": len(shape),
                    "col_range": (left, left + sw - 1),
                }
            )

    cavity_area = inner_h * inner_w
    options: list[dict] = []
    for p in placements:
        remaining = cavity_area - int(p["area"])
        if remaining > 0:
            options.append({"placements": [p], "remaining": remaining})

    for i, p1 in enumerate(placements):
        a0, a1 = p1["col_range"]
        for p2 in placements[i + 1 :]:
            b0, b1 = p2["col_range"]
            if not (a1 + 1 < b0 or b1 + 1 < a0):
                continue
            remaining = cavity_area - int(p1["area"]) - int(p2["area"])
            if remaining <= 0:
                continue
            pair = sorted((p1, p2), key=lambda item: item["col_range"][0])
            options.append({"placements": pair, "remaining": remaining})

    return options


def _choose_container_plan(target_parity: int) -> dict:
    for _ in range(200):
        inner_h = random.randint(2, 4)
        inner_w = random.randint(2, 4)
        if not (4 <= inner_h * inner_w <= 12):
            continue
        options = [opt for opt in _placement_catalog(inner_h, inner_w) if opt["remaining"] % 2 == target_parity]
        if not options:
            continue
        preferred_count = 2 if inner_w >= 3 and random.random() < 0.45 else 1
        preferred = [opt for opt in options if len(opt["placements"]) == preferred_count]
        if preferred:
            options = preferred
        chosen = random.choice(options)
        return {
            "inner_h": inner_h,
            "inner_w": inner_w,
            "outer_h": inner_h + 1,
            "outer_w": inner_w + 2,
            "placements": chosen["placements"],
            "remaining": chosen["remaining"],
        }
    raise RuntimeError("failed to choose container plan")


def _paint_shape(grid: np.ndarray, top: int, left: int, shape: tuple[Coord, ...], color: int) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = color


def _draw_container(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> None:
    bottom = top + inner_h
    right = left + inner_w + 1
    grid[top : bottom + 1, left] = Color.BLUE
    grid[top : bottom + 1, right] = Color.BLUE
    grid[bottom, left : right + 1] = Color.BLUE


def _assign_block_colors(block_count: int) -> list[int]:
    palette = [Color.RED, Color.GREEN, Color.YELLOW]
    if block_count == 1:
        return [random.choice(palette)]
    colors = random.sample(palette, k=min(3, block_count))
    while len(colors) < block_count:
        colors.append(random.choice(palette))
    random.shuffle(colors)
    return colors


def _gen_attempt() -> np.ndarray | None:
    container_count = random.randint(2, 4)
    parities = [0, 1] + [random.randint(0, 1) for _ in range(container_count - 2)]
    random.shuffle(parities)
    plans = [_choose_container_plan(parity) for parity in parities]

    base_width = sum(plan["outer_w"] for plan in plans) + (container_count + 1)
    if base_width > 30:
        return None
    width = base_width + random.randint(0, min(5, 30 - base_width))
    gap_slack = _split_extra(width - base_width, container_count + 1)

    max_outer_h = max(plan["outer_h"] for plan in plans)
    bottom_margin = random.randint(0, 1)
    top_clearance = random.randint(6, 10)
    base_height = max(12, max_outer_h + bottom_margin + top_clearance)
    if base_height > 30:
        return None
    height = base_height + random.randint(0, min(3, 30 - base_height))

    grid = np.zeros((height, width), dtype=int)
    bottom = height - 1 - bottom_margin

    block_specs: list[dict] = []
    c = 1 + gap_slack[0]
    for i, plan in enumerate(plans):
        top = bottom - plan["outer_h"] + 1
        _draw_container(grid, top, c, plan["inner_h"], plan["inner_w"])

        for placement in plan["placements"]:
            cavity_left = c + 1 + int(placement["left"])
            max_start_top = top - int(placement["shape_h"]) - 2
            if max_start_top < 0:
                return None
            start_top = random.randint(0, max_start_top)
            block_specs.append(
                {
                    "shape": placement["shape"],
                    "top": start_top,
                    "left": cavity_left,
                }
            )

        c += plan["outer_w"] + 1 + gap_slack[i + 1]

    colors = _assign_block_colors(len(block_specs))
    for spec, color in zip(block_specs, colors):
        _paint_shape(grid, spec["top"], spec["left"], spec["shape"], color)

    try:
        containers = _detect_containers(grid)
        if len(containers) != container_count:
            return None
        blocks = _find_components(grid, (Color.RED, Color.GREEN, Color.YELLOW))
        if len(blocks) != len(block_specs):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if int(np.max(grid.shape)) > 30 or int(np.max(out.shape)) > 30:
        return None
    if Color.ORANGE not in set(map(int, np.unique(out))):
        return None

    settled = _settle_blocks(grid)
    fill_parities: list[int] = []
    for container in containers:
        r0, c0, r1, c1 = container["cavity_bbox"]
        black_count = int(np.sum(settled[r0 : r1 + 1, c0 : c1 + 1] == Color.BLACK))
        if black_count <= 0:
            return None
        fill_parities.append(black_count % 2)
    if 0 not in fill_parities or 1 not in fill_parities:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-parity input")
