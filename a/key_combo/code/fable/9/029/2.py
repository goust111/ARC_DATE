from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
PAINT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    return cells


def _interior_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _gap_info(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    missing: tuple[int, int] | None,
) -> tuple[tuple[int, int], tuple[int, int]] | tuple[None, None]:
    if missing is None:
        return None, None
    r, c = missing
    if r == r0:
        return missing, (-1, 0)
    if r == r1:
        return missing, (1, 0)
    if c == c0:
        return missing, (0, -1)
    if c == c1:
        return missing, (0, 1)
    return None, None


def _stack_cells(container: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = container["bbox"]
    gap_dir = container["gap_dir"]
    cells: list[tuple[int, int]] = []
    if gap_dir == (-1, 0):
        for r in range(r1 - 1, r0, -1):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif gap_dir == (1, 0):
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif gap_dir == (0, -1):
        for c in range(c1 - 1, c0, -1):
            for r in range(r0 + 1, r1):
                cells.append((r, c))
    elif gap_dir == (0, 1):
        for c in range(c0 + 1, c1):
            for r in range(r0 + 1, r1):
                cells.append((r, c))
    return cells


def _parse_containers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = _component_cells(g == FRAME)
    containers = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        frame_cells = set(comp)
        perimeter = _perimeter_cells(r0, c0, r1, c1)
        if not frame_cells <= perimeter:
            continue
        missing_cells = sorted(perimeter - frame_cells)
        if len(missing_cells) > 1:
            continue
        missing = missing_cells[0] if missing_cells else None
        if missing is not None:
            mr, mc = missing
            if (mr in (r0, r1) and mc in (c0, c1)):
                continue
        gap_cell, gap_dir = _gap_info(r0, c0, r1, c1, missing)
        if missing is None and frame_cells != perimeter:
            continue
        if missing is not None and frame_cells != perimeter - {missing}:
            continue

        interior = _interior_cells(r0, c0, r1, c1)
        colors = sorted({int(g[r, c]) for r, c in interior if int(g[r, c]) not in (BG, FRAME)})
        if len(colors) > 1:
            continue
        paint_color = colors[0] if colors else None
        paint_cells = [(r, c) for r, c in interior if paint_color is not None and int(g[r, c]) == paint_color]
        if paint_cells:
            mask = np.zeros((h, w), dtype=bool)
            for r, c in paint_cells:
                mask[r, c] = True
            if len(_component_cells(mask)) != 1:
                continue

        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap_cell,
                "gap_dir": gap_dir,
                "interior": interior,
                "paint_color": paint_color,
                "paint_cells": paint_cells,
                "paint_count": len(paint_cells),
                "is_empty": len(paint_cells) == 0,
            }
        )
    containers.sort(key=lambda item: item["bbox"])
    return containers


def _find_transfer_target(
    grid: np.ndarray,
    source: dict,
    gap_to_empty: dict[tuple[int, int], dict],
) -> dict | None:
    gap = source["gap"]
    gap_dir = source["gap_dir"]
    if gap is None or gap_dir is None:
        return None
    h, w = grid.shape
    r, c = gap
    dr, dc = gap_dir
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != BG:
            return None
        target = gap_to_empty.get((r, c))
        if target is not None and target["gap_dir"] == (-dr, -dc):
            return target
        r += dr
        c += dc
    return None


def _analyze_actions(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    containers = _parse_containers(grid)
    gap_to_empty = {
        container["gap"]: container
        for container in containers
        if container["gap"] is not None and container["is_empty"]
    }
    actions = []
    for container in containers:
        if container["paint_count"] == 0:
            continue
        if container["gap"] is None:
            actions.append({"kind": "closed", "source": container})
            continue
        target = _find_transfer_target(grid, container, gap_to_empty)
        if target is not None:
            actions.append({"kind": "transfer", "source": container, "target": target})
        else:
            actions.append({"kind": "leak", "source": container})
    return containers, actions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move paint out of punctured grey containers only when two gaps face each other exactly.

    Input structure:
    - The background is black, and every container is a one-cell-thick grey rectangular frame.
    - A container is either closed or has exactly one one-cell gap on a non-corner border cell.
    - Some container interiors hold one 4-connected monochrome paint blob; other containers are empty.

    Rule:
    - A loaded closed container stays unchanged.
    - For each loaded punctured container, follow the outward normal ray starting at its gap.
    - If every cell on that ray is background until it reaches the opposite-facing gap of an empty container,
      clear the source interior and transfer exactly that many paint cells into the target interior.
      The target fills in strips parallel to its broken edge, starting from the strip farthest from the gap;
      within each strip, cells are taken in increasing grid-coordinate order.
    - Otherwise clear the source interior and draw a straight paint trail of the same length, starting on the gap
      cell itself and continuing along the outward normal.

    Key insight:
    - The gap is a directional interface: only exact gap-to-gap alignment through empty space permits whole-blob
      transfer, while every other punctured source leaks its paint count as a ray.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _containers, actions = _analyze_actions(g)

    for action in actions:
        if action["kind"] == "closed":
            continue
        source = action["source"]
        color = int(source["paint_color"])
        count = int(source["paint_count"])
        for r, c in source["paint_cells"]:
            out[r, c] = BG
        if action["kind"] == "transfer":
            stack = _stack_cells(action["target"])
            for r, c in stack[:count]:
                out[r, c] = color
        else:
            r, c = source["gap"]
            dr, dc = source["gap_dir"]
            for _ in range(count):
                out[r, c] = color
                r += dr
                c += dc
    return out


def _random_blob(interior: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    cell_set = set(interior)
    start = random.choice(interior)
    blob = {start}
    while len(blob) < count:
        frontier = []
        for r, c in blob:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) in cell_set and (nr, nc) not in blob:
                    frontier.append((nr, nc))
        if not frontier:
            remaining = [cell for cell in interior if cell not in blob]
            frontier = remaining
        blob.add(random.choice(frontier))
    return sorted(blob)


def _paint_container(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int] | None,
    paint_cells: list[tuple[int, int]],
    color: int | None,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    if gap is not None:
        grid[gap] = BG
    if color is not None:
        for r, c in paint_cells:
            grid[r, c] = color


def _bbox_clear(
    reserved: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad: int = 1,
) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_bbox(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _mark_cells(reserved: np.ndarray, cells: list[tuple[int, int]], pad: int = 0) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    reserved[nr, nc] = True


def _available_line(
    reserved: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = reserved.shape
    r, c = start
    dr, dc = direction
    cells = []
    while 0 <= r < h and 0 <= c < w and not reserved[r, c]:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _sample_gap(side: str, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, random.randint(c0 + 1, c1 - 1)
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    return random.randint(r0 + 1, r1 - 1), c1


def _side_dir(side: str) -> tuple[int, int]:
    return {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]


def _make_transfer_pair(
    reserved: np.ndarray,
    color: int,
) -> tuple[list[dict], list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = reserved.shape
    for _ in range(200):
        horizontal = random.random() < 0.5
        if horizontal:
            src_side, dst_side = ("right", "left") if random.random() < 0.5 else ("left", "right")
        else:
            src_side, dst_side = ("bottom", "top") if random.random() < 0.5 else ("top", "bottom")

        sh, sw = random.randint(4, 7), random.randint(4, 7)
        th, tw = random.randint(4, 7), random.randint(4, 7)
        corridor = random.randint(2, 6)
        src_gap_idx = random.randint(1, (sh if horizontal else sw) - 2)
        dst_gap_idx = random.randint(1, (th if horizontal else tw) - 2)

        if horizontal:
            line = random.randint(1 + max(src_gap_idx, dst_gap_idx), h - 2 - max(sh - 1 - src_gap_idx, th - 1 - dst_gap_idx))
            src_r0 = line - src_gap_idx
            dst_r0 = line - dst_gap_idx
            if src_side == "right":
                src_c0 = random.randint(1, max(1, w - (sw + corridor + tw + 2)))
                dst_c0 = src_c0 + sw + corridor
            else:
                dst_c0 = random.randint(1, max(1, w - (tw + corridor + sw + 2)))
                src_c0 = dst_c0 + tw + corridor
            src_bbox = (src_r0, src_c0, src_r0 + sh - 1, src_c0 + sw - 1)
            dst_bbox = (dst_r0, dst_c0, dst_r0 + th - 1, dst_c0 + tw - 1)
        else:
            line = random.randint(1 + max(src_gap_idx, dst_gap_idx), w - 2 - max(sw - 1 - src_gap_idx, tw - 1 - dst_gap_idx))
            src_c0 = line - src_gap_idx
            dst_c0 = line - dst_gap_idx
            if src_side == "bottom":
                src_r0 = random.randint(1, max(1, h - (sh + corridor + th + 2)))
                dst_r0 = src_r0 + sh + corridor
            else:
                dst_r0 = random.randint(1, max(1, h - (th + corridor + sh + 2)))
                src_r0 = dst_r0 + th + corridor
            src_bbox = (src_r0, src_c0, src_r0 + sh - 1, src_c0 + sw - 1)
            dst_bbox = (dst_r0, dst_c0, dst_r0 + th - 1, dst_c0 + tw - 1)

        if not _bbox_clear(reserved, src_bbox, pad=1) or not _bbox_clear(reserved, dst_bbox, pad=1):
            continue
        if not (0 <= src_bbox[0] and 0 <= src_bbox[1] and src_bbox[2] < h and src_bbox[3] < w):
            continue
        if not (0 <= dst_bbox[0] and 0 <= dst_bbox[1] and dst_bbox[2] < h and dst_bbox[3] < w):
            continue

        src_gap = _sample_gap(src_side, src_bbox)
        dst_gap = _sample_gap(dst_side, dst_bbox)
        if horizontal and src_gap[0] != dst_gap[0]:
            continue
        if not horizontal and src_gap[1] != dst_gap[1]:
            continue

        dr, dc = _side_dir(src_side)
        ray_cells = []
        r, c = src_gap[0] + dr, src_gap[1] + dc
        while (r, c) != dst_gap:
            if not (0 <= r < h and 0 <= c < w) or reserved[r, c]:
                ray_cells = []
                break
            ray_cells.append((r, c))
            r += dr
            c += dc
        if not ray_cells and (src_gap[0] + dr, src_gap[1] + dc) != dst_gap:
            continue

        src_interior = _interior_cells(*src_bbox)
        dst_interior = _interior_cells(*dst_bbox)
        count_cap = min(len(src_interior), len(dst_interior), 8)
        if count_cap < 2:
            continue
        count = random.randint(2, count_cap)
        paint_cells = _random_blob(src_interior, count)

        _mark_bbox(reserved, src_bbox, pad=1)
        _mark_bbox(reserved, dst_bbox, pad=1)
        _mark_cells(reserved, ray_cells, pad=0)
        return (
            [
                {
                    "bbox": src_bbox,
                    "gap": src_gap,
                    "paint": paint_cells,
                    "color": color,
                    "role": "transfer_source",
                },
                {
                    "bbox": dst_bbox,
                    "gap": dst_gap,
                    "paint": [],
                    "color": None,
                    "role": "transfer_target",
                },
            ],
            ray_cells,
            paint_cells,
        )
    return None


def _make_leak_source(
    reserved: np.ndarray,
    color: int,
) -> tuple[dict, list[tuple[int, int]]] | None:
    h, w = reserved.shape
    sides = ["top", "bottom", "left", "right"]
    for _ in range(250):
        side = random.choice(sides)
        bh, bw = random.randint(4, 7), random.randint(4, 7)
        r0 = random.randint(1, h - bh - 1)
        c0 = random.randint(1, w - bw - 1)
        bbox = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        if not _bbox_clear(reserved, bbox, pad=1):
            continue
        gap = _sample_gap(side, bbox)
        gap_dir = _side_dir(side)
        line = _available_line(reserved, gap, gap_dir)
        interior = _interior_cells(*bbox)
        count_cap = min(len(interior), len(line), 8)
        if count_cap < 2:
            continue
        count = random.randint(2, count_cap)
        trail = line[:count]
        paint_cells = _random_blob(interior, count)
        _mark_bbox(reserved, bbox, pad=1)
        _mark_cells(reserved, trail, pad=0)
        return (
            {
                "bbox": bbox,
                "gap": gap,
                "paint": paint_cells,
                "color": color,
                "role": "leak_source",
            },
            trail,
        )
    return None


def _make_closed_source(reserved: np.ndarray, color: int) -> dict | None:
    h, w = reserved.shape
    for _ in range(150):
        bh, bw = random.randint(4, 7), random.randint(4, 7)
        r0 = random.randint(1, h - bh - 1)
        c0 = random.randint(1, w - bw - 1)
        bbox = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        if not _bbox_clear(reserved, bbox, pad=1):
            continue
        interior = _interior_cells(*bbox)
        count = random.randint(2, min(len(interior), 8))
        paint_cells = _random_blob(interior, count)
        _mark_bbox(reserved, bbox, pad=1)
        return {"bbox": bbox, "gap": None, "paint": paint_cells, "color": color, "role": "closed_source"}
    return None


def _make_empty_distractor(reserved: np.ndarray) -> dict | None:
    h, w = reserved.shape
    for _ in range(150):
        bh, bw = random.randint(4, 7), random.randint(4, 7)
        r0 = random.randint(1, h - bh - 1)
        c0 = random.randint(1, w - bw - 1)
        bbox = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        if not _bbox_clear(reserved, bbox, pad=1):
            continue
        gap = None
        if random.random() < 0.65:
            gap = _sample_gap(random.choice(["top", "bottom", "left", "right"]), bbox)
        _mark_bbox(reserved, bbox, pad=1)
        return {"bbox": bbox, "gap": gap, "paint": [], "color": None, "role": "empty"}
    return None


def _render_scene(h: int, w: int, items: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for item in items:
        _paint_container(grid, item["bbox"], item["gap"], item["paint"], item["color"])
    return grid


def _count_colors(grid: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(grid, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _semantic_ok(grid: np.ndarray, plan: dict) -> bool:
    containers, actions = _analyze_actions(grid)
    if len(containers) != plan["container_count"]:
        return False

    action_by_color = {
        int(action["source"]["paint_color"]): action
        for action in actions
        if action["source"]["paint_color"] is not None
    }
    for color, expected in plan["source_roles"].items():
        action = action_by_color.get(color)
        if action is None or action["kind"] != expected:
            return False

    if sum(action["kind"] == "transfer" for action in actions) != plan["transfer_count"]:
        return False
    if sum(action["kind"] == "leak" for action in actions) != plan["leak_count"]:
        return False
    if sum(action["kind"] == "closed" for action in actions) != plan["closed_count"]:
        return False

    colors = [container["paint_color"] for container in containers if container["paint_color"] is not None]
    if len(colors) != len(set(colors)):
        return False

    inp_counts = _count_colors(grid)
    out = main(grid)
    out_counts = _count_colors(out)
    for color in colors:
        if inp_counts.get(color, 0) != out_counts.get(color, 0):
            return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 29)
    w = random.randint(20, 29)
    reserved = np.zeros((h, w), dtype=bool)
    palette = random.sample(PAINT_COLORS, random.randint(3, 6))
    color_idx = 0
    items: list[dict] = []
    source_roles: dict[int, str] = {}
    transfer_count = random.randint(1, 2)
    leak_count = random.randint(1, 2)
    closed_count = random.randint(0, 2)
    extra_empty = random.randint(0, 2)

    for _ in range(transfer_count):
        if color_idx >= len(palette):
            return None
        made = _make_transfer_pair(reserved, palette[color_idx])
        color = palette[color_idx]
        color_idx += 1
        if made is None:
            return None
        pair, _ray_cells, _paint = made
        items.extend(pair)
        source_roles[color] = "transfer"

    for _ in range(leak_count):
        if color_idx >= len(palette):
            return None
        made = _make_leak_source(reserved, palette[color_idx])
        color = palette[color_idx]
        color_idx += 1
        if made is None:
            return None
        item, _trail = made
        items.append(item)
        source_roles[color] = "leak"

    for _ in range(closed_count):
        if color_idx >= len(palette):
            break
        item = _make_closed_source(reserved, palette[color_idx])
        color = palette[color_idx]
        color_idx += 1
        if item is None:
            return None
        items.append(item)
        source_roles[color] = "closed"

    for _ in range(extra_empty):
        item = _make_empty_distractor(reserved)
        if item is None:
            return None
        items.append(item)

    grid = _render_scene(h, w, items)
    plan = {
        "container_count": len(items),
        "source_roles": source_roles,
        "transfer_count": transfer_count,
        "leak_count": leak_count,
        "closed_count": closed_count,
    }
    if not _semantic_ok(grid, plan):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
