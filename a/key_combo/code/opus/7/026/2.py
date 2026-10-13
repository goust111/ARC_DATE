from __future__ import annotations

import random

import numpy as np


BG = 6
BORDER = 0
GREY = 5
ORANGE = 7
MAROON = 9

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _detect_containers(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    boxes: list[tuple[int, int, int, int]] = []
    for comp in _components_of_color(grid, BORDER):
        if len(comp) < 8:
            continue
        r0, c0, r1, c1 = _bbox_from_coords(comp)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue
        border = set(_border_cells((r0, c0, r1, c1)))
        if set(comp) != border:
            continue
        if r0 + 1 > r1 - 1 or c0 + 1 > c1 - 1:
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or not bool(np.isin(inner, [GREY, ORANGE]).all()):
            continue
        boxes.append((r0, c0, r1, c1))
    boxes.sort()
    assert boxes, "expected bordered containers"
    return boxes


def _find_head_and_target(grid: np.ndarray, box: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int], int]:
    h, w = grid.shape
    r0, c0, r1, c1 = box
    heads: list[tuple[int, int, int]] = []
    if c0 - 1 >= 0:
        for r in range(r0, r1 + 1):
            if int(grid[r, c0 - 1]) == MAROON:
                heads.append((r, c0 - 1, -1))
    if c1 + 1 < w:
        for r in range(r0, r1 + 1):
            if int(grid[r, c1 + 1]) == MAROON:
                heads.append((r, c1 + 1, 1))
    assert len(heads) == 1, f"expected one adjacent head for {box}, got {heads}"
    hr, hc, dc = heads[0]
    cc = hc + dc
    while 0 <= cc < w:
        if int(grid[hr, cc]) == MAROON:
            return (hr, hc), (hr, cc), dc
        cc += dc
    raise AssertionError(f"missing destination marker for head {(hr, hc)}")


def _interior_order(box: tuple[int, int, int, int], dc: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cols = range(c0 + 1, c1) if dc < 0 else range(c1 - 1, c0, -1)
    order: list[tuple[int, int]] = []
    for c in cols:
        for r in range(r0 + 1, r1):
            order.append((r, c))
    return order


def _paint_container(out: np.ndarray, box: tuple[int, int, int, int], grey_count: int, dc: int) -> None:
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for r, c in _border_cells(box):
        out[r, c] = BORDER
    order = _interior_order(box, dc)
    grey_count = int(grey_count)
    grey_cells = set(order[-grey_count:]) if grey_count > 0 else set()
    for r, c in order:
        out[r, c] = GREY if (r, c) in grey_cells else ORANGE


def _bridge_regions(boxes: list[tuple[int, int, int, int]]) -> tuple[list[tuple[int, int, int, int]], int]:
    regions: list[tuple[int, int, int, int]] = []
    blocked_count = 0
    for i, a in enumerate(boxes):
        ar0, ac0, ar1, ac1 = a
        for j in range(i + 1, len(boxes)):
            br0, bc0, br1, bc1 = boxes[j]
            if ac1 < bc0:
                left, right = a, boxes[j]
            elif bc1 < ac0:
                left, right = boxes[j], a
            else:
                continue
            lr0, _lc0, lr1, lc1 = left
            rr0, rc0, rr1, _rc1 = right
            row0 = max(lr0, rr0)
            row1 = min(lr1, rr1)
            gap0 = lc1 + 1
            gap1 = rc0 - 1
            if row0 > row1 or gap0 > gap1:
                continue
            blocked = False
            for k, blocker in enumerate(boxes):
                if k in (i, j):
                    continue
                kr0, kc0, kr1, kc1 = blocker
                if max(row0, kr0) <= min(row1, kr1) and max(gap0, kc0) <= min(gap1, kc1):
                    blocked = True
                    blocked_count += 1
                    break
            if not blocked:
                regions.append((row0, gap0, row1, gap1))
    return sorted(set(regions)), blocked_count


def _scene_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    boxes = _detect_containers(g)
    transformed = []
    targets = set()
    for box in boxes:
        head, target, dc = _find_head_and_target(g, box)
        order = _interior_order(box, dc)
        grey_cells = [cell for cell in order if int(g[cell]) == GREY]
        steps = min(abs(target[1] - head[1]), len(grey_cells))
        final_box = (box[0], box[1] + dc * steps, box[2], box[3] + dc * steps)
        transformed.append(
            {
                "box": final_box,
                "dc": dc,
                "grey_left": len(grey_cells) - steps,
                "head": (head[0], head[1] + dc * steps),
                "target": target,
            }
        )
        targets.add(target)

    out = np.full((h, w), BG, dtype=int)
    for target in targets:
        out[target] = MAROON
    for info in transformed:
        _paint_container(out, info["box"], info["grey_left"], info["dc"])
        out[info["head"]] = MAROON

    final_boxes = [info["box"] for info in transformed]
    bridge_regions, _ = _bridge_regions(final_boxes)
    for r0, c0, r1, c1 in bridge_regions:
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BG] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move energy-limited bordered containers horizontally, then bridge unobstructed gaps between their final positions.

    Input structure:
    - The background is magenta (6), with 3-4 black rectangular container borders whose interiors contain orange (7) and grey (5).
    - Each container has one maroon (9) head pixel immediately outside its left or right side, plus a same-row maroon destination marker farther in that outward direction.

    Rule:
    - For each container, count its grey interior cells G and measure the head-to-destination distance D along the row.
    - Move the container together with its head by min(D, G) cells toward the destination; after each step, the next grey cell from the head-facing side inward becomes orange.
    - After all containers finish moving, examine every pair of final container bounding boxes: if their row spans overlap and a nonempty horizontal gap lies between them, that gap rectangle is a bridge candidate.
    - Fill every candidate gap with maroon wherever no third final container bbox crosses both the same rows and the same gap columns; preserve the moved containers, their charged interiors, the moved heads, and the destination markers.

    Key insight:
    - The bridge pattern depends on the containers' final horizontally shifted positions, so the energy-limited movement must be resolved before gap filling can be determined.
    """
    return _scene_output(grid)


def _scene_signature(boxes: list[tuple[int, int, int, int]]) -> tuple[tuple[int, int, int, int], ...]:
    regions, _ = _bridge_regions(boxes)
    return tuple(regions)


def _render_from_spec(
    shape: tuple[int, int],
    specs: list[dict],
    *,
    final: bool,
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        box = spec["final_box"] if final else spec["input_box"]
        grey_count = spec["g"] - spec["steps"] if final else spec["g"]
        _paint_container(grid, box, grey_count, spec["dc"])
        head = spec["final_head"] if final else spec["input_head"]
        hr, hc = head
        if int(grid[hr, hc]) not in (BG, MAROON):
            raise ValueError("head overlaps container")
        grid[hr, hc] = MAROON
    for spec in specs:
        tr, tc = spec["target"]
        if int(grid[tr, tc]) not in (BG, MAROON):
            raise ValueError("target overlaps non-maroon object")
        grid[tr, tc] = MAROON
    return grid


def _boxes_disjoint(boxes: list[tuple[int, int, int, int]]) -> bool:
    for i, a in enumerate(boxes):
        ar0, ac0, ar1, ac1 = a
        for b in boxes[i + 1 :]:
            br0, bc0, br1, bc1 = b
            if not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0):
                return False
    return True


def _markers_clear_of_other_boxes(specs: list[dict]) -> bool:
    for i, spec in enumerate(specs):
        r, c = spec["input_head"]
        for other in specs:
            r0, c0, r1, c1 = other["input_box"]
            if r0 <= r <= r1 and c0 <= c <= c1:
                return False
        r, c = spec["final_head"]
        for other in specs:
            r0, c0, r1, c1 = other["final_box"]
            if r0 <= r <= r1 and c0 <= c <= c1:
                return False
        r, c = spec["target"]
        for j, other in enumerate(specs):
            for box_name in ("input_box", "final_box"):
                r0, c0, r1, c1 = other[box_name]
                if r0 <= r <= r1 and c0 <= c <= c1:
                    if i == j and box_name == "final_box" and (r, c) == spec["final_head"]:
                        continue
                    return False
    return True


def _pairings_unique(inp: np.ndarray) -> bool:
    try:
        for box in _detect_containers(inp):
            _find_head_and_target(inp, box)
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 3, 4])
    split = random.randint(0, n - 2)
    dims = [(random.randint(3, 4), random.randint(5, 6)) for _ in range(n)]
    steps = []
    specs: list[dict] = []

    grey_capped_index = random.randrange(n)
    distance_capped_index = (grey_capped_index + random.randint(1, n - 1)) % n

    for i, (bh, bw) in enumerate(dims):
        inner_area = (bh - 2) * (bw - 2)
        step = random.randint(1, min(4, inner_area))
        if i == grey_capped_index:
            g = step
            d = step + random.randint(1, 3)
        elif i == distance_capped_index:
            d = step
            g = random.randint(step, min(4, inner_area))
        else:
            if random.random() < 0.5 and step < inner_area:
                d = step
                g = random.randint(step, min(4, inner_area))
            else:
                g = step
                d = step + random.randint(1, 3)
        dc = -1 if i <= split else 1
        steps.append(step)
        specs.append({"bh": bh, "bw": bw, "steps": step, "g": g, "d": d, "dc": dc})

    gaps = [random.randint(1, 3) for _ in range(n - 1)]
    split_gap = steps[split] + steps[split + 1] + 1
    if split_gap > 7:
        return None
    gaps[split] = split_gap

    total_width = sum(spec["bw"] for spec in specs) + sum(gaps)
    left_residual = max([spec["d"] - spec["steps"] for spec in specs if spec["dc"] < 0] or [0])
    right_residual = max([spec["d"] - spec["steps"] for spec in specs if spec["dc"] > 0] or [0])
    left_margin = 2 + left_residual
    right_margin = 2 + right_residual
    width = left_margin + total_width + right_margin
    if width > 30:
        return None

    height = random.randint(18, 24)
    band_top = random.randint(2, height - 6)

    c = left_margin
    used_head_rows: set[int] = set()
    for i, spec in enumerate(specs):
        bh, bw = spec["bh"], spec["bw"]
        row_jitter = random.choice([0, 0, 1])
        r0 = band_top + row_jitter
        if r0 + bh > height - 2:
            return None
        c0 = c
        final_box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        input_box = (r0, c0 - spec["dc"] * spec["steps"], r0 + bh - 1, c0 + bw - 1 - spec["dc"] * spec["steps"])
        if input_box[1] < 0 or input_box[3] >= width:
            return None

        possible_rows = [r for r in range(r0, r0 + bh) if r not in used_head_rows]
        if not possible_rows:
            return None
        head_r = random.choice(possible_rows)
        used_head_rows.add(head_r)

        final_head_c = final_box[1] - 1 if spec["dc"] < 0 else final_box[3] + 1
        final_head = (head_r, final_head_c)
        input_head = (head_r, final_head_c - spec["dc"] * spec["steps"])
        target = (head_r, final_head_c + spec["dc"] * (spec["d"] - spec["steps"]))
        if not (0 <= final_head_c < width and 0 <= input_head[1] < width and 0 <= target[1] < width):
            return None

        spec.update(
            {
                "final_box": final_box,
                "input_box": input_box,
                "final_head": final_head,
                "input_head": input_head,
                "target": target,
            }
        )
        c = final_box[3] + 1 + (gaps[i] if i < n - 1 else 0)

    final_boxes = [spec["final_box"] for spec in specs]
    input_boxes = [spec["input_box"] for spec in specs]
    if not _boxes_disjoint(final_boxes) or not _boxes_disjoint(input_boxes):
        return None
    if not _markers_clear_of_other_boxes(specs):
        return None

    final_rows = [spec["final_head"][0] for spec in specs]
    if len(set(final_rows)) != len(final_rows):
        return None

    input_grid = _render_from_spec((height, width), specs, final=False)
    output_grid = _render_from_spec((height, width), specs, final=True)

    final_regions, blocked_count = _bridge_regions(final_boxes)
    input_regions, _ = _bridge_regions(input_boxes)
    if not final_regions:
        return None
    if tuple(final_regions) == tuple(input_regions):
        return None
    if n >= 3 and blocked_count == 0 and random.random() < 0.7:
        return None
    if not _pairings_unique(input_grid):
        return None

    marked_rows = {spec["input_head"][0] for spec in specs}
    if len(marked_rows) != n:
        return None

    for r0, c0, r1, c1 in final_regions:
        patch = output_grid[r0 : r1 + 1, c0 : c1 + 1]
        patch[patch == BG] = MAROON

    solver_out = main(input_grid)
    if not np.array_equal(solver_out, output_grid):
        return None

    if max(input_grid.shape) > 30 or max(output_grid.shape) > 30:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
