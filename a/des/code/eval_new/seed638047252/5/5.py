from collections import Counter, deque
import random

import numpy as np


DIR_FROM_MISSING = {
    (0, 0): "up",
    (0, 1): "right",
    (1, 1): "down",
    (1, 0): "left",
}
MISSING_FROM_DIR = {v: k for k, v in DIR_FROM_MISSING.items()}
DIR_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_TO_SIDE = {"up": "top", "down": "bottom", "left": "left", "right": "right"}
SIDES = ("top", "right", "bottom", "left")
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR_DELTA.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return (r0 + r1) // 2, (c0 + c1) // 2


def _in_box(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("segment endpoints must share a row or a column")


def _l_path_cells(
    src_center: tuple[int, int], dst_center: tuple[int, int], direction: str
) -> list[tuple[int, int]]:
    sr, sc = src_center
    dr, dc = dst_center
    if direction == "up":
        assert dr < sr
        bend = (dr, sc)
    elif direction == "down":
        assert dr > sr
        bend = (dr, sc)
    elif direction == "left":
        assert dc < sc
        bend = (sr, dc)
    else:
        assert direction == "right"
        assert dc > sc
        bend = (sr, dc)
    first = _segment_cells(src_center, bend)
    second = _segment_cells(bend, dst_center)
    return first + second[1:]


def _touches(node_cells: set[tuple[int, int]], comp_cells: set[tuple[int, int]]) -> bool:
    for r, c in comp_cells:
        if (r - 1, c) in node_cells or (r + 1, c) in node_cells:
            return True
        if (r, c - 1) in node_cells or (r, c + 1) in node_cells:
            return True
    return False


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, dict], dict[int, dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    nodes: dict[int, dict] = {}
    arrows_raw: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for comp in _components(g == color):
            box = _bbox(comp)
            area = len(comp)
            r0, c0, r1, c1 = box
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            cells = set(comp)
            if area == 9 and hh == 3 and ww == 3:
                if color in nodes:
                    raise AssertionError("node colors must be unique")
                nodes[color] = {
                    "color": color,
                    "bbox": box,
                    "center": _center(box),
                    "cells": cells,
                }
            elif area == 3 and hh == 2 and ww == 2:
                rel = {(r - r0, c - c0) for r, c in comp}
                missing = next((p for p in ((0, 0), (0, 1), (1, 0), (1, 1)) if p not in rel), None)
                if missing is None:
                    raise AssertionError("invalid arrow shape")
                arrows_raw.append(
                    {
                        "color": color,
                        "bbox": box,
                        "cells": cells,
                        "direction": DIR_FROM_MISSING[missing],
                    }
                )
            else:
                raise AssertionError("unexpected foreground component")

    assert len(nodes) >= 2, "expected at least two nodes"

    arrows_by_owner: dict[int, dict] = {}
    for arrow in arrows_raw:
        owners = [node for node in nodes.values() if _touches(node["cells"], arrow["cells"])]
        assert len(owners) == 1, "each arrow must touch exactly one node"
        owner = owners[0]
        owner_color = int(owner["color"])
        assert owner_color not in arrows_by_owner, "a node can have at most one arrow"
        arrow = dict(arrow)
        arrow["owner_color"] = owner_color
        arrows_by_owner[owner_color] = arrow

    outgoing = {owner_color: int(arrow["color"]) for owner_color, arrow in arrows_by_owner.items()}
    incoming = Counter(outgoing.values())
    mismatch_starts = [color for color, nxt in outgoing.items() if nxt != color]
    if len(mismatch_starts) == 1:
        start_color = mismatch_starts[0]
    else:
        start_candidates = [
            color for color in nodes if color in outgoing and incoming.get(color, 0) == 0
        ]
        assert len(start_candidates) == 1, "expected a unique start node"
        start_color = start_candidates[0]

    return bg, nodes, arrows_by_owner, start_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the scattered 3x3 nodes in arrow-color order with source-colored orthogonal segments.

    Input structure:
    - bg = most common color.
    - Several scattered 3x3 solid monochrome squares act as nodes, with one node per color.
    - Every non-terminal node has one adjacent 3-cell L marker in a 2x2 box. The marker touches exactly one node, its color names the next node in the chain, and its missing corner encodes the first travel direction in clockwise order: upper-left -> up, upper-right -> right, lower-right -> down, lower-left -> left.
    - Exactly one node has no outgoing marker, and exactly one node color is never used by any marker, making that node the start.

    Rule:
    - Read each marker as an outgoing edge from its touched node to the node whose 3x3 block has the same color as the marker.
    - Start from the unique node with no incoming edge.
    - For every edge A -> B, draw a one-pixel orthogonal path colored A from A's center toward B: first move in A's marker direction until aligned with B on the other axis, then turn once toward B.
    - Keep all original node and marker pixels unchanged while adding the path.

    Key insight:
    - Each L marker carries two signals at once: its missing corner gives the first-leg direction, while its color selects the next node in the visitation chain.
    """
    g = np.asarray(grid, dtype=int)
    bg, nodes, arrows_by_owner, start_color = _parse_scene(g)

    out = g.copy()
    seen: set[int] = set()
    current = start_color
    while current in arrows_by_owner:
        assert current not in seen, "cycle detected in node chain"
        seen.add(current)
        arrow = arrows_by_owner[current]
        nxt = int(arrow["color"])
        assert nxt in nodes, "arrow color must match a node color"
        for r, c in _l_path_cells(nodes[current]["center"], nodes[nxt]["center"], arrow["direction"]):
            out[r, c] = current
        current = nxt

    out[g != bg] = g[g != bg]
    assert len(seen) == len(nodes) - 1, "chain should visit every non-terminal node once"
    return out


def _node_box_ok(
    box: tuple[int, int, int, int],
    existing_boxes: list[tuple[int, int, int, int]],
    path_cells: set[tuple[int, int]],
) -> bool:
    for other in existing_boxes:
        if _boxes_too_close(box, other, pad=2):
            return False
    r0, c0, r1, c1 = box
    for r, c in path_cells:
        if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
            return False
    return True


def _sample_target_box(
    src_box: tuple[int, int, int, int], direction: str, h: int, w: int
) -> tuple[int, int, int, int]:
    sr, sc = _center(src_box)
    if direction == "up":
        tr = random.randint(3, sr - 4)
        tc_choices = [c for c in range(3, w - 3) if c != sc]
        tc = random.choice(tc_choices)
    elif direction == "down":
        tr = random.randint(sr + 4, h - 4)
        tc_choices = [c for c in range(3, w - 3) if c != sc]
        tc = random.choice(tc_choices)
    elif direction == "left":
        tc = random.randint(3, sc - 4)
        tr_choices = [r for r in range(3, h - 3) if r != sr]
        tr = random.choice(tr_choices)
    else:
        assert direction == "right"
        tc = random.randint(sc + 4, w - 4)
        tr_choices = [r for r in range(3, h - 3) if r != sr]
        tr = random.choice(tr_choices)
    return (tr - 1, tc - 1, tr + 1, tc + 1)


def _visible_path_cells(
    src_box: tuple[int, int, int, int],
    dst_box: tuple[int, int, int, int],
    direction: str,
) -> set[tuple[int, int]]:
    hidden_boxes = (src_box, dst_box)
    cells = set(_l_path_cells(_center(src_box), _center(dst_box), direction))
    return {
        cell
        for cell in cells
        if not any(_in_box(cell, hidden_box) for hidden_box in hidden_boxes)
    }


def _arrow_box(
    node_box: tuple[int, int, int, int], side: str, offset: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = node_box
    if side == "top":
        return (r0 - 2, c0 + offset, r0 - 1, c0 + offset + 1)
    if side == "bottom":
        return (r1 + 1, c0 + offset, r1 + 2, c0 + offset + 1)
    if side == "left":
        return (r0 + offset, c0 - 2, r0 + offset + 1, c0 - 1)
    assert side == "right"
    return (r0 + offset, c1 + 1, r0 + offset + 1, c1 + 2)


def _arrow_cells(box: tuple[int, int, int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0, _, _ = box
    missing = MISSING_FROM_DIR[direction]
    cells = set()
    for dr in (0, 1):
        for dc in (0, 1):
            if (dr, dc) != missing:
                cells.add((r0 + dr, c0 + dc))
    return cells


def _in_bounds_box(box: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = box
    return 0 <= r0 and 0 <= c0 and r1 < h and c1 < w


def _candidate_arrow_positions(
    node_box: tuple[int, int, int, int],
    direction: str,
    h: int,
    w: int,
) -> list[tuple[tuple[int, int, int, int], set[tuple[int, int]]]]:
    banned_side = DIR_TO_SIDE[direction]
    candidates = []
    for side in SIDES:
        if side == banned_side:
            continue
        for offset in (0, 1):
            box = _arrow_box(node_box, side, offset)
            if not _in_bounds_box(box, h, w):
                continue
            candidates.append((box, _arrow_cells(box, direction)))
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    node_count = random.randint(3, 5)
    node_colors = random.sample(palette, node_count)

    nodes: list[dict] = []
    node_boxes: list[tuple[int, int, int, int]] = []
    node_cells: set[tuple[int, int]] = set()
    visible_paths: set[tuple[int, int]] = set()
    directions: list[str] = []

    for _ in range(200):
        tr = random.randint(2, h - 5)
        tc = random.randint(2, w - 5)
        first_box = (tr, tc, tr + 2, tc + 2)
        if _node_box_ok(first_box, node_boxes, visible_paths):
            break
    else:
        return None

    nodes.append({"color": node_colors[0], "bbox": first_box})
    node_boxes.append(first_box)
    node_cells |= {(r, c) for r in range(first_box[0], first_box[2] + 1) for c in range(first_box[1], first_box[3] + 1)}

    for idx in range(node_count - 1):
        cur_box = nodes[-1]["bbox"]
        placed = False
        for direction in random.sample(list(DIR_DELTA.keys()), 4):
            if direction == "up" and _center(cur_box)[0] <= 6:
                continue
            if direction == "down" and _center(cur_box)[0] >= h - 7:
                continue
            if direction == "left" and _center(cur_box)[1] <= 6:
                continue
            if direction == "right" and _center(cur_box)[1] >= w - 7:
                continue
            for _ in range(80):
                try:
                    next_box = _sample_target_box(cur_box, direction, h, w)
                except ValueError:
                    continue
                if not _node_box_ok(next_box, node_boxes, visible_paths):
                    continue
                vis = _visible_path_cells(cur_box, next_box, direction)
                if not vis:
                    continue
                if vis & visible_paths:
                    continue
                if vis & node_cells:
                    continue
                if any(
                    r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1
                    for (r, c) in vis
                    for (r0, c0, r1, c1) in node_boxes[:-1]
                ):
                    continue
                nodes.append({"color": node_colors[idx + 1], "bbox": next_box})
                node_boxes.append(next_box)
                node_cells |= {
                    (r, c)
                    for r in range(next_box[0], next_box[2] + 1)
                    for c in range(next_box[1], next_box[3] + 1)
                }
                visible_paths |= vis
                directions.append(direction)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    arrows: list[dict] = []
    arrow_cells_taken: set[tuple[int, int]] = set()
    for idx, direction in enumerate(directions):
        owner_box = nodes[idx]["bbox"]
        target_color = nodes[idx + 1]["color"]
        owner_cells = {
            (r, c) for r in range(owner_box[0], owner_box[2] + 1) for c in range(owner_box[1], owner_box[3] + 1)
        }
        placed = False
        for box, cells in _candidate_arrow_positions(owner_box, direction, h, w):
            if cells & node_cells:
                continue
            if cells & arrow_cells_taken:
                continue
            if cells & visible_paths:
                continue
            owners = 0
            for node in nodes:
                other_cells = {
                    (r, c)
                    for r in range(node["bbox"][0], node["bbox"][2] + 1)
                    for c in range(node["bbox"][1], node["bbox"][3] + 1)
                }
                if _touches(other_cells, cells):
                    owners += 1
            if owners != 1 or not _touches(owner_cells, cells):
                continue
            arrows.append({"color": target_color, "cells": cells})
            arrow_cells_taken |= cells
            placed = True
            break
        if not placed:
            return None

    grid = np.full((h, w), bg, dtype=int)
    for node in nodes:
        r0, c0, r1, c1 = node["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(node["color"])
    for arrow in arrows:
        for r, c in arrow["cells"]:
            grid[r, c] = int(arrow["color"])

    try:
        _, parsed_nodes, parsed_arrows, start = _parse_scene(grid)
    except AssertionError:
        return None
    if len(parsed_nodes) != node_count or len(parsed_arrows) != node_count - 1:
        return None
    if start != node_colors[0]:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
