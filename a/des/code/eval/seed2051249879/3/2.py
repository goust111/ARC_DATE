from __future__ import annotations

import random
from typing import Iterable

import numpy as np


MAROON = 9
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = ["U", "D", "L", "R"]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(coords)
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _parse_arrow(coords: list[tuple[int, int]]) -> dict[str, object]:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    rel = {(r - r0, c - c0) for r, c in coords}
    choices: list[dict[str, object]] = []

    if h == 3 and w >= 3 and len(rel) == w + 2:
        row_counts = [sum((rr, cc) in rel for cc in range(w)) for rr in range(3)]
        if row_counts == [1, w, 1]:
            head_cols = []
            for rr in (0, 2):
                cols = [cc for cc in range(w) if (rr, cc) in rel]
                if len(cols) == 1:
                    head_cols.append(cols[0])
            if len(head_cols) == 2 and head_cols[0] == head_cols[1] and head_cols[0] in (0, w - 1):
                direction = "L" if head_cols[0] == 0 else "R"
                tail_rel = (1, w - 1) if direction == "L" else (1, 0)
                head_rel = (1, 0) if direction == "L" else (1, w - 1)
                choices.append(
                    {
                        "dir": direction,
                        "tail": (r0 + tail_rel[0], c0 + tail_rel[1]),
                        "head": (r0 + head_rel[0], c0 + head_rel[1]),
                        "pixels": set(coords),
                    }
                )

    if w == 3 and h >= 3 and len(rel) == h + 2:
        col_counts = [sum((rr, cc) in rel for rr in range(h)) for cc in range(3)]
        if col_counts == [1, h, 1]:
            head_rows = []
            for cc in (0, 2):
                rows = [rr for rr in range(h) if (rr, cc) in rel]
                if len(rows) == 1:
                    head_rows.append(rows[0])
            if len(head_rows) == 2 and head_rows[0] == head_rows[1] and head_rows[0] in (0, h - 1):
                direction = "U" if head_rows[0] == 0 else "D"
                tail_rel = (h - 1, 1) if direction == "U" else (0, 1)
                head_rel = (0, 1) if direction == "U" else (h - 1, 1)
                choices.append(
                    {
                        "dir": direction,
                        "tail": (r0 + tail_rel[0], c0 + tail_rel[1]),
                        "head": (r0 + head_rel[0], c0 + head_rel[1]),
                        "pixels": set(coords),
                    }
                )

    assert len(choices) == 1, "expected one valid maroon arrow shape"
    return choices[0]


def _find_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    colors = sorted(int(color) for color in np.unique(grid) if int(color) not in (bg, MAROON))
    next_id = 0
    for color in colors:
        for coords in _components(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert h == w and h in (2, 3), "objects must be solid 2x2 or 3x3 squares"
            assert len(coords) == h * w, "objects must be solid rectangles"
            objects.append(
                {
                    "id": next_id,
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                }
            )
            next_id += 1
    assert objects, "expected at least one object"
    return objects


def _match_endpoint(
    objects: list[dict[str, object]], direction: str, tail: tuple[int, int], head: tuple[int, int]
) -> tuple[int, int]:
    tr, tc = tail
    hr, hc = head
    if direction == "R":
        src = [obj for obj in objects if obj["c1"] == tc - 1 and obj["r0"] <= tr <= obj["r1"]]
        dst = [obj for obj in objects if obj["c0"] == hc + 1 and obj["r0"] <= hr <= obj["r1"]]
    elif direction == "L":
        src = [obj for obj in objects if obj["c0"] == tc + 1 and obj["r0"] <= tr <= obj["r1"]]
        dst = [obj for obj in objects if obj["c1"] == hc - 1 and obj["r0"] <= hr <= obj["r1"]]
    elif direction == "D":
        src = [obj for obj in objects if obj["r1"] == tr - 1 and obj["c0"] <= tc <= obj["c1"]]
        dst = [obj for obj in objects if obj["r0"] == hr + 1 and obj["c0"] <= hc <= obj["c1"]]
    else:
        src = [obj for obj in objects if obj["r0"] == tr + 1 and obj["c0"] <= tc <= obj["c1"]]
        dst = [obj for obj in objects if obj["r1"] == hr - 1 and obj["c0"] <= hc <= obj["c1"]]
    assert len(src) == 1 and len(dst) == 1, "each arrow endpoint must touch exactly one object"
    return int(src[0]["id"]), int(dst[0]["id"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow each maroon arrow chain and recolor every non-head object with the head color.

    Input structure:
    - The background is the most common color.
    - Non-background, non-maroon objects are separate solid 2x2 or 3x3 squares.
    - Each maroon component is one straight arrow with a triangular head.

    Rule:
    - Reconstruct the directed graph from arrow direction and endpoint contacts.
    - For every chain head (object with no incoming arrow), broadcast its color to
      all later objects in the chain.
    - Erase all maroon arrows in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    assert bg != MAROON, "background must differ from maroon arrow color"

    objects = _find_objects(g, bg)
    arrows = [_parse_arrow(coords) for coords in _components(g == MAROON)]

    out_edge: dict[int, int] = {}
    indegree = {int(obj["id"]): 0 for obj in objects}
    for arrow in arrows:
        src, dst = _match_endpoint(objects, str(arrow["dir"]), tuple(arrow["tail"]), tuple(arrow["head"]))
        assert src != dst, "arrows must connect distinct objects"
        assert src not in out_edge, "each object has at most one outgoing arrow"
        out_edge[src] = dst
        indegree[dst] += 1
        assert indegree[dst] <= 1, "each object has at most one incoming arrow"

    out = g.copy()
    out[g == MAROON] = bg

    colors = {int(obj["id"]): int(obj["color"]) for obj in objects}
    object_map = {int(obj["id"]): obj for obj in objects}
    heads = [node for node, deg in indegree.items() if deg == 0]
    assert heads, "expected at least one chain head"

    visited: set[int] = set()
    for head in heads:
        head_color = colors[head]
        current = head
        first = True
        while current not in visited:
            visited.add(current)
            if not first:
                obj = object_map[current]
                out[int(obj["r0"]) : int(obj["r1"]) + 1, int(obj["c0"]) : int(obj["c1"]) + 1] = head_color
            first = False
            if current not in out_edge:
                break
            current = out_edge[current]

    assert len(visited) == len(objects), "graph must be a disjoint union of chains"
    return out


GRID_MIN = 18
GRID_MAX = 28
LINE_LEN_RANGE = (3, 5)
CHAIN_ATTEMPTS = 120
LINK_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 120
GEN_ATTEMPTS = 300
PATCH_MAX_SPAN = 18
VALID_LENGTHS = {
    1: [(2,), (3,), (4,)],
    2: [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (4, 2)],
    3: [(2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2)],
}


def _rect_cells(obj: dict[str, int]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(int(obj["r0"]), int(obj["r0"]) + int(obj["size"]))
        for c in range(int(obj["c0"]), int(obj["c0"]) + int(obj["size"]))
    }


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _propose_link(source: dict[str, int], size_b: int, direction: str, line_len: int) -> tuple[dict[str, int], set[tuple[int, int]]]:
    if direction == "R":
        row = int(source["r0"]) + random.randrange(int(source["size"]))
        rb = random.randrange(size_b)
        target = {"size": size_b, "r0": row - rb, "c0": int(source["c0"]) + int(source["size"]) + line_len}
        head_c = int(target["c0"]) - 1
        arrow = {(row, c) for c in range(int(source["c0"]) + int(source["size"]), int(target["c0"]))}
        arrow |= {(row - 1, head_c), (row + 1, head_c)}
    elif direction == "L":
        row = int(source["r0"]) + random.randrange(int(source["size"]))
        rb = random.randrange(size_b)
        target = {"size": size_b, "r0": row - rb, "c0": int(source["c0"]) - line_len - size_b}
        head_c = int(target["c0"]) + size_b
        arrow = {(row, c) for c in range(head_c, int(source["c0"]))}
        arrow |= {(row - 1, head_c), (row + 1, head_c)}
    elif direction == "D":
        col = int(source["c0"]) + random.randrange(int(source["size"]))
        cb = random.randrange(size_b)
        target = {"size": size_b, "r0": int(source["r0"]) + int(source["size"]) + line_len, "c0": col - cb}
        head_r = int(target["r0"]) - 1
        arrow = {(r, col) for r in range(int(source["r0"]) + int(source["size"]), int(target["r0"]))}
        arrow |= {(head_r, col - 1), (head_r, col + 1)}
    else:
        col = int(source["c0"]) + random.randrange(int(source["size"]))
        cb = random.randrange(size_b)
        target = {"size": size_b, "r0": int(source["r0"]) - line_len - size_b, "c0": col - cb}
        head_r = int(target["r0"]) + size_b
        arrow = {(r, col) for r in range(head_r, int(source["r0"]))}
        arrow |= {(head_r, col - 1), (head_r, col + 1)}
    return target, arrow


def _can_add_link(
    occupied: set[tuple[int, int]],
    source_cells: set[tuple[int, int]],
    target_cells: set[tuple[int, int]],
    arrow_cells: set[tuple[int, int]],
) -> bool:
    new_cells = target_cells | arrow_cells
    if new_cells & occupied:
        return False
    for cell in target_cells:
        for nbr in _neighbors4(cell):
            if nbr in occupied:
                return False
    for cell in arrow_cells:
        for nbr in _neighbors4(cell):
            if nbr in occupied and nbr not in source_cells:
                return False
    return True


def _normalize_chain(chain: dict[str, object]) -> dict[str, object]:
    all_cells: list[tuple[int, int]] = []
    for obj in chain["objects"]:
        all_cells.extend(_rect_cells(obj))
    for arrow in chain["arrows"]:
        all_cells.extend(list(arrow["pixels"]))
    min_r, min_c, max_r, max_c = _bbox(all_cells)
    shift_r = 1 - min_r
    shift_c = 1 - min_c

    objects = []
    for obj in chain["objects"]:
        moved = dict(obj)
        moved["r0"] = int(obj["r0"]) + shift_r
        moved["c0"] = int(obj["c0"]) + shift_c
        objects.append(moved)

    arrows = []
    for arrow in chain["arrows"]:
        moved_pixels = {(r + shift_r, c + shift_c) for r, c in arrow["pixels"]}
        arrows.append({"dir": arrow["dir"], "pixels": moved_pixels})

    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1
    return {"objects": objects, "arrows": arrows, "h": span_h + 2, "w": span_w + 2}


def _build_chain(length: int) -> dict[str, object] | None:
    for _ in range(CHAIN_ATTEMPTS):
        sizes = [random.choice([2, 3]) for _ in range(length)]
        objects: list[dict[str, int]] = [{"size": sizes[0], "r0": 0, "c0": 0}]
        arrows: list[dict[str, object]] = []
        occupied = _rect_cells(objects[0])
        ok = True

        for idx in range(1, length):
            source = objects[-1]
            source_cells = _rect_cells(source)
            placed = False
            dirs = random.sample(ALL_DIRS, len(ALL_DIRS))
            for direction in dirs:
                for _try in range(LINK_ATTEMPTS):
                    line_len = random.randint(*LINE_LEN_RANGE)
                    target, arrow_cells = _propose_link(source, sizes[idx], direction, line_len)
                    target_cells = _rect_cells(target)
                    if not _can_add_link(occupied, source_cells, target_cells, arrow_cells):
                        continue
                    objects.append(target)
                    arrows.append({"dir": direction, "pixels": arrow_cells})
                    occupied |= target_cells | arrow_cells
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        chain = _normalize_chain({"objects": objects, "arrows": arrows})
        if int(chain["h"]) > PATCH_MAX_SPAN or int(chain["w"]) > PATCH_MAX_SPAN:
            continue
        return chain
    return None


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _place_chains(chains: list[dict[str, object]], h: int, w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(chains)), key=lambda i: int(chains[i]["h"]) * int(chains[i]["w"]), reverse=True)
    placed_boxes: list[tuple[int, int, int, int]] = []
    offsets = [(-1, -1)] * len(chains)

    for idx in order:
        ch = int(chains[idx]["h"])
        cw = int(chains[idx]["w"])
        if ch > h or cw > w:
            return None
        spots = [(r, c) for r in range(h - ch + 1) for c in range(w - cw + 1)]
        random.shuffle(spots)
        found = False
        for r0, c0 in spots[:PLACEMENT_ATTEMPTS]:
            box = (r0, c0, r0 + ch - 1, c0 + cw - 1)
            if any(_boxes_conflict(box, other) for other in placed_boxes):
                continue
            offsets[idx] = (r0, c0)
            placed_boxes.append(box)
            found = True
            break
        if not found:
            return None
    return offsets


def _render_scene(chains: list[dict[str, object]], offsets: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, np.ndarray]:
    max_h = max(int(chain["h"]) + off_r for chain, (off_r, _off_c) in zip(chains, offsets))
    max_w = max(int(chain["w"]) + off_c for chain, (_off_r, off_c) in zip(chains, offsets))
    inp = np.full((max_h, max_w), bg, dtype=int)
    out = np.full((max_h, max_w), bg, dtype=int)

    for chain, (off_r, off_c) in zip(chains, offsets):
        head_color = int(chain["objects"][0]["color"])
        for obj_idx, obj in enumerate(chain["objects"]):
            color = int(obj["color"]) if obj_idx == 0 else head_color
            rr0 = off_r + int(obj["r0"])
            cc0 = off_c + int(obj["c0"])
            size = int(obj["size"])
            inp[rr0 : rr0 + size, cc0 : cc0 + size] = int(obj["color"])
            out[rr0 : rr0 + size, cc0 : cc0 + size] = color
        for arrow in chain["arrows"]:
            for r, c in arrow["pixels"]:
                inp[off_r + r, off_c + c] = MAROON
    return inp, out


def _sample_lengths() -> tuple[int, ...]:
    chain_count = random.choice([1, 2, 3])
    return random.choice(VALID_LENGTHS[chain_count])


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice([color for color in range(10) if color != MAROON])
    lengths = _sample_lengths()
    total_objects = sum(lengths)
    object_colors = random.sample([color for color in range(10) if color not in (bg, MAROON)], total_objects)

    chains: list[dict[str, object]] = []
    color_idx = 0
    for length in lengths:
        chain = _build_chain(length)
        if chain is None:
            return None
        for obj in chain["objects"]:
            obj["color"] = object_colors[color_idx]
            color_idx += 1
        chains.append(chain)

    min_h = max(GRID_MIN, max(int(chain["h"]) for chain in chains))
    min_w = max(GRID_MIN, max(int(chain["w"]) for chain in chains))
    needed_area = sum((int(chain["h"]) + 2) * (int(chain["w"]) + 2) for chain in chains)

    for _ in range(PLACEMENT_ATTEMPTS):
        h = random.randint(min_h, GRID_MAX)
        w = random.randint(min_w, GRID_MAX)
        if h * w < needed_area:
            continue
        offsets = _place_chains(chains, h, w)
        if offsets is None:
            continue
        inp_small, expected_small = _render_scene(chains, offsets, bg)
        if inp_small.shape[0] > h or inp_small.shape[1] > w:
            continue
        inp = np.full((h, w), bg, dtype=int)
        expected = np.full((h, w), bg, dtype=int)
        inp[: inp_small.shape[0], : inp_small.shape[1]] = inp_small
        expected[: expected_small.shape[0], : expected_small.shape[1]] = expected_small

        values, counts = np.unique(inp, return_counts=True)
        bg_count = int(counts[values.tolist().index(bg)])
        if any(int(count) >= bg_count for value, count in zip(values, counts) if int(value) != bg):
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
