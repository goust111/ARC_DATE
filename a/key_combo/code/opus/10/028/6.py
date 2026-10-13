import itertools
import random
from typing import Iterable

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
LEGEND_SEQ = (RED, GREEN, BLUE)
COUNT_TO_COLOR = {1: RED, 2: GREEN, 3: BLUE}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_legend(grid: np.ndarray) -> tuple[int, int]:
    positions = []
    h, w = grid.shape
    for r in range(h - 2):
        for c in range(w):
            if tuple(int(x) for x in grid[r : r + 3, c]) == LEGEND_SEQ:
                positions.append((r, c))
    assert len(positions) == 1, f"expected one legend, found {positions}"
    return positions[0]


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _extract_frames(grid: np.ndarray) -> list[dict]:
    grey_comps = _components(grid == GREY)
    frames = []
    orange_cover = np.zeros(grid.shape, dtype=int)
    for coords in grey_comps:
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        if h < 5 or w < 5:
            continue
        if not np.array_equal(patch == GREY, _frame_mask(h, w)):
            continue
        interior = patch[1:-1, 1:-1]
        assert np.all(np.isin(interior, [BG, ORANGE])), "frame interior must contain only bg/markers"
        count = int(np.sum(interior == ORANGE))
        assert 1 <= count <= 3, f"marker count out of range: {count}"
        rel = np.argwhere(interior == ORANGE)
        for rr, cc in rel:
            orange_cover[r0 + 1 + rr, c0 + 1 + cc] += 1
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "count": count,
                "color": COUNT_TO_COLOR[count],
            }
        )

    assert len(frames) in (5, 6), f"expected 5-6 frames, got {len(frames)}"
    if np.any(grid == ORANGE):
        assert np.all(orange_cover[grid == ORANGE] == 1), "every marker must lie in exactly one frame"
    frames.sort(key=lambda f: f["bbox"])
    return frames


def _recolor_and_clear(grid: np.ndarray, frames: list[dict], legend_pos: tuple[int, int]) -> np.ndarray:
    r_legend, c_legend = legend_pos
    legend_map = {
        1: int(grid[r_legend + 0, c_legend]),
        2: int(grid[r_legend + 1, c_legend]),
        3: int(grid[r_legend + 2, c_legend]),
    }
    assert legend_map == COUNT_TO_COLOR, legend_map

    out = np.array(grid, dtype=int, copy=True)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = legend_map[frame["count"]]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[inner == ORANGE] = BG
    assert np.array_equal(out[r_legend : r_legend + 3, c_legend], np.array(LEGEND_SEQ))
    return out


def _candidate_edges(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    edges = []
    for i, fa in enumerate(frames):
        for j in range(i + 1, len(frames)):
            fb = frames[j]
            a = fa["bbox"]
            b = fb["bbox"]

            if a[3] < b[1] or b[3] < a[1]:
                if a[3] < b[1]:
                    left_i, right_i, left, right = i, j, a, b
                else:
                    left_i, right_i, left, right = j, i, b, a
                overlap0 = max(left[0], right[0])
                overlap1 = min(left[2], right[2])
                if overlap1 - overlap0 + 1 >= 3:
                    r0 = overlap0 + 1
                    r1 = overlap1 - 1
                    c0 = left[3] + 1
                    c1 = right[1] - 1
                    if r0 <= r1 and c0 <= c1 and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG):
                        edges.append(
                            {
                                "kind": "h",
                                "a": left_i,
                                "b": right_i,
                                "gap": c1 - c0 + 1,
                                "r0": r0,
                                "r1": r1,
                                "c0": c0,
                                "c1": c1,
                            }
                        )

            if a[2] < b[0] or b[2] < a[0]:
                if a[2] < b[0]:
                    upper_i, lower_i, upper, lower = i, j, a, b
                else:
                    upper_i, lower_i, upper, lower = j, i, b, a
                overlap0 = max(upper[1], lower[1])
                overlap1 = min(upper[3], lower[3])
                if overlap1 - overlap0 + 1 >= 3:
                    c0 = overlap0 + 1
                    c1 = overlap1 - 1
                    r0 = upper[2] + 1
                    r1 = lower[0] - 1
                    if c0 <= c1 and r0 <= r1 and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG):
                        edges.append(
                            {
                                "kind": "v",
                                "a": upper_i,
                                "b": lower_i,
                                "gap": r1 - r0 + 1,
                                "r0": r0,
                                "r1": r1,
                                "c0": c0,
                                "c1": c1,
                            }
                        )
    return edges


def _mst_edges(edges: list[dict], n_nodes: int) -> list[dict]:
    if n_nodes <= 1:
        return []

    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    chosen = []
    for edge in sorted(
        edges,
        key=lambda e: (e["gap"], e["kind"], e["a"], e["b"], e["r0"], e["c0"], e["r1"], e["c1"]),
    ):
        ra = find(edge["a"])
        rb = find(edge["b"])
        if ra == rb:
            continue
        parent[rb] = ra
        chosen.append(edge)
        if len(chosen) == n_nodes - 1:
            break

    assert len(chosen) == n_nodes - 1, "same-color frame group is not bridge-connectable"
    return chosen


def _apply_edge(out: np.ndarray, frames: list[dict], color: int, edge: dict) -> None:
    r0, r1, c0, c1 = edge["r0"], edge["r1"], edge["c0"], edge["c1"]
    if edge["kind"] == "h":
        left = frames[edge["a"]]["bbox"]
        right = frames[edge["b"]]["bbox"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        if r0 == r1:
            out[r0, left[3]] = BG
            out[r0, right[1]] = BG
        elif r0 + 1 <= r1 - 1:
            out[r0 + 1 : r1, left[3]] = BG
            out[r0 + 1 : r1, right[1]] = BG
    else:
        upper = frames[edge["a"]]["bbox"]
        lower = frames[edge["b"]]["bbox"]
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        if c0 == c1:
            out[upper[2], c0] = BG
            out[lower[0], c0] = BG
        elif c0 + 1 <= c1 - 1:
            out[upper[2], c0 + 1 : c1] = BG
            out[lower[0], c0 + 1 : c1] = BG


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    legend_pos = _find_legend(g)
    frames = _extract_frames(g)
    base = _recolor_and_clear(g, frames, legend_pos)
    out = base.copy()

    by_color: dict[int, list[dict]] = {}
    for frame in frames:
        by_color.setdefault(frame["color"], []).append(frame)

    for color, group in by_color.items():
        if len(group) <= 1:
            continue
        edges = _candidate_edges(base, group)
        chosen = _mst_edges(edges, len(group))
        for edge in chosen:
            _apply_edge(out, group, color, edge)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray marker-filled frame by a legend-indexed count, erase the markers, then bridge same-color frames across shortest visible gaps.

    Input structure:
    - The background is black, with one 3x1 legend column near the upper-left whose rows are red, green, blue.
    - The main objects are 5-6 gray hollow rectangular frames, each containing 1-3 isolated orange marker pixels on background cells.
    - Frames do not overlap, and same-count frames may have clear horizontal or vertical gaps between them.

    Rule:
    - Read the legend as a row-indexed mapping from marker count to new frame color: 1->red, 2->green, 3->blue.
    - For each gray frame, count its interior orange markers, recolor the frame border to the legend color for that count, and erase the orange markers.
    - For each resulting color group, enumerate visible horizontal or vertical gaps whose channel is pure background and whose perpendicular overlap is at least three cells.
    - Connect each color group with a minimum-gap spanning tree, drawing the corridor in that color and cutting openings in the facing frame walls where the corridor passes.

    Key insight:
    - The orange marker count does not directly control the output shape; it first indexes the small legend, and that recoloring changes which frames belong to the same bridge-connected group.
    """
    return _solve(grid)


def _rotate_boxes(
    boxes: list[tuple[int, int, int, int]], patch_h: int, patch_w: int, turns: int
) -> tuple[list[tuple[int, int, int, int]], int, int]:
    turns %= 4
    current = list(boxes)
    h, w = patch_h, patch_w
    for _ in range(turns):
        rotated = []
        for r0, c0, r1, c1 in current:
            corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
            rc = [(c, h - 1 - r) for r, c in corners]
            rs = [p[0] for p in rc]
            cs = [p[1] for p in rc]
            rotated.append((min(rs), min(cs), max(rs), max(cs)))
        current = rotated
        h, w = w, h
    return current, h, w


def _gen_primary_group(count: int) -> dict:
    w_a = random.randint(5, 8)
    h_b = random.randint(5, 8)
    w_b = random.randint(5, 8)
    h_c = random.randint(5, 8)
    w_c = random.randint(5, 8)
    gap_ab = random.randint(2, 4)
    delta = random.randint(1, min(2, w_b - 3, w_c - 3))

    a = (h_b - 3, 0, h_b + 4, w_a - 1)
    b = (0, w_a + gap_ab, h_b - 1, w_a + gap_ab + w_b - 1)
    c = (
        h_b + 2,
        w_a + gap_ab + delta,
        h_b + 2 + h_c - 1,
        w_a + gap_ab + delta + w_c - 1,
    )
    patch_h = max(a[2], b[2], c[2]) + 1
    patch_w = max(a[3], b[3], c[3]) + 1
    turns = random.randrange(4)
    boxes, patch_h, patch_w = _rotate_boxes([a, b, c], patch_h, patch_w, turns)
    return {
        "boxes": boxes,
        "counts": [count, count, count],
        "patch_h": patch_h,
        "patch_w": patch_w,
    }


def _gen_pair_group(count: int) -> dict:
    h_l = random.randint(5, 8)
    w_l = random.randint(5, 8)
    h_r = random.randint(5, 8)
    w_r = random.randint(5, 8)
    overlap = random.randint(4, min(h_l, h_r))
    gap = random.randint(2, 4)

    left = (0, 0, h_l - 1, w_l - 1)
    right_r0 = h_l - overlap
    right = (right_r0, w_l + gap, right_r0 + h_r - 1, w_l + gap + w_r - 1)
    patch_h = max(left[2], right[2]) + 1
    patch_w = max(left[3], right[3]) + 1
    turns = random.randrange(4)
    boxes, patch_h, patch_w = _rotate_boxes([left, right], patch_h, patch_w, turns)
    return {
        "boxes": boxes,
        "counts": [count, count],
        "patch_h": patch_h,
        "patch_w": patch_w,
    }


def _gen_singleton(count: int) -> dict:
    h = random.randint(5, 8)
    w = random.randint(5, 8)
    return {"boxes": [(0, 0, h - 1, w - 1)], "counts": [count], "patch_h": h, "patch_w": w}


def _expand_group(group: dict, top: int, left: int) -> list[dict]:
    frames = []
    for box, count in zip(group["boxes"], group["counts"]):
        r0, c0, r1, c1 = box
        frames.append(
            {
                "bbox": (top + r0, left + c0, top + r1, left + c1),
                "count": count,
                "color": COUNT_TO_COLOR[count],
            }
        )
    return frames


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _sample_markers(interior_h: int, interior_w: int, count: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(interior_h) for c in range(interior_w)]
    random.shuffle(cells)
    chosen = []
    for r, c in cells:
        if all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == count:
                return chosen
    raise RuntimeError("failed to place isolated markers")


def _draw_input(grid_shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)
    grid[0:3, 0] = np.array(LEGEND_SEQ, dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        markers = _sample_markers(r1 - r0 - 1, c1 - c0 - 1, frame["count"])
        for rr, cc in markers:
            grid[r0 + 1 + rr, c0 + 1 + cc] = ORANGE
    return grid


def _gen_groups() -> list[dict]:
    primary_count = random.randint(1, 3)
    others = [c for c in (1, 2, 3) if c != primary_count]
    template = random.choice(["3+2", "3+1+1", "3+2+1"])
    groups = [_gen_primary_group(primary_count)]
    if template == "3+2":
        groups.append(_gen_pair_group(random.choice(others)))
    elif template == "3+1+1":
        random.shuffle(others)
        groups.append(_gen_singleton(others[0]))
        groups.append(_gen_singleton(others[1]))
    else:
        secondary = random.choice(others)
        groups.append(_gen_pair_group(secondary))
        singleton = others[0] if others[1] == secondary else others[1]
        groups.append(_gen_singleton(singleton))
    return groups


def _place_groups(groups: list[dict]) -> tuple[tuple[int, int], list[dict]] | None:
    max_patch_h = max(group["patch_h"] for group in groups)
    max_patch_w = max(group["patch_w"] for group in groups)
    min_h = max(22, max_patch_h + 2)
    min_w = max(22, max_patch_w + 2)

    for _ in range(120):
        h = random.randint(min_h, min(30, max(min_h, 27)))
        w = random.randint(min_w, min(30, max(min_w, 27)))
        placed_boxes: list[tuple[int, int, int, int]] = [(0, 0, 2, 0)]
        frames: list[dict] = []
        ok = True
        for group in sorted(groups, key=lambda g: g["patch_h"] * g["patch_w"], reverse=True):
            candidates = []
            for _ in range(220):
                top = random.randint(0, h - group["patch_h"])
                left = random.randint(0, w - group["patch_w"])
                group_frames = _expand_group(group, top, left)
                boxes = [frame["bbox"] for frame in group_frames]
                if any(not _boxes_separated(box, other) for box in boxes for other in placed_boxes):
                    continue
                if any(not _boxes_separated(box, other) for i, box in enumerate(boxes) for other in boxes[i + 1 :]):
                    continue
                score = top + left + random.random()
                candidates.append((score, group_frames))
            if not candidates:
                ok = False
                break
            _, chosen = min(candidates, key=lambda item: item[0])
            frames.extend(chosen)
            placed_boxes.extend(frame["bbox"] for frame in chosen)
        if ok:
            return (h, w), frames
    return None


def _tree_weight(edge_subset: Iterable[int], edges: list[dict], n_nodes: int) -> int | None:
    subset = list(edge_subset)
    if len(subset) != n_nodes - 1:
        return None
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    total = 0
    for idx in subset:
        edge = edges[idx]
        ra = find(edge["a"])
        rb = find(edge["b"])
        if ra == rb:
            return None
        parent[rb] = ra
        total += int(edge["gap"])
    root = find(0)
    if any(find(i) != root for i in range(n_nodes)):
        return None
    return total


def _assert_unique_mst(edges: list[dict], n_nodes: int) -> None:
    if n_nodes <= 1:
        return
    weights = []
    for subset in itertools.combinations(range(len(edges)), n_nodes - 1):
        weight = _tree_weight(subset, edges, n_nodes)
        if weight is not None:
            weights.append((weight, subset))
    assert weights, "no spanning tree exists"
    best = min(weight for weight, _ in weights)
    best_count = sum(1 for weight, _ in weights if weight == best)
    assert best_count == 1, f"ambiguous MST with weight {best}"


def _validate_input(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert max(grid.shape) <= 30
    assert _get_bg(grid) == BG

    legend_pos = _find_legend(grid)
    assert legend_pos == (0, 0), legend_pos
    frames = _extract_frames(grid)
    assert max(frame["count"] for frame in frames) <= 3
    assert min(frame["count"] for frame in frames) >= 1

    counts = [frame["count"] for frame in frames]
    group_sizes = {count: counts.count(count) for count in (1, 2, 3)}
    assert max(group_sizes.values()) >= 3, group_sizes

    base = _recolor_and_clear(grid, frames, legend_pos)
    chosen_kinds = set()
    redundant_group_seen = False
    for color in (RED, GREEN, BLUE):
        group = [frame for frame in frames if frame["color"] == color]
        if len(group) <= 1:
            continue
        edges = _candidate_edges(base, group)
        assert len(edges) >= len(group) - 1
        _assert_unique_mst(edges, len(group))
        chosen = _mst_edges(edges, len(group))
        chosen_kinds.update(edge["kind"] for edge in chosen)
        if len(edges) > len(group) - 1:
            redundant_group_seen = True
    assert redundant_group_seen, "need at least one color group with distracting extra gap"
    assert chosen_kinds == {"h", "v"}, f"need both corridor orientations, got {chosen_kinds}"

    out = main(grid)
    assert out.shape == grid.shape
    assert np.array_equal(out[0:3, 0], np.array(LEGEND_SEQ))
    assert not np.any(out == ORANGE)


def _gen_attempt() -> np.ndarray | None:
    groups = _gen_groups()
    placed = _place_groups(groups)
    if placed is None:
        return None
    shape, frames = placed
    grid = _draw_input(shape, frames)
    try:
        _validate_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
