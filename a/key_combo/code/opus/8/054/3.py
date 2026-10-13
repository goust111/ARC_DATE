from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations, product

import numpy as np


BACKGROUND = 0
PLACEHOLDER = 5
GROUP_COLORS = [2, 1, 3, 4]  # red, blue, green, yellow

# Irregular 4-connected shapes with 3-6 cells. Rotations/reflections are distinct.
SHAPE_LIBRARY = [
    ((0, 0), (0, 1), (1, 1)),  # 3 cells, 2x2
    ((0, 0), (1, 0), (1, 1), (2, 1)),  # 4 cells, 3x2
    ((0, 1), (1, 0), (1, 1), (1, 2)),  # 4 cells, 2x3
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),  # 5 cells, 3x3
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),  # 5 cells, 3x3
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),  # 5 cells, 3x2
    ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)),  # 6 cells, 3x3
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)),  # 6 cells, 3x3
]

LAYOUT_TEMPLATES = [
    {
        "g4": [(0, 0), (0, 2), (2, 1), (3, 3)],
        "g3": [(1, 0), (1, 3), (3, 1)],
        "g2": [(2, 0), (2, 3)],
    },
    {
        "g4": [(0, 1), (0, 3), (2, 0), (3, 2)],
        "g3": [(1, 1), (1, 2), (3, 0)],
        "g2": [(2, 2), (2, 3)],
    },
    {
        "g4": [(0, 0), (0, 3), (2, 2), (3, 1)],
        "g3": [(1, 0), (1, 2), (3, 3)],
        "g2": [(2, 0), (2, 3)],
    },
]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_placeholder_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != PLACEHOLDER:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == PLACEHOLDER:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_signature(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    shifted = coords - coords.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in shifted))


def _center_from_bbox(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return ((r0 + r1) // 2, (c0 + c1) // 2)


def _component_info(coords: np.ndarray) -> dict:
    box = _bbox(coords)
    return {
        "coords": coords,
        "bbox": box,
        "center": _center_from_bbox(box),
        "signature": _normalize_signature(coords),
    }


def _iter_path_cells(a: tuple[int, int], b: tuple[int, int]):
    if a <= b:
        start, end = a, b
    else:
        start, end = b, a
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            yield r0, c
        return
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            yield r, c0
        return
    step_c = 1 if c1 >= c0 else -1
    for c in range(c0, c1 + step_c, step_c):
        yield r0, c
    step_r = 1 if r1 >= r0 else -1
    for r in range(r0 + step_r, r1 + step_r, step_r):
        yield r, c1


def _group_sort_key_high(group: dict) -> tuple[int, tuple[int, int]]:
    return (-group["size"], group["anchor"])


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    components = [_component_info(coords) for coords in _find_placeholder_components(g)]
    if not components:
        raise ValueError("expected placeholder objects")

    by_sig: dict[tuple[tuple[int, int], ...], list[dict]] = defaultdict(list)
    for info in components:
        by_sig[info["signature"]].append(info)

    groups = []
    singleton_groups = []
    for sig, members in by_sig.items():
        members = sorted(members, key=lambda item: item["bbox"][:2])
        payload = {
            "signature": sig,
            "members": members,
            "size": len(members),
            "anchor": members[0]["bbox"][:2],
        }
        if len(members) > 1:
            groups.append(payload)
        else:
            singleton_groups.append(payload)

    groups_high = sorted(groups, key=_group_sort_key_high)
    color_assignment = {}
    for idx, group in enumerate(groups_high):
        if idx >= len(GROUP_COLORS):
            raise ValueError("too many effective groups for fixed color order")
        color_assignment[id(group)] = GROUP_COLORS[idx]
        group["color"] = GROUP_COLORS[idx]
        group["priority"] = idx

    out = g.copy()
    object_mask = g == PLACEHOLDER

    for group in groups_high:
        color = group["color"]
        for member in group["members"]:
            out[tuple(member["coords"].T)] = color

    line_claims: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for group in reversed(groups_high):
        color = group["color"]
        for left, right in combinations(group["members"], 2):
            start, end = sorted((left["center"], right["center"]))
            for r, c in _iter_path_cells(start, end):
                if object_mask[r, c]:
                    continue
                out[r, c] = color
                line_claims[(r, c)].append(
                    {
                        "group_anchor": group["anchor"],
                        "group_size": group["size"],
                        "color": color,
                    }
                )

    if not return_meta:
        return out

    return out, {
        "components": components,
        "groups_high": groups_high,
        "singleton_groups": singleton_groups,
        "line_claims": line_claims,
        "object_mask": object_mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Group identical placeholder shapes by translation-normalized mask, recolor each group, and connect every intra-group pair with priority-resolved elbow lines.

    Input structure:
    - A black background contains several separated grey 4-connected placeholder objects.
    - Some objects share the same exact translation-normalized shape; 1-2 other shapes appear only once as distractors.
    - Object pixels are never adjacent across components, and all reasoning uses row/column coordinates of their bounding boxes.

    Rule:
    - Normalize each grey object by cropping to its bounding box and translating that mask to the origin.
    - Group objects with identical normalized masks; only groups with at least two members are active, while singleton shapes stay grey.
    - Sort active groups by member count descending and recolor them red, then blue, then green, then yellow; ties break by the earliest member bbox top-left in reading order.
    - For every pair of members inside an active group, connect their bbox centers with a one-cell elbow path drawn horizontal-first then vertical, using the group color and the earlier center as the start.
    - Lines may overwrite background and lower-priority lines, but they never overwrite any object body pixel.

    Key insight:
    - The endpoints and drawing priority are not given directly by color; they must both be inferred from translation-normalized shape grouping and group population size.
    """
    return _solve(grid, return_meta=False)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _slot_coords(row_vals: list[int], col_vals: list[int], pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(row_vals[r_idx], col_vals[c_idx]) for r_idx, c_idx in pairs]


def _candidate_slot_axes(limit: int, rng: np.random.Generator) -> list[int]:
    base = list(range(1, limit - 2, 5))
    if len(base) < 4:
        raise ValueError("grid too small for required spacing")
    chosen = sorted(rng.choice(base, size=4, replace=False).tolist())
    return chosen


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = PLACEHOLDER


def _all_positions(layout: dict[str, list[tuple[int, int]]], distractors: list[tuple[int, int]]):
    all_used = set()
    for pairs in layout.values():
        all_used.update(pairs)
    all_used.update(distractors)
    return all_used


def _sample_layout(rng: np.random.Generator, row_vals: list[int], col_vals: list[int], distractor_count: int):
    template = LAYOUT_TEMPLATES[int(rng.integers(0, len(LAYOUT_TEMPLATES)))]
    remap_rows = rng.permutation(4).tolist()
    remap_cols = rng.permutation(4).tolist()

    layout = {}
    for key, pairs in template.items():
        layout[key] = [(remap_rows[r], remap_cols[c]) for r, c in pairs]

    used = _all_positions(layout, [])
    free = [(r, c) for r, c in product(range(4), repeat=2) if (r, c) not in used]
    rng.shuffle(free)
    distractor_slots = free[:distractor_count]

    concrete = {
        key: _slot_coords(row_vals, col_vals, layout[key])
        for key in ("g4", "g3", "g2")
    }
    concrete["distractors"] = _slot_coords(row_vals, col_vals, distractor_slots)
    return concrete


def _build_input(
    rng: np.random.Generator,
    h: int,
    w: int,
    group_shapes: list[tuple[tuple[int, int], ...]],
    distractor_shapes: list[tuple[tuple[int, int], ...]],
):
    row_vals = _candidate_slot_axes(h, rng)
    col_vals = _candidate_slot_axes(w, rng)
    layout = _sample_layout(rng, row_vals, col_vals, len(distractor_shapes))

    grid = np.zeros((h, w), dtype=int)
    for top_left in layout["g4"]:
        _paint_shape(grid, group_shapes[0], top_left)
    for top_left in layout["g3"]:
        _paint_shape(grid, group_shapes[1], top_left)
    for top_left in layout["g2"]:
        _paint_shape(grid, group_shapes[2], top_left)
    for top_left, shape in zip(layout["distractors"], distractor_shapes):
        _paint_shape(grid, shape, top_left)
    return grid


def _validate_semantics(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        recomputed, meta = _solve(inp, return_meta=True)
    except Exception:
        return False
    if not np.array_equal(recomputed, out):
        return False

    groups = meta["groups_high"]
    singles = meta["singleton_groups"]
    if [group["size"] for group in groups] != [4, 3, 2]:
        return False
    if len(singles) not in (1, 2):
        return False

    for group in groups:
        for member in group["members"]:
            rr, cc = tuple(member["coords"].T)
            if not np.all(out[rr, cc] == group["color"]):
                return False

    for group in singles:
        member = group["members"][0]
        rr, cc = tuple(member["coords"].T)
        if not np.all(out[rr, cc] == PLACEHOLDER):
            return False

    aligned_pair_exists = False
    elbow_pair_exists = False
    for group in groups:
        centers = [member["center"] for member in group["members"]]
        has_aligned = False
        for a, b in combinations(centers, 2):
            if a[0] == b[0] or a[1] == b[1]:
                has_aligned = True
                aligned_pair_exists = True
            else:
                elbow_pair_exists = True
        if not has_aligned:
            return False
    if not aligned_pair_exists or not elbow_pair_exists:
        return False

    crossing_cells = []
    for cell, claims in meta["line_claims"].items():
        distinct_sizes = {claim["group_size"] for claim in claims}
        if len(distinct_sizes) > 1:
            crossing_cells.append((cell, claims))
    if not crossing_cells:
        return False

    # At least one genuine priority-resolved crossing on background.
    if not any(inp[r, c] == BACKGROUND for (r, c), _claims in crossing_cells):
        return False

    # Winner must match the largest participating group size.
    for (r, c), claims in crossing_cells:
        winner = int(out[r, c])
        max_size = max(claim["group_size"] for claim in claims)
        valid_colors = {claim["color"] for claim in claims if claim["group_size"] == max_size}
        if winner not in valid_colors:
            return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(19, 27))
    w = int(rng.integers(19, 27))

    chosen = rng.choice(len(SHAPE_LIBRARY), size=5, replace=False).tolist()
    rng.shuffle(chosen)
    group_shapes = [SHAPE_LIBRARY[idx] for idx in chosen[:3]]
    distractor_count = int(rng.integers(1, 3))
    distractor_shapes = [SHAPE_LIBRARY[idx] for idx in chosen[3 : 3 + distractor_count]]

    grid = _build_input(rng, h, w, group_shapes, distractor_shapes)
    out = main(grid)
    if not _validate_semantics(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
