from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random
from typing import Any

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_for_opus_11_003_3",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


BACKGROUND = 0
RECT_COLORS = (1, 2, 3, 4, 5)
MARKER_COLORS = (6, 7)
GRID_SIZE_RANGE = (18, 24)
GROUP_COUNT_WEIGHTS = ((2, 1), (3, 3))
DEPTH_OPTIONS = (1, 2, 3)
DEPTH_WEIGHTS = (2, 3, 2)
INNER_SIZE_RANGE = (3, 6)
NEST_GAP_VALUES = (1, 1, 2, 2, 3)
PLACEMENT_MARGIN = 2
PLACEMENT_GAP = 2
GENERATION_ATTEMPTS = 500


def _is_rectangular_perimeter(component: dict[str, Any]) -> bool:
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    coords = component["coords"]
    if h == 1 or w == 1:
        return int(component["area"]) == h * w
    border = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    return int(component["area"]) == 2 * h + 2 * w - 4 and bool(np.all(border))


def _strictly_contains_bbox(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = map(int, outer)
    rr0, cc0, rr1, cc1 = map(int, inner)
    return r0 < rr0 and c0 < cc0 and rr1 < r1 and cc1 < c1


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _marker_direction(shape: tuple[int, int], pos: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    r, c = map(int, pos)
    matches: list[tuple[int, int]] = []
    if r == 0:
        matches.append((1, 0))
    if r == h - 1:
        matches.append((-1, 0))
    if c == 0:
        matches.append((0, 1))
    if c == w - 1:
        matches.append((0, -1))
    assert len(matches) == 1, "marker must lie on exactly one canvas side"
    return matches[0]


def _marker_side(shape: tuple[int, int], pos: tuple[int, int]) -> str:
    h, w = map(int, shape)
    r, c = map(int, pos)
    if r == 0:
        return "top"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    assert c == w - 1
    return "right"


def _project_marker(
    grid: np.ndarray, pos: tuple[int, int]
) -> tuple[tuple[int, int], int] | None:
    dr, dc = _marker_direction(grid.shape, pos)
    r, c = pos[0] + dr, pos[1] + dc
    while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
        if int(grid[r, c]) != BACKGROUND:
            return (int(r), int(c)), abs(int(r - pos[0])) + abs(int(c - pos[1]))
        r += dr
        c += dc
    return None


def _winner_key(hit: dict[str, Any]) -> tuple[int, int, int]:
    r, c = hit["marker_pos"]
    return int(hit["distance"]), int(r), int(c)


def _analyze_grid(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    rects = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=RECT_COLORS,
        return_format="dict",
    )
    assert rects, "expected rectangular components"
    assert all(_is_rectangular_perimeter(comp) for comp in rects), "all foreground rectangles must be perimeters"

    coord_to_rect: dict[tuple[int, int], int] = {}
    for idx, rect in enumerate(rects):
        for r_raw, c_raw in rect["coords"]:
            coord = (int(r_raw), int(c_raw))
            assert coord not in coord_to_rect, "rectangles must not overlap"
            coord_to_rect[coord] = idx

    parents: list[int | None] = [None] * len(rects)
    for idx, rect in enumerate(rects):
        containers = [
            parent_idx
            for parent_idx, parent in enumerate(rects)
            if parent_idx != idx and _strictly_contains_bbox(parent["bbox"], rect["bbox"])
        ]
        if containers:
            parents[idx] = min(containers, key=lambda j: _bbox_area(rects[j]["bbox"]))

    root_of: list[int] = []
    for idx in range(len(rects)):
        cur = idx
        while parents[cur] is not None:
            cur = int(parents[cur])
        root_of.append(cur)

    root_indices = sorted({int(root) for root in root_of}, key=lambda i: tuple(map(int, rects[i]["bbox"])))
    members_by_root: dict[int, list[int]] = {root_idx: [] for root_idx in root_indices}
    for idx, root_idx in enumerate(root_of):
        members_by_root[int(root_idx)].append(idx)

    markers: list[dict[str, Any]] = []
    marker_mask = np.zeros(g.shape, dtype=bool)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if r not in (0, g.shape[0] - 1) and c not in (0, g.shape[1] - 1):
                continue
            if int(g[r, c]) not in MARKER_COLORS:
                continue
            marker_mask[r, c] = True
            projection = _project_marker(g, (r, c))
            assert projection is not None, "every marker must hit a rectangle"
            hit_pos, distance = projection
            assert hit_pos in coord_to_rect, "marker must hit a rectangle border"
            rect_idx = coord_to_rect[hit_pos]
            root_idx = int(root_of[rect_idx])
            markers.append(
                {
                    "pos": (int(r), int(c)),
                    "color": int(g[r, c]),
                    "side": _marker_side(g.shape, (r, c)),
                    "hit_pos": hit_pos,
                    "hit_rect_idx": int(rect_idx),
                    "root_idx": root_idx,
                    "distance": int(distance),
                }
            )

    hits_by_root: dict[int, list[dict[str, Any]]] = {root_idx: [] for root_idx in root_indices}
    for marker in markers:
        hits_by_root[int(marker["root_idx"])].append(
            {
                "marker_pos": marker["pos"],
                "color": marker["color"],
                "distance": marker["distance"],
                "hit_pos": marker["hit_pos"],
                "side": marker["side"],
            }
        )

    winners = {
        root_idx: min(hits, key=_winner_key)
        for root_idx, hits in hits_by_root.items()
        if hits
    }
    return {
        "grid": g,
        "bg": bg,
        "rects": rects,
        "parents": parents,
        "root_of": root_of,
        "root_indices": root_indices,
        "members_by_root": members_by_root,
        "markers": markers,
        "marker_mask": marker_mask,
        "hits_by_root": hits_by_root,
        "winners": winners,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project boundary markers inward to select an outer rectangular frame, then recolor that whole contained nesting tree.

    Input structure
    - The grid has a black background, 2-3 disjoint groups of monochromatic rectangular perimeters, and 2-4 single-pixel markers on the canvas boundary.
    - Each group consists of 1-3 strictly nested rectangle borders; colors of rectangle borders come from blue, red, green, yellow, or gray.
    - Boundary markers are magenta or orange and each projects straight inward along its row or column to the first non-background rectangle border.

    Rule
    - Identify every boundary marker and project it inward according to its side: top/bottom markers move vertically and left/right markers move horizontally.
    - The first rectangle border hit by a marker selects that border's outermost containing rectangle group.
    - For each selected outermost group, recolor the outer border and every non-background pixel inside that root bbox to the winning marker's color.
    - If two markers hit the same outermost group, prefer the one with shorter projection distance; if distances tie, prefer the smaller marker row, then the smaller marker column.
    - Leave unhit rectangle groups unchanged, and preserve the boundary marker pixels themselves.

    Key insight
    - A marker does not paint only the collision point: the collision chooses one outer rectangle root, and that choice propagates inward through the full strict-containment hierarchy.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    g = analysis["grid"]
    bg = int(analysis["bg"])
    marker_mask = analysis["marker_mask"]
    out = g.copy()

    for root_idx, winner in analysis["winners"].items():
        r0, c0, r1, c1 = map(int, analysis["rects"][root_idx]["bbox"])
        region = out[r0 : r1 + 1, c0 : c1 + 1]
        source = g[r0 : r1 + 1, c0 : c1 + 1]
        protected = marker_mask[r0 : r1 + 1, c0 : c1 + 1]
        fill_mask = (source != bg) & (~protected)
        region[fill_mask] = int(winner["color"])

    return out


def _weighted_choice(options: tuple[int, ...], weights: tuple[int, ...]) -> int:
    return random.choices(list(options), weights=list(weights), k=1)[0]


def _make_group_spec(depth: int) -> dict[str, Any]:
    while True:
        inner_h = random.randint(*INNER_SIZE_RANGE)
        inner_w = random.randint(*INNER_SIZE_RANGE)
        boxes = [(0, 0, inner_h - 1, inner_w - 1)]
        for _ in range(depth - 1):
            top = random.choice(NEST_GAP_VALUES)
            bottom = random.choice(NEST_GAP_VALUES)
            left = random.choice(NEST_GAP_VALUES)
            right = random.choice(NEST_GAP_VALUES)
            r0, c0, r1, c1 = boxes[-1]
            boxes.append((r0 - top, c0 - left, r1 + bottom, c1 + right))
        outer_r0, outer_c0, _, _ = boxes[-1]
        shifted = []
        for r0, c0, r1, c1 in boxes:
            shifted.append((r0 - outer_r0, c0 - outer_c0, r1 - outer_r0, c1 - outer_c0))
        root_box = shifted[-1]
        h = root_box[2] + 1
        w = root_box[3] + 1
        if 4 <= h <= 12 and 4 <= w <= 12:
            colors = random.sample(list(RECT_COLORS), depth)
            layers = []
            for box, color in zip(reversed(shifted), colors, strict=False):
                layers.append({"bbox": box, "color": int(color)})
            return {"depth": int(depth), "height": int(h), "width": int(w), "layers": layers}


def _bbox_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _draw_group(grid: np.ndarray, placed_group: dict[str, Any]) -> None:
    base_r = int(placed_group["r0"])
    base_c = int(placed_group["c0"])
    for layer in placed_group["layers"]:
        r0, c0, r1, c1 = layer["bbox"]
        color = int(layer["color"])
        rr0, cc0 = base_r + int(r0), base_c + int(c0)
        rr1, cc1 = base_r + int(r1), base_c + int(c1)
        grid[rr0, cc0 : cc1 + 1] = color
        grid[rr1, cc0 : cc1 + 1] = color
        grid[rr0 : rr1 + 1, cc0] = color
        grid[rr0 : rr1 + 1, cc1] = color


def _enumerate_root_candidates(grid: np.ndarray, root_bbox: tuple[int, int, int, int]) -> list[dict[str, Any]]:
    h, w = grid.shape
    r0, c0, r1, c1 = map(int, root_bbox)
    candidates: list[dict[str, Any]] = []

    for c in range(c0 + 1, c1):
        projection = _project_marker(grid, (0, c))
        if projection and projection[0] == (r0, c):
            candidates.append(
                {
                    "marker_pos": (0, c),
                    "side": "top",
                    "hit_pos": (r0, c),
                    "distance": int(projection[1]),
                }
            )
        projection = _project_marker(grid, (h - 1, c))
        if projection and projection[0] == (r1, c):
            candidates.append(
                {
                    "marker_pos": (h - 1, c),
                    "side": "bottom",
                    "hit_pos": (r1, c),
                    "distance": int(projection[1]),
                }
            )

    for r in range(r0 + 1, r1):
        projection = _project_marker(grid, (r, 0))
        if projection and projection[0] == (r, c0):
            candidates.append(
                {
                    "marker_pos": (r, 0),
                    "side": "left",
                    "hit_pos": (r, c0),
                    "distance": int(projection[1]),
                }
            )
        projection = _project_marker(grid, (r, w - 1))
        if projection and projection[0] == (r, c1):
            candidates.append(
                {
                    "marker_pos": (r, w - 1),
                    "side": "right",
                    "hit_pos": (r, c1),
                    "distance": int(projection[1]),
                }
            )
    return candidates


def _pick_conflict_pair(candidates: list[dict[str, Any]], mode: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a = candidates[i]
            b = candidates[j]
            if a["side"] == b["side"]:
                continue
            if a["marker_pos"] == b["marker_pos"]:
                continue
            da = int(a["distance"])
            db = int(b["distance"])
            ra, ca = a["marker_pos"]
            rb, cb = b["marker_pos"]
            if mode == "unequal" and da != db:
                pairs.append((a, b))
            elif mode == "tie_row" and da == db and ra != rb:
                pairs.append((a, b))
            elif mode == "tie_col" and da == db and ra == rb and ca != cb:
                pairs.append((a, b))
    if not pairs:
        return None
    return random.choice(pairs)


def _place_groups(
    group_specs: list[dict[str, Any]], shape: tuple[int, int], conflict_mode: str
) -> list[dict[str, Any]] | None:
    h, w = map(int, shape)
    placed: list[dict[str, Any]] = []

    conflict_spec = dict(group_specs[0])
    for _ in range(120):
        if conflict_mode == "tie_col":
            if (w - conflict_spec["width"]) % 2:
                return None
            c0 = (w - conflict_spec["width"]) // 2
            r_choices = list(
                range(
                    PLACEMENT_MARGIN,
                    h - conflict_spec["height"] - PLACEMENT_MARGIN + 1,
                )
            )
            if not r_choices:
                return None
            r0 = random.choice(r_choices)
        elif conflict_mode == "tie_row":
            if (h - conflict_spec["height"]) % 2:
                return None
            r0 = (h - conflict_spec["height"]) // 2
            c_choices = list(
                range(
                    PLACEMENT_MARGIN,
                    w - conflict_spec["width"] - PLACEMENT_MARGIN + 1,
                )
            )
            if not c_choices:
                return None
            c0 = random.choice(c_choices)
        else:
            r_choices = list(
                range(
                    PLACEMENT_MARGIN,
                    h - conflict_spec["height"] - PLACEMENT_MARGIN + 1,
                )
            )
            c_choices = list(
                range(
                    PLACEMENT_MARGIN,
                    w - conflict_spec["width"] - PLACEMENT_MARGIN + 1,
                )
            )
            if not r_choices or not c_choices:
                return None
            r0 = random.choice(r_choices)
            c0 = random.choice(c_choices)
        bbox = (r0, c0, r0 + conflict_spec["height"] - 1, c0 + conflict_spec["width"] - 1)
        placed = [{**conflict_spec, "r0": int(r0), "c0": int(c0), "root_bbox": bbox}]
        break
    else:
        return None

    for spec in group_specs[1:]:
        options = []
        for r0 in range(PLACEMENT_MARGIN, h - spec["height"] - PLACEMENT_MARGIN + 1):
            for c0 in range(PLACEMENT_MARGIN, w - spec["width"] - PLACEMENT_MARGIN + 1):
                bbox = (r0, c0, r0 + spec["height"] - 1, c0 + spec["width"] - 1)
                if all(_bbox_separated(bbox, other["root_bbox"], PLACEMENT_GAP) for other in placed):
                    options.append((r0, c0, bbox))
        if not options:
            return None
        r0, c0, bbox = random.choice(options)
        placed.append({**spec, "r0": int(r0), "c0": int(c0), "root_bbox": bbox})
    return placed


def _build_instance() -> tuple[np.ndarray, dict[str, Any]]:
    n_groups = random.choices([2, 3], weights=[w for _, w in GROUP_COUNT_WEIGHTS], k=1)[0]
    conflict_mode = random.choices(
        ["unequal", "tie_row", "tie_col"],
        weights=[4, 3, 2],
        k=1,
    )[0]
    grid_h = random.randint(*GRID_SIZE_RANGE)
    grid_w = random.randint(*GRID_SIZE_RANGE)

    depths = [_weighted_choice(DEPTH_OPTIONS, DEPTH_WEIGHTS) for _ in range(n_groups)]
    random.shuffle(depths)
    group_specs = [_make_group_spec(depth) for depth in depths]
    group_specs.sort(key=lambda spec: (-int(spec["depth"]), -int(spec["height"]) * int(spec["width"])))

    placed_groups = _place_groups(group_specs, (grid_h, grid_w), conflict_mode)
    if placed_groups is None:
        raise ValueError("placement failed")

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for group in placed_groups:
        _draw_group(grid, group)

    candidate_by_group = {
        idx: _enumerate_root_candidates(grid, group["root_bbox"]) for idx, group in enumerate(placed_groups)
    }
    conflict_group_choices = [
        idx
        for idx, candidates in candidate_by_group.items()
        if _pick_conflict_pair(candidates, conflict_mode) is not None
    ]
    if not conflict_group_choices:
        raise ValueError("no conflict root")
    conflict_group_idx = random.choice(conflict_group_choices)
    conflict_pair = _pick_conflict_pair(candidate_by_group[conflict_group_idx], conflict_mode)
    assert conflict_pair is not None

    chosen: list[dict[str, Any]] = []
    conflict_colors = list(MARKER_COLORS)
    random.shuffle(conflict_colors)
    for candidate, color in zip(conflict_pair, conflict_colors, strict=False):
        chosen.append(
            {
                **candidate,
                "color": int(color),
                "group_idx": int(conflict_group_idx),
                "role": "conflict",
            }
        )

    if n_groups == 3:
        remaining = [idx for idx in range(3) if idx != conflict_group_idx]
        single_hit_options = [idx for idx in remaining if candidate_by_group[idx]]
        if not single_hit_options:
            raise ValueError("no single-hit group")
        single_group_idx = random.choice(single_hit_options)
        untouched_group_idx = [idx for idx in remaining if idx != single_group_idx][0]
        single_choices = [
            cand
            for cand in candidate_by_group[single_group_idx]
            if cand["marker_pos"] not in {item["marker_pos"] for item in chosen}
        ]
        if not single_choices:
            raise ValueError("single-hit marker collision")
        candidate = random.choice(single_choices)
        chosen.append(
            {
                **candidate,
                "color": int(random.choice(MARKER_COLORS)),
                "group_idx": int(single_group_idx),
                "role": "single",
            }
        )
    else:
        untouched_group_idx = next(idx for idx in range(2) if idx != conflict_group_idx)

    for marker in chosen:
        r, c = marker["marker_pos"]
        assert int(grid[r, c]) == BACKGROUND
        grid[r, c] = int(marker["color"])

    return grid, {
        "conflict_mode": conflict_mode,
        "conflict_group_idx": int(conflict_group_idx),
        "untouched_group_idx": int(untouched_group_idx),
        "chosen_markers": chosen,
        "placed_groups": placed_groups,
    }


def _validate_generated_instance(grid: np.ndarray, meta: dict[str, Any]) -> None:
    analysis = _analyze_grid(grid)
    out = main(grid)
    roots = analysis["root_indices"]
    bbox_to_root = {
        tuple(map(int, analysis["rects"][root_idx]["bbox"])): int(root_idx) for root_idx in roots
    }

    assert 2 <= len(roots) <= 3
    assert 2 <= len(analysis["markers"]) <= 4
    assert len(analysis["markers"]) == len(meta["chosen_markers"])
    assert all(len(hits) <= 2 for hits in analysis["hits_by_root"].values())

    hit_counts = {root_idx: len(analysis["hits_by_root"][root_idx]) for root_idx in roots}
    assert 2 in hit_counts.values(), "need one doubly hit root"
    assert 0 in hit_counts.values(), "need one untouched root"
    if len(roots) == 3:
        assert 1 in hit_counts.values(), "3-root cases must include a single-hit branch"

    conflict_root_idx = next(root_idx for root_idx, cnt in hit_counts.items() if cnt == 2)
    conflict_hits = analysis["hits_by_root"][conflict_root_idx]
    assert {int(hit["color"]) for hit in conflict_hits} == set(MARKER_COLORS)

    winner = min(conflict_hits, key=_winner_key)
    loser = max(conflict_hits, key=_winner_key)
    assert winner != loser
    mode = meta["conflict_mode"]
    if mode == "unequal":
        assert winner["distance"] != loser["distance"]
    elif mode == "tie_row":
        assert winner["distance"] == loser["distance"]
        assert winner["marker_pos"][0] < loser["marker_pos"][0]
    else:
        assert mode == "tie_col"
        assert winner["distance"] == loser["distance"]
        assert winner["marker_pos"][0] == loser["marker_pos"][0]
        assert winner["marker_pos"][1] < loser["marker_pos"][1]

    untouched_root_idx = bbox_to_root[tuple(map(int, meta["placed_groups"][int(meta["untouched_group_idx"])]["root_bbox"]))]
    assert hit_counts[untouched_root_idx] == 0

    changed_pixels = 0
    for root_idx in roots:
        r0, c0, r1, c1 = map(int, analysis["rects"][root_idx]["bbox"])
        source = grid[r0 : r1 + 1, c0 : c1 + 1]
        target = out[r0 : r1 + 1, c0 : c1 + 1]
        protected = analysis["marker_mask"][r0 : r1 + 1, c0 : c1 + 1]
        fg_mask = (source != analysis["bg"]) & (~protected)
        if hit_counts[root_idx] == 0:
            assert np.array_equal(source, target), "untouched group must stay unchanged"
            continue
        fill_color = int(analysis["winners"][root_idx]["color"])
        assert np.all(target[fg_mask] == fill_color), "hit group must recolor to winner color"
        changed_pixels += int(np.sum(source[fg_mask] != target[fg_mask]))

    assert changed_pixels > 0
    for marker in analysis["markers"]:
        assert int(out[marker["pos"]]) == int(marker["color"])


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            grid, meta = _build_instance()
            _validate_generated_instance(grid, meta)
            return grid
        except (AssertionError, ValueError, IndexError, StopIteration):
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
