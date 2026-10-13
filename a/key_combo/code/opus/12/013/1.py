from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
TOKEN_COLORS = (1, 2, 3, 6, 7, 8, 9)

GEN_HEIGHT_RANGE = (14, 18)
GEN_WIDTH_RANGE = (14, 18)
GEN_ANCHOR_COUNTS = (2, 3)
GEN_COLOR_COUNTS = (3, 4)
GEN_MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != color or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen:
                        if int(grid[nr, nc]) == color:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
            comps.append(comp)
    return comps


def _make_anchor(top: int, left: int) -> dict:
    cells = {
        (top, left),
        (top, left + 1),
        (top + 1, left),
        (top + 1, left + 1),
    }
    frame = [
        (r, c)
        for r in range(top - 1, top + 3)
        for c in range(left - 1, left + 3)
        if (r, c) not in cells
    ]
    side_frame = [
        (top, left - 1),
        (top + 1, left - 1),
        (top, left + 2),
        (top + 1, left + 2),
    ]
    return {
        "top": top,
        "left": left,
        "cells": cells,
        "frame": frame,
        "side_frame": side_frame,
    }


def _find_anchors(grid: np.ndarray) -> list[dict]:
    anchors = []
    for comp in _components_of_color(grid, GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(comp) != 4 or (r1 - r0, c1 - c0) != (1, 1):
            raise AssertionError("grey objects must be isolated 2x2 anchors")
        expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if set(comp) != expected:
            raise AssertionError("grey anchor must be a solid 2x2 block")
        anchors.append(_make_anchor(r0, c0))
    anchors.sort(key=lambda a: (a["top"], a["left"]))
    if not (2 <= len(anchors) <= 3):
        raise AssertionError("expected 2 or 3 anchors")
    return anchors


def _chebyshev_to_anchor(cell: tuple[int, int], anchor: dict) -> int:
    r, c = cell
    return min(max(abs(r - ar), abs(c - ac)) for ar, ac in anchor["cells"])


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _snap_pixel(
    cell: tuple[int, int], anchors: list[dict], *, strict: bool
) -> tuple[int, tuple[int, int]] | None:
    anchor_distances = [_chebyshev_to_anchor(cell, anchor) for anchor in anchors]
    best_anchor_distance = min(anchor_distances)
    best_anchor_indices = [
        i for i, d in enumerate(anchor_distances) if d == best_anchor_distance
    ]
    if len(best_anchor_indices) != 1:
        if strict:
            return None
        best_anchor_indices.sort(key=lambda i: (anchors[i]["top"], anchors[i]["left"]))
    anchor_idx = best_anchor_indices[0]
    anchor = anchors[anchor_idx]

    frame_distances = [_manhattan(cell, target) for target in anchor["frame"]]
    best_frame_distance = min(frame_distances)
    best_targets = [
        target
        for target, dist in zip(anchor["frame"], frame_distances, strict=False)
        if dist == best_frame_distance
    ]
    if len(best_targets) != 1:
        if strict:
            return None
        best_targets.sort()
    return anchor_idx, best_targets[0]


def _pair_start_end(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _path_cells(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    start, end = _pair_start_end(a, b)
    step = 1 if end[1] >= start[1] else -1
    cells = [(start[0], c) for c in range(start[1], end[1] + step, step)]
    cells.extend((r, end[1]) for r in range(start[0] + 1, end[0] + 1))
    return start, end, cells


def _token_records(
    grid: np.ndarray, anchors: list[dict]
) -> list[dict]:
    anchor_cells = set().union(*(anchor["cells"] for anchor in anchors))
    colors = sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, GREY))
    records = []
    for color in colors:
        coords = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        if len(coords) != 2:
            raise AssertionError("each token color must appear exactly twice")
        for cell in coords:
            if cell in anchor_cells:
                raise AssertionError("token may not overlap an anchor")
            snap = _snap_pixel(cell, anchors, strict=True)
            if snap is None:
                raise AssertionError("token must have a unique snap target")
            anchor_idx, target = snap
            records.append(
                {
                    "color": color,
                    "source": cell,
                    "anchor_idx": anchor_idx,
                    "target": target,
                }
            )
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Snap paired singleton colors onto grey anchor frames, then either recolor same-anchor pairs or connect cross-anchor pairs by elbow paths.

    Input structure:
    - Black background contains 2 or 3 separated grey 2x2 anchor blocks.
    - Several other colors appear only as isolated single pixels, with each color occurring exactly twice.
    - Each singleton has a unique nearest anchor by Chebyshev distance and a unique nearest outer-frame cell on that anchor by Manhattan distance.

    Rule:
    - Move every colored singleton to its unique nearest cell on the one-cell outer frame of its nearest grey anchor, clearing the original cell.
    - For each color pair, if both snapped pixels land on the same anchor then recolor both snapped pixels yellow.
    - Otherwise sort the two snapped pixels by row then column, draw horizontally from the earlier one to the later one's column, then vertically down that column to the later pixel, while leaving grey anchor cells untouched; if multiple paths cross, the path whose start is earlier in that same ordering keeps the crossing color.

    Key insight:
    - The grey anchors act as sorting stations: only after snapping do you know whether a pair collapses into yellow markers or remains a colored cross-anchor path.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    anchors = _find_anchors(g)
    records = _token_records(g, anchors)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_color[rec["color"]].append(rec)

    out = g.copy()
    for rec in records:
        out[rec["source"]] = BG
    for rec in records:
        out[rec["target"]] = rec["color"]

    path_jobs = []
    for color, pair in by_color.items():
        if len(pair) != 2:
            raise AssertionError("each color must define exactly one pair")
        a, b = pair
        if a["anchor_idx"] == b["anchor_idx"]:
            out[a["target"]] = YELLOW
            out[b["target"]] = YELLOW
        else:
            start, end, cells = _path_cells(a["target"], b["target"])
            path_jobs.append(
                {
                    "color": color,
                    "start": start,
                    "end": end,
                    "cells": cells,
                }
            )

    anchor_cells = set().union(*(anchor["cells"] for anchor in anchors))
    painted_path_cells: set[tuple[int, int]] = set()
    for job in sorted(path_jobs, key=lambda item: item["start"]):
        for cell in job["cells"]:
            if cell in anchor_cells:
                continue
            if cell in painted_path_cells:
                continue
            out[cell] = job["color"]
            painted_path_cells.add(cell)

    return out


def _anchor_separated(existing: list[dict], top: int, left: int) -> bool:
    new_cells = {(top + dr, left + dc) for dr in range(2) for dc in range(2)}
    for anchor in existing:
        for a in new_cells:
            for b in anchor["cells"]:
                if max(abs(a[0] - b[0]), abs(a[1] - b[1])) < 4:
                    return False
    return True


def _place_anchors(h: int, w: int, count: int) -> list[dict] | None:
    positions = [(r, c) for r in range(1, h - 2) for c in range(1, w - 2)]
    random.shuffle(positions)
    anchors: list[dict] = []
    for top, left in positions:
        if _anchor_separated(anchors, top, left):
            anchors.append(_make_anchor(top, left))
            if len(anchors) == count:
                return anchors
    return None


def _sample_pair_specs(n_colors: int, n_anchors: int) -> tuple[list[dict], bool]:
    if n_colors == 3:
        same_count = 1 if random.random() < 0.65 else 2
    else:
        same_count = random.choice([1, 2, 2, 3])
    cross_count = n_colors - same_count
    if cross_count == 0:
        cross_count = 1
        same_count -= 1
    if same_count == 0:
        same_count = 1
        cross_count -= 1

    specs = []
    for _ in range(same_count):
        specs.append({"mode": "same", "anchors": (random.randrange(n_anchors),)})
    anchor_pairs = [(a, b) for a in range(n_anchors) for b in range(a + 1, n_anchors)]
    for _ in range(cross_count):
        specs.append({"mode": "cross", "anchors": random.choice(anchor_pairs)})
    random.shuffle(specs)
    require_crossing = cross_count >= 2 and random.random() < 0.55
    return specs, require_crossing


def _enumerate_source_candidates(
    h: int, w: int, anchors: list[dict]
) -> dict[tuple[int, tuple[int, int]], list[tuple[int, int]]]:
    anchor_cells = set().union(*(anchor["cells"] for anchor in anchors))
    frame_cells = set().union(*(set(anchor["frame"]) for anchor in anchors))
    candidates: dict[tuple[int, tuple[int, int]], list[tuple[int, int]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if cell in anchor_cells or cell in frame_cells:
                continue
            snapped = _snap_pixel(cell, anchors, strict=True)
            if snapped is None:
                continue
            anchor_idx, target = snapped
            if _manhattan(cell, target) < 2:
                continue
            if _chebyshev_to_anchor(cell, anchors[anchor_idx]) < 2:
                continue
            candidates[(anchor_idx, target)].append(cell)
    for cells in candidates.values():
        random.shuffle(cells)
    return candidates


def _usable_frame_cells(
    anchors: list[dict], candidates: dict[tuple[int, tuple[int, int]], list[tuple[int, int]]]
) -> dict[int, list[tuple[int, int]]]:
    usable = {}
    for idx, anchor in enumerate(anchors):
        cells = [cell for cell in anchor["frame"] if candidates.get((idx, cell))]
        random.shuffle(cells)
        usable[idx] = cells
    return usable


def _validate_target_pairs(
    anchors: list[dict], pair_targets: list[dict], require_crossing: bool
) -> bool:
    used_targets = []
    for item in pair_targets:
        used_targets.extend(item["targets"])
    if len(set(used_targets)) != len(used_targets):
        return False

    anchor_cells = set().union(*(anchor["cells"] for anchor in anchors))
    cross_routes = []
    same_targets = set()
    interrupted = False
    for item in pair_targets:
        if item["mode"] == "same":
            same_targets.update(item["targets"])
            continue
        start, end, route = _path_cells(*item["targets"])
        route_set = {cell for cell in route if cell not in anchor_cells}
        if any(cell in same_targets for cell in route_set):
            return False
        if any(cell in set(used_targets) - set(item["targets"]) for cell in route_set):
            return False
        if any(cell in anchor_cells for cell in route):
            interrupted = True
        cross_routes.append({"start": start, "end": end, "cells": route_set})

    counts = Counter(cell for route in cross_routes for cell in route["cells"])
    overlap_cells = {cell for cell, cnt in counts.items() if cnt > 1}
    if any(cnt > 2 for cnt in counts.values()):
        return False
    for cell in overlap_cells:
        for nr, nc in _neighbors4(*cell):
            if (nr, nc) in overlap_cells:
                return False
    if require_crossing and not overlap_cells:
        return False
    if not interrupted:
        return False

    return True


def _pair_options(
    spec: dict, usable: dict[int, list[tuple[int, int]]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if spec["mode"] == "same":
        anchor_idx = spec["anchors"][0]
        cells = usable[anchor_idx]
        options = []
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                options.append((cells[i], cells[j]))
        return options
    left_idx, right_idx = spec["anchors"]
    return [(a, b) for a in usable[left_idx] for b in usable[right_idx]]


def _assign_targets(
    anchors: list[dict],
    specs: list[dict],
    usable: dict[int, list[tuple[int, int]]],
    require_crossing: bool,
) -> list[dict] | None:
    options_by_index = {i: _pair_options(specs[i], usable) for i in range(len(specs))}
    if any(len(options_by_index[i]) == 0 for i in range(len(specs))):
        return None
    for _ in range(200):
        order = sorted(
            range(len(specs)),
            key=lambda i: (len(options_by_index[i]), random.random()),
        )
        chosen: dict[int, dict] = {}
        used: set[tuple[int, int]] = set()
        ok = True
        for spec_idx in order:
            viable = [
                endpoints
                for endpoints in options_by_index[spec_idx]
                if endpoints[0] not in used and endpoints[1] not in used
            ]
            if not viable:
                ok = False
                break
            endpoints = random.choice(viable)
            chosen[spec_idx] = {
                "mode": specs[spec_idx]["mode"],
                "anchors": specs[spec_idx]["anchors"],
                "targets": endpoints,
            }
            used.update(endpoints)
        if not ok:
            continue
        target_list = [chosen[i] for i in range(len(specs))]
        if _validate_target_pairs(anchors, target_list, require_crossing):
            return target_list
    return None


def _far_enough_from_selected(
    cell: tuple[int, int], selected: set[tuple[int, int]]
) -> bool:
    return all(max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) > 1 for other in selected)


def _assign_sources(
    token_defs: list[dict],
    candidates: dict[tuple[int, tuple[int, int]], list[tuple[int, int]]],
) -> list[dict] | None:
    enriched = []
    for item in token_defs:
        options = candidates.get((item["anchor_idx"], item["target"]), [])
        if not options:
            return None
        enriched.append({**item, "options": options[:]})
    for _ in range(200):
        order = sorted(
            range(len(enriched)),
            key=lambda i: (len(enriched[i]["options"]), random.random()),
        )
        chosen: dict[int, tuple[int, int]] = {}
        used: set[tuple[int, int]] = set()
        ok = True
        for idx in order:
            viable = [
                cell
                for cell in enriched[idx]["options"]
                if cell not in used and _far_enough_from_selected(cell, used)
            ]
            if not viable:
                ok = False
                break
            cell = random.choice(viable)
            chosen[idx] = cell
            used.add(cell)
        if not ok:
            continue
        out = []
        for idx, item in enumerate(enriched):
            out.append({k: v for k, v in item.items() if k != "options"} | {"source": chosen[idx]})
        return out
    return None


def _build_grid(
    h: int, w: int, anchors: list[dict], tokens: list[dict]
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for anchor in anchors:
        for cell in anchor["cells"]:
            grid[cell] = GREY
    for token in tokens:
        grid[token["source"]] = token["color"]
    return grid


def _check_generated_semantics(
    grid: np.ndarray, require_crossing: bool
) -> bool:
    try:
        anchors = _find_anchors(grid)
        records = _token_records(grid, anchors)
    except AssertionError:
        return False

    if not (3 <= len({rec["color"] for rec in records}) <= 4):
        return False

    sources = [rec["source"] for rec in records]
    for i, a in enumerate(sources):
        for b in sources[i + 1 :]:
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
                return False

    targets = [rec["target"] for rec in records]
    if len(set(targets)) != len(targets):
        return False

    by_color: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        by_color[rec["color"]].append(rec)
    same_count = 0
    cross_count = 0
    pair_targets = []
    for pair in by_color.values():
        if len(pair) != 2:
            return False
        if pair[0]["anchor_idx"] == pair[1]["anchor_idx"]:
            same_count += 1
            pair_targets.append(
                {
                    "mode": "same",
                    "anchors": (pair[0]["anchor_idx"],),
                    "targets": (pair[0]["target"], pair[1]["target"]),
                }
            )
        else:
            cross_count += 1
            pair_targets.append(
                {
                    "mode": "cross",
                    "anchors": tuple(sorted((pair[0]["anchor_idx"], pair[1]["anchor_idx"]))),
                    "targets": (pair[0]["target"], pair[1]["target"]),
                }
            )
    if same_count == 0 or cross_count == 0:
        return False
    if not _validate_target_pairs(anchors, pair_targets, require_crossing):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.any(out == YELLOW):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        h = random.randint(*GEN_HEIGHT_RANGE)
        w = random.randint(*GEN_WIDTH_RANGE)
        n_anchors = random.choice(GEN_ANCHOR_COUNTS)
        anchors = _place_anchors(h, w, n_anchors)
        if anchors is None:
            continue

        candidates = _enumerate_source_candidates(h, w, anchors)
        usable = _usable_frame_cells(anchors, candidates)
        if any(len(cells) < 2 for cells in usable.values()):
            continue

        n_colors = random.choice(GEN_COLOR_COUNTS)
        colors = random.sample(TOKEN_COLORS, n_colors)
        specs, require_crossing = _sample_pair_specs(n_colors, n_anchors)
        pair_targets = _assign_targets(anchors, specs, usable, require_crossing)
        if pair_targets is None:
            continue

        token_defs = []
        for color, pair in zip(colors, pair_targets, strict=False):
            if pair["mode"] == "same":
                anchor_idx = pair["anchors"][0]
                token_defs.append({"color": color, "anchor_idx": anchor_idx, "target": pair["targets"][0]})
                token_defs.append({"color": color, "anchor_idx": anchor_idx, "target": pair["targets"][1]})
            else:
                left_idx, right_idx = pair["anchors"]
                token_defs.append({"color": color, "anchor_idx": left_idx, "target": pair["targets"][0]})
                token_defs.append({"color": color, "anchor_idx": right_idx, "target": pair["targets"][1]})

        tokens = _assign_sources(token_defs, candidates)
        if tokens is None:
            continue

        grid = _build_grid(h, w, anchors, tokens)
        if _check_generated_semantics(grid, require_crossing):
            return grid

    raise RuntimeError("failed to generate a valid anchor-sorting ARC input")
