from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
BLOCKER = 2
GREY = 5
ANCHOR_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_ATTEMPTS = 300
SELECT_ATTEMPTS = 300
BLOCKER_ATTEMPTS = 60


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _anchor_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _find_anchors(grid: np.ndarray) -> dict[int, dict]:
    anchors: dict[int, dict] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BG, BLOCKER, GREY)):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 9
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3)
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
        anchors[color] = {"color": color, "bbox": bbox, "cells": _anchor_cells(bbox)}
    assert anchors
    return anchors


def _ray_info(
    cell: tuple[int, int],
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> dict | None:
    r, c = cell
    r0, c0, r1, c1 = bbox
    dr = 1 if r < r0 else -1 if r > r1 else 0
    dc = 1 if c < c0 else -1 if c > c1 else 0
    if (dr, dc) == (0, 0):
        return None

    path: list[tuple[int, int]] = []
    cr, cc = r, c
    while True:
        nr, nc = cr + dr, cc + dc
        if not _in_bounds(shape, nr, nc):
            return None
        if r0 <= nr <= r1 and c0 <= nc <= c1:
            return {
                "terminal": (cr, cc),
                "step": (dr, dc),
                "dist": len(path),
                "path": path[:],
            }
        path.append((nr, nc))
        cr, cc = nr, nc


def _blocker_forbidden(shape: tuple[int, int], blockers: list[tuple[int, int]]) -> np.ndarray:
    forbidden = np.zeros(shape, dtype=bool)
    for r, c in blockers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(shape, rr, cc):
                    forbidden[rr, cc] = True
    return forbidden


def _visible_candidates(
    cell: tuple[int, int],
    anchors: dict[int, dict],
    grid: np.ndarray,
    forbidden: np.ndarray,
    *,
    ignore_grey: bool,
) -> list[dict]:
    candidates: list[dict] = []
    for color, anchor in anchors.items():
        ray = _ray_info(cell, anchor["bbox"], grid.shape)
        if ray is None:
            continue
        ok = True
        for pr, pc in ray["path"]:
            if forbidden[pr, pc]:
                ok = False
                break
            value = int(grid[pr, pc])
            if value == BG or (ignore_grey and value == GREY):
                continue
            ok = False
            break
        if ok:
            candidates.append(
                {
                    "color": color,
                    "bbox": anchor["bbox"],
                    "terminal": ray["terminal"],
                    "step": ray["step"],
                    "dist": ray["dist"],
                    "path": ray["path"],
                }
            )
    candidates.sort(key=lambda item: (item["dist"], item["color"]))
    return candidates


def _render_output_from_assignments(grid: np.ndarray, assignments: list[dict]) -> np.ndarray:
    out = np.full_like(grid, BG)
    anchors = _find_anchors(grid)
    for color, anchor in anchors.items():
        r0, c0, r1, c1 = anchor["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    out[grid == BLOCKER] = BLOCKER

    groups: dict[tuple[int, tuple[int, int], tuple[int, int]], list[dict]] = defaultdict(list)
    for item in assignments:
        key = (item["color"], item["terminal"], item["step"])
        groups[key].append(item)

    for (color, terminal, step), members in groups.items():
        tr, tc = terminal
        dr, dc = step
        for index, member in enumerate(sorted(members, key=lambda item: item["dist"])):
            rr = tr - dr * index
            cc = tc - dc * index
            assert _in_bounds(out.shape, rr, cc)
            assert out[rr, cc] == BG
            out[rr, cc] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Move each gray pixel onto the uniquely visible anchor ray and recolor it to that anchor.

    Input structure:
    - A black background contains several solid 3x3 colored anchor blocks with distinct colors, plus isolated gray singleton pixels and a few red blocker pixels.
    - From each gray pixel, looking along one of the eight straight row/column/diagonal directions reaches exactly one anchor through a valid sightline.

    Rule:
    - A sightline is valid when the cells between the gray pixel and the anchor are otherwise empty and none of those intervening cells lies in the 8-neighborhood of any red blocker.
    - Assign each gray pixel to its unique valid anchor ray, then pack all gray pixels from the same ray into consecutive cells adjacent to that anchor while preserving their near-to-far order on that ray.
    - Recolor each moved pixel to the anchor's color, and leave the anchors and red blockers unchanged.

    Key insight:
    - The gray pixels have no intrinsic labels; their destination is determined purely by line-of-sight geometry after red blocker neighborhoods eliminate decoy anchor options.
    """
    g = np.asarray(grid, dtype=int)
    anchors = _find_anchors(g)
    blockers = [tuple(map(int, pos)) for pos in np.argwhere(g == BLOCKER)]
    forbidden = _blocker_forbidden(g.shape, blockers)

    assignments: list[dict] = []
    for raw_r, raw_c in np.argwhere(g == GREY):
        cell = (int(raw_r), int(raw_c))
        candidates = _visible_candidates(cell, anchors, g, forbidden, ignore_grey=True)
        assert len(candidates) == 1
        chosen = dict(candidates[0])
        chosen["cell"] = cell
        assignments.append(chosen)

    return _render_output_from_assignments(g, assignments)


def _anchor_grid(shape: tuple[int, int], anchors: dict[int, dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for color, anchor in anchors.items():
        r0, c0, r1, c1 = anchor["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _random_anchor_layout() -> tuple[tuple[int, int], dict[int, dict]]:
    height = random.randint(20, 27)
    width = random.randint(20, 27)
    corners = [
        (random.randint(1, 3), random.randint(1, 3)),
        (random.randint(1, 3), random.randint(width - 6, width - 4)),
        (random.randint(height - 6, height - 4), random.randint(1, 3)),
        (random.randint(height - 6, height - 4), random.randint(width - 6, width - 4)),
    ]
    count = random.choice([2, 3, 3, 3])
    top_lefts = random.sample(corners, count)
    colors = random.sample(ANCHOR_COLORS, count)
    anchors: dict[int, dict] = {}
    for color, (r0, c0) in zip(colors, top_lefts):
        bbox = (r0, c0, r0 + 2, c0 + 2)
        anchors[color] = {"color": color, "bbox": bbox, "cells": _anchor_cells(bbox)}
    return (height, width), anchors


def _anchor_halo(shape: tuple[int, int], anchors: dict[int, dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for anchor in anchors.values():
        for r, c in anchor["cells"]:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if _in_bounds(shape, rr, cc):
                        mask[rr, cc] = True
    return mask


def _enumerate_candidate_cells(
    anchors: dict[int, dict],
    anchor_grid: np.ndarray,
    blockers: list[tuple[int, int]],
) -> list[dict]:
    blocked_grid = anchor_grid.copy()
    for r, c in blockers:
        blocked_grid[r, c] = BLOCKER

    forbidden = _blocker_forbidden(blocked_grid.shape, blockers)
    halo = _anchor_halo(blocked_grid.shape, anchors).copy()
    for r, c in blockers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(blocked_grid.shape, rr, cc):
                    halo[rr, cc] = True

    infos: list[dict] = []
    empty_for_no_block = anchor_grid
    no_block_forbidden = np.zeros(blocked_grid.shape, dtype=bool)
    for r in range(blocked_grid.shape[0]):
        for c in range(blocked_grid.shape[1]):
            if blocked_grid[r, c] != BG or forbidden[r, c] or halo[r, c]:
                continue
            cell = (r, c)
            blocked_candidates = _visible_candidates(cell, anchors, blocked_grid, forbidden, ignore_grey=True)
            if len(blocked_candidates) != 1:
                continue
            no_block_candidates = _visible_candidates(
                cell,
                anchors,
                empty_for_no_block,
                no_block_forbidden,
                ignore_grey=True,
            )
            if not no_block_candidates:
                continue
            target = blocked_candidates[0]
            nearest = min(item["dist"] for item in no_block_candidates)
            infos.append(
                {
                    "cell": cell,
                    "blocked": target,
                    "no_block": no_block_candidates,
                    "ambiguous": len(no_block_candidates) >= 2,
                    "misleading": len(no_block_candidates) >= 2 and target["dist"] > nearest,
                    "ray_id": (target["color"], target["terminal"], target["step"]),
                    "diag": abs(target["step"][0]) == 1 and abs(target["step"][1]) == 1,
                    "dist": target["dist"],
                }
            )
    return infos


def _enumerate_ambiguous_cells_for_blockers(
    anchors: dict[int, dict],
    anchor_grid: np.ndarray,
    blockers: list[tuple[int, int]],
) -> list[dict]:
    blocked_grid = anchor_grid.copy()
    for r, c in blockers:
        blocked_grid[r, c] = BLOCKER

    forbidden = _blocker_forbidden(blocked_grid.shape, blockers)
    halo = _anchor_halo(blocked_grid.shape, anchors).copy()
    for r, c in blockers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if _in_bounds(blocked_grid.shape, rr, cc):
                    halo[rr, cc] = True

    infos: list[dict] = []
    for r in range(blocked_grid.shape[0]):
        for c in range(blocked_grid.shape[1]):
            if blocked_grid[r, c] != BG or forbidden[r, c] or halo[r, c]:
                continue
            cell = (r, c)
            candidates = _visible_candidates(cell, anchors, blocked_grid, forbidden, ignore_grey=True)
            if len(candidates) >= 2:
                infos.append({"cell": cell, "candidates": candidates})
    return infos


def _candidate_blocker_cells(
    cell: tuple[int, int],
    target: dict,
    decoy: dict,
    anchors: dict[int, dict],
    grid: np.ndarray,
    blockers: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    forbidden = _blocker_forbidden(grid.shape, blockers)
    anchor_halo = _anchor_halo(grid.shape, anchors)
    protected = list(target["path"]) + [cell]
    options: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pr, pc in decoy["path"]:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = pr + dr, pc + dc
                pos = (rr, cc)
                if pos in seen or not _in_bounds(grid.shape, rr, cc):
                    continue
                seen.add(pos)
                if grid[rr, cc] != BG or forbidden[rr, cc] or anchor_halo[rr, cc]:
                    continue
                if _cheb(pos, cell) <= 1:
                    continue
                if any(_cheb(pos, protected_cell) <= 1 for protected_cell in protected):
                    continue
                temp_blockers = blockers + [pos]
                temp_forbidden = _blocker_forbidden(grid.shape, temp_blockers)
                candidates = _visible_candidates(cell, anchors, grid, temp_forbidden, ignore_grey=True)
                if len(candidates) == 1 and candidates[0]["color"] == target["color"]:
                    options.append(pos)
    return options


def _place_blockers(
    anchors: dict[int, dict],
    anchor_grid: np.ndarray,
) -> list[tuple[int, int]] | None:
    blockers: list[tuple[int, int]] = []
    desired = random.choice([1, 2, 2, 3])
    for _ in range(desired):
        placed = False
        ambiguous = _enumerate_ambiguous_cells_for_blockers(anchors, anchor_grid, blockers)
        random.shuffle(ambiguous)
        for info in ambiguous[:BLOCKER_ATTEMPTS]:
            if len(info["candidates"]) != 2:
                continue
            target = max(info["candidates"], key=lambda item: (item["dist"], item["color"]))
            decoy = min(info["candidates"], key=lambda item: (item["dist"], item["color"]))
            if target["color"] == decoy["color"]:
                continue
            options = _candidate_blocker_cells(info["cell"], target, decoy, anchors, anchor_grid, blockers)
            if not options:
                continue
            blockers.append(random.choice(options))
            placed = True
            break
        if not placed:
            break

    if not blockers:
        return None
    return blockers


def _pick_spaced_group(candidates: list[dict], count: int) -> list[dict] | None:
    chosen: list[dict] = []
    for info in sorted(candidates, key=lambda item: item["dist"]):
        if all(abs(info["dist"] - other["dist"]) >= 2 for other in chosen):
            chosen.append(info)
            if len(chosen) == count:
                return chosen
    return None


def _non_adjacent(cell: tuple[int, int], chosen: list[dict]) -> bool:
    return all(_cheb(cell, item["cell"]) > 1 for item in chosen)


def _select_dots(infos: list[dict]) -> list[dict] | None:
    if len(infos) < 4:
        return None
    misleading_pool = [info for info in infos if info["misleading"]]
    if not misleading_pool:
        return None

    groups: dict[tuple[int, tuple[int, int], tuple[int, int]], list[dict]] = defaultdict(list)
    for info in infos:
        groups[info["ray_id"]].append(info)

    multi_rays = [
        ray_id
        for ray_id, members in groups.items()
        if _pick_spaced_group(members, 2) is not None
    ]
    if not multi_rays:
        return None

    for _ in range(SELECT_ATTEMPTS):
        target_total = random.randint(4, min(7, len(infos)))
        require_diag = random.random() < 0.5 and any(info["diag"] for info in infos)

        selected: list[dict] = []
        ray_id = random.choice(multi_rays)
        members = groups[ray_id][:]
        random.shuffle(members)
        multi_count = 3 if random.random() < 0.35 else 2
        multi = _pick_spaced_group(members, multi_count)
        if multi is None:
            multi = _pick_spaced_group(groups[ray_id], 2)
        if multi is None:
            continue
        selected.extend(multi)

        special_pools = [
            misleading_pool,
            [info for info in infos if info["ambiguous"]],
        ]
        if require_diag:
            special_pools.append([info for info in infos if info["diag"]])
        ok = True
        for pool in special_pools:
            if not pool:
                continue
            choices = pool[:]
            random.shuffle(choices)
            picked = False
            for info in choices:
                if info in selected:
                    picked = True
                    break
                if _non_adjacent(info["cell"], selected):
                    selected.append(info)
                    picked = True
                    break
            if not picked:
                ok = False
                break
        if not ok:
            continue

        remaining = infos[:]
        random.shuffle(remaining)
        while len(selected) < target_total:
            added = False
            for info in remaining:
                if info in selected:
                    continue
                if _non_adjacent(info["cell"], selected):
                    selected.append(info)
                    added = True
                    break
            if not added:
                break

        if len(selected) < 4:
            continue
        if not any(info["misleading"] for info in selected):
            continue
        if not any(info["ambiguous"] for info in selected):
            continue
        if require_diag and not any(info["diag"] for info in selected):
            continue
        anchor_colors = {info["blocked"]["color"] for info in selected}
        if len(anchor_colors) < 2:
            continue
        ray_sizes = defaultdict(int)
        for info in selected:
            ray_sizes[info["ray_id"]] += 1
        if max(ray_sizes.values()) < 2:
            continue
        direction_kinds = {info["blocked"]["step"] for info in selected}
        if len(direction_kinds) < 2:
            continue
        return selected
    return None


def _validate_instance(grid: np.ndarray) -> bool:
    anchors = _find_anchors(grid)
    blockers = [tuple(map(int, pos)) for pos in np.argwhere(grid == BLOCKER)]
    if not blockers:
        return False
    forbidden = _blocker_forbidden(grid.shape, blockers)
    anchor_only = _anchor_grid(grid.shape, anchors)
    no_block_forbidden = np.zeros(grid.shape, dtype=bool)

    assignments: list[dict] = []
    ray_sizes: dict[tuple[int, tuple[int, int], tuple[int, int]], int] = defaultdict(int)
    misleading = 0
    ambiguous = 0
    for raw_r, raw_c in np.argwhere(grid == GREY):
        cell = (int(raw_r), int(raw_c))
        with_blockers = _visible_candidates(cell, anchors, grid, forbidden, ignore_grey=True)
        without_blockers = _visible_candidates(cell, anchors, anchor_only, no_block_forbidden, ignore_grey=True)
        if len(with_blockers) != 1 or not without_blockers:
            return False
        target = with_blockers[0]
        if len(without_blockers) >= 2:
            ambiguous += 1
            if target["dist"] > min(item["dist"] for item in without_blockers):
                misleading += 1
        item = dict(target)
        item["cell"] = cell
        assignments.append(item)
        ray_sizes[(target["color"], target["terminal"], target["step"])] += 1

    if len(assignments) < 4:
        return False
    if ambiguous == 0 or misleading == 0:
        return False
    if max(ray_sizes.values(), default=0) < 2:
        return False
    if len({item["color"] for item in assignments}) < 2:
        return False
    if len({item["step"] for item in assignments}) < 2:
        return False

    try:
        expected = _render_output_from_assignments(grid, assignments)
    except AssertionError:
        return False
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.any(actual == GREY):
        return False
    if np.array_equal(actual, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape, anchors = _random_anchor_layout()
    anchor_grid = _anchor_grid(shape, anchors)
    blockers = _place_blockers(anchors, anchor_grid)
    if blockers is None:
        return None

    infos = _enumerate_candidate_cells(anchors, anchor_grid, blockers)
    selected = _select_dots(infos)
    if selected is None:
        return None

    grid = anchor_grid.copy()
    for r, c in blockers:
        grid[r, c] = BLOCKER
    for info in selected:
        r, c = info["cell"]
        grid[r, c] = GREY

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid line-of-sight anchor puzzle")
