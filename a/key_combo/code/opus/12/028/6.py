from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PATH_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NOTCH_OFFSETS = ((0, 0), (0, 2), (2, 2), (2, 0))
RING_POSITIONS = tuple((r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1))
PORT_TOP_LEFT = {
    "N": lambda r, c: (r + 1, c - 1),
    "S": lambda r, c: (r - 3, c - 1),
    "W": lambda r, c: (r - 1, c + 1),
    "E": lambda r, c: (r - 1, c - 3),
}
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _make_sprite(border: int, center: int, notch_index: int) -> np.ndarray:
    patch = np.full((3, 3), int(border), dtype=int)
    patch[1, 1] = int(center)
    nr, nc = NOTCH_OFFSETS[int(notch_index) % 4]
    patch[nr, nc] = BG
    return patch


def _count_bends(path: list[tuple[int, int]]) -> int:
    bends = 0
    directions = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    for d1, d2 in zip(directions, directions[1:]):
        if d1 == d2:
            continue
        if d1[0] == -d2[0] and d1[1] == -d2[1]:
            bends += 2
        else:
            bends += 1
    return bends


def _trace_path(grid: np.ndarray) -> tuple[list[tuple[int, int]], int, tuple[tuple[int, int], tuple[int, int]]]:
    path_mask = grid == PATH_COLOR
    comps = _components4(path_mask)
    assert len(comps) == 1, "expected a single grey path component"
    cells = set(comps[0])
    degrees = {
        cell: sum((nr, nc) in cells for nr, nc in _neighbors4(*cell))
        for cell in cells
    }
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "expected exactly two path endpoints"
    assert all(degree in {1, 2} for degree in degrees.values()), "path must not branch"

    start, end = endpoints
    ordered = [start]
    prev = None
    cur = start
    while cur != end:
        nxt = [nb for nb in _neighbors4(*cur) if nb in cells and nb != prev]
        assert len(nxt) == 1, "path walk must be unique"
        prev, cur = cur, nxt[0]
        ordered.append(cur)

    return ordered, _count_bends(ordered), (start, end)


def _detect_sprites(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != PATH_COLOR)
    sprites: list[dict] = []
    for comp in _components4(mask):
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        center = int(patch[1, 1])
        if center in {BG, PATH_COLOR}:
            continue
        ring = [int(patch[r, c]) for r, c in RING_POSITIONS]
        nonzero_ring = [color for color in ring if color != BG]
        if len(nonzero_ring) not in {7, 8}:
            continue
        if len(set(nonzero_ring)) != 1:
            continue
        border = nonzero_ring[0]
        if border == center or border == PATH_COLOR:
            continue
        if sum(color == BG for color in ring) > 1:
            continue
        sprites.append(
            {
                "top_left": (r0, c0),
                "patch": patch,
                "border": border,
                "center": center,
            }
        )
    return sprites


def _border_cells(sprite: dict) -> list[tuple[int, int]]:
    r0, c0 = sprite["top_left"]
    patch = sprite["patch"]
    cells = []
    for rr, cc in RING_POSITIONS:
        if int(patch[rr, cc]) == sprite["border"]:
            cells.append((r0 + rr, c0 + cc))
    return cells


def _find_targets(sprites: list[dict], endpoints: tuple[tuple[int, int], tuple[int, int]]) -> list[dict]:
    targets: list[dict] = []
    for endpoint in endpoints:
        matched = []
        for sprite in sprites:
            if any(abs(br - endpoint[0]) + abs(bc - endpoint[1]) == 1 for br, bc in _border_cells(sprite)):
                matched.append(sprite)
        assert len(matched) == 1, "each endpoint must identify exactly one target sprite"
        targets.append(matched[0])
    assert targets[0]["top_left"] != targets[1]["top_left"], "targets must be distinct"
    return targets


def _apply_solution(grid: np.ndarray, turns: int | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    path, bends, endpoints = _trace_path(g)
    sprites = _detect_sprites(g)
    assert len(sprites) >= 4, "expected at least four sprites"
    targets = _find_targets(sprites, endpoints)
    target_ids = {id(sprite) for sprite in targets}
    lookup = {}
    for sprite in sprites:
        if id(sprite) in target_ids:
            continue
        assert sprite["border"] not in lookup, "source border colors must be unique"
        lookup[sprite["border"]] = sprite

    quarter_turns = bends % 4 if turns is None else int(turns) % 4
    out = g.copy()

    for sprite in sprites:
        if id(sprite) in target_ids:
            continue
        r0, c0 = sprite["top_left"]
        out[r0 : r0 + 3, c0 : c0 + 3] = BG

    for target in targets:
        assert target["center"] in lookup, "each target center must match a source border"
        source = lookup[target["center"]]
        rotated = np.rot90(source["patch"], k=-quarter_turns)
        r0, c0 = target["top_left"]
        out[r0 : r0 + 3, c0 : c0 + 3] = rotated

    assert len(path) >= 2
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace the endpoint-linked sprites with rotated matched sprites based on the grey path's bend count.

    Input structure:
    - A black background contains one thin grey 4-connected path with exactly two endpoints and no branches.
    - Several separated 3x3 bordered sprites appear away from one another; each sprite has one center color,
      one border color, and a single black missing corner that makes its orientation visible.
    - Exactly two sprites touch the two path endpoints by 4-neighbor border contact; these are the targets.

    Rule:
    - Identify the two target sprites from the two grey path endpoints.
    - For each target, use its center color as a lookup key to find the non-target sprite whose border color matches it.
    - Count how many times the grey path changes direction; rotate each matched source clockwise by that many quarter-turns.
    - Replace each target's full 3x3 area with the rotated source sprite and erase every non-target sprite that was not kept in place.

    Key insight:
    - The grey path carries two pieces of information at once: its endpoints choose which sprites are destinations,
      and its bend topology chooses the rotation that must be applied before copying the matched sources.
    """
    return _apply_solution(grid)


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    raise ValueError("segments must be axis-aligned")


def _polyline(points: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(len(points) - 1):
        segment = _line_cells(points[i], points[i + 1])
        if i:
            segment = segment[1:]
        for cell in segment:
            if cell in seen:
                return None
            seen.add(cell)
            cells.append(cell)
    return cells


def _generate_path(h: int, w: int, bends: int) -> list[tuple[int, int]] | None:
    margin = 4
    rows = list(range(margin, h - margin))
    cols = list(range(margin, w - margin))
    if len(rows) < 3 or len(cols) < 3:
        return None

    for _ in range(120):
        if bends == 0:
            if random.random() < 0.5:
                r = random.choice(rows)
                c1, c2 = sorted(random.sample(cols, 2))
                points = [(r, c1), (r, c2)]
            else:
                c = random.choice(cols)
                r1, r2 = sorted(random.sample(rows, 2))
                points = [(r1, c), (r2, c)]
        elif bends == 1:
            r1, r2 = random.sample(rows, 2)
            c1, c2 = random.sample(cols, 2)
            start = (r1, c1)
            end = (r2, c2)
            corner = random.choice([(r1, c2), (r2, c1)])
            points = [start, corner, end]
        elif bends == 2:
            if random.random() < 0.5:
                r1, r2 = random.sample(rows, 2)
                c1, c2 = random.sample(cols, 2)
                mids = [c for c in cols if c not in {c1, c2}]
                if not mids:
                    continue
                cm = random.choice(mids)
                points = [(r1, c1), (r1, cm), (r2, cm), (r2, c2)]
            else:
                r1, r2 = random.sample(rows, 2)
                c1, c2 = random.sample(cols, 2)
                mids = [r for r in rows if r not in {r1, r2}]
                if not mids:
                    continue
                rm = random.choice(mids)
                points = [(r1, c1), (rm, c1), (rm, c2), (r2, c2)]
        else:
            if random.random() < 0.5:
                r1, r2 = random.sample(rows, 2)
                c1, c2 = random.sample(cols, 2)
                outer_cols = [
                    c for c in cols
                    if c not in {c1, c2} and (c < min(c1, c2) or c > max(c1, c2))
                ]
                mids = [r for r in rows if r not in {r1, r2}]
                if not outer_cols or not mids:
                    continue
                ca = random.choice(outer_cols)
                rb = random.choice(mids)
                points = [(r1, c1), (r1, ca), (rb, ca), (rb, c2), (r2, c2)]
            else:
                r1, r2 = random.sample(rows, 2)
                c1, c2 = random.sample(cols, 2)
                outer_rows = [
                    r for r in rows
                    if r not in {r1, r2} and (r < min(r1, r2) or r > max(r1, r2))
                ]
                mids = [c for c in cols if c not in {c1, c2}]
                if not outer_rows or not mids:
                    continue
                ra = random.choice(outer_rows)
                cb = random.choice(mids)
                points = [(r1, c1), (ra, c1), (ra, cb), (r2, cb), (r2, c2)]

        path = _polyline(points)
        if path is None or len(path) < 5 or len(path) > max(h, w) + 10:
            continue
        cells = set(path)
        degrees = {
            cell: sum((nr, nc) in cells for nr, nc in _neighbors4(*cell))
            for cell in cells
        }
        if any(degree > 2 for degree in degrees.values()):
            continue
        if _count_bends(path) != bends:
            continue
        return path
    return None


def _adjacent_path_cells(cells: list[tuple[int, int]], path_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    adjacent = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in path_set:
                adjacent.add((nr, nc))
    return adjacent


def _rects_too_close(a: tuple[int, int], b: tuple[int, int], gap: int = 1) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 2, ac0 + 2
    br1, bc1 = br0 + 2, bc0 + 2
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_targets(
    path: list[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    path_set = set(path)
    targets: list[tuple[int, int]] = []
    for endpoint in (path[0], path[-1]):
        candidates = []
        for side, fn in PORT_TOP_LEFT.items():
            r0, c0 = fn(*endpoint)
            if not (0 <= r0 and 0 <= c0 and r0 + 2 < h and c0 + 2 < w):
                continue
            cells = [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]
            if any(cell in path_set for cell in cells):
                continue
            if _adjacent_path_cells(cells, path_set) != {endpoint}:
                continue
            if any(_rects_too_close((r0, c0), other, gap=1) for other in targets):
                continue
            candidates.append((r0, c0))
        if not candidates:
            return None
        targets.append(random.choice(candidates))
    return targets


def _place_sources(
    count: int,
    h: int,
    w: int,
    path: list[tuple[int, int]],
    target_boxes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    path_set = set(path)
    candidates = [(r, c) for r in range(h - 2) for c in range(w - 2)]
    random.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    occupied = list(target_boxes)
    for r0, c0 in candidates:
        cells = [(r0 + dr, c0 + dc) for dr in range(3) for dc in range(3)]
        if any(cell in path_set for cell in cells):
            continue
        if _adjacent_path_cells(cells, path_set):
            continue
        if any(_rects_too_close((r0, c0), other, gap=1) for other in occupied):
            continue
        placed.append((r0, c0))
        occupied.append((r0, c0))
        if len(placed) == count:
            return placed
    return None


def _render_example(
    h: int,
    w: int,
    bends: int,
    sprite_count: int,
) -> tuple[np.ndarray, dict] | None:
    path = _generate_path(h, w, bends)
    if path is None:
        return None
    target_boxes = _place_targets(path, h, w)
    if target_boxes is None:
        return None

    source_count = sprite_count - 2
    source_boxes = _place_sources(source_count, h, w, path, target_boxes)
    if source_boxes is None:
        return None

    border_colors = random.sample(PALETTE, sprite_count)
    source_borders = border_colors[:source_count]
    target_borders = border_colors[source_count:]
    remaining_centers = [color for color in PALETTE if color not in border_colors]
    random.shuffle(remaining_centers)
    source_centers = remaining_centers[:source_count]

    matched_indices = random.sample(range(source_count), 2)
    target_centers = [source_borders[idx] for idx in matched_indices]
    random.shuffle(target_centers)

    grid = np.zeros((h, w), dtype=int)
    for r, c in path:
        grid[r, c] = PATH_COLOR

    sprite_specs = []
    for i, (r0, c0) in enumerate(source_boxes):
        sprite_specs.append(
            {
                "kind": "source",
                "top_left": (r0, c0),
                "border": source_borders[i],
                "center": source_centers[i],
                "notch": random.randrange(4),
                "matched": i in matched_indices,
            }
        )
    for i, (r0, c0) in enumerate(target_boxes):
        sprite_specs.append(
            {
                "kind": "target",
                "top_left": (r0, c0),
                "border": target_borders[i],
                "center": target_centers[i],
                "notch": random.randrange(4),
                "matched": True,
            }
        )

    for spec in sprite_specs:
        patch = _make_sprite(spec["border"], spec["center"], spec["notch"])
        r0, c0 = spec["top_left"]
        grid[r0 : r0 + 3, c0 : c0 + 3] = np.maximum(grid[r0 : r0 + 3, c0 : c0 + 3], patch)

    return grid, {"bends": bends, "sprite_count": sprite_count, "specs": sprite_specs}


def _validate_example(grid: np.ndarray, meta: dict) -> bool:
    sprites = _detect_sprites(grid)
    if len(sprites) != meta["sprite_count"]:
        return False

    path, bends, endpoints = _trace_path(grid)
    if bends != meta["bends"] or len(path) < 5:
        return False

    targets = _find_targets(sprites, endpoints)
    target_centers = {sprite["center"] for sprite in targets}
    target_positions = {sprite["top_left"] for sprite in targets}
    source_borders = {
        sprite["border"] for sprite in sprites if sprite["top_left"] not in target_positions
    }
    if not target_centers.issubset(source_borders):
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False

    non_targets = [sprite for sprite in sprites if sprite["top_left"] not in target_positions]
    unmatched = [sprite for sprite in non_targets if sprite["border"] not in target_centers]
    for sprite in unmatched:
        r0, c0 = sprite["top_left"]
        if np.any(out[r0 : r0 + 3, c0 : c0 + 3] != BG):
            return False

    if bends % 4 != 0:
        if np.array_equal(out, _apply_solution(grid, turns=0)):
            return False

    for sprite in targets:
        r0, c0 = sprite["top_left"]
        patch = out[r0 : r0 + 3, c0 : c0 + 3]
        if np.count_nonzero(patch) < 7:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = random.randint(16, 24)
        w = random.randint(16, 24)
        bends = random.choice((0, 1, 2, 3, 3))
        sprite_count = 5 if random.random() < 0.75 else 4
        rendered = _render_example(h, w, bends, sprite_count)
        if rendered is None:
            continue
        grid, meta = rendered
        if _validate_example(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
