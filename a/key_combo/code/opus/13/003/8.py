from __future__ import annotations

import random
from collections import deque
from functools import lru_cache

import numpy as np


BG = 0
TL_COLOR = 1
TR_COLOR = 2
BL_COLOR = 3
BR_COLOR = 4
RECT_COLOR = 5

RAY_SPECS = (
    ("TL", TL_COLOR, (0, 0), (1, 1)),
    ("TR", TR_COLOR, None, (1, -1)),
    ("BL", BL_COLOR, None, (-1, 1)),
    ("BR", BR_COLOR, None, (-1, -1)),
)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != RECT_COLOR:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == RECT_COLOR:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in pts]
            cols = [cc for _, cc in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if (r1 - r0 + 1) * (c1 - c0 + 1) != len(pts):
                raise AssertionError("grey components must be solid rectangles")
            rects.append((r0, c0, r1, c1))
    rects.sort()
    return rects


def _rect_id_grid(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    rid = np.full(shape, -1, dtype=int)
    for idx, (r0, c0, r1, c1) in enumerate(rects):
        rid[r0 : r1 + 1, c0 : c1 + 1] = idx
    return rid


def _touches_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = rect
    return (
        r0 <= r <= r1 and (c == c0 - 1 or c == c1 + 1)
    ) or (
        c0 <= c <= c1 and (r == r0 - 1 or r == r1 + 1)
    )


def _trace_ray(
    rect_ids: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict:
    h, w = rect_ids.shape
    dr, dc = direction
    r, c = start
    path: list[tuple[int, int]] = []
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            return {
                "endpoint": (r, c),
                "path": path,
                "path_len": len(path),
                "hit_rect": None,
            }
        hit = int(rect_ids[nr, nc])
        if hit >= 0:
            return {
                "endpoint": (r, c),
                "path": path,
                "path_len": len(path),
                "hit_rect": hit,
            }
        r, c = nr, nc
        path.append((r, c))


def _scan_ray_hits(
    rect_ids: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[int]:
    h, w = rect_ids.shape
    dr, dc = direction
    r, c = start
    hits: list[int] = []
    current = -1
    while True:
        nr, nc = r + dr, c + dc
        if not _in_bounds((h, w), nr, nc):
            return hits
        current = int(rect_ids[nr, nc])
        if current >= 0 and (not hits or hits[-1] != current):
            hits.append(current)
        r, c = nr, nc


def _ray_specs_for_shape(shape: tuple[int, int]):
    h, w = shape
    return (
        ("TL", TL_COLOR, (0, 0), (1, 1)),
        ("TR", TR_COLOR, (0, w - 1), (1, -1)),
        ("BL", BL_COLOR, (h - 1, 0), (-1, 1)),
        ("BR", BR_COLOR, (h - 1, w - 1), (-1, -1)),
    )


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    rects = _rectangles(g)
    rect_ids = _rect_id_grid(g.shape, rects)

    rays = []
    cell_owner: dict[tuple[int, int], int] = {}
    conflict = False
    full_hits: dict[str, list[int]] = {}
    for ray_idx, (name, color, start, direction) in enumerate(_ray_specs_for_shape(g.shape)):
        info = _trace_ray(rect_ids, start, direction)
        info["name"] = name
        info["color"] = color
        info["start"] = start
        info["direction"] = direction
        info["index"] = ray_idx
        full_hits[name] = _scan_ray_hits(rect_ids, start, direction)
        for cell in info["path"]:
            prev = cell_owner.get(cell)
            if prev is not None and prev != ray_idx:
                conflict = True
            cell_owner[cell] = ray_idx
        rays.append(info)

    contacts: list[list[dict]] = [[] for _ in rects]
    for info in rays:
        end = info["endpoint"]
        for rect_id, rect in enumerate(rects):
            if _touches_rect(end, rect):
                contacts[rect_id].append(info)

    winners: list[int | None] = []
    for items in contacts:
        if not items:
            winners.append(None)
            continue
        best = min(items, key=lambda item: (item["path_len"], item["start"][0], item["start"][1]))
        winners.append(int(best["color"]))

    return {
        "rects": rects,
        "rect_ids": rect_ids,
        "rays": rays,
        "contacts": contacts,
        "winners": winners,
        "full_hits": full_hits,
        "conflict": conflict,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire four inward diagonal rays from the corners and recolor each grey rectangle by the first ray that truly reaches its side.

    Input structure:
    - Black background with four fixed corner pixels: blue at top-left, red at top-right, green at bottom-left, yellow at bottom-right.
    - The interior contains several solid grey axis-aligned rectangles.
    - Rectangles do not overlap the corner markers and are separated from one another.

    Rule:
    - Each corner emits a 45-degree diagonal ray toward the grid interior.
    - A ray paints the black cells on its path and stops just before the next diagonal step would enter a grey rectangle; if no rectangle is encountered, it continues to the boundary.
    - After all four rays are traced, a rectangle checks which ray endpoints lie in its 4-neighborhood; if exactly one ray touches it, the whole rectangle takes that ray's color, and if several touch it, the shortest ray wins with row-then-column tie-breaking on the emitting corner.

    Key insight:
    - Rectangle colors are determined by explicit ray simulation, not by static nearest-corner distance, because earlier rectangles can geometrically shield later ones along the same diagonal.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    analysis = _analyze(g)

    for info in analysis["rays"]:
        for r, c in info["path"]:
            if int(out[r, c]) == BG:
                out[r, c] = int(info["color"])

    for rect_id, rect in enumerate(analysis["rects"]):
        winner = analysis["winners"][rect_id]
        if winner is None:
            continue
        r0, c0, r1, c1 = rect
        out[r0 : r1 + 1, c0 : c1 + 1] = int(winner)

    h, w = g.shape
    out[0, 0] = TL_COLOR
    out[0, w - 1] = TR_COLOR
    out[h - 1, 0] = BL_COLOR
    out[h - 1, w - 1] = BR_COLOR
    return out


def _bbox_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _build_grid(n: int, rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    g = np.full((n, n), BG, dtype=int)
    g[0, 0] = TL_COLOR
    g[0, n - 1] = TR_COLOR
    g[n - 1, 0] = BL_COLOR
    g[n - 1, n - 1] = BR_COLOR
    for r0, c0, r1, c1 in rects:
        g[r0 : r1 + 1, c0 : c1 + 1] = RECT_COLOR
    return g


def _ray_index(name: str) -> int:
    return {"TL": 0, "TR": 1, "BL": 2, "BR": 3}[name]


@lru_cache(maxsize=None)
def _candidate_rects(n: int) -> tuple[dict, ...]:
    grid_shape = (n, n)
    out: list[dict] = []
    center = (n - 1) / 2.0
    for h in (2, 3):
        for w in (2, 3, 4):
            for r0 in range(1, n - h):
                for c0 in range(1, n - w):
                    rect = (r0, c0, r0 + h - 1, c0 + w - 1)
                    g = _build_grid(n, [rect])
                    analysis = _analyze(g)
                    hits = analysis["full_hits"]
                    hit_set = {name for name, ids in hits.items() if ids}
                    touches = {
                        info["name"]
                        for info in analysis["rays"]
                        if any(_touches_rect(info["endpoint"], rect) for _ in [0])
                    }
                    kind = "off"
                    if hit_set == {"TL", "BR"}:
                        kind = "main"
                    elif hit_set == {"TR", "BL"}:
                        kind = "anti"
                    elif hit_set == {"TL", "TR", "BL", "BR"}:
                        kind = "both"
                    out.append(
                        {
                            "bbox": rect,
                            "kind": kind,
                            "hit_steps": {info["name"]: info["path_len"] for info in analysis["rays"] if info["hit_rect"] == 0},
                            "touches": frozenset(touches),
                            "center_dist": abs((r0 + r0 + h - 1) / 2.0 - center) + abs((c0 + c0 + w - 1) / 2.0 - center),
                        }
                    )
    return tuple(out)


def _choose_non_touching(pool: list[dict], chosen: list[dict]) -> dict | None:
    random.shuffle(pool)
    for cand in pool:
        if all(_bbox_clear(cand["bbox"], other["bbox"]) for other in chosen):
            return cand
    return None


def _final_metrics(grid: np.ndarray) -> dict:
    analysis = _analyze(grid)
    full_hits = analysis["full_hits"]
    blocking = any(len(ids) >= 2 for ids in full_hits.values())
    recolored = sum(winner is not None for winner in analysis["winners"])
    multi_touch = any(len(items) >= 2 for items in analysis["contacts"])
    tie_touch = False
    hidden_by_block = False
    for rect_id, items in enumerate(analysis["contacts"]):
        if items:
            best_len = min(item["path_len"] for item in items)
            if sum(item["path_len"] == best_len for item in items) >= 2:
                tie_touch = True
        else:
            for ids in full_hits.values():
                if rect_id in ids:
                    hidden_by_block = True
                    break
    main_covered = bool(full_hits["TL"])
    anti_covered = bool(full_hits["TR"])
    return {
        "analysis": analysis,
        "blocking": blocking,
        "recolored": recolored,
        "multi_touch": multi_touch,
        "tie_touch": tie_touch,
        "hidden_by_block": hidden_by_block,
        "main_covered": main_covered,
        "anti_covered": anti_covered,
    }


def _attempt_generate() -> np.ndarray | None:
    n = random.randint(13, 18)
    want_tie = random.random() < 0.45
    want_hidden = random.random() < 0.6
    competition_axis = random.choice(["main", "anti"])

    candidates = list(_candidate_rects(n))
    both = [c for c in candidates if c["kind"] == "both"]
    main_only = [c for c in candidates if c["kind"] == "main"]
    anti_only = [c for c in candidates if c["kind"] == "anti"]
    off_only = [c for c in candidates if c["kind"] == "off"]

    if competition_axis == "anti":
        comp_pool = [
            c
            for c in both
            if {"TR", "BL"} <= set(c["touches"])
            and (not want_tie or c["hit_steps"].get("TR") == c["hit_steps"].get("BL"))
        ]
        side = random.choice(["TL", "BR"])
        blocker_pool = [c for c in main_only if side in c["touches"]]
        axis_pool = main_only
    else:
        comp_pool = [
            c
            for c in both
            if {"TL", "BR"} <= set(c["touches"])
            and (not want_tie or c["hit_steps"].get("TL") == c["hit_steps"].get("BR"))
        ]
        side = random.choice(["TR", "BL"])
        blocker_pool = [c for c in anti_only if side in c["touches"]]
        axis_pool = anti_only

    if not comp_pool or not blocker_pool or len(off_only) < 1:
        return None

    comp_pool = sorted(comp_pool, key=lambda c: (c["center_dist"], random.random()))
    for comp in comp_pool[:20]:
        chosen = [comp]
        comp_step = comp["hit_steps"][side if side in comp["hit_steps"] else ("TL" if competition_axis == "main" else "TR")]
        front_candidates = [
            c
            for c in blocker_pool
            if c["hit_steps"][side] + 1 < comp_step and _bbox_clear(c["bbox"], comp["bbox"])
        ]
        if not front_candidates:
            continue
        random.shuffle(front_candidates)
        for front in front_candidates[:25]:
            chosen = [comp, front]
            if want_hidden:
                middle_candidates = [
                    c
                    for c in axis_pool
                    if c is not front
                    and front["hit_steps"][side] + 1 < c["hit_steps"][side] < comp_step
                    and all(_bbox_clear(c["bbox"], other["bbox"]) for other in chosen)
                ]
                if not middle_candidates:
                    continue
                middle = random.choice(middle_candidates)
                chosen.append(middle)

            target_count = random.randint(4, 6)
            extras_needed = target_count - len(chosen)
            extras = []
            if extras_needed > 0:
                pool = off_only[:]
                random.shuffle(pool)
                for cand in pool:
                    if len(extras) == extras_needed:
                        break
                    if all(_bbox_clear(cand["bbox"], other["bbox"]) for other in chosen + extras):
                        extras.append(cand)
            if len(extras) != extras_needed:
                continue

            rects = [cand["bbox"] for cand in chosen + extras]
            g = _build_grid(n, rects)
            metrics = _final_metrics(g)
            if metrics["analysis"]["conflict"]:
                continue
            if not metrics["main_covered"] or not metrics["anti_covered"]:
                continue
            if not metrics["blocking"] or not metrics["multi_touch"]:
                continue
            if want_tie and not metrics["tie_touch"]:
                continue
            if want_hidden and not metrics["hidden_by_block"]:
                continue
            if metrics["recolored"] < 2:
                continue
            if np.array_equal(main(g), g):
                continue
            return g
    return None


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray rectangle input")
