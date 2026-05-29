# Combo variant of 800d221b (index 22)
# combo_spec: D17/1+D18/2
# Name: 无效种子+共享路径

"""
Design analysis for the combo variant
=====================================

a) Concretization and behavior trace
------------------------------------
Base task behavior:
  - every seed framed by path-color cells is connected to the hub
  - each non-hub path branch belongs to one seed
  - recolor each branch and framed border to that seed's interior majority color

Combo behavior:
  - add 1-2 framed seed regions whose path-color border is disconnected from the hub;
    those seeds are invalid and stay unchanged in the output
  - let two valid top seeds share one trunk from the hub before splitting left/right
  - shared cells take the majority color of the larger of those two seeds

Concrete trace on one sample:
  - top-left seed area = 18, majority = A
  - top-right seed area = 12, majority = B
  - the upward trunk is adjacent to both top seeds only after the split
  - original logic would assume every seed matters and every branch has one owner
  - combo logic must:
      1. ignore disconnected seeds when recoloring
      2. propagate reachable valid-seed identities back toward the hub
      3. color the shared trunk with A because 18 > 12
  - one invalid bottom seed keeps both its interior and its P-colored border unchanged

b) Complexity calibration
-------------------------
The generator keeps one clean shared conflict only: the two top seeds share a single
vertical trunk. This makes the new rule visually obvious without turning the path
network into a dense graph. Invalid seeds are limited to two framed rectangles, so
the solver must notice them, but the scene stays readable.

c) Hidden-assumption review against the original main()
-------------------------------------------------------
Original assumptions broken by the combo:
  - "every seed bordered by P participates in recoloring":
      false now because disconnected seeds must remain unchanged
  - "every non-hub P component is a simple single-owner branch":
      false now because the hub-connected network contains one fork whose upstream
      cells are jointly attributable to two seeds
  - "degree>2 away from seeds implies an invalid T-junction":
      false now because the shared split is intentional

New solver strategy:
  - still infer bg, P, and the solid square hub from the image
  - separate the hub-connected P component from disconnected P components
  - treat the hub-connected network as a tree rooted at the hub
  - for each P cell, collect which valid seeds are reachable farther from the hub
  - choose the largest-area reachable seed as the owner of contested shared cells

d) Placement legality
---------------------
The generator uses a fixed 27x27 template with disjoint seed rectangles and simple
orthogonal polylines. Borders and paths only overlap at intended attachment cells.
Disconnected seeds are placed far from the hub-connected component, so they never
accidentally touch it. All objects remain in-bounds and non-overlapping.

e) Visual observability
-----------------------
Humans can infer both fused behaviors from 2-4 examples:
  - some framed seeds recolor their border/path, while another framed seed does not
  - the two top seeds have different sizes and different majority colors
  - the shared trunk adopts the larger top seed's color, not a generic "nearest seed"
  - the hub center follows the recolored branches, so the shared-winner effect is
    visible again at the center pixel

f) Visual cleanliness
---------------------
The shared structure is a single top trunk with a left/right split, and disconnected
seeds sit on unused edges. No extra symbols, legends, or overlapping decorations are
added, so the image still reads as a hub with several framed seed regions.

g) Change inferability
----------------------
The color change is structural, not numeric. The deciding cue for the shared trunk is
visible seed area, and the deciding cue for invalid seeds is visible lack of path
connection. The generator enforces opposite majority colors on the two shared seeds,
so the contested recolor is always observable.

h) Anti-degeneration metric
---------------------------
For each generated sample we can measure:
  - invalid_seed_count: framed seeds whose border P component is not hub-connected
  - preserved_invalid_border_cells: disconnected border cells that stay color P
  - shared_cells_recolored_to_larger: shared trunk cells colored with the larger
    top seed's majority color
  - shared_cells_different_from_smaller: contested cells whose output differs from
    the smaller top seed's majority color
The generator is only accepted if invalid_seed_count >= 1 and the shared top majors
are different, so both combo dimensions are active.

i) New solver assumptions that the generator guarantees
-------------------------------------------------------
  - the hub-connected P component is loop-free even though it contains one fork
  - exactly two valid top seeds participate in the shared trunk conflict
  - those two top seeds have different areas and different majority colors
  - disconnected seeds still use the same framed-seed visual language as valid seeds
  - at least one non-top branch shares the winning top color so the hub-center vote
    is not tie-dependent
"""

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, label_mask


GRID_SHAPE = (27, 27)
HUB_BOX = (12, 12, 14, 14)  # r0, c0, r1, c1
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _rect(r0: int, r1: int, c0: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _poly(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for idx, ((r0, c0), (r1, c1)) in enumerate(zip(points, points[1:])):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            seg = [(r0, c) for c in range(c0, c1 + step, step)]
        elif c0 == c1:
            step = 1 if r1 >= r0 else -1
            seg = [(r, c0) for r in range(r0, r1 + step, step)]
        else:
            raise ValueError("non-orthogonal path segment")
        if idx:
            seg = seg[1:]
        out.extend(seg)
    return set(out)


def _counter_winner(counter: Counter[int]) -> int:
    return int(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])


def _comp_deg_score(mask: np.ndarray) -> float:
    h, w = mask.shape
    comps = components_mask(mask, connectivity=4)
    if not comps:
        return -1e9
    best_pts = max(comps, key=lambda coords: int(coords.shape[0]))
    best_sz = int(best_pts.shape[0])
    m = np.zeros(mask.shape, dtype=bool)
    m[best_pts[:, 0], best_pts[:, 1]] = True
    deg2 = 0
    deg4 = 0
    for r, c in best_pts:
        d = 0
        r = int(r)
        c = int(c)
        if r > 0 and bool(m[r - 1, c]):
            d += 1
        if r + 1 < h and bool(m[r + 1, c]):
            d += 1
        if c > 0 and bool(m[r, c - 1]):
            d += 1
        if c + 1 < w and bool(m[r, c + 1]):
            d += 1
        if d == 2:
            deg2 += 1
        elif d == 4:
            deg4 += 1
    frac2 = deg2 / max(1, best_sz)
    frac4 = deg4 / max(1, best_sz)
    return float((frac2 - frac4) * best_sz)


def _infer_path_color(g: np.ndarray, bg: int) -> int:
    colors = sorted({int(x) for x in np.unique(g)} - {int(bg)})
    assert colors
    return int(max(colors, key=lambda c: _comp_deg_score(g == int(c))))


def _find_hub_mask(g: np.ndarray, pcol: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = g.shape
    m = g == int(pcol)
    ps = m.astype(int).cumsum(axis=0).cumsum(axis=1)

    def rect_sum(r0: int, c0: int, r1: int, c1: int) -> int:
        s = int(ps[r1, c1])
        if r0 > 0:
            s -= int(ps[r0 - 1, c1])
        if c0 > 0:
            s -= int(ps[r1, c0 - 1])
        if r0 > 0 and c0 > 0:
            s += int(ps[r0 - 1, c0 - 1])
        return int(s)

    best = None
    for k in range(2, min(h, w) + 1):
        found = None
        for r0 in range(0, h - k + 1):
            r1 = r0 + k - 1
            for c0 in range(0, w - k + 1):
                c1 = c0 + k - 1
                if rect_sum(r0, c0, r1, c1) == k * k:
                    found = (k, r0, c0)
        if found is not None:
            best = found
    assert best is not None
    k, r0, c0 = map(int, best)
    r1, c1 = int(r0 + k - 1), int(c0 + k - 1)
    hub = np.zeros((h, w), dtype=bool)
    hub[r0 : r1 + 1, c0 : c1 + 1] = True
    return hub, (r0, c0, r1, c1)


def _extract_seed_info(g: np.ndarray, bg: int, pcol: int) -> tuple[np.ndarray, dict[int, dict]]:
    seed_mask = (g != int(bg)) & (g != int(pcol))
    seed_lab, seed_n = label_mask(seed_mask, connectivity=4)
    info: dict[int, dict] = {}
    for sid in range(1, int(seed_n) + 1):
        pts = np.argwhere(seed_lab == int(sid))
        vals = [int(g[int(r), int(c)]) for r, c in pts]
        counts = Counter(vals)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        info[int(sid)] = {
            "area": int(len(pts)),
            "major": int(_counter_winner(counts)),
            "bbox": (int(r0), int(c0), int(r1), int(c1)),
            "pts": pts,
        }
    return seed_lab, info


def _hub_component_label(p_lab: np.ndarray, hub: np.ndarray) -> int:
    labels = {int(p_lab[r, c]) for r, c in np.argwhere(hub) if int(p_lab[r, c]) != 0}
    assert len(labels) == 1
    return int(next(iter(labels)))


def _classify_seeds(
    seed_lab: np.ndarray,
    seed_info: dict[int, dict],
    p_lab: np.ndarray,
    hub_label: int,
) -> tuple[set[int], set[int]]:
    h, w = seed_lab.shape
    valid: set[int] = set()
    invalid: set[int] = set()
    for sid, data in seed_info.items():
        touches_hub_comp = False
        for r, c in data["pts"]:
            r = int(r)
            c = int(c)
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and int(p_lab[rr, cc]) == int(hub_label):
                    touches_hub_comp = True
                    break
            if touches_hub_comp:
                break
        if touches_hub_comp:
            valid.add(int(sid))
        else:
            invalid.add(int(sid))
    return valid, invalid


def _hub_distances(pcomp_mask: np.ndarray, hub: np.ndarray) -> np.ndarray:
    h, w = pcomp_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in np.argwhere(hub):
        r = int(r)
        c = int(c)
        dist[r, c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if not bool(pcomp_mask[rr, cc]):
                continue
            if dist[rr, cc] != -1:
                continue
            dist[rr, cc] = int(dist[r, c] + 1)
            q.append((rr, cc))
    return dist


def _reachable_valid_seeds(
    network_mask: np.ndarray,
    seed_lab: np.ndarray,
    valid_seeds: set[int],
    dist: np.ndarray,
) -> dict[tuple[int, int], set[int]]:
    h, w = network_mask.shape
    adj_map: dict[tuple[int, int], set[int]] = {}
    for r, c in np.argwhere(network_mask):
        r = int(r)
        c = int(c)
        here: set[int] = set()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                sid = int(seed_lab[rr, cc])
                if sid in valid_seeds:
                    here.add(int(sid))
        adj_map[(r, c)] = here

    order = sorted(
        [(int(r), int(c)) for r, c in np.argwhere(network_mask)],
        key=lambda rc: int(dist[rc[0], rc[1]]),
        reverse=True,
    )
    reachable: dict[tuple[int, int], set[int]] = {}
    for r, c in order:
        seen = set(adj_map[(r, c)])
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(network_mask[rr, cc]):
                if int(dist[rr, cc]) > int(dist[r, c]):
                    seen |= reachable[(rr, cc)]
        reachable[(r, c)] = seen
    return reachable


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    pcol = int(_infer_path_color(g, bg))
    hub, hub_box = _find_hub_mask(g, pcol)
    p_mask = g == int(pcol)
    p_lab, _ = label_mask(p_mask, connectivity=4)
    hub_label = int(_hub_component_label(p_lab, hub))
    seed_lab, seed_info = _extract_seed_info(g, bg, pcol)
    valid_seeds, invalid_seeds = _classify_seeds(seed_lab, seed_info, p_lab, hub_label)
    pcomp_mask = p_lab == int(hub_label)
    network_mask = pcomp_mask & (~hub)
    dist = _hub_distances(pcomp_mask, hub)
    reachable = _reachable_valid_seeds(network_mask, seed_lab, valid_seeds, dist)
    return {
        "grid": g,
        "bg": bg,
        "pcol": pcol,
        "hub": hub,
        "hub_box": hub_box,
        "seed_lab": seed_lab,
        "seed_info": seed_info,
        "valid_seeds": valid_seeds,
        "invalid_seeds": invalid_seeds,
        "pcomp_mask": pcomp_mask,
        "network_mask": network_mask,
        "dist": dist,
        "reachable": reachable,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a hub-centered path tree with disconnected seeds and one contested trunk.

    Input structure:
    - The most common color is the background.
    - One non-background color forms a solid square hub plus a thin orthogonal network.
    - Colored seed interiors (non-bg, non-path) sit inside framed path-colored borders.
      Some framed seeds are connected to the hub, while others are completely detached.
      Two connected top seeds share an initial path segment before splitting.

    Rule:
    - Infer the path color and locate the solid square hub.
    - Split framed seeds into valid ones whose border lies in the hub-connected path
      component and invalid detached ones whose border does not.
    - Recolor only the hub-connected path cells: each cell inherits the majority color
      of the valid seed(s) reachable farther from the hub; if multiple valid seeds share
      that cell, use the majority color of the largest reachable seed region.
    - Leave detached seeds unchanged, and recolor the hub center pixel to the most
      common color among recolored path cells touching the hub.

    Key insight:
    Treat the path network as a tree rooted at the hub. A path cell is controlled by
    the valid seeds in its downstream subtree, while disconnected seed frames never
    enter that subtree at all.
    """
    parsed = _parse_instance(np.array(grid, dtype=int))
    g = parsed["grid"]
    hub = parsed["hub"]
    seed_info = parsed["seed_info"]
    network_mask = parsed["network_mask"]
    reachable = parsed["reachable"]

    out = g.copy()
    for r, c in np.argwhere(network_mask):
        r = int(r)
        c = int(c)
        owners = reachable[(r, c)]
        if not owners:
            continue
        sid = max(sorted(owners), key=lambda s: (int(seed_info[s]["area"]), -int(s)))
        out[r, c] = int(seed_info[sid]["major"])

    touch_cols: Counter[int] = Counter()
    h, w = g.shape
    for r, c in np.argwhere(hub):
        r = int(r)
        c = int(c)
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bool(network_mask[rr, cc]):
                touch_cols[int(out[rr, cc])] += 1
    assert touch_cols
    hr0, hc0, hr1, hc1 = parsed["hub_box"]
    center = (int((hr0 + hr1) // 2), int((hc0 + hc1) // 2))
    out[center] = int(_counter_winner(touch_cols))
    return out


def _seed_border_cells(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    if r0 > 0:
        border |= {(r0 - 1, c) for c in range(c0, c1 + 1)}
    if r1 < h - 1:
        border |= {(r1 + 1, c) for c in range(c0, c1 + 1)}
    if c0 > 0:
        border |= {(r, c0 - 1) for r in range(r0, r1 + 1)}
    if c1 < w - 1:
        border |= {(r, c1 + 1) for r in range(r0, r1 + 1)}
    if r0 > 0 and c0 > 0:
        border.add((r0 - 1, c0 - 1))
    if r0 > 0 and c1 < w - 1:
        border.add((r0 - 1, c1 + 1))
    if r1 < h - 1 and c0 > 0:
        border.add((r1 + 1, c0 - 1))
    if r1 < h - 1 and c1 < w - 1:
        border.add((r1 + 1, c1 + 1))
    return border


def _paint_seed_interior(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    major: int,
    minor: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    n = len(cells)
    low = max(1, int(round(n * 0.18)))
    high = min(max(low, int(round(n * 0.35))), max(1, n // 2 - 1))
    n_minor = rng.randint(low, high)
    rng.shuffle(cells)
    for r, c in cells[:n_minor]:
        grid[r, c] = int(minor)
    for r, c in cells[n_minor:]:
        grid[r, c] = int(major)


def _build_layout(rng: random.Random) -> dict:
    h, w = GRID_SHAPE
    top_winner = rng.choice(["NL", "NR"])
    if top_winner == "NL":
        seeds = {
            "NL": {"bbox": (0, 1, 2, 6), "major_side": "winner"},
            "NR": {"bbox": (0, 21, 2, 24), "major_side": "loser"},
        }
    else:
        seeds = {
            "NL": {"bbox": (0, 2, 2, 5), "major_side": "loser"},
            "NR": {"bbox": (0, 20, 2, 25), "major_side": "winner"},
        }
    seeds["S"] = {"bbox": (24, 11, 26, 15), "major_side": "winner"}
    seeds["W"] = {"bbox": (18, 0, 22, 2), "major_side": "loser"}
    seeds["X1"] = {"bbox": (9, 24, 11, 26), "major_side": rng.choice(["winner", "loser"])}
    seeds["X2"] = {"bbox": (24, 19, 26, 22), "major_side": rng.choice(["winner", "loser"])}

    hub = _rect(HUB_BOX[0], HUB_BOX[2], HUB_BOX[1], HUB_BOX[3])
    shared_cells = _poly([(11, 13), (6, 13)])
    valid_paths = {
        "NL": shared_cells | _poly([(6, 13), (6, 4), (3, 4)]),
        "NR": shared_cells | _poly([(6, 13), (6, 22), (3, 22)]),
        "S": _poly([(15, 13), (23, 13)]),
        "W": _poly([(13, 11), (13, 4), (20, 4), (20, 3)]),
    }

    for slot, data in seeds.items():
        bbox = data["bbox"]
        data["interior"] = _rect(bbox[0], bbox[2], bbox[1], bbox[3])
        data["border"] = _seed_border_cells(bbox[0], bbox[1], bbox[2], bbox[3], h, w)

    occupied: set[tuple[int, int]] = set(hub)
    for data in seeds.values():
        assert not (occupied & data["interior"])
        occupied |= data["interior"]
        assert not (occupied & data["border"])
        occupied |= data["border"]

    path_cells = set()
    for cells in valid_paths.values():
        path_cells |= cells
    assert not (path_cells & {pt for name, data in seeds.items() if name in {"X1", "X2"} for pt in data["border"]})
    assert not (path_cells & {pt for data in seeds.values() for pt in data["interior"]})

    return {
        "hub": hub,
        "shared_cells": shared_cells,
        "valid_paths": valid_paths,
        "seeds": seeds,
        "top_winner": top_winner,
    }


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    parsed = _parse_instance(grid)
    g = parsed["grid"]
    out = main(g)
    seed_info = parsed["seed_info"]
    valid_top = [
        sid
        for sid in parsed["valid_seeds"]
        if int(seed_info[sid]["bbox"][0]) == 0
    ]
    assert len(valid_top) == 2
    larger = max(valid_top, key=lambda sid: int(seed_info[sid]["area"]))
    smaller = min(valid_top, key=lambda sid: int(seed_info[sid]["area"]))
    shared_cells = list(_poly([(11, 13), (6, 13)]))

    invalid_border_cells = 0
    preserved_invalid_border_cells = 0
    for sid in parsed["invalid_seeds"]:
        pts = seed_info[sid]["pts"]
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                    if int(g[rr, cc]) == int(parsed["pcol"]):
                        invalid_border_cells += 1
                        if int(out[rr, cc]) == int(parsed["pcol"]):
                            preserved_invalid_border_cells += 1

    shared_recolored_to_larger = sum(
        int(out[r, c]) == int(seed_info[larger]["major"]) for r, c in shared_cells
    )
    shared_different_from_smaller = sum(
        int(out[r, c]) != int(seed_info[smaller]["major"]) for r, c in shared_cells
    )
    return {
        "invalid_seed_count": int(len(parsed["invalid_seeds"])),
        "invalid_border_cells": int(invalid_border_cells),
        "preserved_invalid_border_cells": int(preserved_invalid_border_cells),
        "top_area_gap": int(seed_info[larger]["area"] - seed_info[smaller]["area"]),
        "top_major_diff": int(seed_info[larger]["major"] != seed_info[smaller]["major"]),
        "shared_cell_count": int(len(shared_cells)),
        "shared_recolored_to_larger": int(shared_recolored_to_larger),
        "shared_different_from_smaller": int(shared_different_from_smaller),
    }


def generate_input() -> np.ndarray:
    rng = random.Random()
    h, w = GRID_SHAPE
    bg, pcol, color_a, color_b = rng.sample(list(range(10)), 4)
    layout = _build_layout(rng)

    winner_major = color_a
    loser_major = color_b
    g = np.full((h, w), int(bg), dtype=int)

    for slot, data in layout["seeds"].items():
        major = winner_major if data["major_side"] == "winner" else loser_major
        minor = loser_major if major == winner_major else winner_major
        _paint_seed_interior(g, data["bbox"], int(major), int(minor), rng)

    for data in layout["seeds"].values():
        for r, c in data["border"]:
            g[r, c] = int(pcol)
    for cells in layout["valid_paths"].values():
        for r, c in cells:
            g[r, c] = int(pcol)
    for r, c in layout["hub"]:
        g[r, c] = int(pcol)

    metrics = _behavior_metrics(g)
    assert metrics["invalid_seed_count"] >= 1
    assert metrics["top_area_gap"] > 0
    assert metrics["top_major_diff"] == 1
    assert metrics["shared_recolored_to_larger"] == metrics["shared_cell_count"]
    assert metrics["shared_different_from_smaller"] == metrics["shared_cell_count"]
    assert metrics["preserved_invalid_border_cells"] == metrics["invalid_border_cells"]
    assert int(get_bg(g)) == int(bg)
    return g
