from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DOWN = "down"  # wall pixels on the NW-SE diagonal: ↘
UP = "up"      # wall pixels on the NE-SW diagonal: ↗


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                yield r + dr, c + dc


def _colors(g: np.ndarray) -> list[int]:
    return sorted(map(int, np.unique(g)))


def _most_common(values) -> int:
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _label_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    H, W = mask.shape
    labels = np.full((H, W), -1, dtype=int)
    comps: list[list[tuple[int, int]]] = []
    neigh = _neighbors8 if connectivity == 8 else _neighbors4

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            cid = len(comps)
            labels[r, c] = cid
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            comps.append(comp)
    return labels, comps


def _crack_edges(g: np.ndarray, wall: int, labels: np.ndarray) -> list[tuple[int, int, str]]:
    H, W = g.shape
    edges: set[tuple[int, int, str]] = set()
    for r in range(H - 1):
        for c in range(W - 1):
            nw = int(g[r, c])
            ne = int(g[r, c + 1])
            sw = int(g[r + 1, c])
            se = int(g[r + 1, c + 1])

            if nw == wall and se == wall and ne != wall and sw != wall:
                a = int(labels[r, c + 1])
                b = int(labels[r + 1, c])
                if a != -1 and b != -1 and a != b:
                    edges.add((min(a, b), max(a, b), DOWN))

            if ne == wall and sw == wall and nw != wall and se != wall:
                a = int(labels[r, c])
                b = int(labels[r + 1, c + 1])
                if a != -1 and b != -1 and a != b:
                    edges.add((min(a, b), max(a, b), UP))
    return sorted(edges)


def _infer_scene(g: np.ndarray):
    colors = _colors(g)
    candidates = []
    for wall in colors:
        labels, comps = _label_components(g != wall, connectivity=4)
        edges = _crack_edges(g, wall, labels)
        if not edges:
            continue
        count = int(np.count_nonzero(g == wall))
        candidates.append((len(edges), count, wall, labels, comps, edges))

    assert candidates, "expected a wall color with diagonal crack contacts"
    candidates.sort(key=lambda item: (item[0], item[1]))
    _edge_count, _wall_count, wall, labels, comps, edges = candidates[-1]

    open_values = g[g != wall]
    assert open_values.size > 0, "expected background cavities"
    bg = _most_common(open_values)
    seed_colors = [color for color in colors if color not in (wall, bg)]
    assert seed_colors, "expected at least one seed blob color"
    return int(wall), int(bg), labels, comps, edges, seed_colors


def _keys_for_color(g: np.ndarray, color: int) -> set[str]:
    mask = g == int(color)
    H, W = mask.shape
    keys: set[str] = set()
    for r in range(H - 1):
        for c in range(W - 1):
            if mask[r, c] and mask[r + 1, c + 1]:
                keys.add(DOWN)
            if mask[r, c + 1] and mask[r + 1, c]:
                keys.add(UP)
    return keys


def _seed_infos(g: np.ndarray, seed_colors: list[int], labels: np.ndarray) -> list[dict]:
    infos = []
    for color in seed_colors:
        mask = g == int(color)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        assert coords, "seed color must be present"

        _lab8, comps8 = _label_components(mask, connectivity=8)
        assert len(comps8) == 1, "each seed color must form one 8-connected blob"

        comp_ids = {int(labels[r, c]) for r, c in coords}
        assert len(comp_ids) == 1 and -1 not in comp_ids, "a seed blob must lie in one cavity"
        infos.append({"color": int(color), "component": comp_ids.pop(), "keys": _keys_for_color(g, color)})
    return infos


def _reachability(start: int, keys: set[str], edges: list[tuple[int, int, str]]) -> set[int]:
    adj: dict[int, list[tuple[int, str]]] = {}
    for a, b, orient in edges:
        adj.setdefault(a, []).append((b, orient))
        adj.setdefault(b, []).append((a, orient))

    seen = {int(start)}
    q = deque([int(start)])
    while q:
        cur = q.popleft()
        for nxt, orient in adj.get(cur, []):
            if orient not in keys or nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    return seen


def _owners_for_scene(g: np.ndarray):
    wall, bg, labels, comps, edges, seed_colors = _infer_scene(g)
    infos = _seed_infos(g, seed_colors, labels)
    owners: dict[int, int] = {}
    reaches: dict[int, set[int]] = {}
    for info in infos:
        color = int(info["color"])
        reached = _reachability(int(info["component"]), set(info["keys"]), edges)
        reaches[color] = reached
        for cid in reached:
            prev = owners.get(cid)
            assert prev is None or prev == color, "two seed colors may not claim the same cavity"
            owners[cid] = color
    return wall, bg, labels, comps, edges, infos, owners, reaches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let each seed color seep through only the diagonal wall cracks matching the seed blob's chirality keys.

    Input structure:
    - A single wall color divides the grid into 4-connected background cavities; the wall is 8-connected and has a few 2x2 diagonal contacts.
    - Each crack is a 2x2 block with wall pixels on one diagonal and open cavity pixels on the other diagonal, giving either ↘ or ↗ chirality.
    - Several cavities contain one 8-connected monochrome seed blob each; seed colors are distinct and differ from the wall and cavity background.

    Rule:
    - Decode each seed blob's keys from same-color diagonal pixel pairs inside its own shape: NW-SE pairs give ↘, NE-SW pairs give ↗.
    - Starting from the seed's cavity, flood through only cracks whose chirality is one of that seed's keys, repeating across the cavity graph.
    - Fill every reached cavity, including the original seed pixels, with the seed color; keep wall pixels unchanged and leave unreached cavities as background.

    Key insight:
    - The passage labels are hidden in near-invisible diagonal wall contacts, while the matching keys are hidden in each seed's internal diagonal geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _wall, _bg, labels, comps, _edges, _infos, owners, _reaches = _owners_for_scene(g)
    out = g.copy()
    for cid, color in owners.items():
        for r, c in comps[int(cid)]:
            out[r, c] = int(color)
    return out


# -------------------------- generator --------------------------

GEN_BG = 0
GEN_WALL = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300

CLUSTERS = [
    {"kind": "none", "cells": [(0, 0), (1, 1)], "seed": 0},
    {"kind": DOWN, "cells": [(0, 1), (1, 0), (2, 1)], "seed": 0},
    {"kind": UP, "cells": [(0, 0), (1, 1), (2, 0)], "seed": 0},
    {"kind": "both", "cells": [(0, 1), (1, 0), (2, 1)], "seed": 1},
]

BASE_ANCHORS = [(0, 0), (0, 5), (4, 0), (4, 5)]

SHAPES = {
    "none": [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 1]], dtype=bool),
    ],
    DOWN: [
        np.array([[1, 0], [0, 1]], dtype=bool),
        np.array([[1, 0], [0, 1], [0, 1]], dtype=bool),
        np.array([[1, 0, 0], [0, 1, 1]], dtype=bool),
    ],
    UP: [
        np.array([[0, 1], [1, 0]], dtype=bool),
        np.array([[0, 1], [1, 0], [1, 0]], dtype=bool),
        np.array([[0, 0, 1], [1, 1, 0]], dtype=bool),
    ],
    "both": [
        np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 0, 1]], dtype=bool),
        np.array([[1, 0], [0, 1], [1, 0]], dtype=bool),
    ],
}


def _transform_macro(cells: list[tuple[int, int]], n: int, mode: int) -> list[tuple[int, int]]:
    out = []
    for r, c in cells:
        if mode == 0:
            rr, cc = r, c
        elif mode == 1:
            rr, cc = c, n - 1 - r
        elif mode == 2:
            rr, cc = n - 1 - r, n - 1 - c
        elif mode == 3:
            rr, cc = n - 1 - c, r
        elif mode == 4:
            rr, cc = r, n - 1 - c
        elif mode == 5:
            rr, cc = n - 1 - r, c
        elif mode == 6:
            rr, cc = c, r
        else:
            rr, cc = n - 1 - c, n - 1 - r
        out.append((int(rr), int(cc)))
    return out


def _transform_grid(g: np.ndarray, mode: int) -> np.ndarray:
    if mode == 0:
        return g.copy()
    if mode == 1:
        return np.rot90(g, -1).copy()
    if mode == 2:
        return np.rot90(g, 2).copy()
    if mode == 3:
        return np.rot90(g, 1).copy()
    if mode == 4:
        return np.fliplr(g).copy()
    if mode == 5:
        return np.flipud(g).copy()
    if mode == 6:
        return g.T.copy()
    return np.fliplr(np.flipud(g)).T.copy()


def _placed_macro_clusters(mode: int) -> list[dict]:
    n = 7
    placed = []
    for cluster, anchor in zip(CLUSTERS, BASE_ANCHORS):
        ar, ac = anchor
        raw = [(ar + r, ac + c) for r, c in cluster["cells"]]
        cells = _transform_macro(raw, n, mode)
        seed_raw = raw[int(cluster["seed"])]
        seed_cell = _transform_macro([seed_raw], n, mode)[0]
        assert seed_cell in cells
        placed.append({"kind": cluster["kind"], "cells": cells, "seed_cell": seed_cell})
    return placed


def _room_bounds(mr: int, mc: int, room: int, top: int, left: int) -> tuple[int, int, int, int]:
    r0 = int(top + mr * room)
    c0 = int(left + mc * room)
    return r0, c0, r0 + room - 1, c0 + room - 1


def _paint_seed(g: np.ndarray, room_bbox: tuple[int, int, int, int], color: int, kind: str) -> None:
    r0, c0, r1, c1 = room_bbox
    shape = random.choice(SHAPES[kind])
    h, w = shape.shape
    room_h = r1 - r0 + 1
    room_w = c1 - c0 + 1

    # Prefer non-corner placement so the visible crack corners stay easy to read.
    rr_min = r0 + (1 if room_h - h >= 2 else 0)
    cc_min = c0 + (1 if room_w - w >= 2 else 0)
    rr_max = r1 - h + 1 - (1 if room_h - h >= 2 else 0)
    cc_max = c1 - w + 1 - (1 if room_w - w >= 2 else 0)
    rr = random.randint(rr_min, rr_max)
    cc = random.randint(cc_min, cc_max)
    for dr, dc in np.argwhere(shape):
        g[int(rr + dr), int(cc + dc)] = int(color)


def _render_base() -> np.ndarray:
    macro_n = 7
    room = 4
    extra_h = random.randint(0, 2)
    extra_w = random.randint(0, 2)
    top = random.randint(0, extra_h)
    left = random.randint(0, extra_w)
    H = macro_n * room + extra_h
    W = macro_n * room + extra_w

    mode = random.randrange(8)
    clusters = _placed_macro_clusters(mode)
    g = np.full((H, W), GEN_WALL, dtype=int)

    all_rooms: set[tuple[int, int]] = set()
    for cluster in clusters:
        all_rooms.update(cluster["cells"])

    assert len(all_rooms) == sum(len(cluster["cells"]) for cluster in clusters)
    for mr, mc in all_rooms:
        r0, c0, r1, c1 = _room_bounds(mr, mc, room, top, left)
        g[r0 : r1 + 1, c0 : c1 + 1] = GEN_BG

    colors = random.sample(SEED_COLORS, len(clusters))
    random.shuffle(colors)
    for color, cluster in zip(colors, clusters):
        bbox = _room_bounds(*cluster["seed_cell"], room, top, left)
        _paint_seed(g, bbox, color, cluster["kind"])

    # Add a final pixel-level dihedral transform; the seed geometry and cracks transform together.
    return _transform_grid(g, random.randrange(8))


def _semantic_check(inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if inp.shape != out.shape:
        return False
    if np.array_equal(inp, out):
        return False

    try:
        wall, bg, labels, comps, edges, infos, owners, reaches = _owners_for_scene(inp)
    except Exception:
        return False

    if wall == bg or wall not in _colors(inp) or bg not in _colors(inp):
        return False
    if not {DOWN, UP}.issubset({orient for _a, _b, orient in edges}):
        return False

    key_sets = {frozenset(info["keys"]) for info in infos}
    required = {frozenset(), frozenset({DOWN}), frozenset({UP}), frozenset({DOWN, UP})}
    if not required.issubset(key_sets):
        return False

    if len(owners) >= len(comps):
        return False
    if not any(len(reached) > 1 for reached in reaches.values()):
        return False

    for r, c in np.argwhere(inp == wall):
        if int(out[int(r), int(c)]) != wall:
            return False
    for cid, comp in enumerate(comps):
        vals = {int(out[r, c]) for r, c in comp}
        if cid in owners:
            if vals != {int(owners[cid])}:
                return False
        else:
            if vals != {bg}:
                return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _render_base()
        try:
            out = main(inp)
        except Exception:
            continue
        if _semantic_check(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid chiral-crack input")
