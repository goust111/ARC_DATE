from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
HOUSE_COLORS = [1, 2, 3, 4, 6, 9]
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray, *, conn8: bool = False) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = _neighbors4(0, 0) if not conn8 else [
        (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc
    ]
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    return int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())


def _hole_count_from_mask(mask: np.ndarray) -> int:
    inv = ~mask
    count = 0
    for pts in _components(inv):
        if not any(r == 0 or c == 0 or r == inv.shape[0] - 1 or c == inv.shape[1] - 1 for r, c in pts):
            count += 1
    return count


def _object_holes(pts: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = pts - np.array([[r0, c0]])
    mask[rel[:, 0], rel[:, 1]] = True
    return _hole_count_from_mask(mask)


def _find_houses(g: np.ndarray) -> list[dict]:
    houses = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY, AZURE)):
        for pts in _components(g == color):
            if len(pts) < 10:
                continue
            houses.append({"color": color, "pts": pts, "holes": _object_holes(pts)})
    assert 3 <= len(houses) <= 5
    assert len({h["holes"] for h in houses}) == len(houses)
    return houses


def _find_key(g: np.ndarray) -> dict:
    comps = _components(g == GREY, conn8=True)
    assert len(comps) == 1
    pts = comps[0]
    return {"pts": pts, "holes": _object_holes(pts)}


def _azure_chains(g: np.ndarray) -> list[set[tuple[int, int]]]:
    nodes = {tuple(map(int, p)) for p in np.argwhere(g == AZURE)}
    unseen = set(nodes)
    chains = []
    while unseen:
        start = unseen.pop()
        comp = {start}
        q = deque([start])
        while q:
            r, c = q.popleft()
            for nr, nc in ((r - 2, c), (r + 2, c), (r, c - 2), (r, c + 2)):
                if (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    comp.add((nr, nc))
                    q.append((nr, nc))
        chains.append(comp)
    assert 2 <= len(chains) <= 3
    return chains


def _adjacent_houses(chain: set[tuple[int, int]], houses: list[dict]) -> set[int]:
    hits = set()
    chain_neighbors = set()
    for r, c in chain:
        chain_neighbors.update(_neighbors4(r, c))
    for i, house in enumerate(houses):
        if any((int(r), int(c)) in chain_neighbors for r, c in house["pts"]):
            hits.add(i)
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    houses = _find_houses(g)
    key = _find_key(g)
    start_matches = [i for i, h in enumerate(houses) if h["holes"] == key["holes"]]
    assert len(start_matches) == 1
    start = start_matches[0]

    chosen = []
    for chain in _azure_chains(g):
        adj = _adjacent_houses(chain, houses)
        if start in adj:
            chosen.append((chain, adj))
    assert len(chosen) == 1
    _chain, adj = chosen[0]
    targets = [i for i in adj if i != start]
    assert len(targets) == 1
    target_color = int(houses[targets[0]]["color"])

    out = g.copy()
    pts = key["pts"]
    out[pts[:, 0], pts[:, 1]] = target_color
    return out


_HOLE_POS = [(1, 1), (1, 3), (3, 1), (3, 3)]


def _draw_holed_block(g: np.ndarray, r0: int, c0: int, color: int, holes: int, size: int = 5) -> np.ndarray:
    g[r0 : r0 + size, c0 : c0 + size] = color
    for rr, cc in _HOLE_POS[:holes]:
        g[r0 + rr, c0 + cc] = BLACK
    return np.argwhere(g == color)


def _paint_block(g: np.ndarray, r0: int, c0: int, color: int, holes: int) -> None:
    g[r0 : r0 + 5, c0 : c0 + 5] = color
    for rr, cc in random.sample(_HOLE_POS, holes):
        g[r0 + rr, c0 + cc] = BLACK


def _draw_chain(g: np.ndarray, a: str, b: str) -> None:
    mids = {
        "TL": (4, 4), "TR": (4, 20), "BL": (20, 4), "BR": (20, 20)
    }
    ra, ca = mids[a]
    rb, cb = mids[b]
    if ra == rb:
        step = 2 if cb > ca else -2
        for c in range(ca + (3 if step > 0 else -3), cb + (-2 if step > 0 else 2), step):
            g[ra, c] = AZURE
    elif ca == cb:
        step = 2 if rb > ra else -2
        for r in range(ra + (3 if step > 0 else -3), rb + (-2 if step > 0 else 2), step):
            g[r, ca] = AZURE
    else:
        raise ValueError


def _gen_attempt() -> np.ndarray | None:
    g = np.zeros((25, 25), dtype=int)
    slots = {"TL": (2, 2), "TR": (2, 18), "BL": (18, 2), "BR": (18, 18)}
    hole_values = random.sample([1, 2, 3, 4], 4)
    colors = random.sample(HOUSE_COLORS, 4)
    for slot, holes, color in zip(slots, hole_values, colors):
        r0, c0 = slots[slot]
        _paint_block(g, r0, c0, color, holes)

    start_slot = random.choice(list(slots))
    start_holes = hole_values[list(slots).index(start_slot)]
    _paint_block(g, 10, 10, GREY, start_holes)

    edges = [("TL", "TR"), ("BL", "BR"), ("TL", "BL"), ("TR", "BR")]
    start_edges = [e for e in edges if start_slot in e]
    start_edge = random.choice(start_edges)
    other_edges = [e for e in edges if start_slot not in e]
    chain_edges = [start_edge] + other_edges
    for a, b in chain_edges:
        _draw_chain(g, a, b)

    try:
        main(g)
    except Exception:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input")
