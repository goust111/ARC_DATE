from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 500
HOLE_POS = [(1, 1), (1, 3), (3, 1), (3, 3)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    return int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())


def _hole_count(mask: np.ndarray) -> int:
    count = 0
    for pts in _components(~mask):
        if not any(r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1 for r, c in pts):
            count += 1
    return count


def _object_record(pts: np.ndarray, color: int) -> dict:
    r0, c0, r1, c1 = _bbox(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = pts - np.array([[r0, c0]])
    mask[rel[:, 0], rel[:, 1]] = True
    return {
        "color": int(color),
        "pts": pts,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "holes": _hole_count(mask),
    }


def _edge_from_bbox(bbox: tuple[int, int, int, int], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = bbox
    hits = []
    if r0 == 0:
        hits.append("top")
    if r1 == h - 1:
        hits.append("bottom")
    if c0 == 0:
        hits.append("left")
    if c1 == w - 1:
        hits.append("right")
    return hits[0] if len(hits) == 1 else None


def _find_legends(g: np.ndarray) -> dict[int, dict]:
    h, w = g.shape
    legends: dict[int, dict] = {}
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY)):
        for pts in _components(g == color):
            rec = _object_record(pts, color)
            edge = _edge_from_bbox(rec["bbox"], h, w)
            if edge is None:
                continue
            r0, c0, r1, c1 = rec["bbox"]
            if (r1 - r0 + 1) < 3 or (c1 - c0 + 1) < 3:
                continue
            holes = int(rec["holes"])
            assert holes not in legends
            legends[holes] = {**rec, "edge": edge}
    assert 2 <= len(legends) <= 4
    return legends


def _grey_objects(g: np.ndarray) -> list[dict]:
    objects = [_object_record(pts, GREY) for pts in _components(g == GREY)]
    assert objects
    return objects


def _distance_to_edge(obj: dict, edge: str, h: int, w: int) -> int:
    r0, c0, r1, c1 = obj["bbox"]
    if edge == "top":
        return r0
    if edge == "bottom":
        return h - 1 - r1
    if edge == "left":
        return c0
    return w - 1 - c1


def _dir(edge: str) -> tuple[int, int]:
    return {"top": (-1, 0), "bottom": (1, 0), "left": (0, -1), "right": (0, 1)}[edge]


def _can_place(out: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = out.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    target = out[top : top + mh, left : left + mw]
    return bool(np.all(target[mask] == BLACK))


def _paint(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    view = out[top : top + mask.shape[0], left : left + mask.shape[1]]
    view[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legends = _find_legends(g)
    objects = _grey_objects(g)

    out = g.copy()
    for obj in objects:
        pts = obj["pts"]
        out[pts[:, 0], pts[:, 1]] = BLACK

    by_edge: dict[str, list[dict]] = {"top": [], "bottom": [], "left": [], "right": []}
    for obj in objects:
        holes = int(obj["holes"])
        assert holes in legends
        legend = legends[holes]
        by_edge[str(legend["edge"])].append({**obj, "new_color": int(legend["color"])})

    for edge, group in by_edge.items():
        group.sort(key=lambda obj: _distance_to_edge(obj, edge, h, w))
        dr, dc = _dir(edge)
        for obj in group:
            r0, c0, _r1, _c1 = obj["bbox"]
            top, left = int(r0), int(c0)
            while _can_place(out, obj["mask"], top + dr, left + dc):
                top += dr
                left += dc
            _paint(out, obj["mask"], top, left, int(obj["new_color"]))
    return out


def _can_stamp(occ: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = occ.shape
    if r0 < 1 or c0 < 1 or r0 + size >= h - 1 or c0 + size >= w - 1:
        return False
    patch = occ[r0 - 1 : r0 + size + 1, c0 - 1 : c0 + size + 1]
    return not bool(patch.any())


def _stamp_holed(g: np.ndarray, occ: np.ndarray, r0: int, c0: int, color: int, holes: int) -> None:
    size = 5
    g[r0 : r0 + size, c0 : c0 + size] = color
    occ[r0 : r0 + size, c0 : c0 + size] = True
    for rr, cc in random.sample(HOLE_POS, holes):
        g[r0 + rr, c0 + cc] = BLACK
        occ[r0 + rr, c0 + cc] = False


def _place_legend(g: np.ndarray, occ: np.ndarray, edge: str, color: int, holes: int) -> bool:
    h, w = g.shape
    for _ in range(80):
        if edge == "top":
            r0, c0 = 0, random.randint(1, w - 6)
        elif edge == "bottom":
            r0, c0 = h - 5, random.randint(1, w - 6)
        elif edge == "left":
            r0, c0 = random.randint(1, h - 6), 0
        else:
            r0, c0 = random.randint(1, h - 6), w - 5
        if not occ[max(0, r0 - 1) : min(h, r0 + 6), max(0, c0 - 1) : min(w, c0 + 6)].any():
            _stamp_holed(g, occ, r0, c0, color, holes)
            return True
    return False


def _place_object(g: np.ndarray, occ: np.ndarray, holes: int) -> bool:
    h, w = g.shape
    for _ in range(160):
        r0 = random.randint(6, h - 11)
        c0 = random.randint(6, w - 11)
        if _can_stamp(occ, r0, c0, 5):
            _stamp_holed(g, occ, r0, c0, GREY, holes)
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 24)
    w = random.randint(20, 24)
    g = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    n = random.randint(2, 4)
    edges = random.sample(["top", "bottom", "left", "right"], n)
    holes = random.sample([1, 2, 3, 4], n)
    colors = random.sample(PALETTE, n)
    for edge, color, hole_count in zip(edges, colors, holes):
        if not _place_legend(g, occ, edge, color, hole_count):
            return None
    for hole_count in holes:
        if not _place_object(g, occ, hole_count):
            return None
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
