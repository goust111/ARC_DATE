from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "center_key": (int(c0 + c1), int(r0 + r1)),
                }
            )
    return comps


def _is_rule_band(row: np.ndarray, bg: int) -> bool:
    if row.size < 4:
        return False
    if np.any(row == bg):
        return False
    return bool(np.all(row[1:] != row[:-1]))


def _compose_owner(mask_a: np.ndarray, mask_b: np.ndarray, outer_shape: tuple[int, int], off_a: tuple[int, int], off_b: tuple[int, int]) -> np.ndarray | None:
    owner = np.full(outer_shape, -1, dtype=int)
    for owner_id, mask, (dr, dc) in ((0, mask_a, off_a), (1, mask_b, off_b)):
        pts = np.argwhere(mask)
        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < outer_shape[0] and 0 <= cc < outer_shape[1]):
                return None
            if int(owner[rr, cc]) != -1:
                return None
            owner[rr, cc] = owner_id
    if np.any(owner == -1):
        return None
    return owner


def _has_cross_contact(owner: np.ndarray) -> bool:
    h, w = map(int, owner.shape)
    for r in range(h):
        for c in range(w):
            cur = int(owner[r, c])
            if r + 1 < h and int(owner[r + 1, c]) != cur:
                return True
            if c + 1 < w and int(owner[r, c + 1]) != cur:
                return True
    return False


def _pair_candidates(mask_a: np.ndarray, mask_b: np.ndarray) -> list[dict]:
    h, w = map(int, mask_a.shape)
    placements = [
        ((h, w), (0, 0), (0, 0)),
        ((h, w + 1), (0, 0), (0, 1)),
        ((h, w + 1), (0, 1), (0, 0)),
        ((h + 1, w), (0, 0), (1, 0)),
        ((h + 1, w), (1, 0), (0, 0)),
    ]
    out: list[dict] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for shape, off_a, off_b in placements:
        owner = _compose_owner(mask_a, mask_b, shape, off_a, off_b)
        if owner is None:
            continue
        key = (shape, owner.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "owner": owner,
                "shape": shape,
                "offset_a": off_a,
                "offset_b": off_b,
                "touches": _has_cross_contact(owner),
            }
        )
    return out


def _render_pair(owner: np.ndarray, color_a: int, color_b: int, touches: bool) -> np.ndarray:
    rect = np.empty(owner.shape, dtype=int)
    rect[owner == 0] = int(color_a)
    rect[owner == 1] = int(color_b)
    count_a = int(np.sum(owner == 0))
    count_b = int(np.sum(owner == 1))
    if touches and count_a != count_b:
        rect[:, :] = int(color_a if count_a > count_b else color_b)
    return rect


def _matching_solution(comps: list[dict]) -> list[tuple[int, int, dict]]:
    n = len(comps)
    edges: dict[tuple[int, int], dict] = {}
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if comps[i]["mask"].shape != comps[j]["mask"].shape:
                continue
            cands = _pair_candidates(comps[i]["mask"], comps[j]["mask"])
            if len(cands) == 1:
                edges[(i, j)] = cands[0]
                adj[i].append(j)
                adj[j].append(i)

    solutions: list[list[tuple[int, int, dict]]] = []

    def rec(used: set[int], cur: list[tuple[int, int, dict]]) -> None:
        if len(solutions) > 1:
            return
        if len(used) == n:
            solutions.append(cur.copy())
            return
        i = min(idx for idx in range(n) if idx not in used)
        for j in sorted(adj[i], key=lambda idx: comps[idx]["center_key"]):
            if j in used:
                continue
            key = (min(i, j), max(i, j))
            if key not in edges:
                continue
            used.add(i)
            used.add(j)
            cur.append((i, j, edges[key]))
            rec(used, cur)
            cur.pop()
            used.remove(i)
            used.remove(j)

    rec(set(), [])
    assert len(solutions) == 1, "fragment pairing must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-size monochrome fragments by shape complementarity, compose a rectangle for each pair,
    then merge colors only when one color is strictly larger and the two colors touch orthogonally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _mode_color(g)

    core = g[:-1] if g.shape[0] >= 2 and _is_rule_band(g[-1], bg) else g
    comps = _find_components(core, bg)
    assert comps and len(comps) % 2 == 0, "expected an even number of fragments"

    bbox_shapes = {comp["mask"].shape for comp in comps}
    assert len(bbox_shapes) == 1, "all fragment bounding boxes must have the same size"
    assert all(not np.all(comp["mask"]) for comp in comps), "fragments must be non-rectangular"

    pairs = _matching_solution(comps)
    rendered: list[tuple[tuple[int, int], np.ndarray]] = []
    for i, j, info in pairs:
        rep_key = min(comps[i]["center_key"], comps[j]["center_key"])
        rect = _render_pair(info["owner"], comps[i]["color"], comps[j]["color"], info["touches"])
        rendered.append((rep_key, rect))
    rendered.sort(key=lambda item: item[0])

    out_h = max(rect.shape[0] for _, rect in rendered)
    out_w = sum(rect.shape[1] for _, rect in rendered) + max(0, len(rendered) - 1)
    out = np.full((out_h, out_w), bg, dtype=int)
    c = 0
    for _, rect in rendered:
        rh, rw = rect.shape
        out[:rh, c : c + rw] = rect
        c += rw + 1
    return out


GEN_RETRIES = 400
PAIR_COUNT_RANGE = (2, 4)
BBOX_RANGE = (3, 5)
PLACEMENT_RETRIES = 200


def _vector_complement(vec: tuple[int, ...], max_val: int) -> tuple[int, ...]:
    return tuple(int(max_val + 1 - x) for x in vec)


def _vector_pool(h: int, w: int, orientation: str, want_tie: bool) -> list[tuple[int, ...]]:
    if orientation == "horizontal":
        length = h
        max_val = w
        total = h * (w + 1)
    else:
        length = w
        max_val = h
        total = (h + 1) * w
    out: list[tuple[int, ...]] = []
    for vec in itertools.product(range(1, max_val + 1), repeat=length):
        if 1 not in vec or max_val not in vec or len(set(vec)) == 1:
            continue
        is_tie = (sum(vec) * 2 == total)
        if is_tie == want_tie:
            out.append(tuple(int(x) for x in vec))
    random.shuffle(out)
    return out


def _masks_from_vector(vec: tuple[int, ...], h: int, w: int, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((h, w), dtype=bool)
    b = np.zeros((h, w), dtype=bool)
    if orientation == "horizontal":
        for r, t in enumerate(vec):
            a[r, :t] = True
            b[r, t - 1 :] = True
    else:
        for c, t in enumerate(vec):
            a[:t, c] = True
            b[t - 1 :, c] = True
    return a, b


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    pts = np.argwhere(mask)
    for r, c in pts:
        grid[int(top + r), int(left + c)] = int(color)


def _pick_vectors(h: int, w: int, orientation: str, pair_count: int) -> list[tuple[int, ...]] | None:
    if pair_count == 2:
        want_ties = [True, False]
    elif pair_count == 3:
        want_ties = [True, False, random.choice([True, False])]
    else:
        want_ties = [True, True, False, False]
    random.shuffle(want_ties)

    max_val = w if orientation == "horizontal" else h
    chosen: list[tuple[int, ...]] = []
    blocked: set[tuple[int, ...]] = set()
    for want_tie in want_ties:
        pool = _vector_pool(h, w, orientation, want_tie)
        vec = next((v for v in pool if v not in blocked), None)
        if vec is None:
            return None
        chosen.append(vec)
        blocked.add(vec)
        blocked.add(_vector_complement(vec, max_val))
    random.shuffle(chosen)
    return chosen


def _place_bboxes(grid: np.ndarray, bbox_shape: tuple[int, int], count: int) -> list[tuple[int, int]] | None:
    h, w = bbox_shape
    gh, gw = map(int, grid.shape)
    blocked = np.zeros((gh, gw), dtype=bool)
    blocked[gh - 1, :] = True
    if gh >= 2:
        blocked[gh - 2, :] = True
    placed: list[tuple[int, int]] = []
    for _ in range(count):
        ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(0, gh - h - 3)
            c = random.randint(0, gw - w)
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(gh, r + h + 1)
            c1 = min(gw, c + w + 1)
            if np.any(blocked[r0:r1, c0:c1]):
                continue
            blocked[r0:r1, c0:c1] = True
            placed.append((int(r), int(c)))
            ok = True
            break
        if not ok:
            return None
    random.shuffle(placed)
    return placed


def _build_rule_band(width: int, pair_colors: list[tuple[int, int]]) -> np.ndarray:
    flat_colors = [c for pair in pair_colors for c in pair]
    hints: list[tuple[int, int]] = []
    need = width // 2
    while len(hints) < need:
        if len(pair_colors) >= 2 and random.random() < 0.5:
            p1, p2 = random.sample(pair_colors, 2)
            a = random.choice(p1)
            b = random.choice(p2)
            if a == b:
                continue
        else:
            a, b = random.choice(pair_colors)
            if random.random() < 0.5:
                a, b = b, a
        if hints and hints[-1][1] == a:
            a, b = b, a
        if hints and hints[-1][1] == a:
            for cand in flat_colors:
                if cand != hints[-1][1]:
                    a = cand
                    break
            if a == b:
                for cand in flat_colors:
                    if cand != a:
                        b = cand
                        break
        if a == b:
            continue
        hints.append((int(a), int(b)))
    band = np.asarray([c for pair in hints for c in pair], dtype=int)
    assert band.size == width
    assert np.all(band[1:] != band[:-1])
    return band


def _attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    orientation = random.choice(["horizontal", "vertical"])

    dims = [(h, w) for h in range(BBOX_RANGE[0], BBOX_RANGE[1] + 1) for w in range(BBOX_RANGE[0], BBOX_RANGE[1] + 1)]
    random.shuffle(dims)
    picked = None
    for h, w in dims:
        total = h * (w + 1) if orientation == "horizontal" else (h + 1) * w
        if total % 2 != 0:
            continue
        vectors = _pick_vectors(h, w, orientation, pair_count)
        if vectors is not None:
            picked = (h, w, vectors)
            break
    if picked is None:
        return None
    h, w, vectors = picked

    bg = random.randrange(10)
    colors = random.sample([c for c in range(10) if c != bg], 2 * pair_count)

    gh_min = max(10, 2 * h + 5)
    gh_max = max(gh_min, 3 * h + 7)
    gw_min = max(16, pair_count * (w + 5) + 6)
    gw_max = max(gw_min, 2 * pair_count * (w + 4) + 10)
    gh = random.randint(gh_min, gh_max)
    gw = random.randint(gw_min, gw_max)
    if gw % 2 == 1:
        gw += 1

    grid = np.full((gh, gw), bg, dtype=int)
    positions = _place_bboxes(grid, (h, w), 2 * pair_count)
    if positions is None:
        return None

    pair_colors: list[tuple[int, int]] = []
    pos_iter = iter(positions)
    for idx, vec in enumerate(vectors):
        mask_a, mask_b = _masks_from_vector(vec, h, w, orientation)
        color_a = int(colors[2 * idx])
        color_b = int(colors[2 * idx + 1])
        pair_colors.append((color_a, color_b))
        ra, ca = next(pos_iter)
        rb, cb = next(pos_iter)
        _paint_mask(grid, mask_a, ra, ca, color_a)
        _paint_mask(grid, mask_b, rb, cb, color_b)

    grid[-1, :] = _build_rule_band(gw, pair_colors)

    if _mode_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.size == 0 or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
