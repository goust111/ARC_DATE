import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
YELLOW = 4
FRAME_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
PATTERNS = [
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int),  # L
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),  # T
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),  # zig
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),  # hook
]
CORNERS = ["tl", "tr", "bl", "br"]


def _bbox_from_mask(mask: np.ndarray):
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    return (r0, c0, r1, c1)


def _perimeter_cells(box):
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _corner_window(box, corner: str):
    r0, c0, r1, c1 = box
    if corner == "tl":
        return (r0 + 1, c0 + 1)
    if corner == "tr":
        return (r0 + 1, c1 - 3)
    if corner == "bl":
        return (r1 - 3, c0 + 1)
    return (r1 - 3, c1 - 3)


def _component_masks(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comp = np.zeros_like(mask, dtype=bool)
            for rr, cc in coords:
                comp[rr, cc] = True
            comps.append(comp)
    return comps


def _rot_cw(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, -k)


def _extract_frames(grid: np.ndarray):
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, YELLOW)]
    frames = []
    for color in colors:
        box = _bbox_from_mask(grid == color)
        assert box is not None
        r0, c0, r1, c1 = box
        assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5
        frames.append({"color": color, "box": box})
    return frames


def _layer_order(grid: np.ndarray, frames):
    perims = {f["color"]: _perimeter_cells(f["box"]) for f in frames}
    edges = defaultdict(set)
    indeg = {f["color"]: 0 for f in frames}
    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            inter = perims[a["color"]] & perims[b["color"]]
            if not inter:
                continue
            seen = {int(grid[r, c]) for r, c in inter if int(grid[r, c]) in (a["color"], b["color"])}
            if len(seen) != 1:
                continue
            top = next(iter(seen))
            bot = b["color"] if top == a["color"] else a["color"]
            if top not in edges[bot]:
                edges[bot].add(top)
                indeg[top] += 1

    q = deque(sorted([c for c, d in indeg.items() if d == 0]))
    order = []
    while q:
        color = q.popleft()
        order.append(color)
        for nxt in sorted(edges[color]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    assert len(order) == len(frames), "layer order must be recoverable"
    return order


def _assign_patterns(grid: np.ndarray, frames):
    comps = _component_masks(grid == YELLOW)
    assignments = {}
    for comp in comps:
        hit = []
        pts = np.argwhere(comp)
        for frame in frames:
            r0, c0, r1, c1 = frame["box"]
            if bool((pts[:, 0] <= r0).any()) or bool((pts[:, 0] >= r1).any()) or bool((pts[:, 1] <= c0).any()) or bool((pts[:, 1] >= c1).any()):
                continue
            for corner in CORNERS:
                wr, wc = _corner_window(frame["box"], corner)
                if bool(comp[wr : wr + 3, wc : wc + 3].sum() == comp.sum()):
                    hit.append((frame["color"], corner, wr, wc))
        assert len(hit) == 1, "each yellow pattern must belong to exactly one frame corner"
        color, corner, wr, wc = hit[0]
        assignments[color] = {"corner": corner, "mask": comp[wr : wr + 3, wc : wc + 3].astype(int)}
    assert len(assignments) == len(frames)
    return assignments


def _render(frames_by_color, order, assignments, rotations, shape=None):
    if shape is None:
        h = max(f["box"][2] for f in frames_by_color.values()) + 1
        w = max(f["box"][3] for f in frames_by_color.values()) + 1
    else:
        h, w = map(int, shape)
    out = np.full((h, w), BG, dtype=int)
    for color in order:
        r0, c0, r1, c1 = frames_by_color[color]["box"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        wr, wc = _corner_window((r0, c0, r1, c1), assignments[color]["corner"])
        patch = _rot_cw(assignments[color]["mask"], rotations[color])
        out[wr : wr + 3, wc : wc + 3][patch.astype(bool)] = YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapping rectangle frames, infer their bottom-to-top order from
    visible overlap colors, and rotate each frame's yellow corner motif by an
    order-dependent amount.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    order = _layer_order(g, frames)
    assignments = _assign_patterns(g, frames)
    frames_by_color = {f["color"]: f for f in frames}
    rotations = {color: k for k, color in enumerate(order)}
    return _render(frames_by_color, order, assignments, rotations, shape=g.shape)


def _make_rectangles(rng: random.Random, h: int, w: int, n: int):
    for _ in range(300):
        bh = rng.randint(7, 11)
        bw = rng.randint(7, 11)
        step_r = rng.randint(2, 4)
        step_c = rng.randint(2, 4)
        base_r = rng.randint(1, max(1, h - bh - step_r * (n - 1) - 2))
        base_c = rng.randint(1, max(1, w - bw - step_c * (n - 1) - 2))
        rects = []
        ok = True
        for i in range(n):
            rh = bh + rng.randint(-1, 1)
            rw = bw + rng.randint(-1, 1)
            r0 = base_r + i * step_r + rng.randint(0, 1)
            c0 = base_c + i * step_c + rng.randint(0, 1)
            r1 = r0 + rh - 1
            c1 = c0 + rw - 1
            if r1 >= h - 1 or c1 >= w - 1:
                ok = False
                break
            rects.append({"box": (r0, c0, r1, c1)})
        if not ok:
            continue
        good = True
        for i in range(n - 1):
            a = _perimeter_cells(rects[i]["box"])
            b = _perimeter_cells(rects[i + 1]["box"])
            if len(a & b) < 2:
                good = False
                break
        if good:
            return rects
    return None


def _pick_corner(rect, others, rng: random.Random):
    for corner in rng.sample(CORNERS, len(CORNERS)):
        wr, wc = _corner_window(rect["box"], corner)
        cells = {(wr + r, wc + c) for r in range(3) for c in range(3)}
        ambiguous = False
        for other in others:
            or0, oc0, or1, oc1 = other["box"]
            inside_other = all(or0 < r < or1 and oc0 < c < oc1 for r, c in cells)
            if not inside_other:
                continue
            for ocorner in CORNERS:
                owr, owc = _corner_window(other["box"], ocorner)
                owcells = {(owr + r, owc + c) for r in range(3) for c in range(3)}
                if cells <= owcells:
                    ambiguous = True
                    break
            if ambiguous:
                break
        if ambiguous:
            continue
        hits_border = False
        for other in others:
            if other is rect:
                continue
            oper = _perimeter_cells(other["box"])
            if cells & oper:
                hits_border = True
                break
        if not hits_border:
            return corner
    return None


def _visible_side_counts(grid, rect):
    color = rect["color"]
    r0, c0, r1, c1 = rect["box"]
    return [
        int((grid[r0, c0 : c1 + 1] == color).sum()),
        int((grid[r1, c0 : c1 + 1] == color).sum()),
        int((grid[r0 : r1 + 1, c0] == color).sum()),
        int((grid[r0 : r1 + 1, c1] == color).sum()),
    ]


def _build_instance(rng: random.Random):
    h = rng.randint(22, 31)
    w = rng.randint(22, 31)
    n = rng.randint(3, 5)
    rects = _make_rectangles(rng, h, w, n)
    if rects is None:
        return None

    colors = rng.sample(FRAME_COLORS, n)
    for i, rect in enumerate(rects):
        rect["color"] = colors[i]
        rect["corner"] = "tl"
        rect["pattern"] = rng.choice(PATTERNS).copy()

    order = colors[:]
    frames_by_color = {rect["color"]: rect for rect in rects}
    assignments = {rect["color"]: {"corner": rect["corner"], "mask": rect["pattern"]} for rect in rects}
    raw = _render(frames_by_color, order, assignments, {c: 0 for c in order}, shape=(h, w))

    # Require each adjacent pair to expose a witness overlap cell of the upper color.
    for low, high in zip(order, order[1:]):
        inter = _perimeter_cells(frames_by_color[low]["box"]) & _perimeter_cells(frames_by_color[high]["box"])
        if not inter:
            return None
        if not any(int(raw[r, c]) == high for r, c in inter):
            return None

    # Keep all frames recoverable from visible border remnants.
    for rect in rects:
        counts = _visible_side_counts(raw, rect)
        if min(counts) < 1:
            return None
        if sum(counts) < 8:
            return None

    # Ensure pattern assignment remains unique on the visible input.
    try:
        inferred = _assign_patterns(raw, [{"color": r["color"], "box": r["box"]} for r in rects])
    except AssertionError:
        return None
    if set(inferred) != set(order):
        return None

    # Ensure the oracle can fully recover the intended layer order.
    try:
        out = main(raw)
    except AssertionError:
        return None
    if out.shape != raw.shape or not np.array_equal(out, main(raw)):
        return None
    return raw


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        grid = _build_instance(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
