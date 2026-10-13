from __future__ import annotations

from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()

GEN_RETRIES = 500
PLACE_RETRIES = 160
MASK_RETRIES = 120

GRID_H_RANGE = (14, 21)
GRID_W_RANGE = (14, 21)
PALETTE = tuple(range(10))

ADJACENT_EDGE_PAIRS = {
    frozenset(("top", "left")),
    frozenset(("top", "right")),
    frozenset(("bottom", "left")),
    frozenset(("bottom", "right")),
}


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(lo, hi + 1))


def _bg_color(grid: np.ndarray) -> int:
    return Counter(map(int, grid.flat)).most_common(1)[0][0]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _coords_to_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_to_coords(mask: np.ndarray, r0: int, c0: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        coords.append((int(r0 + r), int(c0 + c)))
    return coords


def _touched_edges(coords: list[tuple[int, int]], h: int, w: int) -> frozenset[str]:
    edges = set()
    for r, c in coords:
        if r == 0:
            edges.add("top")
        if r == h - 1:
            edges.add("bottom")
        if c == 0:
            edges.add("left")
        if c == w - 1:
            edges.add("right")
    return frozenset(edges)


def _branch_name(edges: frozenset[str]) -> str:
    if len(edges) == 0:
        return "none"
    if len(edges) == 1:
        return "one"
    if len(edges) == 2 and edges in ADJACENT_EDGE_PAIRS:
        return "corner"
    return "fixed"


def _transform_mask(mask: np.ndarray, edges: frozenset[str]) -> np.ndarray:
    branch = _branch_name(edges)
    if branch == "one":
        if "top" in edges or "bottom" in edges:
            return np.flipud(mask)
        return np.fliplr(mask)
    if branch == "corner":
        return np.rot90(mask, -1)
    return mask.copy()


def _transformed_anchor(
    mask: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    edges: frozenset[str],
) -> tuple[int, int]:
    if _branch_name(edges) != "corner":
        return r0, c0
    new_h, new_w = np.rot90(mask, -1).shape
    if edges == frozenset(("top", "left")):
        return 0, 0
    if edges == frozenset(("top", "right")):
        return 0, w - new_w
    if edges == frozenset(("bottom", "left")):
        return h - new_h, 0
    return h - new_h, w - new_w


def _simulate_object(
    mask: np.ndarray,
    r0: int,
    c0: int,
    color: int,
    h: int,
    w: int,
) -> dict:
    in_coords = _mask_to_coords(mask, r0, c0)
    edges = _touched_edges(in_coords, h, w)
    out_mask = _transform_mask(mask, edges)
    out_r0, out_c0 = _transformed_anchor(mask, r0, c0, h, w, edges)
    out_coords = _mask_to_coords(out_mask, out_r0, out_c0)
    return {
        "color": int(color),
        "mask": mask,
        "r0": int(r0),
        "c0": int(c0),
        "edges": edges,
        "branch": _branch_name(edges),
        "in_coords": in_coords,
        "out_mask": out_mask,
        "out_r0": int(out_r0),
        "out_c0": int(out_c0),
        "out_coords": out_coords,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each foreground object according to which grid edges it touches.

    - 0 touched edges: unchanged.
    - 1 touched edge: flip across that edge-aligned axis within the object's bbox.
    - 2 adjacent touched edges: rotate 90 degrees clockwise within the bbox frame,
      keeping the object anchored to the same corner.
    - 2 opposite edges or 3+ edges: unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _bg_color(g)

    out = np.full_like(g, bg)
    for comp in _find_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]
        mask = _coords_to_mask(comp["coords"], comp["bbox"])
        sim = _simulate_object(mask, r0, c0, int(comp["color"]), h, w)
        for rr, cc in sim["out_coords"]:
            out[rr, cc] = int(comp["color"])
    return out


def _random_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    r, c = start
    gr, gc = goal
    path = [(r, c)]
    moves = ["v"] * abs(gr - r) + ["h"] * abs(gc - c)
    RNG.shuffle(moves)
    for move in moves:
        if move == "v":
            r += 1 if gr > r else -1
        else:
            c += 1 if gc > c else -1
        assert 0 <= r < h and 0 <= c < w
        path.append((r, c))
    return path


def _sample_anchor(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, _randint(0, w - 1)
    if side == "bottom":
        return h - 1, _randint(0, w - 1)
    if side == "left":
        return _randint(0, h - 1), 0
    return _randint(0, h - 1), w - 1


def _side_ok(mask: np.ndarray, required_sides: frozenset[str]) -> bool:
    if "top" in required_sides and not bool(mask[0].any()):
        return False
    if "bottom" in required_sides and not bool(mask[-1].any()):
        return False
    if "left" in required_sides and not bool(mask[:, 0].any()):
        return False
    if "right" in required_sides and not bool(mask[:, -1].any()):
        return False
    return True


def _is_visibly_changed(mask: np.ndarray, branch: str, side_hint: str | None) -> bool:
    if branch == "one":
        if side_hint in {"top", "bottom"}:
            return not np.array_equal(mask, np.flipud(mask))
        return not np.array_equal(mask, np.fliplr(mask))
    if branch == "corner":
        rot = np.rot90(mask, -1)
        return mask.shape != rot.shape or not np.array_equal(mask, rot)
    return True


def _grow_mask(
    raw_h: int,
    raw_w: int,
    required_sides: frozenset[str],
    min_area: int,
    max_area: int,
    branch: str,
    side_hint: str | None,
) -> np.ndarray | None:
    total = raw_h * raw_w
    max_area = min(max_area, total)
    for _ in range(MASK_RETRIES):
        anchors = [_sample_anchor(side, raw_h, raw_w) for side in sorted(required_sides)]
        if not anchors:
            anchors = [(_randint(0, raw_h - 1), _randint(0, raw_w - 1))]

        cells: set[tuple[int, int]] = set()
        prev = anchors[0]
        cells.add(prev)
        for anchor in anchors[1:]:
            for cell in _random_path(prev, anchor, raw_h, raw_w):
                cells.add(cell)
            prev = anchor

        lo = max(min_area, len(cells))
        hi = max(lo, max_area)
        target = _randint(lo, hi)

        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, raw_h, raw_w):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier_list = list(frontier)
            cell = frontier_list[_randint(0, len(frontier_list) - 1)]
            cells.add(cell)

        mask = np.zeros((raw_h, raw_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)

        if int(mask.sum()) < min_area:
            continue
        if not _side_ok(mask, required_sides):
            continue
        if branch in {"one", "corner"} and not _is_visibly_changed(mask, branch, side_hint):
            continue
        return mask
    return None


def _coords_fit_forbidden(coords: list[tuple[int, int]], forbidden: np.ndarray) -> bool:
    return all(not bool(forbidden[r, c]) for r, c in coords)


def _mark_forbidden(forbidden: np.ndarray, coords: list[tuple[int, int]]):
    h, w = forbidden.shape
    for r, c in coords:
        forbidden[r, c] = True
        for nr, nc in _neighbors4(r, c, h, w):
            forbidden[nr, nc] = True


def _render(records: list[dict], bg: int, h: int, w: int, key: str) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for rec in records:
        coords = rec[key]
        for r, c in coords:
            grid[r, c] = int(rec["color"])
    return grid


def _candidate(branch: str, color: int, h: int, w: int, in_forbidden: np.ndarray, out_forbidden: np.ndarray) -> dict | None:
    for _ in range(PLACE_RETRIES):
        side_hint: str | None = None
        required_sides: frozenset[str]
        if branch == "fixed":
            orientation = "vertical" if RNG.random() < 0.5 else "horizontal"
            if orientation == "vertical":
                raw_h = h
                raw_w = _randint(2, 4)
                required_sides = frozenset(("top", "bottom"))
                mask = _grow_mask(raw_h, raw_w, required_sides, raw_h, raw_h + 7, branch, None)
                if mask is None or mask.shape[0] != h:
                    continue
                obj_h, obj_w = mask.shape
                if w - obj_w - 1 < 1:
                    continue
                r0 = 0
                c0 = _randint(1, w - obj_w - 1)
            else:
                raw_h = _randint(2, 4)
                raw_w = w
                required_sides = frozenset(("left", "right"))
                mask = _grow_mask(raw_h, raw_w, required_sides, raw_w, raw_w + 7, branch, None)
                if mask is None or mask.shape[1] != w:
                    continue
                obj_h, obj_w = mask.shape
                if h - obj_h - 1 < 1:
                    continue
                r0 = _randint(1, h - obj_h - 1)
                c0 = 0
        elif branch == "corner":
            corner = ("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")
            required_sides = frozenset(corner[_randint(0, 3)])
            raw_h = _randint(2, 5)
            raw_w = _randint(2, 5)
            if RNG.random() < 0.75:
                while raw_w == raw_h:
                    raw_w = _randint(2, 5)
            mask = _grow_mask(raw_h, raw_w, required_sides, 4, raw_h * raw_w, branch, None)
            if mask is None:
                continue
            obj_h, obj_w = mask.shape
            r0 = 0 if "top" in required_sides else h - obj_h
            c0 = 0 if "left" in required_sides else w - obj_w
        elif branch == "one":
            side_hint = ("top", "bottom", "left", "right")[_randint(0, 3)]
            required_sides = frozenset((side_hint,))
            raw_h = _randint(2, 5)
            raw_w = _randint(2, 5)
            mask = _grow_mask(raw_h, raw_w, required_sides, 3, raw_h * raw_w, branch, side_hint)
            if mask is None:
                continue
            obj_h, obj_w = mask.shape
            if side_hint in {"top", "bottom"}:
                if w - obj_w - 1 < 1:
                    continue
                r0 = 0 if side_hint == "top" else h - obj_h
                c0 = _randint(1, w - obj_w - 1)
            else:
                if h - obj_h - 1 < 1:
                    continue
                r0 = _randint(1, h - obj_h - 1)
                c0 = 0 if side_hint == "left" else w - obj_w
        else:
            required_sides = frozenset()
            raw_h = _randint(2, 5)
            raw_w = _randint(2, 5)
            mask = _grow_mask(raw_h, raw_w, required_sides, 3, raw_h * raw_w, branch, None)
            if mask is None:
                continue
            obj_h, obj_w = mask.shape
            if h - obj_h - 1 < 1 or w - obj_w - 1 < 1:
                continue
            r0 = _randint(1, h - obj_h - 1)
            c0 = _randint(1, w - obj_w - 1)

        rec = _simulate_object(mask, r0, c0, color, h, w)
        if rec["branch"] != branch:
            continue
        if not _coords_fit_forbidden(rec["in_coords"], in_forbidden):
            continue
        if not _coords_fit_forbidden(rec["out_coords"], out_forbidden):
            continue
        return rec
    return None


def _validate_sample(grid: np.ndarray, records: list[dict], bg: int) -> bool:
    h, w = grid.shape
    comps = _find_components(grid, bg)
    if len(comps) != len(records):
        return False
    branch_counts = Counter(_branch_name(_touched_edges(comp["coords"], h, w)) for comp in comps)
    if not all(branch_counts[branch] >= 1 for branch in ("none", "one", "corner", "fixed")):
        return False

    for rec in records:
        if rec["branch"] in {"one", "corner"} and rec["in_coords"] == rec["out_coords"]:
            return False

    expected = _render(records, bg, h, w, "out_coords")
    out = main(grid)
    if not np.array_equal(out, expected):
        return False
    if len(_find_components(out, bg)) != len(records):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = _randint(*GRID_H_RANGE)
    w = _randint(*GRID_W_RANGE)
    bg = PALETTE[_randint(0, len(PALETTE) - 1)]

    specs = ["fixed", "corner", "one", "none"]
    if RNG.random() < 0.5:
        specs.append(("none", "one", "corner")[_randint(0, 2)])

    if len(specs) > len(PALETTE) - 1:
        return None

    colors = [color for color in PALETTE if color != bg]
    RNG.shuffle(colors)
    in_forbidden = np.zeros((h, w), dtype=bool)
    out_forbidden = np.zeros((h, w), dtype=bool)
    records: list[dict] = []

    priority = {"fixed": 0, "corner": 1, "one": 2, "none": 3}
    for branch in sorted(specs, key=lambda name: priority[name]):
        rec = _candidate(branch, colors[len(records)], h, w, in_forbidden, out_forbidden)
        if rec is None:
            return None
        records.append(rec)
        _mark_forbidden(in_forbidden, rec["in_coords"])
        _mark_forbidden(out_forbidden, rec["out_coords"])

    grid = _render(records, bg, h, w, "in_coords")
    if not _validate_sample(grid, records, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
