import random
from collections import deque

import numpy as np


_AXIS_ORDER = {"v": 0, "h": 1, "d": 2, "a": 3}
_AXIS_KINDS = ("v", "h", "d", "a")
_GEN_RETRIES = 400
_LOCAL_RETRIES = 200
_GRID_H_RANGE = (12, 18)
_GRID_W_RANGE = (17, 26)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), max(rs), min(cs), max(cs)


def _parse_legend(row: np.ndarray, bg: int) -> dict[int, int]:
    mapping: dict[int, int] = {}
    c = 0
    w = len(row)
    while c < w - 1:
        if int(row[c]) != bg and int(row[c + 1]) != bg:
            mapping[int(row[c])] = int(row[c + 1])
            c += 2
        else:
            c += 1
    return mapping


def _reflect_point(r: int, c: int, kind: str, param: int) -> tuple[int, int]:
    if kind == "v":
        return int(r), int(param - c)
    if kind == "h":
        return int(param - r), int(c)
    if kind == "d":
        return int(c + param), int(r - param)
    return int(param - c), int(param - r)


def _axis_candidates(h: int, w: int):
    for param in range(2 * w - 1):
        yield "v", int(param)
    for param in range(2 * h - 1):
        yield "h", int(param)
    for param in range(-(w - 1), h):
        yield "d", int(param)
    for param in range(h + w - 1):
        yield "a", int(param)


def _evaluate_axis(
    coords: set[tuple[int, int]],
    axis: tuple[str, int],
    grid: np.ndarray,
    bg: int,
    forbid_top_row: bool,
):
    h, w = grid.shape
    kind, param = axis
    reflected: set[tuple[int, int]] = set()
    for r, c in coords:
        rr, cc = _reflect_point(r, c, kind, param)
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if forbid_top_row and rr == 0:
            return None
        if (rr, cc) not in coords and int(grid[rr, cc]) != bg:
            return None
        reflected.add((rr, cc))

    added = reflected - coords
    union = coords | reflected
    r0, r1, c0, c1 = _bbox(union)
    return {
        "axis": axis,
        "added": added,
        "add_count": len(added),
        "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
    }


def _rank_axes(
    coords: list[tuple[int, int]] | set[tuple[int, int]],
    grid: np.ndarray,
    bg: int,
    forbid_top_row: bool = True,
):
    coord_set = set(coords)
    infos = []
    for axis in _axis_candidates(*grid.shape):
        info = _evaluate_axis(coord_set, axis, grid, bg, forbid_top_row)
        if info is not None:
            infos.append(info)
    infos.sort(
        key=lambda info: (
            int(info["add_count"]),
            int(info["bbox_area"]),
            _AXIS_ORDER[info["axis"][0]],
            int(info["axis"][1]),
        )
    )
    return infos


def _best_axis_and_missing(
    coords: list[tuple[int, int]] | set[tuple[int, int]],
    grid: np.ndarray,
    bg: int,
    forbid_top_row: bool = True,
):
    infos = _rank_axes(coords, grid, bg, forbid_top_row)
    bounded = [info for info in infos if 1 <= int(info["add_count"]) <= 3]
    if bounded:
        return bounded[0]
    return infos[0] if infos else None


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 4

    bg = _most_common_color(g)
    mapping = _parse_legend(g[0], bg)
    assert mapping, "legend row must contain at least one color pair"

    mask = g != bg
    mask[0, :] = False
    comps = _components(mask)
    assert comps, "expected at least one foreground object"

    out = g.copy()
    out[0, :] = bg

    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each object must be monochrome"
        obj_color = int(next(iter(colors)))
        assert obj_color in mapping, "every object color must appear in the legend"
        repair_color = int(mapping[obj_color])

        info = _best_axis_and_missing(comp, g, bg, forbid_top_row=True)
        assert info is not None, "failed to infer a symmetry axis"
        for r, c in info["added"]:
            out[r, c] = repair_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find each almost-symmetric object, infer its missing reflected cells, and
    fill those cells with the repair color specified in the top legend row.
    """
    return _solve_grid(grid)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    pts = int(mask.sum())
    return pts > 0 and len(_components(mask)) == 1


def _reflect_local(r: int, c: int, kind: str, h: int, w: int) -> tuple[int, int]:
    if kind == "v":
        return int(r), int(w - 1 - c)
    if kind == "h":
        return int(h - 1 - r), int(c)
    if kind == "d":
        return int(c), int(r)
    return int(w - 1 - c), int(h - 1 - r)


def _in_domain(r: int, c: int, kind: str, h: int, w: int) -> bool:
    if kind == "v":
        return c <= (w - 1) // 2
    if kind == "h":
        return r <= (h - 1) // 2
    if kind == "d":
        return r <= c
    return r + c <= w - 1


def _seed_choices(kind: str, h: int, w: int) -> list[tuple[int, int]]:
    if kind == "v":
        bc = w // 2 if w % 2 else w // 2 - 1
        return [(r, bc) for r in range(h)]
    if kind == "h":
        br = h // 2 if h % 2 else h // 2 - 1
        return [(br, c) for c in range(w)]
    if kind == "d":
        return [(i, i) for i in range(h)]
    return [(i, w - 1 - i) for i in range(h)]


def _pair_orbits(mask: np.ndarray, kind: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r, c in map(tuple, np.argwhere(mask)):
        if (r, c) in seen:
            continue
        rr, cc = _reflect_local(int(r), int(c), kind, h, w)
        if not mask[rr, cc]:
            return []
        seen.add((int(r), int(c)))
        seen.add((int(rr), int(cc)))
        if (int(r), int(c)) != (int(rr), int(cc)):
            orbits.append(((int(r), int(c)), (int(rr), int(cc))))
    return orbits


def _unique_zero_axis(mask: np.ndarray, kind: str) -> bool:
    h, w = mask.shape
    pad = 2
    grid = np.zeros((h + 2 * pad, w + 2 * pad), dtype=int)
    coords = []
    for r, c in map(tuple, np.argwhere(mask)):
        rr, cc = pad + int(r), pad + int(c)
        grid[rr, cc] = 1
        coords.append((rr, cc))
    zeros = [info for info in _rank_axes(coords, grid, 0, forbid_top_row=False) if int(info["add_count"]) == 0]
    return len(zeros) == 1 and zeros[0]["axis"][0] == kind


def _validate_removed(mask: np.ndarray, missing: set[tuple[int, int]], kind: str) -> bool:
    h, w = mask.shape
    pad = 2
    grid = np.zeros((h + 2 * pad, w + 2 * pad), dtype=int)
    coords = []
    for r, c in map(tuple, np.argwhere(mask)):
        rr, cc = pad + int(r), pad + int(c)
        grid[rr, cc] = 1
        coords.append((rr, cc))

    infos = [info for info in _rank_axes(coords, grid, 0, forbid_top_row=False) if 1 <= int(info["add_count"]) <= 3]
    if not infos:
        return False
    best = infos[0]
    min_add = int(best["add_count"])
    if len([info for info in infos if int(info["add_count"]) == min_add]) != 1:
        return False
    expected = {(pad + r, pad + c) for r, c in missing}
    return best["axis"][0] == kind and best["added"] == expected


def _make_symmetric_mask(kind: str) -> np.ndarray | None:
    for _ in range(_LOCAL_RETRIES):
        if kind in ("v", "h"):
            h = random.randint(4, 8)
            w = random.randint(4, 8)
        else:
            h = w = random.randint(4, 7)

        domain = [(r, c) for r in range(h) for c in range(w) if _in_domain(r, c, kind, h, w)]
        seed = random.choice(_seed_choices(kind, h, w))
        cells = {seed}
        target = random.randint(4, min(11, len(domain)))

        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and _in_domain(nr, nc, kind, h, w) and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            frontier = list(dict.fromkeys(frontier))
            if not frontier:
                break
            cells.add(random.choice(frontier))
            if len(cells) >= 5 and random.random() < 0.12:
                break

        if len(cells) < 4:
            continue

        full = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            full[r, c] = True
            rr, cc = _reflect_local(r, c, kind, h, w)
            full[rr, cc] = True
        full = _crop_mask(full)

        if not _is_connected(full):
            continue
        if min(full.shape) < 3:
            continue
        if not (8 <= int(full.sum()) <= 24):
            continue
        if len(_pair_orbits(full, kind)) < 3:
            continue
        if not _unique_zero_axis(full, kind):
            continue
        return full
    return None


def _remove_cells(
    full: np.ndarray, kind: str, missing_count: int
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    orbits = _pair_orbits(full, kind)
    if len(orbits) < missing_count:
        return None

    for _ in range(_LOCAL_RETRIES):
        observed = full.copy()
        missing: set[tuple[int, int]] = set()
        for p, q in random.sample(orbits, missing_count):
            rem = p if random.random() < 0.5 else q
            observed[rem] = False
            missing.add(rem)
        if not _is_connected(observed):
            continue
        if not _validate_removed(observed, missing, kind):
            continue
        return observed, missing
    return None


def _make_object(kind: str):
    for _ in range(_LOCAL_RETRIES):
        full = _make_symmetric_mask(kind)
        if full is None:
            continue
        missing_count = random.choice((1, 2, 3))
        removed = _remove_cells(full, kind, missing_count)
        if removed is None:
            continue
        observed, missing = removed
        return {
            "kind": kind,
            "full": full,
            "observed": observed,
            "missing": missing,
        }
    return None


def _legend_positions(width: int, count: int) -> list[int] | None:
    gaps = [random.randint(0, 1) for _ in range(max(0, count - 1))]
    need = 2 * count + sum(gaps)
    if need > width:
        return None
    start = random.randint(0, width - need)
    out = []
    c = start
    for i in range(count):
        out.append(c)
        c += 2
        if i < count - 1:
            c += gaps[i]
    return out


def _place_objects(objects: list[dict], h: int, w: int):
    order = sorted(range(len(objects)), key=lambda i: int(objects[i]["full"].sum()), reverse=True)
    occupied = np.zeros((h, w), dtype=bool)
    positions = [None] * len(objects)

    for idx in order:
        mask = objects[idx]["full"]
        mh, mw = mask.shape
        if h - mh < 2 or w - mw < 0:
            return None
        placed = False
        for _ in range(_LOCAL_RETRIES):
            r0 = random.randint(2, h - mh)
            c0 = random.randint(0, w - mw)
            rr0 = max(1, r0 - 1)
            rr1 = min(h, r0 + mh + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + mw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            positions[idx] = (r0, c0)
            placed = True
            break
        if not placed:
            return None
    return positions


def _build_case():
    object_count = random.randint(2, 4)
    kinds = list(_AXIS_KINDS)
    random.shuffle(kinds)
    kinds = kinds[:object_count]

    objects = []
    for kind in kinds:
        obj = _make_object(kind)
        if obj is None:
            return None
        objects.append(obj)

    legend_cols = _legend_positions(_GRID_W_RANGE[1], object_count)
    if legend_cols is None:
        return None

    heights = [obj["full"].shape[0] for obj in objects]
    widths = [obj["full"].shape[1] for obj in objects]
    min_h = max(_GRID_H_RANGE[0], max(heights) + 3, 3 + (sum(heights) + 1) // 2)
    min_w = max(_GRID_W_RANGE[0], max(widths), 2 * object_count + 1, 2 + (sum(widths) + 1) // 2)
    if min_h > _GRID_H_RANGE[1] or min_w > _GRID_W_RANGE[1]:
        return None

    h = random.randint(min_h, _GRID_H_RANGE[1])
    w = random.randint(min_w, _GRID_W_RANGE[1])
    legend_cols = _legend_positions(w, object_count)
    if legend_cols is None:
        return None

    positions = _place_objects(objects, h, w)
    if positions is None:
        return None

    colors = list(range(10))
    random.shuffle(colors)
    bg = colors.pop()
    obj_colors = [colors.pop() for _ in range(object_count)]
    repair_colors = [colors.pop() for _ in range(object_count)]

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    legend_order = list(range(object_count))
    random.shuffle(legend_order)
    for idx, c0 in zip(legend_order, legend_cols):
        inp[0, c0] = obj_colors[idx]
        inp[0, c0 + 1] = repair_colors[idx]

    for idx, obj in enumerate(objects):
        r0, c0 = positions[idx]
        for r, c in map(tuple, np.argwhere(obj["observed"])):
            inp[r0 + int(r), c0 + int(c)] = obj_colors[idx]
        for r, c in map(tuple, np.argwhere(obj["full"])):
            color = repair_colors[idx] if (int(r), int(c)) in obj["missing"] else obj_colors[idx]
            out[r0 + int(r), c0 + int(c)] = color

    if _most_common_color(inp) != bg:
        return None
    if not np.array_equal(_solve_grid(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _build_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
