from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
STAMP_COLORS = np.array([1, 2, 3, 4, 6, 7, 9], dtype=int)
OPS = ("H", "V", "R")
TYPE_ORDER = ("H", "V", "R", "I")
MARKERS_TO_OP = {
    frozenset({"tl", "tr"}): "H",
    frozenset({"tl", "bl"}): "V",
    frozenset({"tl", "br"}): "R",
    frozenset({"tl"}): "I",
}
RNG = np.random.default_rng()
GEN_RETRIES = 300


def _neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 8:
        return (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    return ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in _neighbors(connectivity):
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _transform(mask: np.ndarray, op: str) -> np.ndarray:
    if op == "H":
        return np.flipud(mask)
    if op == "V":
        return np.fliplr(mask)
    if op == "R":
        return np.rot90(mask, 2)
    return mask.copy()


def _symmetry_diffs(mask: np.ndarray) -> dict[str, int]:
    return {op: int(np.count_nonzero(mask != _transform(mask, op))) for op in OPS}


def _classify_mask(mask: np.ndarray) -> tuple[str, dict[str, int]]:
    diffs = _symmetry_diffs(mask)
    best = min(OPS, key=lambda op: (diffs[op], OPS.index(op)))
    if diffs[best] <= 2:
        return best, diffs
    return "I", diffs


def _repair_mask(mask: np.ndarray, op: str) -> np.ndarray:
    if op == "I":
        return mask.copy()
    return mask | _transform(mask, op)


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask, connectivity=4)
    return len(comps) == 1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(grid == GREY, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        oh, ow = sub.shape
        if oh < 5 or ow < 5:
            continue
        border = np.zeros((oh, ow), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.array_equal(sub == GREY, border):
            continue

        inside = sub[1:-1, 1:-1]
        if inside.shape[0] < 3 or inside.shape[1] < 3:
            continue
        if not np.all((inside == BLACK) | (inside == AZURE)):
            continue

        corners = {
            "tl": (0, 0),
            "tr": (0, inside.shape[1] - 1),
            "bl": (inside.shape[0] - 1, 0),
            "br": (inside.shape[0] - 1, inside.shape[1] - 1),
        }
        marks = frozenset(name for name, (rr, cc) in corners.items() if int(inside[rr, cc]) == AZURE)
        if int(np.count_nonzero(inside == AZURE)) != len(marks):
            continue
        op = MARKERS_TO_OP.get(marks)
        if op is None:
            continue

        frames.append(
            {
                "op": op,
                "inner_r0": int(r0 + 1),
                "inner_c0": int(c0 + 1),
                "shape": tuple(map(int, inside.shape)),
                "sort_key": (int(r0), int(c0)),
            }
        )
    return frames


def _detect_stamps(grid: np.ndarray) -> list[dict]:
    mask = (grid != BLACK) & (grid != GREY) & (grid != AZURE)
    stamps: list[dict] = []
    for coords in _components(mask, connectivity=8):
        r0, c0, r1, c1 = _bbox(coords)
        rel = coords - np.array([r0, c0], dtype=int)
        shape = (int(r1 - r0 + 1), int(c1 - c0 + 1))
        patch = _mask_from_coords(rel, shape)
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1
        color = colors.pop()
        op, _diffs = _classify_mask(patch)
        stamps.append(
            {
                "op": op,
                "color": int(color),
                "coords": coords,
                "shape": shape,
                "mask": patch,
                "repaired": _repair_mask(patch, op),
                "sort_key": (int(r0), int(c0)),
            }
        )
    return stamps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    stamps = _detect_stamps(g)
    frames = _detect_frames(g)
    assert stamps and frames

    groups_s: dict[tuple[str, int, int], list[dict]] = {}
    groups_f: dict[tuple[str, int, int], list[dict]] = {}
    for stamp in stamps:
        groups_s.setdefault((stamp["op"], *stamp["shape"]), []).append(stamp)
    for frame in frames:
        groups_f.setdefault((frame["op"], *frame["shape"]), []).append(frame)

    assert set(groups_s) == set(groups_f)
    for key in groups_s:
        assert len(groups_s[key]) == len(groups_f[key])

    for stamp in stamps:
        out[stamp["coords"][:, 0], stamp["coords"][:, 1]] = BLACK
    for frame in frames:
        r0 = frame["inner_r0"]
        c0 = frame["inner_c0"]
        h, w = frame["shape"]
        out[r0 : r0 + h, c0 : c0 + w] = BLACK

    for key in sorted(groups_s):
        srcs = sorted(groups_s[key], key=lambda item: item["sort_key"])
        dsts = sorted(groups_f[key], key=lambda item: item["sort_key"])
        for stamp, frame in zip(srcs, dsts):
            rr, cc = np.nonzero(stamp["repaired"])
            out[frame["inner_r0"] + rr, frame["inner_c0"] + cc] = int(stamp["color"])

    return out


def _map_cell(h: int, w: int, r: int, c: int, op: str) -> tuple[int, int]:
    if op == "H":
        return h - 1 - r, c
    if op == "V":
        return r, w - 1 - c
    if op == "R":
        return h - 1 - r, w - 1 - c
    return r, c


def _orbits(h: int, w: int, op: str) -> list[list[tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            orbit = tuple(sorted({(r, c), _map_cell(h, w, r, c, op)}))
            if orbit not in seen:
                seen.add(orbit)
                out.append(list(orbit))
    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    seed_r = int(RNG.integers(0, h))
    seed_c = int(RNG.integers(0, w))
    mask[seed_r, seed_c] = True
    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in _neighbors(4):
                nr = int(r) + dr
                nc = int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = list(frontier)[int(RNG.integers(0, len(frontier)))]
        mask[nr, nc] = True
    return mask


def _make_near_symmetric(op: str) -> np.ndarray:
    for _ in range(4000):
        h = int(RNG.integers(3, 6))
        w = int(RNG.integers(3, 6))
        orbit_list = _orbits(h, w, op)
        pick_count = int(RNG.integers(max(3, len(orbit_list) // 2), len(orbit_list) + 1))
        chosen = RNG.choice(len(orbit_list), size=pick_count, replace=False)
        mask = np.zeros((h, w), dtype=bool)
        for idx in chosen:
            for r, c in orbit_list[int(idx)]:
                mask[r, c] = True
        diffs = _symmetry_diffs(mask)
        if (
            int(mask.sum()) < 5
            or int(mask.sum()) >= h * w
            or not _is_connected(mask)
            or not _touches_all_sides(mask)
            or diffs[op] != 0
            or not all(diffs[other] > 0 for other in OPS if other != op)
        ):
            continue

        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if mask[r, c] and _map_cell(h, w, r, c, op) != (r, c)
        ]
        RNG.shuffle(candidates)
        for r, c in candidates:
            broken = mask.copy()
            broken[r, c] = False
            label, broken_diffs = _classify_mask(broken)
            if (
                _is_connected(broken)
                and _touches_all_sides(broken)
                and label == op
                and broken_diffs[op] == 2
                and all(broken_diffs[other] > broken_diffs[op] for other in OPS if other != op)
            ):
                return broken
    raise RuntimeError(f"failed to build near-symmetric mask for {op}")


def _make_asymmetric() -> np.ndarray:
    for _ in range(4000):
        h = int(RNG.integers(3, 6))
        w = int(RNG.integers(3, 6))
        area = int(RNG.integers(max(5, (h + w) // 2), h * w))
        mask = _grow_connected_mask(h, w, area)
        if mask is None or int(mask.sum()) >= h * w:
            continue
        label, diffs = _classify_mask(mask)
        if label == "I" and min(diffs.values()) > 2 and _touches_all_sides(mask):
            return mask
    raise RuntimeError("failed to build asymmetric mask")


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.tobytes()


def _build_mask_library() -> dict[str, list[np.ndarray]]:
    lib: dict[str, list[np.ndarray]] = {op: [] for op in TYPE_ORDER}
    seen: dict[str, set[tuple[tuple[int, int], bytes]]] = {op: set() for op in TYPE_ORDER}
    targets = {"H": 10, "V": 10, "R": 10, "I": 12}
    for op in TYPE_ORDER:
        maker = _make_asymmetric if op == "I" else (lambda cur=op: _make_near_symmetric(cur))
        while len(lib[op]) < targets[op]:
            mask = maker()
            key = _mask_key(mask)
            if key in seen[op]:
                continue
            seen[op].add(key)
            lib[op].append(mask)
    return lib


MASK_LIBRARY = _build_mask_library()


def _marker_names_for_op(op: str) -> tuple[str, ...]:
    for marks, mapped in MARKERS_TO_OP.items():
        if mapped == op:
            return tuple(sorted(marks))
    raise AssertionError(op)


def _frame_patch(inner_shape: tuple[int, int], op: str) -> np.ndarray:
    h, w = inner_shape
    patch = np.full((h + 2, w + 2), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    corners = {
        "tl": (1, 1),
        "tr": (1, w),
        "bl": (h, 1),
        "br": (h, w),
    }
    for name in _marker_names_for_op(op):
        rr, cc = corners[name]
        patch[rr, cc] = AZURE
    return patch


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _place_in_region(
    occupied: np.ndarray,
    region: tuple[int, int, int, int],
    shape: tuple[int, int],
    attempts: int = 120,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = region
    h, w = shape
    if r1 - r0 + 1 < h or c1 - c0 + 1 < w:
        return None
    max_r = r1 - h + 1
    max_c = c1 - w + 1
    for _ in range(attempts):
        top = int(RNG.integers(r0, max_r + 1))
        left = int(RNG.integers(c0, max_c + 1))
        if _can_place(occupied, top, left, h, w):
            _mark(occupied, top, left, h, w)
            return top, left
    return None


def _draw_stamp(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rr, cc = np.nonzero(mask)
    grid[top + rr, left + cc] = int(color)


def _attempt_generate() -> np.ndarray | None:
    count = int(RNG.integers(2, 5))
    types = list(RNG.choice(np.array(TYPE_ORDER, dtype=object), size=count, replace=False))
    colors = [int(c) for c in RNG.choice(STAMP_COLORS, size=count, replace=False)]

    stamps: list[dict] = []
    for op, color in zip(types, colors):
        mask = MASK_LIBRARY[op][int(RNG.integers(0, len(MASK_LIBRARY[op])))]
        stamps.append({"op": str(op), "color": color, "mask": mask, "shape": tuple(map(int, mask.shape))})

    H = int(RNG.integers(20, 29))
    W = int(RNG.integers(24, 35))
    mid = W // 2
    left_region = (1, 1, H - 2, mid - 2)
    right_region = (1, mid + 1, H - 2, W - 2)
    occupied = np.zeros((H, W), dtype=bool)
    grid = np.full((H, W), BLACK, dtype=int)

    order = list(range(count))
    RNG.shuffle(order)

    for idx in order:
        stamp = stamps[idx]
        pos = _place_in_region(occupied, left_region, stamp["shape"])
        if pos is None:
            return None
        stamp["pos"] = pos
        _draw_stamp(grid, pos[0], pos[1], stamp["mask"], stamp["color"])

    for idx in order:
        stamp = stamps[idx]
        fh = stamp["shape"][0] + 2
        fw = stamp["shape"][1] + 2
        pos = _place_in_region(occupied, right_region, (fh, fw))
        if pos is None:
            return None
        patch = _frame_patch(stamp["shape"], stamp["op"])
        r0, c0 = pos
        grid[r0 : r0 + fh, c0 : c0 + fw] = np.where(patch != BLACK, patch, grid[r0 : r0 + fh, c0 : c0 + fw])

    try:
        out = main(grid)
    except Exception:
        return None
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
