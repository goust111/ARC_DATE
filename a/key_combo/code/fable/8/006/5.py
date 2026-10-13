from __future__ import annotations

import random
from collections import Counter, deque
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


YELLOW = 4
GREEN = 3
OBJECT_COLORS = (1, 2, 5, 6, 7, 8, 9)
ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


def _load_shared_utils():
    try:
        if ARC_UTILS_PATH.exists():
            mod = SourceFileLoader("arc_utils_fable_8_006_5", str(ARC_UTILS_PATH)).load_module()
            return mod.get_bg, mod.components_mask
    except Exception:
        pass
    return None, None


_shared_get_bg, _shared_components_mask = _load_shared_utils()


def _get_bg_local(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def get_bg(grid: np.ndarray) -> int:
    if _shared_get_bg is not None:
        return int(_shared_get_bg(grid))
    return _get_bg_local(grid)


def _components_mask_local(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    if _shared_components_mask is not None:
        return _shared_components_mask(mask, connectivity=connectivity)
    return _components_mask_local(mask, connectivity=connectivity)


def _border_segment_info(coords: np.ndarray, shape: tuple[int, int]) -> dict | None:
    h, w = map(int, shape)
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    if not all(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
        return None

    if len(rows) == 1:
        row = rows[0]
        if row not in (0, h - 1):
            return None
        c0, c1 = cols[0], cols[-1]
        if len(coords) != c1 - c0 + 1:
            return None
        if {(int(r), int(c)) for r, c in coords} != {(row, c) for c in range(c0, c1 + 1)}:
            return None
        return {
            "orientation": "horizontal",
            "edge": "top" if row == 0 else "bottom",
            "length": int(len(coords)),
            "span": (int(c0), int(c1)),
            "coords": coords,
        }

    if len(cols) == 1:
        col = cols[0]
        if col not in (0, w - 1):
            return None
        r0, r1 = rows[0], rows[-1]
        if len(coords) != r1 - r0 + 1:
            return None
        if {(int(r), int(c)) for r, c in coords} != {(r, col) for r in range(r0, r1 + 1)}:
            return None
        return {
            "orientation": "vertical",
            "edge": "left" if col == 0 else "right",
            "length": int(len(coords)),
            "span": (int(r0), int(r1)),
            "coords": coords,
        }
    return None


def _stripe_indices(size: int, edge: str, period: int) -> set[int]:
    if edge in ("top", "left"):
        return set(range(0, int(size), int(period)))
    if edge in ("bottom", "right"):
        return set(range(int(size) - 1, -1, -int(period)))
    raise ValueError(f"unknown edge: {edge}")


def _detect_spec(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    yellow_comps = components_mask(g == YELLOW, connectivity=4)
    assert len(yellow_comps) == 2, "expected exactly two yellow border segments"

    infos = []
    for comp in yellow_comps:
        info = _border_segment_info(comp, g.shape)
        assert info is not None, "yellow components must be straight border segments"
        infos.append(info)

    horizontals = [info for info in infos if info["orientation"] == "horizontal"]
    verticals = [info for info in infos if info["orientation"] == "vertical"]
    assert len(horizontals) == 1 and len(verticals) == 1, "segments must be perpendicular"
    hseg = horizontals[0]
    vseg = verticals[0]
    assert hseg["length"] != vseg["length"], "segment lengths must differ"

    segment_mask = np.zeros_like(g, dtype=bool)
    for info in infos:
        pts = info["coords"]
        segment_mask[pts[:, 0], pts[:, 1]] = True

    return {
        "bg": int(bg),
        "horizontal": hseg,
        "vertical": vseg,
        "segment_mask": segment_mask,
        "stripe_rows": _stripe_indices(g.shape[0], hseg["edge"], hseg["length"]),
        "stripe_cols": _stripe_indices(g.shape[1], vseg["edge"], vseg["length"]),
    }


def _component_hits(coords: np.ndarray, stripe_rows: set[int], stripe_cols: set[int]) -> tuple[bool, bool, bool]:
    hit_h = any(int(r) in stripe_rows for r, _c in coords)
    hit_v = any(int(c) in stripe_cols for _r, c in coords)
    same_pixel_both = any(int(r) in stripe_rows and int(c) in stripe_cols for r, c in coords)
    return bool(hit_h), bool(hit_v), bool(same_pixel_both)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor scattered objects by how many invisible stripe families from two yellow edge segments pass through them.

    Input structure:
    - The background is the dominant color.
    - Exactly two yellow straight segments lie flush on perpendicular grid edges, one horizontal and one vertical.
    - Interior foreground objects are separated 4-connected components in a single non-yellow color.

    Rule:
    - Each yellow segment defines an invisible family of full-grid lines parallel to its edge, starting from that edge and repeating every number of cells equal to the segment length.
    - For each interior object, test its actual pixels against the horizontal and vertical line families.
    - Recolor the whole object green if both families hit at least one of its pixels, yellow if exactly one family hits, and leave it unchanged if neither hits; preserve the control segments and background.

    Key insight:
    - The segment positions and spans are decoys: only their lengths matter, because those lengths encode the row and column stripe periods.
    """
    g = np.asarray(grid, dtype=int)
    spec = _detect_spec(g)
    out = g.copy()
    obj_mask = (g != spec["bg"]) & ~spec["segment_mask"]

    for comp in components_mask(obj_mask, connectivity=4):
        hit_h, hit_v, _same_pixel_both = _component_hits(comp, spec["stripe_rows"], spec["stripe_cols"])
        if hit_h and hit_v:
            color = GREEN
        elif hit_h or hit_v:
            color = YELLOW
        else:
            continue
        out[comp[:, 0], comp[:, 1]] = int(color)

    return out


SHAPES: dict[str, tuple[tuple[int, int], ...]] = {
    "single": ((0, 0),),
    "h2": ((0, 0), (0, 1)),
    "v2": ((0, 0), (1, 0)),
    "h3": ((0, 0), (0, 1), (0, 2)),
    "v3": ((0, 0), (1, 0), (2, 0)),
    "l3": ((0, 0), (1, 0), (1, 1)),
    "j3": ((0, 1), (1, 0), (1, 1)),
}


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _c in shape), 1 + max(c for _r, c in shape)


def _coords_from_origin(origin: tuple[int, int], shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0, c0 = origin
    return tuple((int(r0 + dr), int(c0 + dc)) for dr, dc in shape)


def _can_place(coords: tuple[tuple[int, int], ...], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if occupied[r, c]:
            return False
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if occupied[rr, cc]:
                    return False
    return True


def _place_object(
    occupied: np.ndarray,
    shape_names: tuple[str, ...],
    predicate,
) -> tuple[tuple[int, int], ...] | None:
    h, w = occupied.shape
    candidates: list[tuple[tuple[int, int], ...]] = []
    for name in shape_names:
        shape = SHAPES[name]
        sh, sw = _shape_dims(shape)
        for r0 in range(1, h - sh):
            for c0 in range(1, w - sw):
                coords = _coords_from_origin((r0, c0), shape)
                if _can_place(coords, occupied) and predicate(coords):
                    candidates.append(coords)
    if not candidates:
        return None
    coords = random.choice(candidates)
    for r, c in coords:
        occupied[r, c] = True
    return coords


def _wrong_span_hits(
    coords: tuple[tuple[int, int], ...],
    stripe_rows: set[int],
    stripe_cols: set[int],
    h_span_cols: set[int],
    v_span_rows: set[int],
) -> tuple[bool, bool]:
    hit_h = any(r in stripe_rows and c in h_span_cols for r, c in coords)
    hit_v = any(c in stripe_cols and r in v_span_rows for r, c in coords)
    return bool(hit_h), bool(hit_v)


def _recolor(color: int, hit_h: bool, hit_v: bool) -> int:
    if hit_h and hit_v:
        return GREEN
    if hit_h or hit_v:
        return YELLOW
    return int(color)


def _expected_output(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    for obj in meta["objects"]:
        arr = np.array(obj, dtype=int)
        hit_h, hit_v, _same_pixel_both = _component_hits(arr, meta["stripe_rows"], meta["stripe_cols"])
        color = _recolor(meta["obj_color"], hit_h, hit_v)
        for r, c in obj:
            out[r, c] = color
    return out


def _semantic_validate(grid: np.ndarray, meta: dict) -> bool:
    expected = _expected_output(grid, meta)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    obj_mask = (grid != meta["bg"]) & ~meta["segment_mask"]
    comps = components_mask(obj_mask, connectivity=4)
    seen = {frozenset(map(tuple, comp.tolist())) for comp in comps}
    built = {frozenset(obj) for obj in meta["objects"]}
    if seen != built:
        return False

    hit_pairs = set()
    has_split_both = False
    span_rule_diff = False
    for obj in meta["objects"]:
        arr = np.array(obj, dtype=int)
        hit_h, hit_v, same_pixel_both = _component_hits(arr, meta["stripe_rows"], meta["stripe_cols"])
        hit_pairs.add((hit_h, hit_v))
        wrong_h, wrong_v = _wrong_span_hits(
            obj, meta["stripe_rows"], meta["stripe_cols"], meta["h_span_cols"], meta["v_span_rows"]
        )
        span_rule_diff |= (wrong_h, wrong_v) != (hit_h, hit_v)
        has_split_both |= hit_h and hit_v and not same_pixel_both

    if hit_pairs != {(False, False), (True, False), (False, True), (True, True)}:
        return False
    if not has_split_both or not span_rule_diff:
        return False
    if np.array_equal(grid, actual):
        return False
    return True


def _build_attempt() -> tuple[np.ndarray, dict] | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    bg = 0
    obj_color = random.choice(OBJECT_COLORS)

    period_choices = list(range(2, min(7, h - 2, w - 2)))
    if len(period_choices) < 2:
        return None
    h_period, v_period = random.sample(period_choices, 2)
    h_edge = random.choice(("top", "bottom"))
    v_edge = random.choice(("left", "right"))

    h_start = random.randint(1, w - h_period - 1)
    v_start = random.randint(1, h - v_period - 1)
    h_seg_cols = set(range(h_start, h_start + h_period))
    v_seg_rows = set(range(v_start, v_start + v_period))

    grid = np.full((h, w), bg, dtype=int)
    if h_edge == "top":
        grid[0, h_start : h_start + h_period] = YELLOW
    else:
        grid[h - 1, h_start : h_start + h_period] = YELLOW
    if v_edge == "left":
        grid[v_start : v_start + v_period, 0] = YELLOW
    else:
        grid[v_start : v_start + v_period, w - 1] = YELLOW

    segment_mask = grid == YELLOW
    occupied = segment_mask.copy()
    stripe_rows = _stripe_indices(h, h_edge, h_period)
    stripe_cols = _stripe_indices(w, v_edge, v_period)

    objects: list[tuple[tuple[int, int], ...]] = []

    def rows_of(coords):
        return {r for r, _c in coords}

    def cols_of(coords):
        return {c for _r, c in coords}

    def sig(coords):
        arr = np.array(coords, dtype=int)
        return _component_hits(arr, stripe_rows, stripe_cols)

    witnesses = [
        (
            ("l3", "j3", "h2", "v2"),
            lambda coords: (lambda a, b, same: a and b and not same)(*sig(coords)),
        ),
        (
            ("single", "h2", "h3", "l3"),
            lambda coords: (lambda a, b, _same: a and not b and cols_of(coords).isdisjoint(h_seg_cols))(*sig(coords)),
        ),
        (
            ("single", "v2", "v3", "j3"),
            lambda coords: (lambda a, b, _same: (not a) and b and rows_of(coords).isdisjoint(v_seg_rows))(*sig(coords)),
        ),
        (
            ("single", "h2", "v2", "h3", "v3"),
            lambda coords: (lambda a, b, _same: (not a) and (not b))(*sig(coords)),
        ),
    ]

    for shape_names, predicate in witnesses:
        coords = _place_object(occupied, shape_names, predicate)
        if coords is None:
            return None
        objects.append(coords)

    target_objects = random.randint(7, 11)
    categories = [
        lambda coords: (lambda a, b, _same: (not a) and (not b))(*sig(coords)),
        lambda coords: (lambda a, b, _same: a and not b)(*sig(coords)),
        lambda coords: (lambda a, b, _same: (not a) and b)(*sig(coords)),
        lambda coords: (lambda a, b, _same: a and b)(*sig(coords)),
    ]
    shape_banks = [
        ("single", "h2", "v2", "h3", "v3", "l3", "j3"),
        ("single", "h2", "h3", "l3", "j3"),
        ("single", "v2", "v3", "l3", "j3"),
        ("single", "h2", "v2", "l3", "j3"),
    ]
    misses = 0
    while len(objects) < target_objects and misses < 20:
        idx = random.randrange(4)
        coords = _place_object(occupied, shape_banks[idx], categories[idx])
        if coords is None:
            misses += 1
            continue
        objects.append(coords)
        misses = 0
    if len(objects) < 6:
        return None

    for obj in objects:
        for r, c in obj:
            grid[r, c] = obj_color

    meta = {
        "bg": bg,
        "obj_color": obj_color,
        "segment_mask": segment_mask,
        "stripe_rows": stripe_rows,
        "stripe_cols": stripe_cols,
        "h_span_cols": h_seg_cols,
        "v_span_rows": v_seg_rows,
        "objects": objects,
    }
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(400):
        attempt = _build_attempt()
        if attempt is None:
            continue
        grid, meta = attempt
        if _semantic_validate(grid, meta):
            return grid
    raise RuntimeError("generate_input failed after 400 attempts")
