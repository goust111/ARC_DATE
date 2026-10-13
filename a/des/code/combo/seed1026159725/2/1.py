import math
from collections import Counter, deque

import numpy as np


CORNER_ORDER = ("TL", "TR", "BL", "BR")
CORNER_INDEX = {name: idx for idx, name in enumerate(CORNER_ORDER)}
CORNER_SIGNS = {
    "TL": (-1, -1),
    "TR": (-1, 1),
    "BL": (1, -1),
    "BR": (1, 1),
}
SINGLE_MOVES = {
    "TL": (1, 1),
    "TR": (1, -1),
    "BL": (-1, 1),
    "BR": (-1, -1),
}
DOUBLE_MOVES = {
    frozenset(("TL", "TR")): (2, 0),
    frozenset(("BL", "BR")): (-2, 0),
    frozenset(("TL", "BL")): (0, 2),
    frozenset(("TR", "BR")): (0, -2),
}
TRIPLE_PATTERNS = [
    ("TR", "BL", "BR"),
    ("TL", "BL", "BR"),
    ("TL", "TR", "BR"),
    ("TL", "TR", "BL"),
]
BRANCH_PATTERNS = {
    1: [("TL",), ("TR",), ("BL",), ("BR",)],
    2: [("TL", "TR"), ("BL", "BR"), ("TL", "BL"), ("TR", "BR")],
    3: TRIPLE_PATTERNS,
}
MAX_ATTEMPTS = 400
MAX_SIZE = 30


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_positions(h: int, w: int) -> dict[str, tuple[int, int]]:
    return {
        "TL": (0, 0),
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }


def _movement_from_missing(missing: tuple[str, ...]) -> tuple[int, int]:
    missing = tuple(sorted(missing, key=lambda name: CORNER_INDEX[name]))
    miss_set = frozenset(missing)
    if len(missing) == 1:
        return SINGLE_MOVES[missing[0]]
    if len(missing) == 2:
        if miss_set not in DOUBLE_MOVES:
            raise AssertionError("two missing corners must lie on one side")
        return DOUBLE_MOVES[miss_set]
    if len(missing) == 3:
        kept = next(iter(set(CORNER_ORDER) - miss_set))
        sr, sc = CORNER_SIGNS[kept]
        return 3 * sr, 3 * sc
    raise AssertionError("expected 1 to 3 missing corners")


def _mask_from_missing(h: int, w: int, missing: tuple[str, ...]) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    for name, (r, c) in _corner_positions(h, w).items():
        if name in missing:
            mask[r, c] = False
    return mask


def _analyze_component(grid: np.ndarray, bg: int, coords: np.ndarray) -> dict:
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        raise AssertionError("each object must be monochrome")
    color = colors.pop()
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2:
        raise AssertionError("objects must have a meaningful rectangular bbox")

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    missing = tuple(
        name
        for name, (rr, cc) in _corner_positions(h, w).items()
        if not local[rr, cc]
    )
    if len(missing) == 3 and (h < 3 or w < 3):
        raise AssertionError("three-missing-corner objects need at least 3x3 support")
    expected = _mask_from_missing(h, w, missing)
    if not np.array_equal(local, expected):
        raise AssertionError("object must be a solid rectangle minus corner pixels")

    dr, dc = _movement_from_missing(missing)
    patch = np.full((h, w), bg, dtype=int)
    patch[local] = color
    return {
        "coords": coords,
        "color": color,
        "top": r0,
        "left": c0,
        "patch": patch,
        "delta": (dr, dc),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each rectangle-minus-corners object according to its missing corners.

    Single missing corner: move 1 cell toward the opposite diagonal.
    Two missing corners on one side: move 2 cells away from that side.
    Three missing corners: move 3 cells toward the unique preserved corner.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    objects = [_analyze_component(g, bg, coords) for coords in _components_mask(g != bg)]
    out = g.copy()
    for obj in objects:
        for r, c in obj["coords"]:
            out[int(r), int(c)] = bg

    for obj in objects:
        patch = obj["patch"]
        ph, pw = patch.shape
        dr, dc = obj["delta"]
        nr = int(obj["top"] + dr)
        nc = int(obj["left"] + dc)
        if not (0 <= nr and 0 <= nc and nr + ph <= h and nc + pw <= w):
            raise AssertionError("translated object left the grid")
        mask = patch != bg
        out[nr : nr + ph, nc : nc + pw][mask] = int(obj["color"])
    return out


def _branch_of_pattern(pattern: tuple[str, ...]) -> int:
    return len(pattern)


def _pick_patterns(rng: np.random.Generator, n_objects: int) -> list[tuple[str, ...]]:
    pools = {branch: list(patterns) for branch, patterns in BRANCH_PATTERNS.items()}
    chosen: list[tuple[str, ...]] = []
    branch_counts = {1: 0, 2: 0, 3: 0}

    for branch in (1, 2, 3):
        idx = int(rng.integers(0, len(pools[branch])))
        chosen.append(pools[branch].pop(idx))
        branch_counts[branch] += 1

    while len(chosen) < n_objects:
        min_count = min(branch_counts.values())
        branch_options = [b for b, count in branch_counts.items() if count == min_count and pools[b]]
        if not branch_options:
            break
        branch = int(branch_options[int(rng.integers(0, len(branch_options)))])
        idx = int(rng.integers(0, len(pools[branch])))
        chosen.append(pools[branch].pop(idx))
        branch_counts[branch] += 1

    rng.shuffle(chosen)
    return chosen


def _make_patch(bg: int, color: int, h: int, w: int, missing: tuple[str, ...]) -> np.ndarray:
    patch = np.full((h, w), bg, dtype=int)
    patch[_mask_from_missing(h, w, missing)] = color
    return patch


def _reserved_rect(r0: int, c0: int, h: int, w: int, H: int, W: int) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - 1),
        max(0, c0 - 1),
        min(H, r0 + h + 1),
        min(W, c0 + w + 1),
    )


def _candidate_range(limit: int, size: int, delta: int) -> tuple[int, int]:
    lo = max(1, 1 - delta)
    hi = min(limit - size - 1, limit - size - 1 - delta)
    return int(lo), int(hi)


def _place_specs(rng: np.random.Generator, H: int, W: int, specs: list[dict]) -> bool:
    reserved = np.zeros((H, W), dtype=bool)
    order = sorted(range(len(specs)), key=lambda idx: -specs[idx]["reserve_area"])
    for idx in order:
        spec = specs[idx]
        h = int(spec["h"])
        w = int(spec["w"])
        dr = int(spec["delta"][0])
        dc = int(spec["delta"][1])
        r_lo, r_hi = _candidate_range(H, h, dr)
        c_lo, c_hi = _candidate_range(W, w, dc)
        if r_lo > r_hi or c_lo > c_hi:
            return False

        candidates = [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]
        rng.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            nr = r0 + dr
            nc = c0 + dc
            rects = [
                _reserved_rect(r0, c0, h, w, H, W),
                _reserved_rect(nr, nc, h, w, H, W),
            ]
            if any(reserved[rr0:rr1, cc0:cc1].any() for rr0, cc0, rr1, cc1 in rects):
                continue
            for rr0, cc0, rr1, cc1 in rects:
                reserved[rr0:rr1, cc0:cc1] = True
            spec["top"] = int(r0)
            spec["left"] = int(c0)
            placed = True
            break
        if not placed:
            return False
    return True


def _stamp(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    grid[top : top + patch.shape[0], left : left + patch.shape[1]][mask] = patch[mask]


def _manual_output(H: int, W: int, bg: int, specs: list[dict]) -> np.ndarray:
    out = np.full((H, W), bg, dtype=int)
    for spec in specs:
        dr, dc = spec["delta"]
        _stamp(out, spec["patch"], int(spec["top"] + dr), int(spec["left"] + dc), bg)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_objects = int(rng.integers(3, 7))
    bg = int(rng.integers(0, 10))
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    colors = palette[:n_objects]
    patterns = _pick_patterns(rng, n_objects)

    specs: list[dict] = []
    for color, missing in zip(colors, patterns):
        branch = _branch_of_pattern(missing)
        h = int(rng.integers(3, 7))
        w = int(rng.integers(3, 7))
        if branch == 3:
            h = max(h, 3)
            w = max(w, 3)
        delta = _movement_from_missing(missing)
        reserve_h = h + abs(delta[0]) + 2
        reserve_w = w + abs(delta[1]) + 2
        specs.append(
            {
                "color": int(color),
                "missing": missing,
                "h": h,
                "w": w,
                "delta": delta,
                "patch": _make_patch(bg, int(color), h, w, missing),
                "reserve_area": reserve_h * reserve_w,
            }
        )

    footprint = sum(int(spec["reserve_area"]) for spec in specs)
    min_h = max(
        max(int(spec["h"]) + abs(int(spec["delta"][0])) + 4 for spec in specs),
        int(math.ceil(math.sqrt(footprint))) - 1,
        14,
    )
    min_w = max(
        max(int(spec["w"]) + abs(int(spec["delta"][1])) + 4 for spec in specs),
        int(math.ceil(math.sqrt(footprint))) - 1,
        14,
    )
    if min_h > MAX_SIZE or min_w > MAX_SIZE:
        return None

    H = None
    W = None
    for _ in range(40):
        cand_h = int(rng.integers(min_h, min(MAX_SIZE, min_h + 5) + 1))
        cand_w = int(rng.integers(min_w, min(MAX_SIZE, min_w + 5) + 1))
        if cand_h * cand_w >= footprint + 24:
            H, W = cand_h, cand_w
            break
    if H is None or W is None:
        return None

    if not _place_specs(rng, H, W, specs):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for spec in specs:
        _stamp(inp, spec["patch"], int(spec["top"]), int(spec["left"]), bg)

    color_counts = Counter(map(int, inp.flat))
    if any(color != bg and color_counts[bg] <= count for color, count in color_counts.items()):
        return None

    expected = _manual_output(H, W, bg, specs)
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
