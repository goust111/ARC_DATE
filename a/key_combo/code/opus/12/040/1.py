from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
CRACK = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 500

WIDE_DIMS = ((4, 5), (4, 6), (4, 7), (5, 6), (5, 7))
TALL_DIMS = tuple((w, h) for h, w in WIDE_DIMS)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


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
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.tolist()))
    color, amount = counts.most_common(1)[0]
    assert list(counts.values()).count(amount) == 1, "majority color must be unique"
    return int(color)


def _detect_orientation_and_crack(region: np.ndarray) -> tuple[str, int]:
    h, w = region.shape
    crack_rows = [r for r in range(h) if np.all(region[r, :] == CRACK)]
    crack_cols = [c for c in range(w) if np.all(region[:, c] == CRACK)]

    if w > h:
        assert len(crack_rows) == 1, "wide object must have one full horizontal crack row"
        return "horizontal", int(crack_rows[0])
    if h > w:
        assert len(crack_cols) == 1, "tall object must have one full vertical crack column"
        return "vertical", int(crack_cols[0])

    if len(crack_rows) == 1 and len(crack_cols) != 1:
        return "horizontal", int(crack_rows[0])
    if len(crack_cols) == 1 and len(crack_rows) != 1:
        return "vertical", int(crack_cols[0])
    raise AssertionError("square objects are only allowed when the crack orientation is still unique")


def _paint_shifted_half(out: np.ndarray, coords: np.ndarray, color: int, dr: int, dc: int) -> None:
    for r, c in coords:
        rr = int(r + dr)
        cc = int(c + dc)
        if not _in_bounds(out.shape, rr, cc):
            continue
        assert out[rr, cc] == BG, "shifted halves must not collide"
        out[rr, cc] = int(color)


def _solve_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for comp in _components(grid != BG):
        vals = set(map(int, grid[comp[:, 0], comp[:, 1]].tolist()))
        assert CRACK in vals, "every foreground object must include the crack color"
        paint_colors = sorted(vals - {CRACK})
        assert len(paint_colors) == 2, "each object must use exactly two voting colors"

        r0, c0, r1, c1 = _bbox(comp)
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        orientation, crack_idx = _detect_orientation_and_crack(region)

        body_mask = (region != BG) & (region != CRACK)
        halves = _components(body_mask)
        assert len(halves) == 2, "erasing the crack must split the object into exactly two halves"

        crack_abs = r0 + crack_idx if orientation == "horizontal" else c0 + crack_idx
        half_specs = []
        for half in halves:
            global_coords = half + np.array([[r0, c0]])
            colors = grid[global_coords[:, 0], global_coords[:, 1]]
            majority = _majority_color(colors)
            if orientation == "horizontal":
                side = "upper" if int(global_coords[:, 0].max()) < crack_abs else "lower"
                shift = (-1, 0) if side == "upper" else (1, 0)
            else:
                side = "left" if int(global_coords[:, 1].max()) < crack_abs else "right"
                shift = (0, -1) if side == "left" else (0, 1)
            half_specs.append(
                {
                    "coords": global_coords,
                    "majority": majority,
                    "side": side,
                    "shift": shift,
                }
            )

        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "orientation": orientation,
                "crack_idx": crack_idx,
                "halves": half_specs,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each yellow crack, let each side vote for its local majority color, and slide the two recolored halves apart.

    Input structure:
    - A black background contains 2-3 separated 4-connected foreground objects.
    - Every object is a non-square filled rectangle made from exactly two non-black colors plus one full yellow crack line.
    - The crack spans the whole bbox horizontally in wide objects or vertically in tall objects, so deleting it splits the object into two connected halves.

    Rule:
    - Process each foreground object independently.
    - Remove the yellow crack pixels, split the remaining pixels into the two halves defined by that crack, and count the two object colors inside each half.
    - Recolor every pixel in a half to that half's unique majority color.
    - Move the two recolored halves one cell away from the crack: upper/lower for a horizontal crack, left/right for a vertical crack.
    - Leave all background pixels black; shifted pixels that would leave the grid are discarded.

    Key insight:
    - The yellow crack is both the physical split line and the voting boundary, so the same object can produce two different final colors depending on how the crack partitions its mixed pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    out = np.zeros_like(g)
    for obj in _solve_objects(g):
        for half in obj["halves"]:
            dr, dc = half["shift"]
            _paint_shifted_half(out, half["coords"], half["majority"], dr, dc)
    return out


def _interval_sep(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _cheb_gap_at_least(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    dr = _interval_sep(a[0], a[2], b[0], b[2])
    dc = _interval_sep(a[1], a[3], b[1], b[3])
    return max(dr, dc) >= gap


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded_bbox(top: int, left: int, h: int, w: int, orientation: str) -> tuple[int, int, int, int]:
    if orientation == "horizontal":
        return top - 1, left, top + h, left + w - 1
    return top, left - 1, top + h - 1, left + w


def _candidate_positions(
    shape: tuple[int, int],
    h: int,
    w: int,
    orientation: str,
    originals: list[tuple[int, int, int, int]],
    expanded: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    grid_h, grid_w = shape
    if orientation == "horizontal":
        rows = range(1, grid_h - h)
        cols = range(0, grid_w - w + 1)
    else:
        rows = range(0, grid_h - h + 1)
        cols = range(1, grid_w - w)

    out: list[tuple[int, int]] = []
    for top in rows:
        for left in cols:
            orig = (top, left, top + h - 1, left + w - 1)
            exp = _expanded_bbox(top, left, h, w, orientation)
            if any(not _cheb_gap_at_least(orig, old, 2) for old in originals):
                continue
            if any(_overlap(exp, old) for old in expanded):
                continue
            out.append((top, left))
    random.shuffle(out)
    return out


def _choose_two_colors() -> tuple[int, int]:
    a, b = random.sample(PALETTE, 2)
    return int(a), int(b)


def _target_majority_count(size: int) -> int:
    minority = random.randint(2, max(2, size // 3))
    if size - minority <= minority:
        minority = max(1, size // 2 - 1)
    return int(size - minority)


def _make_mixed_region(height: int, width: int, colors: tuple[int, int], majority: int) -> np.ndarray:
    other = colors[0] if colors[1] == majority else colors[1]
    region = np.fromfunction(
        lambda r, c: np.where((r + c + random.randint(0, 1)) % 2 == 0, colors[0], colors[1]),
        (height, width),
        dtype=int,
    ).astype(int)
    target = _target_majority_count(height * width)
    current = int(np.count_nonzero(region == majority))
    coords = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(coords)
    if current < target:
        for r, c in coords:
            if region[r, c] == other:
                region[r, c] = majority
                current += 1
                if current == target:
                    break
    elif current > target:
        for r, c in coords:
            if region[r, c] == majority:
                region[r, c] = other
                current -= 1
                if current == target:
                    break
    assert 0 < np.count_nonzero(region == majority) < region.size
    assert 0 < np.count_nonzero(region == other) < region.size
    return region


def _build_object_patch(spec: dict) -> tuple[np.ndarray, dict]:
    h = int(spec["h"])
    w = int(spec["w"])
    colors = spec["colors"]
    orientation = spec["orientation"]

    patch = np.zeros((h, w), dtype=int)
    oracle = {"halves": []}

    if orientation == "horizontal":
        crack = int(spec["crack_idx"])
        upper = _make_mixed_region(crack, w, colors, int(spec["majorities"][0]))
        lower = _make_mixed_region(h - crack - 1, w, colors, int(spec["majorities"][1]))
        patch[:crack, :] = upper
        patch[crack, :] = CRACK
        patch[crack + 1 :, :] = lower
        oracle["halves"].append({"slice": (slice(0, crack), slice(0, w)), "shift": (-1, 0), "majority": int(spec["majorities"][0])})
        oracle["halves"].append({"slice": (slice(crack + 1, h), slice(0, w)), "shift": (1, 0), "majority": int(spec["majorities"][1])})
    else:
        crack = int(spec["crack_idx"])
        left = _make_mixed_region(h, crack, colors, int(spec["majorities"][0]))
        right = _make_mixed_region(h, w - crack - 1, colors, int(spec["majorities"][1]))
        patch[:, :crack] = left
        patch[:, crack] = CRACK
        patch[:, crack + 1 :] = right
        oracle["halves"].append({"slice": (slice(0, h), slice(0, crack)), "shift": (0, -1), "majority": int(spec["majorities"][0])})
        oracle["halves"].append({"slice": (slice(0, h), slice(crack + 1, w)), "shift": (0, 1), "majority": int(spec["majorities"][1])})

    return patch, oracle


def _stamp_expected(grid: np.ndarray, patch: np.ndarray, top: int, left: int, oracle: dict) -> None:
    for half in oracle["halves"]:
        rs, cs = half["slice"]
        dr, dc = half["shift"]
        majority = int(half["majority"])
        coords = np.argwhere(patch[rs, cs] != BG)
        r_base = rs.start or 0
        c_base = cs.start or 0
        for rr, cc in coords:
            tr = top + r_base + int(rr) + dr
            tc = left + c_base + int(cc) + dc
            if _in_bounds(grid.shape, tr, tc):
                assert grid[tr, tc] == BG
                grid[tr, tc] = majority


def _stamp_shift_only(grid: np.ndarray, patch: np.ndarray, top: int, left: int, oracle: dict) -> None:
    for half in oracle["halves"]:
        rs, cs = half["slice"]
        dr, dc = half["shift"]
        view = patch[rs, cs]
        for rr, cc in np.argwhere(view != BG):
            tr = top + (rs.start or 0) + int(rr) + dr
            tc = left + (cs.start or 0) + int(cc) + dc
            if _in_bounds(grid.shape, tr, tc):
                assert grid[tr, tc] == BG
                grid[tr, tc] = int(view[rr, cc])


def _stamp_global_majority(grid: np.ndarray, patch: np.ndarray, top: int, left: int, oracle: dict) -> bool:
    body = patch[(patch != BG) & (patch != CRACK)]
    counts = Counter(map(int, body.tolist()))
    amount = max(counts.values())
    if list(counts.values()).count(amount) != 1:
        return False
    majority = int(counts.most_common(1)[0][0])
    for half in oracle["halves"]:
        rs, cs = half["slice"]
        dr, dc = half["shift"]
        for rr, cc in np.argwhere(patch[rs, cs] != BG):
            tr = top + (rs.start or 0) + int(rr) + dr
            tc = left + (cs.start or 0) + int(cc) + dc
            if _in_bounds(grid.shape, tr, tc):
                assert grid[tr, tc] == BG
                grid[tr, tc] = majority
    return True


def _stamp_recolor_without_shift(grid: np.ndarray, patch: np.ndarray, top: int, left: int, oracle: dict) -> None:
    for half in oracle["halves"]:
        rs, cs = half["slice"]
        majority = int(half["majority"])
        for rr, cc in np.argwhere(patch[rs, cs] != BG):
            tr = top + (rs.start or 0) + int(rr)
            tc = left + (cs.start or 0) + int(cc)
            assert grid[tr, tc] == BG
            grid[tr, tc] = majority


def _gen_object_specs(count: int) -> list[dict]:
    if count == 2:
        orientations = ["horizontal", "vertical"]
    else:
        orientations = ["horizontal", "vertical", random.choice(["horizontal", "vertical"])]
    random.shuffle(orientations)

    specs: list[dict] = []
    force_split = random.randrange(count)
    for i, orientation in enumerate(orientations):
        h, w = random.choice(WIDE_DIMS if orientation == "horizontal" else TALL_DIMS)
        if orientation == "horizontal":
            crack_idx = random.randint(1, h - 2)
        else:
            crack_idx = random.randint(1, w - 2)
        colors = _choose_two_colors()
        if i == force_split:
            majorities = (colors[0], colors[1])
        else:
            majorities = random.choice(
                [
                    (colors[0], colors[0]),
                    (colors[1], colors[1]),
                    (colors[0], colors[1]),
                    (colors[1], colors[0]),
                ]
            )
        specs.append(
            {
                "orientation": orientation,
                "h": h,
                "w": w,
                "crack_idx": crack_idx,
                "colors": colors,
                "majorities": majorities,
            }
        )
    assert any(spec["majorities"][0] != spec["majorities"][1] for spec in specs)
    return specs


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> bool:
    originals: list[tuple[int, int, int, int]] = []
    expanded: list[tuple[int, int, int, int]] = []
    for spec in specs:
        options = _candidate_positions(shape, spec["h"], spec["w"], spec["orientation"], originals, expanded)
        if not options:
            return False
        top, left = options[0]
        spec["top"] = top
        spec["left"] = left
        originals.append((top, left, top + spec["h"] - 1, left + spec["w"] - 1))
        expanded.append(_expanded_bbox(top, left, spec["h"], spec["w"], spec["orientation"]))
    return True


def _generate_pair() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.choice([2, 3])
    shape = (random.randint(13, 18), random.randint(13, 18))
    specs = _gen_object_specs(count)
    specs.sort(key=lambda spec: spec["h"] * spec["w"], reverse=True)
    if not _place_specs(shape, specs):
        return None

    inp = np.zeros(shape, dtype=int)
    expected = np.zeros(shape, dtype=int)
    alt_shift_only = np.zeros(shape, dtype=int)
    alt_no_shift = np.zeros(shape, dtype=int)
    alt_global = np.zeros(shape, dtype=int)
    global_ok = True

    for spec in specs:
        patch, oracle = _build_object_patch(spec)
        top = int(spec["top"])
        left = int(spec["left"])
        assert np.all(inp[top : top + patch.shape[0], left : left + patch.shape[1]] == BG)
        inp[top : top + patch.shape[0], left : left + patch.shape[1]] = patch
        _stamp_expected(expected, patch, top, left, oracle)
        _stamp_shift_only(alt_shift_only, patch, top, left, oracle)
        _stamp_recolor_without_shift(alt_no_shift, patch, top, left, oracle)
        global_ok = _stamp_global_majority(alt_global, patch, top, left, oracle) and global_ok

    if not np.array_equal(main(inp), expected):
        return None
    if np.array_equal(expected, alt_shift_only):
        return None
    if np.array_equal(expected, alt_no_shift):
        return None
    if global_ok and np.array_equal(expected, alt_global):
        return None

    objects = _solve_objects(inp)
    if len(objects) != count:
        return None
    if not any(obj["orientation"] == "horizontal" for obj in objects):
        return None
    if not any(obj["orientation"] == "vertical" for obj in objects):
        return None
    if not any(obj["halves"][0]["majority"] != obj["halves"][1]["majority"] for obj in objects):
        return None
    for obj in objects:
        for half in obj["halves"]:
            values = inp[half["coords"][:, 0], half["coords"][:, 1]]
            uniq = set(map(int, values.tolist()))
            if len(uniq) != 2:
                return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        pair = _generate_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate a valid crack-majority ARC input")
