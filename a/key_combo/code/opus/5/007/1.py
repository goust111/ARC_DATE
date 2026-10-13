from __future__ import annotations

import itertools
import random
from collections import Counter, deque
from functools import lru_cache

import numpy as np


BG = 0
FG_COLORS = (1, 2, 3, 4)
ORTHOGONALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(grid: np.ndarray, *, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == int(bg) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHOGONALS:
                    nr, nc = rr + dr, cc + dc
                    if (
                        in_bounds(grid.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _coords_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ORTHOGONALS:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(coords)


def _reflect_coord(r: int, c: int, orientation: str, axis2: int) -> tuple[int, int]:
    if orientation == "vertical":
        return int(r), int(axis2 - c)
    return int(axis2 - r), int(c)


def _evaluate_axis(
    coords: np.ndarray, shape: tuple[int, int], orientation: str, axis2: int
) -> dict | None:
    original = {(int(r), int(c)) for r, c in coords}
    reflected: set[tuple[int, int]] = set()
    for r, c in original:
        rr, cc = _reflect_coord(r, c, orientation, axis2)
        if not in_bounds(shape, rr, cc):
            return None
        reflected.add((rr, cc))
    completed = original | reflected
    if not _coords_connected(completed):
        return None
    additions = len(completed) - len(original)
    return {
        "orientation": orientation,
        "axis2": int(axis2),
        "coords": completed,
        "additions": int(additions),
    }


def _recover_completed_object(coords: np.ndarray, shape: tuple[int, int]) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    candidates: list[dict] = []
    for axis2 in range(2 * c0, 2 * c1 + 1):
        cand = _evaluate_axis(coords, shape, "vertical", axis2)
        if cand is not None:
            candidates.append(cand)
    for axis2 in range(2 * r0, 2 * r1 + 1):
        cand = _evaluate_axis(coords, shape, "horizontal", axis2)
        if cand is not None:
            candidates.append(cand)
    if not candidates:
        raise ValueError("no valid reflection axis")
    min_add = min(c["additions"] for c in candidates)
    best = [c for c in candidates if c["additions"] == min_add]
    if len(best) != 1:
        raise ValueError("reflection axis is not uniquely determined")
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each broken mirror-symmetric object, then drop it downward by its completed area.

    Input structure:
    - Black background with 3-4 separated single-color 4-connected foreground objects.
    - Each object uses a distinct color and is nearly symmetric about one horizontal or vertical reflection axis.
    - Every object is missing 1-3 pixels on exactly one side of that axis, so the visible shape is not yet fully symmetric.

    Rule:
    - For each object, find the unique horizontal or vertical reflection axis that requires the fewest added pixels to make the object perfectly symmetric.
    - Add the reflected missing pixels in the object's own color and measure the completed area A.
    - Clear the original objects, then translate each completed object downward by exactly A rows, clipping any pixels that move beyond the bottom edge.

    Key insight:
    - The fall distance is determined by the completed symmetric shape, not the visible incomplete one, so solving the symmetry completion step is necessary before gravity can be applied correctly.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    out = np.full((h, w), bg, dtype=int)

    components = find_components(g, bg=bg)
    for comp in components:
        recovered = _recover_completed_object(comp["coords"], g.shape)
        drop = len(recovered["coords"])
        for r, c in recovered["coords"]:
            rr = r + drop
            if 0 <= rr < h:
                if int(out[rr, c]) not in (bg, int(comp["color"])):
                    raise ValueError("falling objects overlap")
                out[rr, c] = int(comp["color"])
    return out


def _build_symmetric_coords(lengths: tuple[int, ...]) -> set[tuple[int, int]]:
    axis = max(lengths)
    coords: set[tuple[int, int]] = set()
    for r, arm in enumerate(lengths):
        for c in range(axis - arm, axis + arm + 1):
            coords.add((r, c))
    return coords


def _transpose_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(c, r) for r, c in coords}


def _normalize_coords(coords: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    shifted = {(r - r0, c - c0) for r, c in coords}
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    return shifted, (h, w)


def _allocation_options(lengths: tuple[int, ...], total: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(idx: int, remaining: int, cur: list[int]) -> None:
        if idx == len(lengths):
            if remaining == 0:
                out.append(tuple(cur))
            return
        upper = min(lengths[idx], remaining)
        for val in range(upper + 1):
            cur.append(val)
            rec(idx + 1, remaining - val, cur)
            cur.pop()

    rec(0, total, [])
    return out


@lru_cache(None)
def _object_library() -> tuple[dict, ...]:
    specs: list[dict] = []
    seen: set[tuple] = set()
    for orientation in ("vertical", "horizontal"):
        max_arm_cap = 3 if orientation == "vertical" else 2
        for base_len in (2, 3):
            for lengths in itertools.product(range(max_arm_cap + 1), repeat=base_len):
                if max(lengths) == 0:
                    continue
                if lengths == lengths[::-1]:
                    continue
                base_coords = _build_symmetric_coords(lengths)
                capacity = sum(lengths)
                for missing_count in range(1, min(3, capacity) + 1):
                    for alloc in _allocation_options(lengths, missing_count):
                        if sum(alloc) != missing_count:
                            continue
                        for side in ("low", "high"):
                            broken = set(base_coords)
                            axis = max(lengths)
                            for r, arm in enumerate(lengths):
                                cut = alloc[r]
                                if cut == 0:
                                    continue
                                if side == "low":
                                    remove_cols = range(axis - arm, axis - arm + cut)
                                else:
                                    remove_cols = range(axis + arm - cut + 1, axis + arm + 1)
                                for c in remove_cols:
                                    broken.remove((r, c))
                            if orientation == "horizontal":
                                base_oriented = _transpose_coords(base_coords)
                                broken_oriented = _transpose_coords(broken)
                            else:
                                base_oriented = set(base_coords)
                                broken_oriented = set(broken)
                            broken_norm, shape = _normalize_coords(broken_oriented)
                            base_norm, _ = _normalize_coords(base_oriented)
                            if not _coords_connected(broken_norm):
                                continue
                            try:
                                recovered = _recover_completed_object(
                                    np.array(sorted(broken_norm), dtype=int), shape
                                )
                            except ValueError:
                                continue
                            if recovered["coords"] != base_norm:
                                continue
                            key = (
                                orientation,
                                shape,
                                tuple(sorted(broken_norm)),
                                tuple(sorted(base_norm)),
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            specs.append(
                                {
                                    "orientation": orientation,
                                    "shape": shape,
                                    "input_coords": broken_norm,
                                    "completed_coords": base_norm,
                                    "visible_area": len(broken_norm),
                                    "completed_area": len(base_norm),
                                    "missing_count": missing_count,
                                    "profile": lengths,
                                }
                            )
    return tuple(specs)


def _combo_has_visible_pair(specs: list[dict]) -> bool:
    for a, b in itertools.combinations(specs, 2):
        if a["visible_area"] == b["visible_area"] and a["completed_area"] != b["completed_area"]:
            return True
    return False


def _select_specs() -> list[dict] | None:
    library = list(_object_library())
    pairs = [
        (a, b)
        for a, b in itertools.combinations(library, 2)
        if a["visible_area"] == b["visible_area"] and a["completed_area"] != b["completed_area"]
    ]
    if not pairs:
        return None
    for _ in range(600):
        target_count = random.choice((3, 4))
        a, b = random.choice(pairs)
        used_completed = {a["completed_area"], b["completed_area"]}
        others = [s for s in library if s not in (a, b) and s["completed_area"] not in used_completed]
        need = target_count - 2
        if len(others) < need:
            continue
        for _ in range(40):
            rest = random.sample(others, need)
            specs = [a, b] + list(rest)
            if len({s["orientation"] for s in specs}) < 2:
                continue
            if sum(int(s["shape"][0]) for s in specs) > 9:
                continue
            min_w = sum(int(s["shape"][1]) for s in specs) + (target_count - 1)
            if min_w > 24:
                continue
            if max(int(s["missing_count"]) for s in specs) < 2:
                continue
            return specs
    return None


def _sample_starts(lengths: list[int], total_span: int, *, min_gap: int) -> list[int] | None:
    needed = sum(lengths) + min_gap * (len(lengths) - 1)
    if needed > total_span:
        return None
    slack = total_span - needed
    parts = len(lengths) + 1
    if parts == 1:
        extras = [slack]
    else:
        cuts = sorted(random.sample(range(slack + parts - 1), parts - 1))
        extras = []
        prev = -1
        for cut in cuts + [slack + parts - 1]:
            extras.append(cut - prev - 1)
            prev = cut
    starts: list[int] = []
    pos = extras[0]
    for idx, length in enumerate(lengths):
        starts.append(pos)
        pos += length
        if idx + 1 < len(lengths):
            pos += min_gap + extras[idx + 1]
    return starts


def _render_from_specs(
    placed: list[dict],
    shape: tuple[int, int],
    *,
    use_completed_shape: bool,
    use_completed_area: bool,
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    for item in placed:
        coords = item["completed_abs"] if use_completed_shape else item["input_abs"]
        drop = item["completed_area"] if use_completed_area else item["visible_area"]
        for r, c in coords:
            rr = r + drop
            if 0 <= rr < h:
                if int(out[rr, c]) not in (BG, item["color"]):
                    raise ValueError("render overlap")
                out[rr, c] = item["color"]
    return out


def _build_example() -> np.ndarray | None:
    specs = _select_specs()
    if specs is None:
        return None

    n = len(specs)
    row_order = list(range(n))
    col_order = list(range(n))
    random.shuffle(row_order)
    random.shuffle(col_order)

    row_lengths = [int(specs[i]["shape"][0]) for i in row_order]
    row_starts_ordered = _sample_starts(row_lengths, 9, min_gap=0)
    row_starts = [0] * n
    if row_starts_ordered is None:
        return None
    for idx, spec_idx in enumerate(row_order):
        row_starts[spec_idx] = row_starts_ordered[idx]
    if row_starts is None:
        return None

    width_min = sum(int(s["shape"][1]) for s in specs) + (n - 1)
    width_max = min(24, width_min + 5)
    if width_min > width_max:
        return None
    h = random.randint(12, 16)
    w = random.randint(width_min, width_max)
    col_lengths = [int(specs[i]["shape"][1]) for i in col_order]
    col_starts_ordered = _sample_starts(col_lengths, w, min_gap=1)
    if col_starts_ordered is None:
        return None
    col_starts = [0] * n
    for idx, spec_idx in enumerate(col_order):
        col_starts[spec_idx] = col_starts_ordered[idx]
    colors = random.sample(FG_COLORS, n)

    inp = np.full((h, w), BG, dtype=int)
    placed: list[dict] = []
    for idx, spec in enumerate(specs):
        top = row_starts[idx]
        left = col_starts[idx]
        color = colors[idx]
        input_abs = {(top + r, left + c) for r, c in spec["input_coords"]}
        completed_abs = {(top + r, left + c) for r, c in spec["completed_coords"]}
        for r, c in input_abs:
            inp[r, c] = color
        placed.append(
            {
                **spec,
                "color": color,
                "top": top,
                "left": left,
                "input_abs": input_abs,
                "completed_abs": completed_abs,
            }
        )

    row_intervals = sorted((p["top"], p["top"] + p["shape"][0] - 1) for p in placed)
    col_intervals = sorted((p["left"], p["left"] + p["shape"][1] - 1) for p in placed)
    if any(a[1] >= b[0] for a, b in zip(row_intervals, row_intervals[1:])):
        return None
    if any(a[1] >= b[0] for a, b in zip(col_intervals, col_intervals[1:])):
        return None

    for item in placed:
        try:
            recovered = _recover_completed_object(
                np.array(sorted(item["input_abs"]), dtype=int), inp.shape
            )
        except ValueError:
            return None
        if recovered["coords"] != item["completed_abs"]:
            return None

    if any(p["top"] + p["shape"][0] - 1 > 8 for p in placed):
        return None
    if len(find_components(inp, bg=BG)) != n:
        return None
    if len({p["completed_area"] for p in placed}) != n:
        return None
    if not _combo_has_visible_pair(placed):
        return None

    expected = _render_from_specs(placed, (h, w), use_completed_shape=True, use_completed_area=True)
    shortcut = _render_from_specs(placed, (h, w), use_completed_shape=True, use_completed_area=False)
    if np.array_equal(expected, shortcut):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
