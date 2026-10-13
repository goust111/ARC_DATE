from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


Coord = tuple[int, int]


def _mode(values: np.ndarray) -> int:
    """Return the most frequent visible color in a non-empty array."""
    assert values.size > 0
    return int(Counter(map(int, values.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    unseen = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    result: list[np.ndarray] = []
    while unseen:
        seed = unseen.pop()
        queue = deque([seed])
        cells = [seed]
        while queue:
            r, c = queue.popleft()
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) in unseen:
                    unseen.remove((rr, cc))
                    queue.append((rr, cc))
                    cells.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _layout(grid: np.ndarray) -> dict:
    """Locate the unique divider, the reference edge, and both visible backgrounds."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[float, int, int, int, int]] = []
    for c in range(1, w - 1):
        if not bool(np.all(g[:, c] == g[0, c])):
            continue
        line_color = int(g[0, c])
        left_bg = _mode(g[:, :c])
        right_bg = _mode(g[:, c + 1 :])
        if left_bg != right_bg and line_color not in {left_bg, right_bg}:
            # The real divider is bordered by the two side backgrounds.  This
            # visible local support rejects a constant background column beside
            # a very dense reference edge without changing the side coordinate frame.
            support = float(np.mean(g[:, c - 1] == left_bg) + np.mean(g[:, c + 1] == right_bg))
            candidates.append((support, c, line_color, left_bg, right_bg))
    assert candidates, "the full-height separator must exist"
    candidates.sort(reverse=True)
    assert len(candidates) == 1 or candidates[0][0] > candidates[1][0], \
        "the full-height separator must be unique"

    _support, sep, sep_color, left_bg, right_bg = candidates[0]
    left_marks = int(np.count_nonzero(g[:, 0] != left_bg))
    right_marks = int(np.count_nonzero(g[:, -1] != right_bg))
    assert (left_marks > 0) != (right_marks > 0), "exactly one outer edge is the reference column"

    ref_left = left_marks > 0
    return {
        "sep": int(sep),
        "sep_color": int(sep_color),
        "ref_left": bool(ref_left),
        "ref_col": 0 if ref_left else w - 1,
        "ref_bg": int(left_bg if ref_left else right_bg),
        "obj_bg": int(right_bg if ref_left else left_bg),
        "obj_c0": int(sep + 1 if ref_left else 0),
        "obj_c1": int(w - 1 if ref_left else sep - 1),
    }


def _objects(grid: np.ndarray, info: dict) -> list[dict]:
    """Extract monochrome objects only in the object-side coordinate frame."""
    g = np.asarray(grid, dtype=int)
    c0, c1 = int(info["obj_c0"]), int(info["obj_c1"])
    obj_bg = int(info["obj_bg"])
    region = g[:, c0 : c1 + 1]
    result: list[dict] = []
    for rel in _components(region != obj_bg):
        coords = rel.copy()
        coords[:, 1] += c0
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "every object must be monochrome"
        color = colors.pop()
        assert color not in {obj_bg, int(info["sep_color"])}
        if bool(info["ref_left"]):
            distance = int(coords[:, 1].min() - int(info["sep"]) - 1)
        else:
            distance = int(int(info["sep"]) - coords[:, 1].max() - 1)
        assert distance >= 0
        result.append({"color": color, "coords": coords, "distance": distance})

    distances = [int(obj["distance"]) for obj in result]
    assert result and len(distances) == len(set(distances)), "object distances must determine a unique order"
    return sorted(result, key=lambda obj: int(obj["distance"]))


def _would_collide(coords: np.ndarray, placed: set[Coord], c0: int, c1: int) -> bool:
    return any(not (c0 <= int(c) <= c1) or (int(r), int(c)) in placed for r, c in coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move objects by consumable same-color census quotas, exiling later unpaid objects.

    Input structure:
    - A unique full-height vertical separator divides two differently colored background regions.
    - The outer edge of one region is a reference column containing colored quota pixels.
    - The other region contains separated monochrome objects, including repeated colors and varied shapes.

    Rule:
    - Process objects from nearest to farthest from the separator, using distance in the object-side frame.
    - Count the currently remaining same-color pixels in the reference column.  If the count is k>0,
      erase all of them and move the object k horizontal steps toward the separator, stopping before the
      separator or an already placed object.  If k=0, translate the object to the far outer edge.
    - Every source position is restored to the object-side background and object shapes are preserved.

    Key insight:
    - The reference column is a consumable color census: the first object of a repeated color spends the
      entire quota, so processing order makes every later object of that color receive zero and be exiled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0
    info = _layout(g)
    objects = _objects(g, info)

    out = g.copy()
    obj_bg = int(info["obj_bg"])
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = obj_bg

    toward = -1 if bool(info["ref_left"]) else 1
    ref_col = int(info["ref_col"])
    ref_bg = int(info["ref_bg"])
    c0, c1 = int(info["obj_c0"]), int(info["obj_c1"])
    placed: set[Coord] = set()

    for obj in objects:
        color = int(obj["color"])
        coords = obj["coords"].copy()
        quota_rows = np.flatnonzero(out[:, ref_col] == color)
        quota = int(quota_rows.size)

        if quota > 0:
            out[quota_rows, ref_col] = ref_bg
            for _ in range(quota):
                trial = coords.copy()
                trial[:, 1] += toward
                if _would_collide(trial, placed, c0, c1):
                    break
                coords = trial
        else:
            shift = (c1 - int(coords[:, 1].max())) if toward == -1 else (c0 - int(coords[:, 1].min()))
            trial = coords.copy()
            trial[:, 1] += int(shift)
            assert not _would_collide(trial, placed, c0, c1), "valid unpaid objects must have a clear exile edge"
            coords = trial

        out[coords[:, 0], coords[:, 1]] = color
        placed.update((int(r), int(c)) for r, c in coords)

    return out


# ------------------------------ generator ------------------------------


_SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
]


def _mask_coords(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += int(r0)
    coords[:, 1] += int(c0)
    return coords.astype(int)


def _independent_expected(grid: np.ndarray, records: list[dict], info: dict) -> tuple[np.ndarray, dict[str, dict]]:
    """Render the written rule from generator metadata, independently of layout/component inference."""
    out = grid.copy()
    obj_bg = int(info["obj_bg"])
    for record in records:
        coords = record["coords"]
        out[coords[:, 0], coords[:, 1]] = obj_bg

    toward = -1 if bool(info["ref_left"]) else 1
    ref_col = int(info["ref_col"])
    c0, c1 = int(info["obj_c0"]), int(info["obj_c1"])
    placed: set[Coord] = set()
    trace: dict[str, dict] = {}
    remaining = Counter(int(out[r, ref_col]) for r in range(out.shape[0]))

    for record in sorted(records, key=lambda item: int(item["distance"])):
        coords = record["coords"].copy()
        color = int(record["color"])
        quota = int(remaining[color])
        moved = 0
        branch = "paid"
        if quota:
            rows = np.flatnonzero(out[:, ref_col] == color)
            out[rows, ref_col] = int(info["ref_bg"])
            remaining[color] = 0
            for _ in range(quota):
                trial = coords.copy()
                trial[:, 1] += toward
                if _would_collide(trial, placed, c0, c1):
                    break
                coords = trial
                moved += 1
        else:
            branch = "exile"
            shift = (c1 - int(coords[:, 1].max())) if toward == -1 else (c0 - int(coords[:, 1].min()))
            coords[:, 1] += int(shift)
            moved = abs(int(shift))

        out[coords[:, 0], coords[:, 1]] = color
        placed.update((int(r), int(c)) for r, c in coords)
        trace[str(record["role"])] = {
            "branch": branch,
            "quota": quota,
            "moved": moved,
            "coords": coords.copy(),
        }
    return out, trace


def _static_quota_counterfactual(grid: np.ndarray, records: list[dict], info: dict) -> np.ndarray:
    """Wrong shortcut oracle: quotas are counted once and never consumed."""
    out = grid.copy()
    for record in records:
        xy = record["coords"]
        out[xy[:, 0], xy[:, 1]] = int(info["obj_bg"])
    initial = Counter(map(int, out[:, int(info["ref_col"])]))
    # Keep the same visible erasure as the real rule, so any disagreement is
    # caused by reusing a spent quota, not by leaving the ledger untouched.
    ref_col = int(info["ref_col"])
    quota_colors = {int(record["color"]) for record in records}
    for color in quota_colors:
        out[out[:, ref_col] == color, ref_col] = int(info["ref_bg"])
    placed: set[Coord] = set()
    toward = -1 if bool(info["ref_left"]) else 1
    c0, c1 = int(info["obj_c0"]), int(info["obj_c1"])
    for record in sorted(records, key=lambda item: int(item["distance"])):
        xy = record["coords"].copy()
        for _ in range(int(initial[int(record["color"])])):
            trial = xy.copy()
            trial[:, 1] += toward
            if _would_collide(trial, placed, c0, c1):
                break
            xy = trial
        out[xy[:, 0], xy[:, 1]] = int(record["color"])
        placed.update((int(r), int(c)) for r, c in xy)
    return out


def _pick_row_starts(h: int, count: int) -> list[int] | None:
    candidates = list(range(1, h - 2))
    random.shuffle(candidates)
    chosen: list[int] = []
    for r in candidates:
        if all(abs(r - old) >= 3 for old in chosen):
            chosen.append(r)
            if len(chosen) == count:
                return chosen
    return None


def _build_attempt(force_cap: bool | None = None) -> np.ndarray | None:
    n_objects = random.randint(4, 6)
    if n_objects == 4:
        roles = ["B", "A1", "C", "A2"]
    elif n_objects == 5:
        roles = random.choice([
            ["B", "A1", "D", "C", "A2"],
            ["B", "A1", "C", "D", "A2"],
        ])
    else:
        roles = random.choice([
            ["B", "A1", "D", "C", "E", "A2"],
            ["B", "D", "A1", "C", "E", "A2"],
        ])

    gaps = [random.randint(2, 4)]
    for _ in range(1, n_objects):
        gaps.append(gaps[-1] + random.randint(2, 3))
    role_gap = dict(zip(roles, gaps))

    # B and C are vertically aligned dominoes.  C's quota deliberately exceeds
    # the free route, so only ordered collision simulation gives its destination.
    b_gap = role_gap["B"]
    cap_at_divider = bool(force_cap) if force_cap is not None else random.random() < 0.35
    quota_b = b_gap + random.randint(1, 2) if cap_at_divider else random.randint(1, b_gap - 1)
    final_b_gap = max(0, b_gap - quota_b)
    collision_free_steps = role_gap["C"] - final_b_gap - 1
    quota_c = collision_free_steps + random.randint(1, 2)
    quota_a = random.randint(2, min(5, role_gap["A1"]))
    quotas = {"A": quota_a, "B": quota_b, "C": quota_c}
    for role in ("D", "E"):
        if role in roles:
            quotas[role] = random.randint(1, min(4, role_gap[role]))

    total_quota = sum(quotas.values())
    h_min = max(16, total_quota + 2, 3 * (n_objects - 1) + 4)
    if h_min > 28:
        return None
    h = random.randint(h_min, min(30, h_min + 3))

    palette = random.sample(range(10), 3 + len(quotas))
    ref_bg, obj_bg, sep_color = palette[:3]
    role_color = {name: palette[3 + i] for i, name in enumerate(quotas)}

    ref_width = random.randint(2, 4)
    shape_for: dict[str, np.ndarray] = {
        "B": np.array([[1], [1]], dtype=bool),
        "C": np.array([[1], [1]], dtype=bool),
    }
    a_choices = [0, 2, 3, 4, 5]
    a1_idx, a2_idx = random.sample(a_choices, 2)
    shape_for["A1"] = _SHAPES[a1_idx].copy()
    shape_for["A2"] = _SHAPES[a2_idx].copy()
    for role in ("D", "E"):
        if role in roles:
            shape_for[role] = random.choice(_SHAPES).copy()
    for role, mask in list(shape_for.items()):
        if mask.shape[1] > 1 and random.random() < 0.5:
            shape_for[role] = np.fliplr(mask)

    areas = [int(shape_for[role].sum()) for role in roles]
    if len(set(areas)) < 3:
        return None
    required_obj_width = max(gap + int(shape_for[role].shape[1]) + 1 for role, gap in zip(roles, gaps))
    max_obj_width = 29 - ref_width
    if required_obj_width > max_obj_width:
        return None
    obj_width = random.randint(required_obj_width, min(max_obj_width, required_obj_width + 4))
    w = ref_width + 1 + obj_width

    ref_left = bool(random.getrandbits(1))
    if ref_left:
        sep = ref_width
        ref_col = 0
        obj_c0, obj_c1 = sep + 1, w - 1
        grid = np.full((h, w), obj_bg, dtype=int)
        grid[:, :sep] = ref_bg
    else:
        sep = obj_width
        ref_col = w - 1
        obj_c0, obj_c1 = 0, sep - 1
        grid = np.full((h, w), ref_bg, dtype=int)
        grid[:, :sep] = obj_bg
    grid[:, sep] = sep_color

    quota_cells: list[int] = []
    for name, count in quotas.items():
        quota_cells.extend([int(role_color[name])] * int(count))
    random.shuffle(quota_cells)
    ref_rows = random.sample(range(h), len(quota_cells))
    for r, color in zip(ref_rows, quota_cells):
        grid[r, ref_col] = color

    bands = _pick_row_starts(h, n_objects - 1)
    if bands is None:
        return None
    collision_row = bands[0]
    other_rows = iter(bands[1:])
    records: list[dict] = []
    for role, gap in zip(roles, gaps):
        mask = shape_for[role]
        r0 = collision_row if role in {"B", "C"} else next(other_rows)
        if ref_left:
            c0 = sep + 1 + gap
        else:
            nearest = sep - 1 - gap
            c0 = nearest - int(mask.shape[1]) + 1
        coords = _mask_coords(mask, r0, c0)
        color_name = "A" if role.startswith("A") else role
        color = int(role_color[color_name])
        if np.any(grid[coords[:, 0], coords[:, 1]] != obj_bg):
            return None
        grid[coords[:, 0], coords[:, 1]] = color
        records.append({"role": role, "color": color, "coords": coords, "distance": gap})

    info = {
        "sep": sep,
        "sep_color": sep_color,
        "ref_left": ref_left,
        "ref_col": ref_col,
        "ref_bg": ref_bg,
        "obj_bg": obj_bg,
        "obj_c0": obj_c0,
        "obj_c1": obj_c1,
    }
    expected, trace = _independent_expected(grid, records, info)

    # Structural and semantic rejection checks, rather than only main(grid)==expected.
    parsed_info = _layout(grid)
    parsed = _objects(grid, parsed_info)
    intended = sorted((r["color"], r["distance"], frozenset(map(tuple, r["coords"]))) for r in records)
    observed = sorted((o["color"], o["distance"], frozenset(map(tuple, o["coords"]))) for o in parsed)
    if parsed_info != info or intended != observed:
        return None
    if trace["A1"]["branch"] != "paid" or trace["A1"]["moved"] != quota_a:
        return None
    if trace["A2"]["branch"] != "exile":
        return None
    if trace["C"]["moved"] >= trace["C"]["quota"]:
        return None
    b_cells = set(map(tuple, trace["B"]["coords"]))
    c_cells = set(map(tuple, trace["C"]["coords"]))
    if not any(abs(r1 - r2) + abs(c1 - c2) == 1 for r1, c1 in b_cells for r2, c2 in c_cells):
        return None
    if all(int(shape_for[role].sum()) == int(quotas.get("A" if role.startswith("A") else role, 0)) for role in roles):
        return None
    if np.array_equal(_static_quota_counterfactual(grid, records, info), expected):
        return None
    swapped = []
    for record in records:
        clone = dict(record)
        if record["role"] == "A1":
            clone["distance"] = role_gap["A2"]
        elif record["role"] == "A2":
            clone["distance"] = role_gap["A1"]
        swapped.append(clone)
    if np.array_equal(_independent_expected(grid, swapped, info)[0], expected):
        return None
    if np.array_equal(main(grid), grid) or not np.array_equal(main(grid), expected):
        return None

    # Changing the visible A census by one must change A1's uncapped destination,
    # proving that exact counting (not merely color presence) affects the result.
    a_rows = np.flatnonzero(grid[:, ref_col] == int(role_color["A"]))
    if len(a_rows) <= 1:
        return None
    fewer = grid.copy()
    fewer[int(a_rows[0]), ref_col] = ref_bg
    if np.array_equal(main(fewer), expected):
        return None
    return grid


_GEN_CALL_COUNT = 0


def generate_input() -> np.ndarray:
    """Generate a varied, semantically audited census-ration instance."""
    global _GEN_CALL_COUNT
    # One divider-capped paid object in every three consecutive generated
    # examples guarantees coverage of that less obvious branch.
    force_cap = (_GEN_CALL_COUNT % 3 == 0)
    for _ in range(400):
        grid = _build_attempt(force_cap=force_cap)
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid census-ration input after 400 attempts")
