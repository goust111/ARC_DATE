from __future__ import annotations

import random

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

STRIPE_CYCLE = (YELLOW, YELLOW, GREEN)
GENERATION_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _stripe_kind_at_col(col: int) -> int | None:
    if col < 0 or col % 2 == 1:
        return None
    return int(STRIPE_CYCLE[(col // 2) % len(STRIPE_CYCLE)])


def _find_segment_rows(grid: np.ndarray) -> tuple[int, ...]:
    g = np.asarray(grid, dtype=int)
    yellow_coords = np.argwhere(g == YELLOW)
    if yellow_coords.size == 0:
        raise AssertionError("expected a yellow border segment")
    if not np.all(yellow_coords[:, 1] == 0):
        raise AssertionError("all yellow pixels must belong to the left border segment")

    rows = sorted(int(r) for r in yellow_coords[:, 0])
    runs: list[list[int]] = []
    run = [rows[0]]
    for row in rows[1:]:
        if row == run[-1] + 1:
            run.append(row)
        else:
            runs.append(run)
            run = [row]
    runs.append(run)

    best = max(runs, key=lambda cur: (len(cur), -cur[0]))
    if len(best) < 3:
        raise AssertionError("border segment is too short")
    if len(runs) != 1:
        raise AssertionError("expected a single contiguous left-edge segment")
    return tuple(best)


def _frame_from_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, int] | None:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if r0 <= 0 or h < 4 or w < 4:
        return None

    top = g[r0, c0 : c1 + 1]
    if not np.all(np.isin(top, (BG, GRAY))):
        return None
    gap_offsets = np.flatnonzero(top == BG)
    if gap_offsets.size != 1:
        return None

    gap_off = int(gap_offsets[0])
    gap_c = int(c0 + gap_off)
    left_len = int(gap_off)
    right_len = int(w - gap_off - 1)
    if (left_len == 2) == (right_len == 2):
        return None
    direction = -1 if left_len == 2 else 1

    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        if c != gap_c:
            border.add((int(r0), int(c)))
        border.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        border.add((int(r), int(c0)))
        border.add((int(r), int(c1)))

    if set(coords) != border:
        return None
    if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BG):
        return None

    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "h": int(h),
        "w": int(w),
        "gap_c": int(gap_c),
        "dir": int(direction),
    }


def _parse_frames(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g == GRAY)
    if not comps:
        raise AssertionError("expected at least one gray frame")

    frames: list[dict[str, int]] = []
    covered = 0
    for comp in comps:
        frame = _frame_from_component(g, comp)
        if frame is None:
            raise AssertionError("all gray components must be valid open-top frames")
        frames.append(frame)
        covered += len(comp)

    if covered != int(np.count_nonzero(g == GRAY)):
        raise AssertionError("gray pixels must belong only to frames")
    frames.sort(key=lambda frame: (int(frame["r0"]), int(frame["c0"])))
    return frames


def _gate_exact(row: int, col: int, segment_rows: set[int]) -> int | None:
    kind = _stripe_kind_at_col(int(col))
    if kind is None:
        return None
    if kind == GREEN:
        return GREEN
    return YELLOW if int(row) in segment_rows else None


def _gate_yellow_full(row: int, col: int, segment_rows: set[int]) -> int | None:
    del row, segment_rows
    kind = _stripe_kind_at_col(int(col))
    return None if kind is None else int(kind)


def _gate_green_narrow(row: int, col: int, segment_rows: set[int]) -> int | None:
    kind = _stripe_kind_at_col(int(col))
    if kind is None or int(row) not in segment_rows:
        return None
    return int(kind)


def _gate_row_only(row: int, col: int, segment_rows: set[int]) -> int | None:
    if int(row) in segment_rows:
        kind = _stripe_kind_at_col(int(col))
        return YELLOW if kind is None else int(kind)
    return _gate_exact(int(row), int(col), segment_rows)


def _ray_segment(frame: dict[str, int], width: int) -> tuple[int, int, int]:
    gap_c = int(frame["gap_c"])
    ray_r = int(frame["r0"] - 1)
    if int(frame["dir"]) < 0:
        return int(ray_r), 0, int(gap_c)
    return int(ray_r), int(gap_c), int(width - 1)


def _render_from_parsed(
    grid: np.ndarray,
    segment_rows: tuple[int, ...],
    frames: list[dict[str, int]],
    gate_fn,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    blocked = np.zeros_like(g, dtype=bool)
    for frame in frames:
        blocked[int(frame["r0"]) : int(frame["r1"]) + 1, int(frame["c0"]) : int(frame["c1"]) + 1] = True

    for c in range(0, w, 2):
        kind = int(STRIPE_CYCLE[(c // 2) % len(STRIPE_CYCLE)])
        rows = range(h) if kind == GREEN else segment_rows
        for r in rows:
            rr = int(r)
            if not bool(blocked[rr, c]):
                out[rr, c] = int(kind)

    seg_set = set(int(r) for r in segment_rows)
    for frame in frames:
        gate = gate_fn(int(frame["r0"]), int(frame["gap_c"]), seg_set)
        if gate is None:
            ray_r, c_start, c_end = _ray_segment(frame, w)
            for c in range(c_start, c_end + 1):
                if g[ray_r, c] != GRAY:
                    out[ray_r, c] = RED
            continue

        out[int(frame["r0"]), int(frame["gap_c"])] = int(gate)
        out[int(frame["r0"]) + 1 : int(frame["r1"]), int(frame["c0"]) + 1 : int(frame["c1"])] = int(gate)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate striped columns from the left border and use each frame's top gap either as a fill gate or as a ray outlet.

    Input structure:
    - The grid is a black background with a short yellow vertical segment on the left border.
    - Several gray rectangular frames have solid left, right, and bottom sides plus a single black gap in the top side.
    - On exactly one side of each top gap, that side of the top border is exactly two gray pixels long, marking the horizontal cap direction.

    Rule:
    - Starting from the left border segment, propagate 1-pixel stripes every other column: yellow, yellow, green, repeating.
    - Yellow stripes exist only on the border-segment rows, while green stripes color their full columns.
    - Treat each frame's top-gap cell as the gate test location: if a stripe is present there, color the gap and fill the frame interior with that stripe color.
    - If no stripe reaches the gap cell, keep the frame interior black and draw a red horizontal ray on the row above the frame from the gap toward the cap side until the grid edge.
    - Preserve all gray frame pixels, and draw stripe pixels normally everywhere outside frame bounding boxes.

    Key insight:
    - The stripes are not decorative: the gap cell is a spatial gate, so stripe scope decides which frames flood-fill and which frames instead emit red overflow rays.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    segment_rows = _find_segment_rows(g)
    frames = _parse_frames(g)
    return _render_from_parsed(g, segment_rows, frames, _gate_exact)


def _boxes_overlap_or_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _ray_hits_box(ray: tuple[int, int, int], box: tuple[int, int, int, int]) -> bool:
    ray_r, c_start, c_end = ray
    r0, c0, r1, c1 = box
    if not (r0 <= ray_r <= r1):
        return False
    return not (c_end < c0 or c1 < c_start)


def _draw_frame(grid: np.ndarray, spec: dict[str, int]) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    r1 = int(spec["r1"])
    c1 = int(spec["c1"])
    gap_c = int(spec["gap_c"])

    grid[r0, c0 : c1 + 1] = GRAY
    grid[r0, gap_c] = BG
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 + 1 : r1, c0] = GRAY
    grid[r0 + 1 : r1, c1] = GRAY


def _expected_from_specs(
    shape: tuple[int, int],
    segment_rows: tuple[int, ...],
    specs: list[dict[str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    inp = np.zeros((h, w), dtype=int)
    inp[list(segment_rows), 0] = YELLOW
    for spec in specs:
        _draw_frame(inp, spec)
    expected = _render_from_parsed(inp, segment_rows, specs, _gate_exact)
    return inp, expected


def _gap_columns_for_role(role: str, width: int, direction: int, grid_w: int) -> list[int]:
    gap_off = 2 if direction < 0 else width - 3
    cols: list[int] = []
    for gap_c in range(1, grid_w - 1):
        c0 = int(gap_c - gap_off)
        c1 = int(c0 + width - 1)
        if c0 < 2 or c1 > grid_w - 2:
            continue
        kind = _stripe_kind_at_col(int(gap_c))
        if role in {"yellow_hit", "yellow_miss"} and kind == YELLOW:
            cols.append(int(gap_c))
        elif role == "green_hit" and kind == GREEN:
            cols.append(int(gap_c))
        elif role == "odd_miss" and kind is None:
            cols.append(int(gap_c))
    return cols


def _candidate_rows(role: str, height: int, grid_h: int, segment_rows: tuple[int, ...]) -> list[int]:
    seg_set = set(int(r) for r in segment_rows)
    rows = [r for r in range(1, grid_h - height + 1)]
    if role in {"yellow_hit", "odd_miss"}:
        return [int(r) for r in rows if int(r) in seg_set]
    return [int(r) for r in rows if int(r) not in seg_set]


def _spec_box(spec: dict[str, int]) -> tuple[int, int, int, int]:
    return int(spec["r0"]), int(spec["c0"]), int(spec["r1"]), int(spec["c1"])


def _make_spec(role: str, top: int, gap_c: int, height: int, width: int, direction: int) -> dict[str, int]:
    gap_off = 2 if direction < 0 else width - 3
    c0 = int(gap_c - gap_off)
    r1 = int(top + height - 1)
    c1 = int(c0 + width - 1)
    return {
        "role": role,
        "r0": int(top),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "h": int(height),
        "w": int(width),
        "gap_c": int(gap_c),
        "dir": int(direction),
    }


def _place_specs(
    grid_h: int,
    grid_w: int,
    segment_rows: tuple[int, ...],
    roles: list[str],
) -> list[dict[str, int]] | None:
    placed: list[dict[str, int]] = []
    rays: list[tuple[int, int, int]] = []

    miss_roles = [role for role in roles if role.endswith("miss")]
    hit_roles = [role for role in roles if not role.endswith("miss")]
    order = miss_roles + hit_roles

    direction_by_role: dict[str, int] = {}
    if "yellow_miss" in miss_roles and "odd_miss" in miss_roles:
        left_first = random.random() < 0.5
        direction_by_role["yellow_miss"] = -1 if left_first else 1
        direction_by_role["odd_miss"] = 1 if left_first else -1
    for role in order:
        direction_by_role.setdefault(role, random.choice((-1, 1)))

    for role in order:
        width = 6
        height_choices = [4, 5]
        random.shuffle(height_choices)
        candidates: list[dict[str, int]] = []
        for height in height_choices:
            top_rows = _candidate_rows(role, height, grid_h, segment_rows)
            random.shuffle(top_rows)
            gap_cols = _gap_columns_for_role(role, width, direction_by_role[role], grid_w)
            random.shuffle(gap_cols)
            for top in top_rows:
                for gap_c in gap_cols:
                    spec = _make_spec(role, top, gap_c, height, width, direction_by_role[role])
                    if _gate_exact(int(spec["r0"]), int(spec["gap_c"]), set(segment_rows)) is None:
                        if role.endswith("hit"):
                            continue
                    else:
                        if role.endswith("miss"):
                            continue

                    box = _spec_box(spec)
                    if any(_boxes_overlap_or_touch(box, _spec_box(prev), pad=1) for prev in placed):
                        continue
                    if any(_ray_hits_box(ray, box) for ray in rays):
                        continue

                    if role.endswith("miss"):
                        ray = _ray_segment(spec, grid_w)
                        if any(existing[0] == ray[0] for existing in rays):
                            continue
                        if any(_ray_hits_box(ray, _spec_box(prev)) for prev in placed):
                            continue
                    candidates.append(spec)
            if candidates:
                break

        if not candidates:
            return None

        chosen = random.choice(candidates)
        placed.append(chosen)
        if role.endswith("miss"):
            rays.append(_ray_segment(chosen, grid_w))

    placed.sort(key=lambda spec: (int(spec["r0"]), int(spec["c0"])))
    return placed


def _build_case() -> tuple[np.ndarray, np.ndarray, list[dict[str, int]], tuple[int, ...]] | None:
    grid_h = random.randint(14, 18)
    grid_w = random.randint(18, 24)
    seg_len = random.randint(3, 4)
    seg_start_hi = grid_h - seg_len - 5
    if seg_start_hi < 2:
        return None
    seg_start = random.randint(2, seg_start_hi)
    segment_rows = tuple(range(seg_start, seg_start + seg_len))

    if random.random() < 0.8:
        roles = ["yellow_hit", "green_hit", "yellow_miss", "odd_miss"]
    else:
        roles = random.choice(
            [
                ["yellow_hit", "green_hit", "yellow_miss"],
                ["green_hit", "yellow_miss", "odd_miss"],
            ]
        )

    specs = _place_specs(grid_h, grid_w, segment_rows, roles)
    if specs is None:
        return None

    inp, expected = _expected_from_specs((grid_h, grid_w), segment_rows, specs)
    return inp, expected, specs, segment_rows


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    specs: list[dict[str, int]],
    segment_rows: tuple[int, ...],
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    seg_set = set(int(r) for r in segment_rows)
    roles = {str(spec["role"]) for spec in specs}
    if "green_hit" not in roles or "yellow_miss" not in roles:
        return False
    if "green_hit" in roles:
        if not any(
            spec["role"] == "green_hit"
            and _gate_exact(int(spec["r0"]), int(spec["gap_c"]), seg_set) == GREEN
            and int(spec["r0"]) not in seg_set
            for spec in specs
        ):
            return False
    if "yellow_miss" in roles:
        if not any(
            spec["role"] == "yellow_miss"
            and _stripe_kind_at_col(int(spec["gap_c"])) == YELLOW
            and _gate_exact(int(spec["r0"]), int(spec["gap_c"]), seg_set) is None
            for spec in specs
        ):
            return False
    if "odd_miss" in roles:
        if not any(
            spec["role"] == "odd_miss"
            and _stripe_kind_at_col(int(spec["gap_c"])) is None
            and int(spec["r0"]) in seg_set
            for spec in specs
        ):
            return False
    if "yellow_hit" in roles:
        if not any(
            spec["role"] == "yellow_hit"
            and _gate_exact(int(spec["r0"]), int(spec["gap_c"]), seg_set) == YELLOW
            for spec in specs
        ):
            return False

    try:
        parsed_segment_rows = _find_segment_rows(inp)
        parsed_frames = _parse_frames(inp)
    except Exception:
        return False
    if tuple(parsed_segment_rows) != tuple(segment_rows):
        return False
    if len(parsed_frames) != len(specs):
        return False

    wrong_full_yellow = _render_from_parsed(inp, parsed_segment_rows, parsed_frames, _gate_yellow_full)
    if np.array_equal(wrong_full_yellow, expected):
        return False

    wrong_narrow_green = _render_from_parsed(inp, parsed_segment_rows, parsed_frames, _gate_green_narrow)
    if np.array_equal(wrong_narrow_green, expected):
        return False

    if "odd_miss" in roles:
        wrong_row_only = _render_from_parsed(inp, parsed_segment_rows, parsed_frames, _gate_row_only)
        if np.array_equal(wrong_row_only, expected):
            return False

    for spec in specs:
        if str(spec["role"]).endswith("miss"):
            ray_r, c_start, c_end = _ray_segment(spec, inp.shape[1])
            if not np.all(expected[ray_r, c_start : c_end + 1] == RED):
                return False
        else:
            gate = _gate_exact(int(spec["r0"]), int(spec["gap_c"]), seg_set)
            if gate is None:
                return False
            interior = expected[int(spec["r0"]) + 1 : int(spec["r1"]), int(spec["c0"]) + 1 : int(spec["c1"])]
            if interior.size == 0 or not np.all(interior == gate):
                return False
            if int(expected[int(spec["r0"]), int(spec["gap_c"])]) != int(gate):
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        built = _build_case()
        if built is None:
            continue
        inp, expected, specs, segment_rows = built
        if _semantic_checks(inp, expected, specs, segment_rows):
            return inp
    raise RuntimeError("failed to generate a valid striped-gate frame input")
