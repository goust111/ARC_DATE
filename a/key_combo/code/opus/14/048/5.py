from __future__ import annotations

import numpy as np


BG = 0
RED = 2
YELLOW = 4
GREY = 5

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _is_hollow_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )
    return set(coords) == expected


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for coords in _components_of_color(grid, GREY):
        if _is_hollow_frame(coords):
            frames.append(_bbox(coords))
    frames.sort()
    return frames


def _apply_gravity_to_frame(
    src: np.ndarray,
    dst: np.ndarray,
    frame: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = frame
    inner_r0 = r0 + 1
    inner_c0 = c0 + 1
    inner_r1 = r1 - 1
    inner_c1 = c1 - 1

    freeze_cols = set()
    if r1 + 1 < src.shape[0]:
        for c in range(inner_c0, inner_c1 + 1):
            if int(src[r1 + 1, c]) == RED:
                freeze_cols.add(c)

    yellow_positions: list[tuple[int, int]] = []
    for r in range(inner_r0, inner_r1 + 1):
        for c in range(inner_c0, inner_c1 + 1):
            if int(src[r, c]) == YELLOW:
                yellow_positions.append((r, c))
                dst[r, c] = BG

    counts: dict[int, int] = {}
    for r, c in yellow_positions:
        if c in freeze_cols:
            dst[r, c] = YELLOW
        else:
            counts[c] = counts.get(c, 0) + 1

    for c, count in counts.items():
        for r in range(inner_r1 - count + 1, inner_r1 + 1):
            dst[r, c] = YELLOW


def main(grid: np.ndarray) -> np.ndarray:
    """
    Freeze marked columns inside each grey frame while the other yellow pixels fall and stack, then erase the red markers.

    Input structure:
    - The grid has a black background with one or two grey one-pixel-thick hollow rectangular frames.
    - Yellow movable pixels lie strictly inside each frame, and red control pixels sit on the row immediately below a frame, aligned with some interior columns.
    - Grey borders stay intact; the red controls are outside the enclosed region and indicate which interior columns are frozen.

    Rule:
    - Detect every grey hollow rectangular frame and read its freeze columns from the red pixels directly below that frame.
    - Inside the frame, yellow pixels in frozen columns stay at their original cells.
    - Yellow pixels in all other columns fall straight down within their own columns and stack against the frame's bottom interior edge.
    - After processing all frames, erase every red control pixel to black; keep the grey borders and other black cells unchanged.

    Key insight:
    - The crucial clue is cross-region control: red pixels outside a frame determine which yellow columns inside that frame ignore gravity, while the remaining columns still undergo normal vertical stacking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one grey hollow frame"

    for frame in frames:
        _apply_gravity_to_frame(g, out, frame)

    out[out == RED] = BG
    return out


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [total]
    return list(map(int, rng.multinomial(total, np.full(parts, 1.0 / parts))))


def _choose_layout(
    rng: np.random.Generator,
    region_sizes: list[tuple[int, int]],
) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    count = len(region_sizes)
    outer_sizes = [(h + 2, w + 2) for h, w in region_sizes]

    if count == 1:
        outer_h, outer_w = outer_sizes[0]
        h = max(10, outer_h + 3) + int(rng.integers(0, 3))
        w = max(10, outer_w + 3) + int(rng.integers(0, 3))
        r0_low = 1
        r0_high = h - outer_h - 1
        c0_low = 1
        c0_high = w - outer_w - 1
        r0 = int(rng.integers(r0_low, r0_high + 1))
        c0 = int(rng.integers(c0_low, c0_high + 1))
        return h, w, [(r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)]

    (outer_h_a, outer_w_a), (outer_h_b, outer_w_b) = outer_sizes
    h_min = outer_h_a + outer_h_b + 2
    w_min = outer_w_a + outer_w_b + 2
    h = h_min + int(rng.integers(0, 2))
    w = w_min + int(rng.integers(0, 2))

    extra_h = h - h_min
    extra_w = w - w_min
    top_margin, middle_h, bottom_margin = _random_composition(extra_h, 3, rng)
    left_margin, middle_w, right_margin = _random_composition(extra_w, 3, rng)

    mode = "tl_br" if rng.random() < 0.5 else "tr_bl"
    if mode == "tl_br":
        a_r0 = top_margin
        a_c0 = left_margin
        b_r0 = a_r0 + outer_h_a + 1 + middle_h
        b_c0 = a_c0 + outer_w_a + 1 + middle_w
    else:
        a_r0 = top_margin
        a_c0 = w - right_margin - outer_w_a
        b_r0 = a_r0 + outer_h_a + 1 + middle_h
        b_c0 = left_margin

    assert b_r0 + outer_h_b < h
    assert a_r0 + outer_h_a < h
    assert a_c0 + outer_w_a <= w
    assert b_c0 + outer_w_b <= w
    _ = bottom_margin  # kept for symmetry with the sampled layout composition

    return h, w, [
        (a_r0, a_c0, a_r0 + outer_h_a - 1, a_c0 + outer_w_a - 1),
        (b_r0, b_c0, b_r0 + outer_h_b - 1, b_c0 + outer_w_b - 1),
    ]


def _sample_region_sizes(rng: np.random.Generator, region_count: int) -> list[tuple[int, int]]:
    sizes = []
    for _ in range(region_count):
        if region_count == 1:
            inner_h = int(rng.integers(5, 9))
            inner_w = int(rng.integers(5, 9))
        else:
            inner_h = int(rng.integers(5, 7))
            inner_w = int(rng.integers(5, 7))
        sizes.append((inner_h, inner_w))
    return sizes


def _choose_column_counts(
    inner_h: int,
    inner_w: int,
    rng: np.random.Generator,
) -> tuple[list[int], dict[int, int]]:
    usable_cols = list(range(1, inner_w - 1))
    max_freeze = min(3, len(usable_cols) - 1)
    freeze_count = int(rng.integers(1, max_freeze + 1))
    freeze_cols = sorted(map(int, rng.choice(usable_cols, size=freeze_count, replace=False)))
    unfrozen_cols = [c for c in usable_cols if c not in freeze_cols]

    counts: dict[int, int] = {}
    for col in freeze_cols:
        counts[col] = 1 + int(rng.random() < 0.35)

    stack_col = int(rng.choice(unfrozen_cols))
    max_stack = 2 if inner_h == 5 else 3
    counts[stack_col] = int(rng.integers(2, max_stack + 1))

    if len(unfrozen_cols) >= 2 and rng.random() < 0.7:
        other_choices = [c for c in unfrozen_cols if c != stack_col]
        other_col = int(rng.choice(other_choices))
        counts[other_col] = 1 + int(rng.random() < 0.4)

    target_total = int(rng.integers(4, 9))
    max_per_col = inner_h - 2
    all_cols = usable_cols[:]
    while sum(counts.values()) < target_total:
        candidates = [c for c in all_cols if counts.get(c, 0) < max_per_col]
        if not candidates:
            break
        chosen = int(rng.choice(candidates))
        counts[chosen] = counts.get(chosen, 0) + 1

    while sum(counts.values()) > 8:
        shrinkable = [c for c in all_cols if counts.get(c, 0) > (1 if c in freeze_cols else 0)]
        if not shrinkable:
            break
        chosen = int(rng.choice(shrinkable))
        counts[chosen] -= 1
        if counts[chosen] == 0:
            del counts[chosen]

    assert any(col not in freeze_cols and counts[col] >= 2 for col in counts)
    assert all(counts.get(col, 0) >= 1 for col in freeze_cols)
    return freeze_cols, counts


def _choose_input_rows(
    inner_h: int,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    valid_rows = list(range(1, inner_h - 1))
    rows = list(map(int, rng.choice(valid_rows, size=count, replace=False)))
    rows.sort()
    return rows


def _build_region_spec(
    bbox: tuple[int, int, int, int],
    inner_h: int,
    inner_w: int,
    rng: np.random.Generator,
) -> dict:
    r0, c0, r1, c1 = bbox
    freeze_cols_local, counts = _choose_column_counts(inner_h, inner_w, rng)
    yellow_positions: list[tuple[int, int]] = []

    for local_c, count in counts.items():
        rows = _choose_input_rows(inner_h, count, rng)
        for local_r in rows:
            yellow_positions.append((r0 + 1 + local_r, c0 + 1 + local_c))

    spec = {
        "bbox": bbox,
        "freeze_cols": [c0 + 1 + local_c for local_c in freeze_cols_local],
        "yellow_positions": sorted(yellow_positions),
        "inner_h": inner_h,
        "inner_w": inner_w,
    }
    return spec


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0, c0 : c1 + 1] = GREY
        grid[r1, c0 : c1 + 1] = GREY
        grid[r0 : r1 + 1, c0] = GREY
        grid[r0 : r1 + 1, c1] = GREY
        for col in spec["freeze_cols"]:
            grid[r1 + 1, col] = RED
        for r, c in spec["yellow_positions"]:
            grid[r, c] = YELLOW
    return grid


def _expected_from_specs(inp: np.ndarray, specs: list[dict], *, freeze_all: bool = False) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        inner_r1 = r1 - 1
        inner_c0 = c0 + 1
        inner_c1 = c1 - 1
        freeze_cols = set() if freeze_all else set(spec["freeze_cols"])

        counts: dict[int, int] = {}
        for r, c in spec["yellow_positions"]:
            out[r, c] = BG
            if c in freeze_cols:
                out[r, c] = YELLOW
            else:
                counts[c] = counts.get(c, 0) + 1

        for c, count in counts.items():
            for r in range(inner_r1 - count + 1, inner_r1 + 1):
                assert inner_c0 <= c <= inner_c1
                out[r, c] = YELLOW

    out[out == RED] = BG
    return out


def _check_semantics(inp: np.ndarray, specs: list[dict]) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False

    expected = _expected_from_specs(inp, specs)
    if not np.array_equal(actual, expected):
        return False

    erase_only = inp.copy()
    erase_only[erase_only == RED] = BG
    if np.array_equal(erase_only, expected):
        return False

    all_drop = _expected_from_specs(inp, specs, freeze_all=True)
    if np.array_equal(all_drop, expected):
        return False

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        inner_r1 = r1 - 1
        freeze_cols = set(spec["freeze_cols"])
        by_col: dict[int, list[int]] = {}
        for r, c in spec["yellow_positions"]:
            by_col.setdefault(c, []).append(r)

        if not freeze_cols or not all(col in by_col for col in freeze_cols):
            return False

        moved_nonfreeze = False
        changed_frozen_if_dropped = False
        for col, rows in by_col.items():
            rows = sorted(rows)
            if col in freeze_cols:
                dropped_rows = list(range(inner_r1 - len(rows) + 1, inner_r1 + 1))
                if rows != dropped_rows:
                    changed_frozen_if_dropped = True
            else:
                dropped_rows = list(range(inner_r1 - len(rows) + 1, inner_r1 + 1))
                if rows != dropped_rows:
                    moved_nonfreeze = True

        if not moved_nonfreeze or not changed_frozen_if_dropped:
            return False

        controls = {c for c in range(c0 + 1, c1) if int(inp[r1 + 1, c]) == RED}
        if controls != freeze_cols:
            return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    region_count = 1 if rng.random() < 0.55 else 2
    region_sizes = _sample_region_sizes(rng, region_count)
    h, w, boxes = _choose_layout(rng, region_sizes)
    specs = [
        _build_region_spec(box, inner_h, inner_w, rng)
        for box, (inner_h, inner_w) in zip(boxes, region_sizes)
    ]
    inp = _render_input((h, w), specs)

    if not _check_semantics(inp, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frozen-column gravity input")
