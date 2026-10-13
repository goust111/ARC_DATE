import numpy as np


BG = 0
FRAME_COLOR = 5
SEED_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (18, 27)
FRAME_COUNT_CHOICES = (1, 2, 3)
OUTER_H_RANGE = (6, 10)
OUTER_W_RANGE = (6, 10)
MAX_GENERATION_ATTEMPTS = 400
MAX_FRAME_ATTEMPTS = 200
MAX_SEED_LAYOUT_ATTEMPTS = 200

DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERP_DIRS = {
    "top": ((0, -1), (0, 1)),
    "bottom": ((0, -1), (0, 1)),
    "left": ((-1, 0), (1, 0)),
    "right": ((-1, 0), (1, 0)),
}


def _sorted_seed_ranking(
    seeds: dict[int, list[tuple[int, int]]],
    gap_cell: tuple[int, int],
) -> list[tuple[int, int]]:
    gr, gc = gap_cell
    ranked = []
    visible_keys = set()
    for color, coords in seeds.items():
        ordered = tuple(sorted((int(r), int(c)) for r, c in coords))
        nearest = min(
            ordered,
            key=lambda pos: (abs(pos[0] - gr) + abs(pos[1] - gc), pos[0], pos[1]),
        )
        dist = abs(nearest[0] - gr) + abs(nearest[1] - gc)
        key = (-len(ordered), dist, nearest[0], nearest[1], ordered)
        assert key not in visible_keys, "tied colors must be distinguishable by position"
        visible_keys.add(key)
        ranked.append((key, int(color)))
    ranked.sort(key=lambda item: item[0])
    return [(-item[0][0], item[1]) for item in ranked]


def _gap_growth_commands(
    gap: dict,
    ranking: list[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    gr, gc = gap["cell"]
    side = gap["side"]
    dr, dc = DIRS[side]
    start_r = gr + dr
    start_c = gc + dc
    commands: list[tuple[int, int, int]] = []

    main_len, main_color = ranking[0]
    for step in range(main_len):
        commands.append((start_r + step * dr, start_c + step * dc, main_color))

    end_r = start_r + (main_len - 1) * dr
    end_c = start_c + (main_len - 1) * dc
    if len(ranking) >= 2:
        second_color = ranking[1][1]
        arm_cells = []
        for pdr, pdc in PERP_DIRS[side]:
            rr = end_r + pdr
            cc = end_c + pdc
            arm_cells.append((rr, cc))
            commands.append((rr, cc, second_color))
        if len(ranking) >= 3:
            third_color = ranking[2][1]
            for arm_r, arm_c in arm_cells:
                commands.append((arm_r + dr, arm_c + dc, third_color))

    return commands


def _draw_frame_growth(out: np.ndarray, frame: dict) -> None:
    h, w = out.shape
    for gap in frame["gaps"]:
        ranking = _sorted_seed_ranking(frame["seeds"], gap["cell"])
        for rr, cc, color in _gap_growth_commands(gap, ranking):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color


def _validate_frame_candidate(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
    bg: int,
) -> dict | None:
    if any(int(g[r, c]) != color for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
        return None

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior == color):
        return None

    seed_mask = (interior != bg) & (interior != color)
    if not np.any(seed_mask):
        return None

    gaps = []
    side_slices = {
        "top": g[r0, c0 + 1 : c1],
        "bottom": g[r1, c0 + 1 : c1],
        "left": g[r0 + 1 : r1, c0],
        "right": g[r0 + 1 : r1, c1],
    }
    for side, arr in side_slices.items():
        if np.any((arr != bg) & (arr != color)):
            return None
        gap_idx = np.flatnonzero(arr == bg)
        if gap_idx.size > 0 and np.any(np.diff(gap_idx) == 1):
            return None
        for idx in map(int, gap_idx):
            if side == "top":
                cell = (r0, c0 + 1 + idx)
            elif side == "bottom":
                cell = (r1, c0 + 1 + idx)
            elif side == "left":
                cell = (r0 + 1 + idx, c0)
            else:
                cell = (r0 + 1 + idx, c1)
            dr, dc = DIRS[side]
            orow = cell[0] + dr
            ocol = cell[1] + dc
            if 0 <= orow < g.shape[0] and 0 <= ocol < g.shape[1] and int(g[orow, ocol]) != bg:
                return None
            gaps.append({"side": side, "cell": cell})

    if not (1 <= len(gaps) <= 3):
        return None

    seeds: dict[int, list[tuple[int, int]]] = {}
    for rr, cc in np.argwhere(seed_mask):
        abs_r = int(r0 + 1 + rr)
        abs_c = int(c0 + 1 + cc)
        seed_color = int(g[abs_r, abs_c])
        seeds.setdefault(seed_color, []).append((abs_r, abs_c))

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "frame_color": int(color),
        "gaps": gaps,
        "seeds": {k: sorted(v) for k, v in seeds.items()},
    }


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _find_frames(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    candidates: dict[tuple[int, int, int, int], dict] = {}

    for color in colors:
        coords = {(int(r), int(c)) for r, c in np.argwhere(g == color)}
        if len(coords) < 8:
            continue
        rows: dict[int, list[int]] = {}
        for r, c in coords:
            rows.setdefault(r, []).append(c)
        for row_cols in rows.values():
            row_cols.sort()

        for r0, cols in rows.items():
            if len(cols) < 2:
                continue
            for i in range(len(cols) - 1):
                c0 = cols[i]
                for j in range(i + 1, len(cols)):
                    c1 = cols[j]
                    if c1 - c0 < 2:
                        continue
                    for r1 in range(r0 + 2, h):
                        if (r1, c0) not in coords or (r1, c1) not in coords:
                            continue
                        info = _validate_frame_candidate(g, r0, c0, r1, c1, color, bg)
                        if info is not None:
                            candidates[(r0, c0, r1, c1)] = info

    ordered = sorted(
        candidates.values(),
        key=lambda info: (
            -((info["bbox"][2] - info["bbox"][0] + 1) * (info["bbox"][3] - info["bbox"][1] + 1)),
            info["bbox"],
        ),
    )
    chosen: list[dict] = []
    used: set[tuple[int, int]] = set()
    for info in ordered:
        cells = _bbox_cells(info["bbox"])
        if cells & used:
            continue
        chosen.append(info)
        used |= cells

    chosen.sort(key=lambda info: info["bbox"])
    assert chosen, "expected at least one frame"
    return chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow color-coded branches outward from frame gaps.

    Each grey rectangular frame has 1-3 single-cell border gaps and colored seed
    pixels inside. For each gap, the gap side gives the outward direction. The
    most frequent internal color draws a straight external branch whose length is
    that color's seed count. Ties are broken by which tied color has a seed
    closest to the current gap. If a second ranked color exists, it forms a
    one-pixel T-branch at the end of the main segment. If a third ranked color
    exists, it extends one more step forward from both T tips.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    frames = _find_frames(g, BG)
    for frame in frames:
        _draw_frame_growth(out, frame)
    return out


def _sample_color_counts(rng: np.random.Generator, level: int) -> list[int]:
    if level == 1:
        return [int(rng.integers(1, 5))]
    if level == 2:
        if float(rng.random()) < 0.3:
            top = int(rng.integers(2, 5))
            return [top, top]
        top = int(rng.integers(2, 5))
        return [top, int(rng.integers(1, top))]

    if float(rng.random()) < 0.3:
        top = int(rng.integers(2, 5))
        third = int(rng.integers(1, top))
        counts = [top, top, third]
    else:
        top = int(rng.integers(3, 6))
        second = int(rng.integers(2, top))
        third = int(rng.integers(1, second))
        counts = [top, second, third]
    if float(rng.random()) < 0.2:
        counts.append(1)
    return counts


def _make_gap_specs(
    rng: np.random.Generator,
    bbox: tuple[int, int, int, int],
    gap_count: int,
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    sides = list(rng.choice(np.array(["top", "bottom", "left", "right"]), size=gap_count, replace=False))
    gaps = []
    for side in sides:
        if side in ("top", "bottom"):
            offset = int(rng.integers(1, c1 - c0))
            cell = (r0 if side == "top" else r1, c0 + offset)
        else:
            offset = int(rng.integers(1, r1 - r0))
            cell = (r0 + offset, c0 if side == "left" else c1)
        gaps.append({"side": str(side), "cell": cell})
    return gaps


def _seed_layout_valid(seeds: dict[int, list[tuple[int, int]]], gaps: list[dict]) -> bool:
    for gap in gaps:
        gr, gc = gap["cell"]
        per_count: dict[int, list[int]] = {}
        for coords in seeds.values():
            dist = min(abs(r - gr) + abs(c - gc) for r, c in coords)
            per_count.setdefault(len(coords), []).append(int(dist))
        for dist_list in per_count.values():
            if len(dist_list) != len(set(dist_list)):
                return False
    return True


def _place_seeds(
    rng: np.random.Generator,
    bbox: tuple[int, int, int, int],
    gaps: list[dict],
    colors: list[int],
    counts: list[int],
) -> dict[int, list[tuple[int, int]]] | None:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    total = sum(counts)
    if total > len(cells):
        return None

    for _ in range(MAX_SEED_LAYOUT_ATTEMPTS):
        chosen_idx = rng.choice(len(cells), size=total, replace=False)
        chosen = [cells[int(idx)] for idx in chosen_idx]
        rng.shuffle(chosen)
        seeds: dict[int, list[tuple[int, int]]] = {}
        start = 0
        for color, count in zip(colors, counts):
            seeds[int(color)] = sorted(chosen[start : start + count])
            start += count
        if _seed_layout_valid(seeds, gaps):
            return seeds
    return None


def _render_input(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        for gap in frame["gaps"]:
            gr, gc = gap["cell"]
            grid[gr, gc] = BG
        for color, coords in frame["seeds"].items():
            for rr, cc in coords:
                grid[rr, cc] = int(color)
    return grid


def _render_expected(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    out = _render_input(shape, frames)
    for frame in frames:
        _draw_frame_growth(out, frame)
    return out


def _expanded_bbox_cells(shape: tuple[int, int], bbox: tuple[int, int, int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return {
        (r, c)
        for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1)
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1)
    }


def _visible_branch_cells(shape: tuple[int, int], frame: dict) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for gap in frame["gaps"]:
        ranking = _sorted_seed_ranking(frame["seeds"], gap["cell"])
        for rr, cc, _ in _gap_growth_commands(gap, ranking):
            if 0 <= rr < h and 0 <= cc < w:
                cells.add((rr, cc))
    return cells


def _sample_frame(
    rng: np.random.Generator,
    shape: tuple[int, int],
    used_input: set[tuple[int, int]],
    used_output: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    for _ in range(MAX_FRAME_ATTEMPTS):
        outer_h = int(rng.integers(*OUTER_H_RANGE))
        outer_w = int(rng.integers(*OUTER_W_RANGE))
        if outer_h >= h - 1 or outer_w >= w - 1:
            continue

        r0 = int(rng.integers(1, h - outer_h))
        c0 = int(rng.integers(1, w - outer_w))
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        expanded = _expanded_bbox_cells(shape, bbox, pad=1)
        if expanded & used_input:
            continue
        if expanded & used_output:
            continue

        gap_count = int(rng.integers(1, 4))
        gaps = _make_gap_specs(rng, bbox, gap_count)
        level = int(rng.integers(1, 4))
        counts = _sample_color_counts(rng, level)
        interior_area = (outer_h - 2) * (outer_w - 2)
        if sum(counts) > interior_area - 2:
            continue

        colors = [int(color) for color in rng.choice(SEED_COLORS, size=len(counts), replace=False)]
        seeds = _place_seeds(rng, bbox, gaps, colors, counts)
        if seeds is None:
            continue

        frame = {
            "bbox": bbox,
            "frame_color": FRAME_COLOR,
            "gaps": gaps,
            "seeds": seeds,
        }
        branch_cells = _visible_branch_cells(shape, frame)
        bbox_cells = _bbox_cells(bbox)
        if branch_cells & bbox_cells:
            continue
        if branch_cells & used_input:
            continue
        if branch_cells & used_output:
            continue

        return frame
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    frame_count = int(rng.choice(np.array(FRAME_COUNT_CHOICES)))
    shape = (h, w)

    frames = []
    used_input: set[tuple[int, int]] = set()
    used_output: set[tuple[int, int]] = set()
    for _ in range(frame_count):
        frame = _sample_frame(rng, shape, used_input, used_output)
        if frame is None:
            return None
        frames.append(frame)
        used_input |= _expanded_bbox_cells(shape, frame["bbox"], pad=1)
        used_output |= _visible_branch_cells(shape, frame)

    inp = _render_input(shape, frames)
    expected = _render_expected(shape, frames)
    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
