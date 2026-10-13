from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS = ("up", "right", "down", "left")
DIR_INDEX = {name: idx for idx, name in enumerate(DIRS)}
DIR_VEC = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
GAP_LOCAL = {
    "up": (0, 1),
    "right": (1, 2),
    "down": (2, 1),
    "left": (1, 0),
}
BORDER_LOCAL = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
PALETTE = list(range(1, 10))
MAX_RETRIES = 300


def _detect_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rings: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            gap_dir = None
            border_color = None
            ok = True
            for direction, (rr, cc) in GAP_LOCAL.items():
                value = int(patch[rr, cc])
                if value == BG:
                    if gap_dir is not None:
                        ok = False
                        break
                    gap_dir = direction
                else:
                    if border_color is None:
                        border_color = value
                    elif value != border_color:
                        ok = False
                        break
            if not ok or gap_dir is None:
                continue

            gap_coord = GAP_LOCAL[gap_dir]
            for rr, cc in BORDER_LOCAL:
                value = int(patch[rr, cc])
                if (rr, cc) == gap_coord:
                    if value != BG:
                        ok = False
                        break
                else:
                    if value == BG:
                        ok = False
                        break
                    if border_color is None:
                        border_color = value
                    elif value != border_color:
                        ok = False
                        break
            if not ok or border_color in (None, BG):
                continue

            center_color = int(patch[1, 1])
            if center_color == border_color:
                continue

            rings.append(
                {
                    "top": int(r0),
                    "left": int(c0),
                    "border_color": int(border_color),
                    "center_color": int(center_color),
                    "gap_dir": gap_dir,
                    "gap_pos": (int(r0 + gap_coord[0]), int(c0 + gap_coord[1])),
                    "center_pos": (int(r0 + 1), int(c0 + 1)),
                }
            )
    return rings


def _transform_from_rings(grid: np.ndarray, rings: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    grouped: dict[int, list[dict]] = {}
    for ring in rings:
        grouped.setdefault(int(ring["border_color"]), []).append(ring)

    for border_color, items in grouped.items():
        sources = [item for item in items if int(item["center_color"]) != BG]
        targets = [item for item in items if int(item["center_color"]) == BG]
        assert len(sources) == 1, f"expected exactly one source ring for color {border_color}"
        assert targets, f"expected at least one target ring for color {border_color}"

        source = sources[0]
        pattern_color = int(source["center_color"])
        source_idx = DIR_INDEX[str(source["gap_dir"])]
        for target in targets:
            tr, tc = map(int, target["center_pos"])
            out[tr, tc] = pattern_color

            target_idx = DIR_INDEX[str(target["gap_dir"])]
            ray_len = 1 + ((target_idx - source_idx) % 4)
            gr, gc = map(int, target["gap_pos"])
            dr, dc = DIR_VEC[str(target["gap_dir"])]
            for step in range(ray_len):
                rr = int(gr + dr * step)
                cc = int(gc + dc * step)
                assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
                out[rr, cc] = pattern_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each source-ring center color into same-border target rings and emit a ray
    whose length depends on the clockwise gap-direction difference.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    rings = _detect_rings(g)
    assert rings, "expected at least one ring"
    return _transform_from_rings(g, rings)


def _object_cells(spec: dict) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(3) for c in range(3)}
    if spec["kind"] == "target":
        gap_r, gap_c = GAP_LOCAL[str(spec["gap_dir"])]
        dr, dc = DIR_VEC[str(spec["gap_dir"])]
        for step in range(1, int(spec["ray_len"])):
            cells.add((int(gap_r + dr * step), int(gap_c + dc * step)))
    return cells


def _dilate_cells(cells: set[tuple[int, int]], pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                out.add((int(r + dr), int(c + dc)))
    return out


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _sample_specs() -> list[dict]:
    n_groups = random.randint(1, 3)
    colors = random.sample(PALETTE, 2 * n_groups)
    border_colors = colors[:n_groups]
    pattern_colors = colors[n_groups:]
    coverage_group = random.randrange(n_groups)
    specs: list[dict] = []

    for group_id in range(n_groups):
        source_dir = random.choice(DIRS)
        border_color = int(border_colors[group_id])
        pattern_color = int(pattern_colors[group_id])
        specs.append(
            {
                "kind": "source",
                "group_id": int(group_id),
                "border_color": border_color,
                "center_color": pattern_color,
                "gap_dir": source_dir,
            }
        )

        if group_id == coverage_group:
            diffs = [0, 1, 2, 3]
            extra_targets = random.randint(0, 2)
            diffs.extend(random.randrange(4) for _ in range(extra_targets))
            random.shuffle(diffs)
        else:
            diffs = [random.randrange(4) for _ in range(random.randint(1, 3))]

        source_idx = DIR_INDEX[source_dir]
        for diff in diffs:
            target_dir = DIRS[(source_idx + diff) % 4]
            specs.append(
                {
                    "kind": "target",
                    "group_id": int(group_id),
                    "border_color": border_color,
                    "center_color": BG,
                    "gap_dir": target_dir,
                    "ray_len": int(diff + 1),
                }
            )

    return specs


def _grid_shape(n_objects: int) -> tuple[int, int]:
    side_min = max(14, int(np.ceil(np.sqrt(n_objects))) * 8)
    side_max = side_min + 8
    height = random.randint(side_min, side_max)
    width = random.randint(side_min, side_max)
    return height, width


def _place_specs(specs: list[dict], height: int, width: int) -> list[dict] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict] = []
    order = specs[:]
    random.shuffle(order)

    for spec in order:
        cells = _object_cells(spec)
        padded = _dilate_cells(cells, pad=1)
        min_r, min_c, max_r, max_c = _bbox(cells)
        r_lo = int(-min_r)
        c_lo = int(-min_c)
        r_hi = int(height - max_r - 1)
        c_hi = int(width - max_c - 1)
        if r_lo > r_hi or c_lo > c_hi:
            return None

        placed_here = None
        for _ in range(300):
            top = random.randint(r_lo, r_hi)
            left = random.randint(c_lo, c_hi)
            if any(
                0 <= top + rr < height and 0 <= left + cc < width and occupied[top + rr, left + cc]
                for rr, cc in padded
            ):
                continue
            placed_here = dict(spec)
            placed_here["top"] = int(top)
            placed_here["left"] = int(left)
            for rr, cc in padded:
                r = int(top + rr)
                c = int(left + cc)
                if 0 <= r < height and 0 <= c < width:
                    occupied[r, c] = True
            break
        if placed_here is None:
            return None
        placed.append(placed_here)

    return placed


def _draw_input(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        gap = GAP_LOCAL[str(spec["gap_dir"])]
        for rr, cc in BORDER_LOCAL:
            if (rr, cc) == gap:
                continue
            grid[top + rr, left + cc] = int(spec["border_color"])
        grid[top + 1, left + 1] = int(spec["center_color"])
    return grid


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    sources = {int(spec["group_id"]): spec for spec in specs if spec["kind"] == "source"}
    for spec in specs:
        if spec["kind"] != "target":
            continue
        source = sources[int(spec["group_id"])]
        color = int(source["center_color"])
        top = int(spec["top"])
        left = int(spec["left"])
        out[top + 1, left + 1] = color
        gap_r, gap_c = GAP_LOCAL[str(spec["gap_dir"])]
        dr, dc = DIR_VEC[str(spec["gap_dir"])]
        for step in range(int(spec["ray_len"])):
            rr = int(top + gap_r + dr * step)
            cc = int(left + gap_c + dc * step)
            out[rr, cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    height, width = _grid_shape(len(specs))
    placed = _place_specs(specs, height, width)
    if placed is None:
        return None

    inp = _draw_input(height, width, placed)
    expected = _expected_output(inp, placed)
    rings = _detect_rings(inp)
    if len(rings) != len(placed):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
