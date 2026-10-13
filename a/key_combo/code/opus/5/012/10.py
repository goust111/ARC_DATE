import random
from typing import Any

import numpy as np


COLORS = (1, 2, 3)
PATTERNS = (
    ("valid", "single", "valid"),
    ("valid", "uniform", "valid"),
    ("single", "valid", "uniform", "valid"),
    ("uniform", "valid", "single", "valid"),
    ("valid", "uniform", "valid", "single"),
)
GEN_RETRIES = 400


def _extract_row_runs(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    h, w = g.shape

    runs: list[dict[str, int]] = []
    for r in range(h):
        nz = np.flatnonzero(g[r] != 0)
        assert nz.size > 0, "each row must contain exactly one colored run"
        left = int(nz[0])
        right = int(nz[-1])
        assert left > 0 and right < w - 1, "input runs must have background on both sides"
        assert nz.size == right - left + 1, "each row must contain one contiguous run"
        color = int(g[r, left])
        assert color in COLORS, "foreground colors must be blue/red/green"
        assert np.all(g[r, left : right + 1] == color), "run must be monochromatic"
        runs.append(
            {
                "row": int(r),
                "color": color,
                "left": left,
                "right": right,
                "width": int(right - left + 1),
            }
        )
    return runs


def _align_runs(runs: list[dict[str, int]]) -> tuple[int, list[dict[str, int]]]:
    r_min = min(run["right"] for run in runs)
    aligned: list[dict[str, int]] = []
    for run in runs:
        new_left = max(0, int(r_min - run["width"] + 1))
        aligned.append(
            {
                **run,
                "aligned_left": new_left,
                "aligned_right": int(r_min),
                "effective_width": int(r_min - new_left + 1),
                "clipped": int(run["width"] > r_min + 1),
            }
        )
    return int(r_min), aligned


def _group_aligned_rows(aligned: list[dict[str, int]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    start = 0
    while start < len(aligned):
        end = start
        color = int(aligned[start]["color"])
        while end + 1 < len(aligned) and int(aligned[end + 1]["color"]) == color:
            end += 1
        members = aligned[start : end + 1]
        lefts = [int(m["aligned_left"]) for m in members]
        l_min = min(lefts)
        l_max = max(lefts)
        valid = len(members) >= 2 and l_max > l_min
        groups.append(
            {
                "start": int(start),
                "end": int(end),
                "color": color,
                "lefts": lefts,
                "l_min": int(l_min),
                "l_max": int(l_max),
                "height": int(len(members)),
                "valid": bool(valid),
                "has_interior_fill": bool(
                    len(members) >= 3 and any(left > l_min for left in lefts[1:-1])
                ),
            }
        )
        start = end + 1
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Right-align each row segment, keep only same-color groups that close into rectangles, and fill those rectangles.

    Input structure:
    - The background is black.
    - Every row contains exactly one contiguous horizontal run in blue, red, or green, with black on both sides.
    - Consecutive rows may share the same color, but their run widths and right endpoints can differ.

    Rule:
    - For each row, locate its unique colored run and compute the minimum right endpoint across all rows.
    - Slide every run horizontally so its right endpoint lands on that shared minimum; if the shifted run would cross column 0, truncate it at column 0.
    - Split the aligned rows into maximal vertical groups of consecutive rows with the same color.
    - A same-color group survives only if it has at least two rows and its aligned left endpoints are not all equal; surviving groups become solid rectangles spanning the group's rows and the group's leftmost-to-shared-right columns.
    - Erase every isolated row and every same-color group whose aligned runs fully overlap.

    Key insight:
    - Right-edge alignment turns per-row width differences into left-edge gaps; only those gaps let a vertical same-color run close into a fillable rectangle, so the alignment step directly determines what remains.
    """
    runs = _extract_row_runs(grid)
    r_min, aligned = _align_runs(runs)
    _ = r_min

    out = np.zeros_like(np.asarray(grid, dtype=int))
    for group in _group_aligned_rows(aligned):
        if not group["valid"]:
            continue
        out[group["start"] : group["end"] + 1, group["l_min"] : aligned[0]["aligned_right"] + 1] = int(
            group["color"]
        )
    return out


def _sample_group_colors(rng: random.Random, count: int) -> list[int]:
    colors: list[int] = []
    for _ in range(count):
        choices = [color for color in COLORS if not colors or color != colors[-1]]
        colors.append(int(rng.choice(choices)))
    return colors


def _sample_valid_widths(
    rng: random.Random,
    length: int,
    width_cap: int,
    r_min: int,
    *,
    use_clip: bool,
    require_interior_fill: bool,
) -> list[int]:
    nonclip = list(range(3, min(width_cap, r_min + 1) + 1))
    clip = list(range(r_min + 2, width_cap + 1))
    assert len(nonclip) >= 2

    for _ in range(200):
        widths = [int(rng.choice(nonclip)) for _ in range(length)]
        if use_clip:
            if not clip:
                continue
            clip_idx = rng.randrange(length)
            widths[clip_idx] = int(rng.choice(clip))
        if len({min(w, r_min + 1) for w in widths}) < 2:
            continue
        rng.shuffle(widths)
        effective = [min(w, r_min + 1) for w in widths]
        widest = max(effective)
        if require_interior_fill:
            if length < 3:
                continue
            if all(e == widest for e in effective[1:-1]):
                interior = rng.randrange(1, length - 1)
                narrower = [w for w in nonclip if w < widest]
                if not narrower:
                    continue
                widths[interior] = int(rng.choice(narrower))
                effective = [min(w, r_min + 1) for w in widths]
            if not any(e < max(effective) for e in effective[1:-1]):
                continue
        return widths
    raise RuntimeError("failed to sample a non-degenerate valid-width pattern")


def _sample_uniform_width(rng: random.Random, width_cap: int, r_min: int) -> int:
    choices = list(range(3, width_cap + 1))
    if width_cap > r_min + 1 and rng.random() < 0.35:
        clipped = [w for w in choices if w > r_min + 1]
        if clipped:
            return int(rng.choice(clipped))
    return int(rng.choice(choices))


def _plan_groups(rng: random.Random, width_cap: int, r_min: int) -> list[dict[str, Any]]:
    pattern = list(rng.choice(PATTERNS))
    colors = _sample_group_colors(rng, len(pattern))
    valid_indices = [idx for idx, kind in enumerate(pattern) if kind == "valid"]
    fill_group = int(rng.choice(valid_indices))
    clip_group = None
    if width_cap > r_min + 1 and rng.random() < 0.5:
        clip_group = int(rng.choice(valid_indices))

    groups: list[dict[str, Any]] = []
    for idx, (kind, color) in enumerate(zip(pattern, colors)):
        if kind == "single":
            length = 1
            widths = [int(rng.randint(3, width_cap))]
        elif kind == "uniform":
            length = int(rng.randint(2, 3))
            widths = [_sample_uniform_width(rng, width_cap, r_min)] * length
        else:
            low = 3 if idx == fill_group else 2
            length = int(rng.randint(low, 4))
            widths = _sample_valid_widths(
                rng,
                length,
                width_cap,
                r_min,
                use_clip=idx == clip_group,
                require_interior_fill=idx == fill_group,
            )
        groups.append({"kind": kind, "color": int(color), "length": int(length), "widths": widths})
    return groups


def _rows_from_groups(
    rng: random.Random,
    groups: list[dict[str, Any]],
    width: int,
    r_min: int,
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for group in groups:
        for run_width in group["widths"]:
            rows.append({"color": int(group["color"]), "width": int(run_width)})

    eligible = [idx for idx, row in enumerate(rows) if row["width"] <= r_min]
    assert eligible, "at least one row must be able to realize the shared minimum right edge"
    anchor_idx = int(rng.choice(eligible))

    varied_idx = [idx for idx, row in enumerate(rows) if max(r_min, row["width"]) < width - 2]
    force_varied = int(rng.choice(varied_idx)) if varied_idx else None

    for idx, row in enumerate(rows):
        run_width = int(row["width"])
        min_right = max(r_min, run_width)
        max_right = width - 2
        assert min_right <= max_right
        if idx == anchor_idx:
            right = int(r_min)
        else:
            right = int(rng.randint(min_right, max_right))
        if force_varied is not None and idx == force_varied and idx != anchor_idx:
            right = int(rng.randint(max(min_right, r_min + 1), max_right))
        row["right"] = right
        row["left"] = int(right - run_width + 1)
        assert row["left"] > 0 and row["right"] < width - 1
    return rows


def _build_grid(height: int, width: int, rows: list[dict[str, int]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for r, row in enumerate(rows):
        grid[r, row["left"] : row["right"] + 1] = int(row["color"])
    return grid


def _semantic_ok(
    grid: np.ndarray,
    planned_groups: list[dict[str, Any]],
    *,
    expect_clip_optional: bool,
) -> bool:
    runs = _extract_row_runs(grid)
    r_min, aligned = _align_runs(runs)
    groups = _group_aligned_rows(aligned)

    if len(groups) != len(planned_groups):
        return False
    if min(run["right"] for run in runs) != r_min:
        return False
    if not any(run["right"] > r_min for run in runs):
        return False

    planned_valid = [idx for idx, group in enumerate(planned_groups) if group["kind"] == "valid"]
    actual_valid = [idx for idx, group in enumerate(groups) if group["valid"]]
    if planned_valid != actual_valid or len(actual_valid) < 2:
        return False

    if len({int(groups[idx]["color"]) for idx in actual_valid}) < 2:
        return False

    has_single = False
    has_uniform = False
    for planned, actual in zip(planned_groups, groups):
        if planned["length"] != actual["height"] or planned["color"] != actual["color"]:
            return False
        if planned["kind"] == "single":
            has_single = True
            if actual["height"] != 1 or actual["valid"]:
                return False
        elif planned["kind"] == "uniform":
            has_uniform = True
            if actual["height"] < 2 or actual["valid"] or actual["l_min"] != actual["l_max"]:
                return False
        else:
            if not actual["valid"]:
                return False

    if not any(group["has_interior_fill"] for group in groups if group["valid"]):
        return False

    clipped = any(run["width"] > r_min + 1 for run in runs)
    if expect_clip_optional and clipped:
        pass

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if not np.any(out):
        return False
    if has_single or has_uniform:
        erased_rows = [r for r in range(grid.shape[0]) if not np.any(out[r] != 0)]
        if not erased_rows:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        width = int(rng.randint(10, 16))
        width_cap = min(8, width - 2)
        if width_cap < 4:
            continue
        r_min_hi = min(6, width - 4)
        if r_min_hi < 4:
            continue
        r_min = int(rng.randint(4, r_min_hi))
        groups = _plan_groups(rng, width_cap, r_min)
        rows = _rows_from_groups(rng, groups, width, r_min)
        height = len(rows)
        if not (8 <= height <= 14):
            continue
        grid = _build_grid(height, width, rows)
        if _semantic_ok(grid, groups, expect_clip_optional=True):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
