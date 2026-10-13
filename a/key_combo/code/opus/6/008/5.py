from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREEN = 3
OBJECT_COLORS = (1, 2, 4, 7)
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

GEN_RETRIES = 400
PLACE_RETRIES = 160


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr, dc in NEIGHBORS8:
        yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    target = int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != target:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if (
                        _in_bounds((h, w), nr, nc)
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == target
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors8(rr, cc):
            if _in_bounds((h, w), nr, nc):
                out[nr, nc] = True
    return out


def _line_positions(grid: np.ndarray) -> tuple[list[int], list[int]]:
    row_counts = np.count_nonzero(grid == GREEN, axis=1)
    col_counts = np.count_nonzero(grid == GREEN, axis=0)
    if int(row_counts.max()) <= 0 or int(col_counts.max()) <= 0:
        raise AssertionError("expected green lattice lines")

    line_rows = [int(r) for r, cnt in enumerate(row_counts) if int(cnt) >= int(row_counts.max()) - 2]
    line_cols = [int(c) for c, cnt in enumerate(col_counts) if int(cnt) >= int(col_counts.max()) - 2]
    assert 3 <= len(line_rows) <= 5
    assert 3 <= len(line_cols) <= 5

    row_steps = {b - a for a, b in zip(line_rows, line_rows[1:])}
    col_steps = {b - a for a, b in zip(line_cols, line_cols[1:])}
    assert len(row_steps) == 1
    assert len(col_steps) == 1
    return line_rows, line_cols


def _partition_info(grid: np.ndarray) -> dict:
    line_rows, line_cols = _line_positions(grid)
    missing = [
        (int(r), int(c))
        for r in line_rows
        for c in line_cols
        if int(grid[r, c]) == BG
    ]
    assert len(missing) == 4

    miss_rows = sorted({r for r, _ in missing})
    miss_cols = sorted({c for _, c in missing})
    assert len(miss_rows) == 2
    assert len(miss_cols) == 2
    expected_corners = {
        (miss_rows[0], miss_cols[0]),
        (miss_rows[0], miss_cols[1]),
        (miss_rows[1], miss_cols[0]),
        (miss_rows[1], miss_cols[1]),
    }
    assert set(missing) == expected_corners

    rect = (miss_rows[0], miss_cols[0], miss_rows[1], miss_cols[1])
    return {
        "line_rows": line_rows,
        "line_cols": line_cols,
        "missing": missing,
        "rect": rect,
    }


def _component_side(coords: np.ndarray, rect: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = rect
    inside = (
        (coords[:, 0] >= r0)
        & (coords[:, 0] <= r1)
        & (coords[:, 1] >= c0)
        & (coords[:, 1] <= c1)
    )
    if bool(np.all(inside)):
        return "inside"
    if not bool(np.any(inside)):
        return "outside"
    raise AssertionError("component crosses the partition boundary")


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    info = _partition_info(g)
    rect = info["rect"]
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, GREEN))
    by_color: dict[int, dict] = {}
    for color in colors:
        comps = _components_of_color(g, color)
        inside_comps: list[np.ndarray] = []
        outside_comps: list[np.ndarray] = []
        for comp in comps:
            side = _component_side(comp, rect)
            if side == "inside":
                inside_comps.append(comp)
            else:
                outside_comps.append(comp)
        by_color[color] = {
            "components": comps,
            "inside_components": inside_comps,
            "outside_components": outside_comps,
            "inside_count": len(inside_comps),
            "outside_count": len(outside_comps),
        }
    info["colors"] = colors
    info["by_color"] = by_color
    return info


def _apply_rule(grid: np.ndarray, info: dict) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for color in info["colors"]:
        meta = info["by_color"][color]
        keep_inside = meta["inside_count"] > meta["outside_count"]
        erase = meta["outside_components"] if keep_inside else meta["inside_components"]
        for comp in erase:
            out[comp[:, 0], comp[:, 1]] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep each object color only on the majority side of the rectangle implied by four missing grid intersections.

    Input structure:
    - Black background with thin green horizontal and vertical lattice lines at equal spacing.
    - Exactly four lattice intersections are missing: those green pixels are black and form the corners of one axis-aligned rectangle.
    - Non-green foreground objects use colors from {blue, red, yellow, orange}, are 8-connected, and never overlap the green lattice.

    Rule:
    - Detect the four missing green intersections and use their two rows and two columns as the partition rectangle, including its boundary rows and columns.
    - For each non-green foreground color independently, count how many 8-connected components lie entirely inside that rectangle and how many lie entirely outside it.
    - If a color has more components inside than outside, erase all of that color outside the rectangle; otherwise erase all of that color inside the rectangle.
    - Leave the green lattice and the four missing black intersections unchanged, and keep the full grid size.

    Key insight:
    - The missing intersections define a counting partition, not a crop: each color compares its own inside-versus-outside component counts and may preserve a different side from the others.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    return _apply_rule(grid, info)


def _sample_axis() -> tuple[int, list[int]]:
    for _ in range(100):
        line_count = random.randint(3, 5)
        spacing = random.randint(5, 8)
        top_margin = random.randint(1, 2)
        bottom_margin = random.randint(1, 2)
        size = top_margin + bottom_margin + (line_count - 1) * spacing + 1
        if size > 30:
            continue
        lines = [top_margin + i * spacing for i in range(line_count)]
        return size, lines
    raise RuntimeError("failed to sample axis")


def _cell_interior_mask(shape: tuple[int, int], line_rows: list[int], line_cols: list[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, r1 in zip(line_rows, line_rows[1:]):
        for c0, c1 in zip(line_cols, line_cols[1:]):
            mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _sample_partition(line_rows: list[int], line_cols: list[int]) -> tuple[int, int, int, int]:
    total_cells = (len(line_rows) - 1) * (len(line_cols) - 1)
    options: list[tuple[int, int, int, int]] = []
    for i in range(len(line_rows) - 1):
        for j in range(i + 1, len(line_rows)):
            for k in range(len(line_cols) - 1):
                for m in range(k + 1, len(line_cols)):
                    inside_cells = (j - i) * (m - k)
                    outside_cells = total_cells - inside_cells
                    if inside_cells < 2 or outside_cells < 2:
                        continue
                    options.append((line_rows[i], line_cols[k], line_rows[j], line_cols[m]))
    if not options:
        raise RuntimeError("failed to sample partition")
    return random.choice(options)


def _strict_in_splits() -> list[tuple[int, int]]:
    return [(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2), (4, 3)]


def _strict_out_splits() -> list[tuple[int, int]]:
    return [(b, a) for a, b in _strict_in_splits()]


def _tie_splits() -> list[tuple[int, int]]:
    return [(2, 2), (3, 3)]


def _sample_color_profiles(area_in: int, area_out: int) -> dict[int, tuple[int, int]] | None:
    strict_in = _strict_in_splits()
    strict_out = _strict_out_splits()
    ties = _tie_splits()
    for _ in range(120):
        color_count = 3 if random.random() < 0.65 else 4
        colors = random.sample(OBJECT_COLORS, color_count)
        roles = ["in", "out"]
        while len(roles) < color_count:
            roles.append(random.choices(["in", "out", "tie"], weights=[3, 3, 2], k=1)[0])
        random.shuffle(roles)

        profiles: dict[int, tuple[int, int]] = {}
        inside_total = 0
        outside_total = 0
        for color, role in zip(colors, roles):
            if role == "in":
                cin, cout = random.choice(strict_in)
            elif role == "out":
                cin, cout = random.choice(strict_out)
            else:
                cin, cout = random.choice(ties)
            profiles[int(color)] = (int(cin), int(cout))
            inside_total += int(cin)
            outside_total += int(cout)

        if inside_total * 6 > area_in or outside_total * 6 > area_out:
            continue
        if sum(sum(v) for v in profiles.values()) > 18:
            continue
        return profiles
    return None


def _sample_blob_size() -> int:
    sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    weights = [8, 9, 9, 8, 6, 4, 3, 1, 1]
    return int(random.choices(sizes, weights=weights, k=1)[0])


def _place_blob(
    grid: np.ndarray,
    occupancy: np.ndarray,
    allowed_mask: np.ndarray,
    color: int,
    size: int,
) -> bool:
    h, w = grid.shape
    free_mask = allowed_mask & ~_dilate8(occupancy)
    starts = np.argwhere(free_mask)
    if starts.size == 0:
        return False

    for _ in range(PLACE_RETRIES):
        sr, sc = map(int, starts[random.randrange(len(starts))])
        coords = {(sr, sc)}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors8(sr, sc)
            if _in_bounds((h, w), nr, nc) and bool(free_mask[nr, nc])
        }
        while frontier and len(coords) < size:
            candidates = list(frontier)
            weights = [
                1 + sum((rr, cc) in coords for rr, cc in _neighbors8(r, c))
                for r, c in candidates
            ]
            cell = random.choices(candidates, weights=weights, k=1)[0]
            frontier.remove(cell)
            coords.add(cell)
            r, c = cell
            for nr, nc in _neighbors8(r, c):
                if (
                    _in_bounds((h, w), nr, nc)
                    and bool(free_mask[nr, nc])
                    and (nr, nc) not in coords
                ):
                    frontier.add((nr, nc))

        if len(coords) != size:
            continue
        rr, cc = np.array(sorted(coords), dtype=int).T
        grid[rr, cc] = int(color)
        occupancy[rr, cc] = True
        return True
    return False


def _build_case() -> np.ndarray | None:
    h, line_rows = _sample_axis()
    w, line_cols = _sample_axis()
    grid = np.full((h, w), BG, dtype=int)
    grid[line_rows, :] = GREEN
    grid[:, line_cols] = GREEN

    rect = _sample_partition(line_rows, line_cols)
    r0, c0, r1, c1 = rect
    for rr in (r0, r1):
        for cc in (c0, c1):
            grid[rr, cc] = BG

    cell_mask = _cell_interior_mask(grid.shape, line_rows, line_cols)
    rect_mask = np.zeros_like(cell_mask)
    rect_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    inside_mask = cell_mask & rect_mask
    outside_mask = cell_mask & ~rect_mask

    profiles = _sample_color_profiles(int(inside_mask.sum()), int(outside_mask.sum()))
    if profiles is None:
        return None

    occupancy = np.zeros_like(cell_mask)
    requests: list[tuple[int, str, int]] = []
    for color, (inside_count, outside_count) in profiles.items():
        for _ in range(inside_count):
            requests.append((color, "inside", _sample_blob_size()))
        for _ in range(outside_count):
            requests.append((color, "outside", _sample_blob_size()))
    random.shuffle(requests)
    requests.sort(key=lambda item: item[2], reverse=True)

    for color, side, size in requests:
        target_mask = inside_mask if side == "inside" else outside_mask
        if not _place_blob(grid, occupancy, target_mask, color, size):
            return None
    return grid


def _validate_case(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    info = _analyze_grid(grid)
    line_rows = info["line_rows"]
    line_cols = info["line_cols"]
    assert 3 <= len(line_rows) <= 5
    assert 3 <= len(line_cols) <= 5
    assert set(info["colors"]).issubset(set(OBJECT_COLORS))
    assert 3 <= len(info["colors"]) <= 4

    strict_inside = 0
    strict_outside = 0
    unequal = 0
    for color in info["colors"]:
        meta = info["by_color"][color]
        total = meta["inside_count"] + meta["outside_count"]
        assert 3 <= total <= 7
        assert meta["inside_count"] >= 1
        assert meta["outside_count"] >= 1
        if meta["inside_count"] > meta["outside_count"]:
            strict_inside += 1
        if meta["outside_count"] > meta["inside_count"]:
            strict_outside += 1
        if meta["inside_count"] != meta["outside_count"]:
            unequal += 1
        sizes = [int(comp.shape[0]) for comp in meta["components"]]
        assert all(2 <= s <= 10 for s in sizes)

    assert strict_inside >= 1
    assert strict_outside >= 1
    assert unequal >= 2

    out = main(grid)
    expected = _apply_rule(grid, info)
    assert np.array_equal(out, expected)

    out_info = _analyze_grid(out)
    for color in info["colors"]:
        before = info["by_color"][color]
        after = out_info["by_color"].get(
            color,
            {
                "inside_count": 0,
                "outside_count": 0,
            },
        )
        if before["inside_count"] > before["outside_count"]:
            assert after["inside_count"] == before["inside_count"]
            assert after["outside_count"] == 0
        else:
            assert after["inside_count"] == 0
            assert after["outside_count"] == before["outside_count"]

    for r, c in info["missing"]:
        assert int(out[r, c]) == BG
    assert np.array_equal(out == GREEN, grid == GREEN)
    assert not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_case()
        if grid is None:
            continue
        try:
            _validate_case(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
