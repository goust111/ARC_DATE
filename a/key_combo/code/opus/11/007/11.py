from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
MARK = 6
PLUS_OFFSETS = ((-1, 0), (0, -1), (0, 1), (1, 0))
X_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
SHAPES = (PLUS_OFFSETS, X_OFFSETS)
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
GRID_SIZES = (19, 21, 23)
GENERATION_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _d4_relatives(dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (dr, dc),
        (dc, dr),
        (-dr, dc),
        (dr, -dc),
        (-dr, -dc),
        (-dc, -dr),
        (dc, -dr),
        (-dc, dr),
    }
    return tuple(sorted(pts))


def _d4_orbit(r: int, c: int, n: int) -> tuple[tuple[int, int], ...]:
    mid = n // 2
    pts = [(mid + dr, mid + dc) for dr, dc in _d4_relatives(r - mid, c - mid)]
    return tuple(sorted((rr, cc) for rr, cc in pts if 0 <= rr < n and 0 <= cc < n))


def _offsets_d4_symmetric(offsets: tuple[tuple[int, int], ...]) -> bool:
    s = set(offsets)
    if not s or (0, 0) in s:
        return False
    expanded = set()
    for dr, dc in s:
        expanded.update(_d4_relatives(dr, dc))
    return expanded == s


def _draw_template(
    grid: np.ndarray,
    r: int,
    c: int,
    center_color: int,
    templates: dict[int, tuple[int, tuple[tuple[int, int], ...]]],
) -> bool:
    shell_color, offsets = templates[center_color]
    changed = False

    current = int(grid[r, c])
    if current in (BG, MARK):
        grid[r, c] = center_color
        changed = True
    else:
        assert current == center_color, "center color conflict"

    for dr, dc in offsets:
        rr, cc = r + dr, c + dc
        assert _in_bounds(grid.shape, rr, cc), "template must fit"
        current = int(grid[rr, cc])
        if current in (BG, MARK):
            grid[rr, cc] = shell_color
            changed = True
        else:
            assert current == shell_color, "shell color conflict"
    return changed


def _infer_templates(grid: np.ndarray) -> dict[int, tuple[int, tuple[tuple[int, int], ...]]]:
    g = np.asarray(grid, dtype=int)
    candidates: dict[int, set[tuple[int, tuple[tuple[int, int], ...]]]] = defaultdict(set)
    shell_to_centers: dict[int, set[int]] = defaultdict(set)

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            center_color = int(g[r, c])
            if center_color in (BG, MARK):
                continue

            neighbors: list[tuple[int, int]] = []
            shell_colors: set[int] = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if not _in_bounds(g.shape, rr, cc):
                        continue
                    value = int(g[rr, cc])
                    if value in (BG, MARK):
                        continue
                    neighbors.append((dr, dc))
                    shell_colors.add(value)

            if len(neighbors) < 2 or len(shell_colors) != 1:
                continue
            shell_color = next(iter(shell_colors))
            if shell_color == center_color:
                continue
            offsets = tuple(sorted(neighbors))
            if not _offsets_d4_symmetric(offsets):
                continue
            candidates[center_color].add((shell_color, offsets))
            shell_to_centers[shell_color].add(center_color)

    assert candidates, "need at least one complete motif"
    assert all(len(v) == 1 for v in candidates.values()), "center colors must map to one template"
    assert all(len(v) == 1 for v in shell_to_centers.values()), "shell colors must map back uniquely"

    templates = {center: next(iter(specs)) for center, specs in candidates.items()}
    assert len(templates) >= 2, "expected multiple motif types"
    return templates


def _step_fill_shells(
    grid: np.ndarray,
    templates: dict[int, tuple[int, tuple[tuple[int, int], ...]]],
) -> bool:
    changed = False
    for center_color, (shell_color, offsets) in templates.items():
        for r, c in np.argwhere(grid == center_color):
            r = int(r)
            c = int(c)
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                current = int(grid[rr, cc])
                if current in (BG, MARK):
                    grid[rr, cc] = shell_color
                    changed = True
                else:
                    assert current == shell_color, "center-only repair must be consistent"
    return changed


def _step_fill_centers(
    grid: np.ndarray,
    templates: dict[int, tuple[int, tuple[tuple[int, int], ...]]],
) -> bool:
    changed = False
    height, width = grid.shape
    for r in range(height):
        for c in range(width):
            if int(grid[r, c]) not in (BG, MARK):
                continue
            matches: list[int] = []
            for center_color, (shell_color, offsets) in templates.items():
                if all(
                    _in_bounds(grid.shape, r + dr, c + dc) and int(grid[r + dr, c + dc]) == shell_color
                    for dr, dc in offsets
                ):
                    matches.append(center_color)
            if matches:
                assert len(set(matches)) == 1, "shell-only motif must choose one center"
                grid[r, c] = matches[0]
                changed = True
    return changed


def _is_fully_missing(
    grid: np.ndarray,
    r: int,
    c: int,
    center_color: int,
    templates: dict[int, tuple[int, tuple[tuple[int, int], ...]]],
) -> bool:
    if int(grid[r, c]) == center_color:
        return False
    shell_color, offsets = templates[center_color]
    if any(int(grid[r + dr, c + dc]) == shell_color for dr, dc in offsets):
        return False
    return True


def _step_fill_from_symmetry(
    grid: np.ndarray,
    templates: dict[int, tuple[int, tuple[tuple[int, int], ...]]],
) -> bool:
    n = grid.shape[0]
    center_colors = set(templates)
    orbit_to_color: dict[tuple[tuple[int, int], ...], int] = {}

    for r, c in np.argwhere(np.isin(grid, list(center_colors))):
        r = int(r)
        c = int(c)
        orbit = _d4_orbit(r, c, n)
        color = int(grid[r, c])
        previous = orbit_to_color.get(orbit)
        if previous is None:
            orbit_to_color[orbit] = color
        else:
            assert previous == color, "every symmetry orbit must keep one template type"

    changed = False
    for orbit, center_color in orbit_to_color.items():
        for r, c in orbit:
            if _is_fully_missing(grid, r, c, center_color, templates):
                changed |= _draw_template(grid, r, c, center_color, templates)
    return changed


def _solve_with_flags(
    grid: np.ndarray,
    *,
    do_step2: bool = True,
    do_step3: bool = True,
    do_step4: bool = True,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    templates = _infer_templates(out)

    while True:
        changed = False
        if do_step2:
            changed |= _step_fill_shells(out, templates)
        if do_step3:
            changed |= _step_fill_centers(out, templates)
        if do_step4:
            changed |= _step_fill_from_symmetry(out, templates)
        if not changed:
            break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair every missing 3x3 sprite by combining local motif templates with D4 symmetry orbits.

    Input structure:
    - The grid is an odd square on a black background.
    - Foreground objects are 3x3 centered sprites: each has one center color and one D4-symmetric shell color in its 8-neighborhood.
    - Some sprites are complete, some keep only their center, some keep only their shell, and some orbit positions are fully absent except for optional pink hints.

    Rule:
    - Infer each complete sprite as center color -> (shell color, shell offsets).
    - Fill shell pixels around every visible center using its inferred local template.
    - Fill a missing center wherever one complete shell uniquely identifies a template.
    - For any D4 orbit of sprite centers, once one member's center color is known, draw the full sprite only at orbit positions that are still completely missing.
    - Repeat these repairs until no missing pink or black placeholder remains inside the intended sprites.

    Key insight:
    Local template completion and global D4 symmetry solve different gaps: shell-only copies reveal hidden center types, and those recovered center types unlock fully absent orbit mates that local evidence alone cannot place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1] and g.shape[0] % 2 == 1, "input must be an odd square"
    out = _solve_with_flags(g)
    assert not np.any(out == MARK), "all pink hints must be resolved"
    return out


def _orbit_from_kind(kind: str, radius: int, n: int) -> tuple[tuple[int, int], ...]:
    mid = n // 2
    if kind == "diag":
        return _d4_orbit(mid + radius, mid + radius, n)
    if kind == "cross":
        return _d4_orbit(mid, mid + radius, n)
    raise ValueError(kind)


def _sample_templates() -> dict[int, tuple[int, tuple[tuple[int, int], ...]]]:
    colors = list(PALETTE)
    random.shuffle(colors)
    center_colors = colors[:2]
    shell_colors = colors[2:4]
    shape_order = list(SHAPES)
    random.shuffle(shape_order)
    return {
        center_colors[0]: (shell_colors[0], shape_order[0]),
        center_colors[1]: (shell_colors[1], shape_order[1]),
    }


def _candidate_radii(n: int) -> tuple[int, int, int]:
    mid = n // 2
    small_choices = [d for d in range(4, mid - 4)]
    small = random.choice(small_choices)
    large_choices = [d for d in range(small + 4, mid)]
    large = random.choice(large_choices)
    cross_choices = [d for d in range(4, mid)]
    cross = random.choice(cross_choices)
    return small, cross, large


def _remove_cells_with_optional_marks(grid: np.ndarray, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    removed: list[tuple[int, int]] = []
    for r, c in cells:
        grid[r, c] = BG
        removed.append((r, c))

    if not cells:
        return removed

    max_marks = min(2, len(cells))
    mark_count = random.randint(0, max_marks)
    if mark_count:
        for r, c in random.sample(cells, mark_count):
            grid[r, c] = MARK
    return removed


def _state_plan_for_orbit(orbit: tuple[tuple[int, int], ...], kind: str) -> dict[tuple[int, int], str]:
    positions = list(orbit)
    random.shuffle(positions)
    plan: dict[tuple[int, int], str] = {}
    if kind == "complete":
        return {pos: "complete" for pos in positions}
    if kind == "center":
        n_complete = random.choice((1, 2))
        for pos in positions[:n_complete]:
            plan[pos] = "complete"
        for pos in positions[n_complete:]:
            plan[pos] = "center_only"
        return plan
    if kind == "chain":
        n_shell_only = random.choice((1, 2, 3))
        for pos in positions[:n_shell_only]:
            plan[pos] = "shell_only"
        for pos in positions[n_shell_only:]:
            plan[pos] = "full_missing"
        return plan
    raise ValueError(kind)


def _paint_output(
    n: int,
    templates: dict[int, tuple[int, tuple[tuple[int, int], ...]]],
    orbit_roles: list[tuple[tuple[tuple[int, int], ...], int, str]],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output = np.zeros((n, n), dtype=int)
    records: list[dict[str, object]] = []
    for orbit, center_color, role in orbit_roles:
        state_plan = _state_plan_for_orbit(orbit, role)
        for center in orbit:
            _draw_template(output, center[0], center[1], center_color, templates)
            records.append(
                {
                    "center": center,
                    "center_color": center_color,
                    "shell_color": templates[center_color][0],
                    "offsets": templates[center_color][1],
                    "state": state_plan[center],
                    "role": role,
                    "orbit": orbit,
                }
            )
    return output, records


def _derive_input(
    output: np.ndarray,
    records: list[dict[str, object]],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    inp = output.copy()
    removed_cells: list[tuple[int, int]] = []
    for rec in records:
        r, c = rec["center"]
        offsets = rec["offsets"]
        state = rec["state"]
        shell_cells = [(r + dr, c + dc) for dr, dc in offsets]
        if state == "center_only":
            removed_cells.extend(_remove_cells_with_optional_marks(inp, shell_cells))
        elif state == "shell_only":
            removed_cells.extend(_remove_cells_with_optional_marks(inp, [(r, c)]))
        elif state == "full_missing":
            removed_cells.extend(_remove_cells_with_optional_marks(inp, [(r, c), *shell_cells]))
    return inp, removed_cells


def _force_marker_mix(inp: np.ndarray, removed_cells: list[tuple[int, int]]) -> None:
    assert removed_cells, "need some removed cells"
    pink_cells = [(r, c) for r, c in removed_cells if int(inp[r, c]) == MARK]
    zero_cells = [(r, c) for r, c in removed_cells if int(inp[r, c]) == BG]

    if not pink_cells:
        r, c = random.choice(removed_cells)
        inp[r, c] = MARK
        pink_cells.append((r, c))
        zero_cells = [(rr, cc) for rr, cc in zero_cells if (rr, cc) != (r, c)]

    if not zero_cells:
        r, c = random.choice(pink_cells)
        inp[r, c] = BG


def _is_complete_at(
    grid: np.ndarray,
    r: int,
    c: int,
    center_color: int,
    templates: dict[int, tuple[int, tuple[tuple[int, int], ...]]],
) -> bool:
    shell_color, offsets = templates[center_color]
    return int(grid[r, c]) == center_color and all(int(grid[r + dr, c + dc]) == shell_color for dr, dc in offsets)


def _validate_instance(inp: np.ndarray, out: np.ndarray, records: list[dict[str, object]]) -> None:
    templates = _infer_templates(inp)
    assert len(templates) == 2
    assert np.any(inp == MARK), "at least one pink hint should appear"
    assert np.array_equal(main(inp), out), "solver must match designed output"

    states = [str(rec["state"]) for rec in records]
    assert "center_only" in states
    assert "shell_only" in states
    assert "full_missing" in states

    no_step2 = _solve_with_flags(inp, do_step2=False, do_step3=True, do_step4=True)
    no_step3 = _solve_with_flags(inp, do_step2=True, do_step3=False, do_step4=True)
    no_step4 = _solve_with_flags(inp, do_step2=True, do_step3=True, do_step4=False)
    assert not np.array_equal(no_step2, out), "step 2 must matter"
    assert not np.array_equal(no_step3, out), "step 3 must matter"
    assert not np.array_equal(no_step4, out), "step 4 must matter"

    chain_records = [rec for rec in records if rec["role"] == "chain"]
    assert chain_records
    chain_orbit = chain_records[0]["orbit"]
    chain_color = int(chain_records[0]["center_color"])
    assert all(rec["center_color"] == chain_color for rec in chain_records)
    assert all(int(inp[r, c]) != chain_color for r, c in chain_orbit), "chain orbit must start without known centers"

    local_only = _solve_with_flags(inp, do_step2=True, do_step3=True, do_step4=False)
    missing_centers = [tuple(rec["center"]) for rec in chain_records if rec["state"] == "full_missing"]
    assert missing_centers
    assert any(not _is_complete_at(local_only, r, c, chain_color, templates) for r, c in missing_centers)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray]:
    n = random.choice(GRID_SIZES)
    templates = _sample_templates()
    center_colors = list(templates)
    chain_color = random.choice(center_colors)
    other_color = next(color for color in center_colors if color != chain_color)

    small, cross, large = _candidate_radii(n)
    orbits = [
        _orbit_from_kind("diag", small, n),
        _orbit_from_kind("cross", cross, n),
        _orbit_from_kind("diag", large, n),
    ]
    random.shuffle(orbits)

    orbit_roles = [
        (orbits[0], other_color, "complete"),
        (orbits[1], chain_color, "center"),
        (orbits[2], chain_color, "chain"),
    ]
    out, records = _paint_output(n, templates, orbit_roles)
    inp, removed_cells = _derive_input(out, records)
    _force_marker_mix(inp, removed_cells)
    _validate_instance(inp, out, records)
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            inp, _out = _gen_attempt()
            return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
