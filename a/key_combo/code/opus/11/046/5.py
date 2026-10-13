from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FILL = 6
TARGET_POOL = (1, 2, 3)
EXTRA_POOL = (4, 5, 8, 9)
OBJECT_COLOR_POOL = TARGET_POOL + EXTRA_POOL
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _triangle_mask(shape: tuple[int, int], corner: str, side: int) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    if corner == "TR":
        return rr + (w - 1 - cc) < side
    if corner == "BL":
        return (h - 1 - rr) + cc < side
    assert corner == "BR"
    return (h - 1 - rr) + (w - 1 - cc) < side


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
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
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_target_colors(grid: np.ndarray) -> tuple[int, int]:
    legend = grid[:4, :4]
    assert legend.shape == (4, 4)
    border_mask = np.zeros((4, 4), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    assert np.all(legend[border_mask] == FILL), "legend border must be magenta"
    inner = legend[1:3, 1:3]
    vals = [int(v) for v in inner.flat if int(v) != BG]
    assert len(vals) == 4, "legend interior must be fully populated"
    counts = Counter(vals)
    assert len(counts) == 2 and set(counts.values()) == {2}, "legend must show exactly two duplicated target colors"
    return tuple(sorted(counts))


def _corner_from_control(grid: np.ndarray) -> str:
    h, w = grid.shape
    corners = {
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }
    found = [name for name, (r, c) in corners.items() if int(grid[r, c]) == FILL]
    assert len(found) == 1, "expected exactly one non-top-left magenta corner control"
    return found[0]


def _object_components(grid: np.ndarray) -> list[np.ndarray]:
    mask = grid != BG
    mask[:4, :4] = False
    h, w = grid.shape
    for r, c in ((0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        if int(grid[r, c]) == FILL:
            mask[r, c] = False
    return _find_components(mask)


def _component_colors(grid: np.ndarray, comp: np.ndarray) -> set[int]:
    return {int(grid[r, c]) for r, c in comp}


def _analyze(grid: np.ndarray) -> dict:
    targets = set(_extract_target_colors(grid))
    corner = _corner_from_control(grid)
    comps = _object_components(grid)
    deleted = [comp for comp in comps if targets.issubset(_component_colors(grid, comp))]
    kept = [comp for comp in comps if not targets.issubset(_component_colors(grid, comp))]
    return {
        "targets": targets,
        "corner": corner,
        "components": comps,
        "deleted": deleted,
        "kept": kept,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Filter objects by the legend colors, count the survivors, then paint a corner triangle of that size.

    Input structure:
    - The background is black.
    - The top-left 4x4 region is a legend: a magenta border surrounds a 2x2 interior
      containing exactly two duplicated target colors.
    - Outside the legend are several 4-connected multicolor objects plus exactly one
      magenta control pixel in the top-right, bottom-left, or bottom-right corner.

    Rule:
    - Read the two target colors from the legend interior.
    - For each non-legend, non-control 4-connected object, erase it if it contains both
      target colors; otherwise keep it.
    - Count the surviving objects, and use that count as the leg length of a solid
      isosceles right triangle drawn from the marked corner in magenta.
    - The triangle overwrites anything it covers, while the legend remains unchanged.

    Key insight:
    - The legend is shared state across the whole pipeline: its interior says which
      objects to delete, and its border color is also the triangle fill color.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    out = g.copy()
    for comp in info["deleted"]:
        out[comp[:, 0], comp[:, 1]] = BG

    side = len(info["kept"])
    tri = _triangle_mask(out.shape, info["corner"], side)
    out[tri] = FILL
    out[:4, :4] = g[:4, :4]
    return out


def _grow_connected_cells(area: int) -> set[tuple[int, int]]:
    dims = [
        (h, w)
        for h in range(2, 5)
        for w in range(2, 5)
        if h * w >= area and area >= max(h, w)
    ]
    h, w = random.choice(dims)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(random.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    norm = {(r - r0, c - c0) for r, c in cells}
    if len({r for r, _ in norm}) == 1 or len({c for _, c in norm}) == 1:
        return _grow_connected_cells(area)
    return norm


def _make_patch(area: int, colors: tuple[int, ...]) -> np.ndarray:
    cells = sorted(_grow_connected_cells(area))
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    h, w = max(rows) + 1, max(cols) + 1
    patch = np.zeros((h, w), dtype=int)
    shuffled_cells = cells[:]
    random.shuffle(shuffled_cells)
    assigned = list(colors)
    while len(assigned) < len(shuffled_cells):
        assigned.append(random.choice(colors))
    random.shuffle(assigned)
    for (r, c), color in zip(shuffled_cells, assigned):
        patch[r, c] = int(color)
    return patch


def _object_role_colors(targets: tuple[int, int], role: str, role_index: int) -> tuple[int, ...]:
    target_set = set(targets)
    extras = [c for c in EXTRA_POOL if c not in target_set]
    if role == "delete_exact":
        return tuple(targets)
    if role == "delete_extra":
        return tuple((*targets, random.choice(extras)))
    if role.startswith("one_target_"):
        kept_target = targets[int(role[-1])]
        extra_count = random.choice((1, 2))
        chosen = random.sample(extras, extra_count)
        return tuple((kept_target, *chosen))
    assert role == "no_target"
    extra_count = 2 if role_index % 2 == 0 else 3
    return tuple(random.sample(extras, extra_count))


def _candidate_positions(
    shape: tuple[int, int],
    patch: np.ndarray,
    blocked: np.ndarray,
    require_partial_overlap: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    h, w = shape
    ph, pw = patch.shape
    mask = patch != BG
    out: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if bool((blocked[r0 : r0 + ph, c0 : c0 + pw] & mask).any()):
                continue
            if require_partial_overlap is not None:
                local = require_partial_overlap[r0 : r0 + ph, c0 : c0 + pw] & mask
                if not local.any():
                    continue
                if local.sum() == mask.sum():
                    continue
            out.append((r0, c0))
    return out


def _place_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    ph, pw = patch.shape
    mask = patch != BG
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]


def _build_legend(grid: np.ndarray, targets: tuple[int, int]) -> None:
    grid[:4, :4] = FILL
    vals = [targets[0], targets[0], targets[1], targets[1]]
    random.shuffle(vals)
    grid[1:3, 1:3] = np.array(vals, dtype=int).reshape(2, 2)


def _check_generated_input(grid: np.ndarray, expect_overlap: bool) -> None:
    info = _analyze(grid)
    assert 5 <= len(info["components"]) <= 8
    assert 2 <= len(info["deleted"]) <= 3
    assert 3 <= len(info["kept"]) <= 5

    targets = info["targets"]
    one_target_kept = 0
    no_target_kept = 0
    for comp in info["kept"]:
        colors = _component_colors(grid, comp)
        hits = len(colors & targets)
        assert hits < 2
        if hits == 1:
            one_target_kept += 1
        else:
            no_target_kept += 1
    assert one_target_kept >= 2
    assert no_target_kept >= 1

    deleted_with_extra = 0
    for comp in info["deleted"]:
        colors = _component_colors(grid, comp)
        assert targets.issubset(colors)
        if len(colors) == 3:
            deleted_with_extra += 1
    assert deleted_with_extra >= 1

    out = main(grid)
    assert out.shape == grid.shape
    assert np.array_equal(out[:4, :4], grid[:4, :4])
    assert not np.array_equal(out, grid)

    side = len(info["kept"])
    tri = _triangle_mask(grid.shape, info["corner"], side)
    assert np.all(out[tri] == FILL)

    if expect_overlap:
        survivor_mask = np.zeros_like(grid, dtype=bool)
        for comp in info["kept"]:
            survivor_mask[comp[:, 0], comp[:, 1]] = True
        overlap = survivor_mask & tri
        assert overlap.any()
        assert (survivor_mask & ~tri).any()


def _generate_once() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    grid = np.zeros((h, w), dtype=int)

    targets = tuple(sorted(random.sample(TARGET_POOL, 2)))
    _build_legend(grid, targets)

    survivor_count = random.randint(3, 5)
    deleted_count = random.randint(2, 3)
    if survivor_count + deleted_count > 8:
        deleted_count = 8 - survivor_count
    if survivor_count + deleted_count < 5:
        deleted_count = 5 - survivor_count
    if deleted_count < 2 or deleted_count > 3:
        return None

    corner = random.choice(("TR", "BL", "BR"))
    corner_pos = {
        "TR": (0, w - 1),
        "BL": (h - 1, 0),
        "BR": (h - 1, w - 1),
    }[corner]
    grid[corner_pos] = FILL

    tri = _triangle_mask(grid.shape, corner, survivor_count)
    request_overlap = random.random() < 0.5

    roles: list[str] = []
    roles.extend(["delete_exact", "delete_extra"][:deleted_count])
    if deleted_count == 3:
        roles.append(random.choice(("delete_exact", "delete_extra")))
    roles.extend(["one_target_0", "one_target_1", "no_target"])
    while len(roles) < survivor_count + deleted_count:
        roles.append(random.choice(("one_target_0", "one_target_1", "no_target")))

    object_specs = []
    for idx, role in enumerate(roles):
        colors = _object_role_colors(targets, role, idx)
        area = random.randint(4, 8)
        if request_overlap and role.startswith("one_target_") and not any(spec["overlap"] for spec in object_specs):
            area = random.randint(4, 6)
            overlap = True
        else:
            overlap = False
        object_specs.append(
            {
                "role": role,
                "area": area,
                "colors": colors,
                "patch": _make_patch(area, colors),
                "overlap": overlap,
            }
        )
    if request_overlap and not any(spec["overlap"] for spec in object_specs):
        return None

    order = sorted(range(len(object_specs)), key=lambda i: (not object_specs[i]["overlap"], -object_specs[i]["area"]))
    occupied = grid != BG
    protected = np.zeros_like(grid, dtype=bool)
    protected[:4, :4] = True
    protected = _dilate8(protected)
    control_mask = np.zeros_like(grid, dtype=bool)
    control_mask[corner_pos] = True
    protected |= _dilate8(control_mask)

    for i in order:
        spec = object_specs[i]
        blocked = _dilate8(occupied) | protected
        candidates = _candidate_positions(
            grid.shape,
            spec["patch"],
            blocked,
            tri if spec["overlap"] else None,
        )
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        _place_patch(grid, spec["patch"], r0, c0)
        occupied = grid != BG

    info = _analyze(grid)
    if len(info["components"]) != survivor_count + deleted_count:
        return None
    if len(info["kept"]) != survivor_count or len(info["deleted"]) != deleted_count:
        return None
    if survivor_count > min(h, w) // 2:
        return None

    _check_generated_input(grid, request_overlap)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
