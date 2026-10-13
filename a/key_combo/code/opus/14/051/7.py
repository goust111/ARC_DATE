from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
ORANGE = 7

CARDINALS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

SIDE_ORDER = ("up", "right", "down", "left")

UP_TEMPLATES = (
    ((-2, 0), (-1, 0), (-1, 1)),
    ((-2, 0), (-1, -1), (-1, 0)),
    ((-3, 0), (-2, 0), (-2, 1), (-1, 0), (-1, 1), (-1, 2)),
    ((-3, 0), (-2, -1), (-2, 0), (-1, -2), (-1, -1), (-1, 0)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox_of_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_contains(box: tuple[int, int, int, int], cell: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = box
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _visible_pattern_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere((grid == BLUE) | (grid == RED))]
    assert coords, "expected visible red/blue pattern pixels"
    box = _bbox_of_coords(coords)
    r0, c0, r1, c1 = box
    assert r1 - r0 == c1 - c0, "central pattern must be square"
    return box


def _rot4_orbit(size: int, r: int, c: int) -> tuple[tuple[int, int], ...]:
    last = size - 1
    return tuple(
        sorted(
            {
                (r, c),
                (c, last - r),
                (last - r, last - c),
                (last - c, r),
            }
        )
    )


def _orbit_groups(size: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    groups: list[tuple[tuple[int, int], ...]] = []
    for r in range(size):
        for c in range(size):
            orbit = _rot4_orbit(size, r, c)
            if orbit not in seen:
                seen.add(orbit)
                groups.append(orbit)
    return groups


def _complete_pattern(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    size = patch.shape[0]
    for orbit in _orbit_groups(size):
        values = [int(patch[r, c]) for r, c in orbit]
        visible = {v for v in values if v in (BLUE, RED)}
        assert len(visible) == 1, "each fourfold orbit must expose one visible color"
        color = visible.pop()
        for r, c in orbit:
            if int(patch[r, c]) == GREY:
                patch[r, c] = color
    out = grid.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


def _principal_direction(center: tuple[int, int], tip: tuple[int, int]) -> tuple[int, int]:
    dr = int(tip[0] - center[0])
    dc = int(tip[1] - center[1])
    assert abs(dr) != abs(dc), "principal direction must be unambiguous"
    if abs(dr) > abs(dc):
        return (-1, 0) if dr < 0 else (1, 0)
    return (0, -1) if dc < 0 else (0, 1)


def _shadow_cell(
    box: tuple[int, int, int, int],
    tip: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        cell = (r0 + 1, tip[1])
    elif (dr, dc) == (1, 0):
        cell = (r1 - 1, tip[1])
    elif (dr, dc) == (0, -1):
        cell = (tip[0], c0 + 1)
    else:
        cell = (tip[0], c1 - 1)
    assert _bbox_contains(box, cell), "guide tip must map to an inner shadow cell"
    return cell


def _draw_ray(
    out: np.ndarray,
    blockers: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    color: int,
) -> None:
    h, w = out.shape
    r, c = start
    dr, dc = direction
    while 0 <= r < h and 0 <= c < w and int(blockers[r, c]) != ORANGE:
        out[r, c] = color
        r += dr
        c += dc


def _rotate_up_template(template: tuple[tuple[int, int], ...], side: str) -> tuple[tuple[int, int], ...]:
    steps = SIDE_ORDER.index(side)
    coords = list(template)
    for _ in range(steps):
        coords = [(c, -r) for r, c in coords]
    return tuple(coords)


def _guide_candidates(
    side: str,
    box: tuple[int, int, int, int],
    shape: tuple[tuple[int, int], ...],
) -> list[dict]:
    r0, c0, r1, c1 = box
    cells = list(shape)
    cand: list[dict] = []
    if side in ("up", "down"):
        min_dc = min(c for _, c in cells)
        max_dc = max(c for _, c in cells)
        start = max(1, -min_dc)
        stop = min((c1 - c0) - 1, (c1 - c0) - max_dc)
        for local in range(start, stop + 1):
            anchor = (r0, c0 + local) if side == "up" else (r1, c0 + local)
            placed = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in cells]
            query = (r0 + 1, anchor[1]) if side == "up" else (r1 - 1, anchor[1])
            cand.append({"anchor": anchor, "cells": placed, "query": query})
    else:
        min_dr = min(r for r, _ in cells)
        max_dr = max(r for r, _ in cells)
        start = max(1, -min_dr)
        stop = min((r1 - r0) - 1, (r1 - r0) - max_dr)
        for local in range(start, stop + 1):
            anchor = (r0 + local, c0) if side == "left" else (r0 + local, c1)
            placed = [(anchor[0] + dr, anchor[1] + dc) for dr, dc in cells]
            query = (anchor[0], c0 + 1) if side == "left" else (anchor[0], c1 - 1)
            cand.append({"anchor": anchor, "cells": placed, "query": query})
    return cand


def _tip_of_component(comp: list[tuple[int, int]], center: tuple[int, int]) -> tuple[int, int]:
    scored = sorted(
        comp,
        key=lambda rc: (
            (rc[0] - center[0]) ** 2 + (rc[1] - center[1]) ** 2,
            rc[0],
            rc[1],
        ),
    )
    tip = scored[-1]
    best = (tip[0] - center[0]) ** 2 + (tip[1] - center[1]) ** 2
    assert sum(((r - center[0]) ** 2 + (c - center[1]) ** 2) == best for r, c in comp) == 1
    return tip


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the masked rotational pattern, color guide-tip rays from orbit-mapped pattern cells, and erase helpers.

    Input structure:
    - A square black-background grid contains a centered red/blue square pattern with fourfold rotational symmetry.
    - Some pattern cells are replaced by gray mask pixels, and 2-4 gray right-triangle guides sit just outside the pattern on different sides.
    - Orange single-pixel blockers may lie farther out along the guide-tip ray paths.

    Rule:
    - Infer the central square from the visible red/blue pattern, then restore gray mask cells by copying the unique visible color in each 90-degree rotation orbit.
    - For each gray guide, take the gray pixel farthest from the pattern center as the tip and reduce the center-to-tip vector to its nearest cardinal direction.
    - Map that tip to the corresponding one-cell-inside shadow position on the touched side of the completed pattern; the shadow cell's orbit color determines the ray color.
    - Cast a ray from the tip outward in the inferred direction until an orange blocker or the grid edge, then erase all remaining gray and orange helper pixels.

    Key insight:
    - The gray guide supplies geometry only: its tip fixes where the ray starts and which side of the symmetric center it references, while the completed rotational pattern supplies the ray's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    box = _visible_pattern_bbox(g)
    restored = _complete_pattern(g, box)
    r0, c0, r1, c1 = box
    center = ((r0 + r1) // 2, (c0 + c1) // 2)

    grey_components = _components(g == GREY)
    guides = [comp for comp in grey_components if any(not _bbox_contains(box, cell) for cell in comp)]
    assert len(guides) >= 2, "expected at least two guide components"

    out = restored.copy()
    for comp in guides:
        tip = _tip_of_component(comp, center)
        direction = _principal_direction(center, tip)
        shadow = _shadow_cell(box, tip, direction)
        ray_color = int(restored[shadow])
        assert ray_color in (BLUE, RED), "shadow cell must resolve to a pattern color"
        _draw_ray(out, g, tip, direction, ray_color)

    out[out == GREY] = BLACK
    out[g == ORANGE] = BLACK
    return out


def _path_to_edge(n: int, tip: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = tip
    dr, dc = direction
    path: list[tuple[int, int]] = []
    while 0 <= r < n and 0 <= c < n:
        path.append((r, c))
        r += dr
        c += dc
    return path


def _pattern_from_orbits(size: int) -> np.ndarray:
    patch = np.zeros((size, size), dtype=int)
    orbits = _orbit_groups(size)
    for orbit in orbits:
        color = random.choice((BLUE, RED))
        for r, c in orbit:
            patch[r, c] = color
    if len(set(map(int, patch.flatten()))) == 1:
        orbit = random.choice([o for o in orbits if len(o) > 1])
        new_color = RED if int(patch[orbit[0]]) == BLUE else BLUE
        for r, c in orbit:
            patch[r, c] = new_color
    return patch


def _paint_pattern(grid: np.ndarray, box: tuple[int, int, int, int], patch: np.ndarray) -> None:
    r0, c0, r1, c1 = box
    assert patch.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch


def _pick_guides(
    n: int,
    box: tuple[int, int, int, int],
    patch: np.ndarray,
) -> list[dict] | None:
    r0, c0, r1, c1 = box
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    side_count = random.randint(2, 4)
    sides = random.sample(SIDE_ORDER, side_count)

    used = np.zeros((n, n), dtype=bool)
    guides: list[dict] = []
    for side in sides:
        candidates: list[dict] = []
        for template in UP_TEMPLATES:
            rotated = _rotate_up_template(template, side)
            for cand in _guide_candidates(side, box, rotated):
                cells = cand["cells"]
                if any(not (0 <= r < n and 0 <= c < n) for r, c in cells):
                    continue
                if any(_bbox_contains(box, cell) for cell in cells):
                    continue
                if any(used[r, c] for r, c in cells):
                    continue
                tip = _tip_of_component(cells, center)
                direction = _principal_direction(center, tip)
                shadow = _shadow_cell(box, tip, direction)
                local_shadow = (shadow[0] - r0, shadow[1] - c0)
                candidates.append(
                    {
                        "side": side,
                        "cells": cells,
                        "tip": tip,
                        "direction": direction,
                        "shadow": shadow,
                        "local_shadow": local_shadow,
                    }
                )
        if not candidates:
            return None
        chosen = random.choice(candidates)
        for r, c in chosen["cells"]:
            used[r, c] = True
        guides.append(chosen)

    colors = [int(patch[g["local_shadow"]]) for g in guides]
    if len(guides) >= 2 and len(set(colors)) == 1:
        idx = random.randrange(len(guides))
        orbit = _rot4_orbit(patch.shape[0], *guides[idx]["local_shadow"])
        flip_to = RED if colors[0] == BLUE else BLUE
        for rr, cc in orbit:
            patch[rr, cc] = flip_to
    return guides


def _mask_pattern(
    patch: np.ndarray,
    guides: list[dict],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    size = patch.shape[0]
    masked = patch.copy()
    orbit_caps = {orbit: len(orbit) - 1 for orbit in _orbit_groups(size) if len(orbit) > 1}
    orbit_use = defaultdict(int)
    mask_cells: set[tuple[int, int]] = set()

    forced_queries = random.sample(guides, random.randint(1, min(2, len(guides))))
    for guide in forced_queries:
        cell = guide["local_shadow"]
        orbit = _rot4_orbit(size, *cell)
        if orbit not in orbit_caps or orbit_use[orbit] >= orbit_caps[orbit]:
            continue
        mask_cells.add(cell)
        orbit_use[orbit] += 1

    interior = [
        (r, c)
        for r in range(1, size - 1)
        for c in range(1, size - 1)
        if (r, c) not in mask_cells
    ]
    random.shuffle(interior)
    target = random.randint(len(mask_cells) + 1, min(size + 1, len(interior) + len(mask_cells)))
    for cell in interior:
        if len(mask_cells) >= target:
            break
        orbit = _rot4_orbit(size, *cell)
        if orbit not in orbit_caps or orbit_use[orbit] >= orbit_caps[orbit]:
            continue
        mask_cells.add(cell)
        orbit_use[orbit] += 1

    for r, c in mask_cells:
        masked[r, c] = GREY
    return masked, mask_cells


def _reserve_mask(n: int, box: tuple[int, int, int, int], guides: list[dict]) -> np.ndarray:
    reserved = np.zeros((n, n), dtype=bool)
    r0, c0, r1, c1 = box
    reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    for guide in guides:
        for r, c in guide["cells"]:
            reserved[r, c] = True
    return reserved


def _place_blockers(
    n: int,
    guides: list[dict],
    reserved: np.ndarray,
) -> set[tuple[int, int]]:
    blockers: set[tuple[int, int]] = set()
    ray_paths = {idx: _path_to_edge(n, guide["tip"], guide["direction"]) for idx, guide in enumerate(guides)}

    truncatable = [idx for idx, path in ray_paths.items() if len(path) >= 4]
    random.shuffle(truncatable)
    trunc_count = 0 if not truncatable else random.randint(1, len(truncatable))
    for idx in truncatable[:trunc_count]:
        path = ray_paths[idx]
        cell = random.choice(path[2:])
        blockers.add(cell)
        reserved[cell] = True

    extras = random.randint(0, 2)
    free = [(r, c) for r in range(n) for c in range(n) if not reserved[r, c]]
    random.shuffle(free)
    for cell in free[:extras]:
        blockers.add(cell)
        reserved[cell] = True

    return blockers


def _oracle_output(
    n: int,
    box: tuple[int, int, int, int],
    patch: np.ndarray,
    guides: list[dict],
    blockers: set[tuple[int, int]],
) -> np.ndarray:
    out = np.zeros((n, n), dtype=int)
    _paint_pattern(out, box, patch)
    blocker_grid = np.zeros((n, n), dtype=int)
    for r, c in blockers:
        blocker_grid[r, c] = ORANGE
    for guide in guides:
        color = int(patch[guide["local_shadow"]])
        _draw_ray(out, blocker_grid, guide["tip"], guide["direction"], color)
    return out


def _semantic_checks(
    inp: np.ndarray,
    out: np.ndarray,
    expected: np.ndarray,
    box: tuple[int, int, int, int],
    guides: list[dict],
    mask_cells: set[tuple[int, int]],
    blockers: set[tuple[int, int]],
) -> bool:
    if not np.array_equal(out, expected):
        return False
    if np.any(out == GREY) or np.any(out == ORANGE):
        return False
    r0, c0, r1, c1 = box
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    size = patch.shape[0]
    for orbit in _orbit_groups(size):
        vals = {int(patch[r, c]) for r, c in orbit}
        if len(vals) != 1:
            return False
    if not any(guide["local_shadow"] in mask_cells for guide in guides):
        return False
    grey_guides = [comp for comp in _components(inp == GREY) if any(not _bbox_contains(box, cell) for cell in comp)]
    if len(grey_guides) != len(guides):
        return False
    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    for comp in grey_guides:
        tip = _tip_of_component(comp, center)
        direction = _principal_direction(center, tip)
        shadow = _shadow_cell(box, tip, direction)
        if int(out[shadow]) not in (BLUE, RED):
            return False
    for r, c in blockers:
        if int(out[r, c]) != BLACK:
            return False
    if len(guides) >= 2:
        colors = {int(out[guide["tip"]]) for guide in guides}
        if len(colors) < 2:
            return False
    return True


def _attempt_case() -> tuple[np.ndarray, np.ndarray] | None:
    n, size = random.choice(((13, 5), (15, 5), (15, 7), (17, 7)))
    margin = (n - size) // 2
    box = (margin, margin, margin + size - 1, margin + size - 1)

    patch = _pattern_from_orbits(size)
    guides = _pick_guides(n, box, patch)
    if not guides:
        return None

    masked_patch, mask_cells = _mask_pattern(patch, guides)
    reserved = _reserve_mask(n, box, guides)
    blockers = _place_blockers(n, guides, reserved)

    expected = _oracle_output(n, box, patch, guides, blockers)
    inp = np.zeros((n, n), dtype=int)
    _paint_pattern(inp, box, masked_patch)
    for guide in guides:
        for r, c in guide["cells"]:
            inp[r, c] = GREY
    for r, c in blockers:
        inp[r, c] = ORANGE

    out = main(inp)
    if not _semantic_checks(inp, out, expected, box, guides, mask_cells, blockers):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        case = _attempt_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid symmetric-guide ray task")
