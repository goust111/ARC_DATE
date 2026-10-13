from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
RED = 2
ORIENTATIONS = ("tl", "tr", "bl", "br")
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int):
    yield int(r) - 1, int(c)
    yield int(r) + 1, int(c)
    yield int(r), int(c) - 1
    yield int(r), int(c) + 1


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return mask.astype(int).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    total = int(prefix[r1, c1])
    if r0 > 0:
        total -= int(prefix[r0 - 1, c1])
    if c0 > 0:
        total -= int(prefix[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        total += int(prefix[r0 - 1, c0 - 1])
    return int(total)


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            p = (int(nr), int(nc))
            if p in coords and p not in seen:
                seen.add(p)
                q.append(p)
    return len(seen) == len(coords)


def _to_local(r: int, c: int, box: tuple[int, int, int, int], orient: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if orient == "tl":
        return int(r - r0), int(c - c0)
    if orient == "tr":
        return int(r - r0), int(c1 - c)
    if orient == "bl":
        return int(r1 - r), int(c - c0)
    if orient == "br":
        return int(r1 - r), int(c1 - c)
    raise ValueError(f"unknown orientation: {orient}")


def _from_local(u: int, v: int, box: tuple[int, int, int, int], orient: str) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if orient == "tl":
        return int(r0 + u), int(c0 + v)
    if orient == "tr":
        return int(r0 + u), int(c1 - v)
    if orient == "bl":
        return int(r1 - u), int(c0 + v)
    if orient == "br":
        return int(r1 - u), int(c1 - v)
    raise ValueError(f"unknown orientation: {orient}")


def _classify_local_cells(
    local_cells: set[tuple[int, int]], ph: int, pw: int
) -> dict[str, set[tuple[int, int]]] | None:
    groups = {"inside": set(), "top": set(), "left": set(), "corner": set()}
    for u, v in local_cells:
        if u >= ph or v >= pw:
            return None
        if u >= 0 and v >= 0:
            groups["inside"].add((int(u), int(v)))
            continue
        if u < 0 and 0 <= v < pw:
            if -u > ph:
                return None
            groups["top"].add((int(u), int(v)))
            continue
        if v < 0 and 0 <= u < ph:
            if -v > pw:
                return None
            groups["left"].add((int(u), int(v)))
            continue
        if u < 0 and v < 0:
            if -u > ph or -v > pw:
                return None
            groups["corner"].add((int(u), int(v)))
            continue
        return None
    return groups


def _fold_target(cell: tuple[int, int]) -> tuple[int, int]:
    u, v = cell
    return int(-u - 1 if u < 0 else u), int(-v - 1 if v < 0 else v)


def _compose_patch(
    ph: int,
    pw: int,
    paper_color: int,
    sprite_color: int,
    groups: dict[str, set[tuple[int, int]]],
) -> np.ndarray:
    patch = np.full((ph, pw), int(paper_color), dtype=int)
    hits: Counter[tuple[int, int]] = Counter()
    for cell in groups["inside"]:
        hits[(int(cell[0]), int(cell[1]))] += 1
    for key in ("top", "left", "corner"):
        for cell in groups[key]:
            hits[_fold_target(cell)] += 1

    for (u, v), count in hits.items():
        patch[u, v] = int(RED if count >= 2 else sprite_color)
    return patch


def _render_scene(
    shape: tuple[int, int],
    box: tuple[int, int, int, int],
    paper_color: int,
    sprite_color: int,
    orient: str,
    local_cells: set[tuple[int, int]],
    *,
    output: bool,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = box
    ph = int(r1 - r0 + 1)
    pw = int(c1 - c0 + 1)
    if output:
        groups = _classify_local_cells(local_cells, ph, pw)
        assert groups is not None
        patch = _compose_patch(ph, pw, paper_color, sprite_color, groups)
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
        return grid

    grid[r0 : r1 + 1, c0 : c1 + 1] = int(paper_color)
    for u, v in local_cells:
        rr, cc = _from_local(int(u), int(v), box, orient)
        grid[rr, cc] = int(sprite_color)
    return grid


def _validate_candidate(
    grid: np.ndarray,
    bg: int,
    paper_color: int,
    sprite_color: int,
    box: tuple[int, int, int, int],
    orient: str,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = box
    ph = int(r1 - r0 + 1)
    pw = int(c1 - c0 + 1)
    if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] == int(bg)):
        return None

    sprite_coords = np.argwhere(grid == int(sprite_color))
    if sprite_coords.size == 0:
        return None

    local_cells = {
        _to_local(int(r), int(c), box, orient)
        for r, c in sprite_coords
    }
    groups = _classify_local_cells(local_cells, ph, pw)
    if groups is None:
        return None
    if not all(groups[name] for name in ("inside", "top", "left", "corner")):
        return None

    patch = _compose_patch(ph, pw, paper_color, sprite_color, groups)
    return {
        "bg": int(bg),
        "paper_color": int(paper_color),
        "sprite_color": int(sprite_color),
        "box": tuple(map(int, box)),
        "orient": orient,
        "local_cells": set(local_cells),
        "groups": groups,
        "patch": patch,
    }


def _find_unique_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(set(map(int, np.unique(g))) - {int(bg)})
    assert len(colors) == 2, "expected exactly two non-background colors"

    nonbg_prefix = _prefix_sum(g != int(bg))
    H, W = map(int, g.shape)
    candidates: list[dict[str, object]] = []

    for paper_color, sprite_color in ((colors[0], colors[1]), (colors[1], colors[0])):
        paper_coords = np.argwhere(g == int(paper_color))
        sprite_coords = np.argwhere(g == int(sprite_color))
        if paper_coords.size == 0 or sprite_coords.size == 0:
            continue
        pr0, pc0, pr1, pc1 = _bbox_coords(paper_coords)

        for r0 in range(pr0 + 1):
            for r1 in range(pr1, H):
                for c0 in range(pc0 + 1):
                    for c1 in range(pc1, W):
                        area = (r1 - r0 + 1) * (c1 - c0 + 1)
                        if _rect_sum(nonbg_prefix, r0, c0, r1, c1) != area:
                            continue
                        box = (int(r0), int(c0), int(r1), int(c1))
                        for orient in ORIENTATIONS:
                            candidate = _validate_candidate(g, bg, paper_color, sprite_color, box, orient)
                            if candidate is not None:
                                candidates.append(candidate)

    assert len(candidates) == 1, f"expected one valid interpretation, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the sprite's two single-edge overhangs and corner overhang back onto the paper rectangle.

    Input structure:
    - The background is black, with one visible paper rectangle in one color and one sprite in another color.
    - The paper is an axis-aligned solid rectangle under the sprite, so every paper cell is either paper color or sprite color.
    - The sprite crosses exactly one paper corner: it has pixels on the paper, beyond each of the two adjacent edges, and beyond the corner outside both edges.

    Rule:
    - Infer the unique paper rectangle, the sprite color, and which paper corner is crossed.
    - Keep the sprite pixels that already lie on the paper.
    - Reflect pixels hanging over only one paper edge across that edge, and reflect corner-outside pixels across both edges.
    - Inside the paper, empty paper cells hit once become the sprite color, while any landing cell hit at least twice becomes red; erase every sprite pixel outside the paper.

    Key insight:
    - After standardizing the crossed corner to a top-left local frame, the three outside regions become one horizontal mirror, one vertical mirror, and one double mirror whose landing counts determine sprite versus red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    scene = _find_unique_scene(g)

    out = np.full_like(g, int(scene["bg"]))
    r0, c0, r1, c1 = scene["box"]
    out[r0 : r1 + 1, c0 : c1 + 1] = scene["patch"]
    return out


def _manhattan_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    u, v = a
    path = [(int(u), int(v))]
    axes = [0, 1]
    random.shuffle(axes)
    for axis in axes:
        if axis == 0:
            while u != b[0]:
                u += 1 if b[0] > u else -1
                path.append((int(u), int(v)))
        else:
            while v != b[1]:
                v += 1 if b[1] > v else -1
                path.append((int(u), int(v)))
    return path


def _grow_connected_shape(
    cells: set[tuple[int, int]],
    *,
    u_min: int,
    u_max: int,
    v_min: int,
    v_max: int,
    target_area: int,
) -> set[tuple[int, int]]:
    cells = set(cells)
    attempts = 0
    while len(cells) < target_area and attempts < target_area * 50:
        attempts += 1
        u, v = random.choice(tuple(cells))
        du, dv = random.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
        nu, nv = int(u + du), int(v + dv)
        if not (u_min <= nu <= u_max and v_min <= nv <= v_max):
            continue
        cells.add((nu, nv))
    return cells


def _count_red(patch: np.ndarray) -> int:
    return int(np.sum(patch == RED))


def _patch_changes_without(
    ph: int,
    pw: int,
    paper_color: int,
    sprite_color: int,
    groups: dict[str, set[tuple[int, int]]],
    removed: str,
) -> bool:
    base = _compose_patch(ph, pw, paper_color, sprite_color, groups)
    changed_groups = {key: set(value) for key, value in groups.items()}
    changed_groups[removed] = set()
    alt = _compose_patch(ph, pw, paper_color, sprite_color, changed_groups)
    return not np.array_equal(base, alt)


def _canonical_input_checks(
    ph: int,
    pw: int,
    paper_color: int,
    sprite_color: int,
    local_cells: set[tuple[int, int]],
) -> tuple[dict[str, set[tuple[int, int]]], np.ndarray] | None:
    if not _is_connected(local_cells):
        return None
    groups = _classify_local_cells(local_cells, ph, pw)
    if groups is None:
        return None
    if not all(groups[name] for name in ("inside", "top", "left", "corner")):
        return None

    patch = _compose_patch(ph, pw, paper_color, sprite_color, groups)
    has_collision = _count_red(patch) > 0
    if not has_collision:
        return None

    if not _patch_changes_without(ph, pw, paper_color, sprite_color, groups, "inside"):
        return None
    if not _patch_changes_without(ph, pw, paper_color, sprite_color, groups, "top"):
        return None
    if not _patch_changes_without(ph, pw, paper_color, sprite_color, groups, "left"):
        return None
    if not _patch_changes_without(ph, pw, paper_color, sprite_color, groups, "corner"):
        return None

    if _count_red(patch) > max(4, (ph * pw) // 6):
        return None
    if int(np.sum(patch == int(sprite_color))) < 4:
        return None
    if int(np.sum(patch == int(paper_color))) < max(6, (ph * pw) // 3):
        return None
    return groups, patch


def _sample_canonical_sprite(
    ph: int,
    pw: int,
) -> set[tuple[int, int]] | None:
    max_up = min(4, ph - 2)
    max_left = min(4, pw - 2)
    if max_up < 1 or max_left < 1:
        return None

    up = random.randint(1, max_up)
    left = random.randint(1, max_left)
    ih = random.randint(max(3, up + 1), min(ph, 7))
    iw = random.randint(max(3, left + 1), min(pw, 7))

    inside_seed = (random.randint(0, ih - 1), random.randint(0, iw - 1))
    top_seed = (-random.randint(1, up), random.randint(0, iw - 1))
    left_seed = (random.randint(0, ih - 1), -random.randint(1, left))
    corner_seed = (-random.randint(1, up), -random.randint(1, left))

    cells: set[tuple[int, int]] = set()
    for a, b in (
        (corner_seed, top_seed),
        (corner_seed, left_seed),
        (top_seed, inside_seed),
        (left_seed, inside_seed),
    ):
        cells.update(_manhattan_path(a, b))

    collision_target = (-corner_seed[0] - 1, -corner_seed[1] - 1)
    if not (0 <= collision_target[0] < ih and 0 <= collision_target[1] < iw):
        return None
    cells.add(collision_target)

    local_area = (up + ih) * (left + iw)
    target_area = random.randint(max(8, len(cells) + 1), min(local_area - 1, len(cells) + 12))
    cells = _grow_connected_shape(
        cells,
        u_min=-up,
        u_max=ih - 1,
        v_min=-left,
        v_max=iw - 1,
        target_area=target_area,
    )
    return cells


def _paper_edge_visibility(
    input_grid: np.ndarray,
    box: tuple[int, int, int, int],
    paper_color: int,
) -> bool:
    r0, c0, r1, c1 = box
    patch = input_grid[r0 : r1 + 1, c0 : c1 + 1]
    return bool(
        np.any(patch[0, :] == int(paper_color))
        and np.any(patch[-1, :] == int(paper_color))
        and np.any(patch[:, 0] == int(paper_color))
        and np.any(patch[:, -1] == int(paper_color))
    )


def _place_box(
    H: int,
    W: int,
    ph: int,
    pw: int,
    orient: str,
    local_cells: set[tuple[int, int]],
) -> tuple[int, int, int, int] | None:
    u_over = int(-min(u for u, _ in local_cells))
    v_over = int(-min(v for _, v in local_cells))

    if orient == "tl":
        r0_lo, r0_hi = u_over + 1, H - ph - 1
        c0_lo, c0_hi = v_over + 1, W - pw - 1
    elif orient == "tr":
        r0_lo, r0_hi = u_over + 1, H - ph - 1
        c0_lo, c0_hi = 1, W - pw - v_over - 1
    elif orient == "bl":
        r0_lo, r0_hi = 1, H - ph - u_over - 1
        c0_lo, c0_hi = v_over + 1, W - pw - 1
    else:
        r0_lo, r0_hi = 1, H - ph - u_over - 1
        c0_lo, c0_hi = 1, W - pw - v_over - 1

    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    r0 = random.randint(r0_lo, r0_hi)
    c0 = random.randint(c0_lo, c0_hi)
    return int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1)


def _gen_attempt() -> np.ndarray | None:
    ph = random.randint(6, 11)
    pw = random.randint(6, 11)
    paper_color, sprite_color = random.sample(PALETTE, 2)
    orient = random.choice(ORIENTATIONS)

    local_cells = _sample_canonical_sprite(ph, pw)
    if local_cells is None:
        return None

    checked = _canonical_input_checks(ph, pw, paper_color, sprite_color, local_cells)
    if checked is None:
        return None
    groups, _patch = checked

    u_over = int(-min(u for u, _ in local_cells))
    v_over = int(-min(v for _, v in local_cells))
    min_H = ph + u_over + 3
    min_W = pw + v_over + 3
    H = random.randint(min_H, min(30, min_H + 7))
    W = random.randint(min_W, min(30, min_W + 7))

    box = _place_box(H, W, ph, pw, orient, local_cells)
    if box is None:
        return None

    inp = _render_scene((H, W), box, paper_color, sprite_color, orient, local_cells, output=False)
    out = _render_scene((H, W), box, paper_color, sprite_color, orient, local_cells, output=True)

    if not _paper_edge_visibility(inp, box, paper_color):
        return None
    if np.array_equal(inp, out):
        return None
    if len(np.unique(inp)) != 3:
        return None
    if len(np.unique(out)) < 3:
        return None
    if int(np.sum(out == RED)) == 0:
        return None
    if int(np.sum(inp == int(sprite_color))) == int(np.sum(out == int(sprite_color))):
        return None

    try:
        scene = _find_unique_scene(inp)
    except AssertionError:
        return None
    if tuple(scene["box"]) != tuple(box):
        return None
    if str(scene["orient"]) != orient:
        return None
    if int(scene["paper_color"]) != int(paper_color) or int(scene["sprite_color"]) != int(sprite_color):
        return None
    if not np.array_equal(main(inp), out):
        return None

    # Make sure every outside category changes the rendered output on the full canvas.
    for removed in ("inside", "top", "left", "corner"):
        changed = {key: set(value) for key, value in groups.items()}
        changed[removed] = set()
        reduced_cells = set().union(*changed.values())
        alt = _render_scene((H, W), box, paper_color, sprite_color, orient, reduced_cells, output=True)
        if np.array_equal(alt, out):
            return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-corner fold input")
