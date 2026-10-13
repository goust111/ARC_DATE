from __future__ import annotations

from collections import Counter

import numpy as np


GRAY = 5


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_component(
    grid: np.ndarray, coords: np.ndarray, bg: int
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    local = coords - np.array([r0, c0], dtype=int)
    patch[local[:, 0], local[:, 1]] = grid[coords[:, 0], coords[:, 1]]
    anchor_local = np.argwhere(patch == GRAY)
    assert anchor_local.shape == (1, 2), "source sprite must contain exactly one gray anchor"
    return patch, (r0, c0), tuple(map(int, anchor_local[0]))


def _flip_patch(patch: np.ndarray, axis: str) -> np.ndarray:
    if axis == "lr":
        return np.fliplr(patch)
    if axis == "ud":
        return np.flipud(patch)
    raise ValueError(f"unknown flip axis: {axis}")


def _mirrored_anchor(anchor: tuple[int, int], shape: tuple[int, int], axis: str) -> tuple[int, int]:
    ar, ac = anchor
    h, w = shape
    if axis == "lr":
        return ar, w - 1 - ac
    return h - 1 - ar, ac


def _visible_cells_for_anchor(
    patch: np.ndarray,
    bg: int,
    anchor_pos: tuple[int, int],
    local_anchor: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int, int]]:
    top = int(anchor_pos[0] - local_anchor[0])
    left = int(anchor_pos[1] - local_anchor[1])
    cells: list[tuple[int, int, int]] = []
    for pr, pc in np.argwhere(patch != bg):
        rr = top + int(pr)
        cc = left + int(pc)
        if _in_bounds(shape, rr, cc):
            cells.append((rr, cc, int(patch[int(pr), int(pc)])))
    return cells


def _stamp_cells(grid: np.ndarray, cells: list[tuple[int, int, int]]) -> None:
    for r, c, color in cells:
        grid[r, c] = int(color)


def _extract_scene(
    grid: np.ndarray,
) -> tuple[int, np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _find_components(g != bg)

    source_comp: np.ndarray | None = None
    target_anchor: tuple[int, int] | None = None
    for coords in comps:
        vals = g[coords[:, 0], coords[:, 1]]
        gray_locs = coords[vals == GRAY]
        if gray_locs.shape == (1, 2) and coords.shape[0] > 1 and np.any(vals != GRAY):
            assert source_comp is None, "expected exactly one source sprite"
            source_comp = coords
        elif coords.shape == (1, 2) and int(vals[0]) == GRAY:
            assert target_anchor is None, "expected exactly one isolated target anchor"
            target_anchor = tuple(map(int, coords[0]))

    assert source_comp is not None, "source sprite not found"
    assert target_anchor is not None, "isolated target anchor not found"

    patch, origin, anchor_local = _crop_component(g, source_comp, bg)
    source_anchor = (origin[0] + anchor_local[0], origin[1] + anchor_local[1])
    return bg, patch, origin, anchor_local, target_anchor


def _render_orbit(
    base: np.ndarray,
    patch: np.ndarray,
    source_origin: tuple[int, int],
    anchor_local: tuple[int, int],
    offset: tuple[int, int],
    axis: str,
    variant: str = "true",
) -> np.ndarray:
    bg = _background_color(base)
    out = np.array(base, dtype=int, copy=True)
    source_anchor = (source_origin[0] + anchor_local[0], source_origin[1] + anchor_local[1])

    mirror_patch = _flip_patch(patch, axis)
    wrong_axis = "ud" if axis == "lr" else "lr"
    wrong_mirror_patch = _flip_patch(patch, wrong_axis)
    mirror_anchor = _mirrored_anchor(anchor_local, patch.shape, axis)
    wrong_mirror_anchor = _mirrored_anchor(anchor_local, patch.shape, wrong_axis)

    step = 1
    anchor_pos = (source_anchor[0] + offset[0], source_anchor[1] + offset[1])
    while True:
        if variant == "true":
            use_mirror = step % 2 == 1
            step_patch = mirror_patch if use_mirror else patch
            step_anchor = mirror_anchor if use_mirror else anchor_local
        elif variant == "original_only":
            step_patch = patch
            step_anchor = anchor_local
        elif variant == "mirror_only":
            step_patch = mirror_patch
            step_anchor = mirror_anchor
        elif variant == "wrong_axis":
            use_mirror = step % 2 == 1
            step_patch = wrong_mirror_patch if use_mirror else patch
            step_anchor = wrong_mirror_anchor if use_mirror else anchor_local
        else:
            raise ValueError(f"unknown variant: {variant}")

        cells = _visible_cells_for_anchor(step_patch, bg, anchor_pos, step_anchor, out.shape)
        if not cells:
            break
        _stamp_cells(out, cells)
        anchor_pos = (anchor_pos[0] + offset[0], anchor_pos[1] + offset[1])
        step += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the sprite repeatedly along the anchor offset, alternating original and mirrored copies.

    Input structure:
    - The dominant color is the background.
    - One 4-connected multicolor sprite contains exactly one gray anchor pixel inside it.
    - A second gray pixel appears alone elsewhere on the background as the target anchor.

    Rule:
    - Compute the translation vector from the sprite's gray anchor to the isolated gray anchor.
    - If the horizontal component has larger magnitude than the vertical component, use a left-right mirror; otherwise use an up-down mirror.
    - Keep the original sprite unchanged, then place a first mirrored copy so the mirrored anchor lands on the isolated anchor.
    - Continue stepping by the same vector, alternating mirrored and original copies, clipping any out-of-bounds cells, until the next copy is fully outside the grid.

    Key insight:
    - The two gray pixels define an orbit generator, not a one-off alignment: the offset drives every later stamp, and its dominant component also chooses the mirror axis.
    """
    g = np.asarray(grid, dtype=int)
    bg, patch, source_origin, anchor_local, target_anchor = _extract_scene(g)

    source_anchor = (source_origin[0] + anchor_local[0], source_origin[1] + anchor_local[1])
    offset = (target_anchor[0] - source_anchor[0], target_anchor[1] - source_anchor[1])
    assert offset != (0, 0), "the two anchors must be distinct"
    axis = "lr" if abs(offset[1]) > abs(offset[0]) else "ud"

    return _render_orbit(g, patch, source_origin, anchor_local, offset, axis, variant="true")


# --------------- generator (o5x) ---------------

_GRID_MIN = 14
_GRID_MAX = 24
_SPRITE_H_RANGE = (3, 6)
_SPRITE_W_RANGE = (3, 6)
_SPRITE_CELLS_RANGE = (6, 10)
_MAX_TRIES = 500
_PALETTE = tuple(color for color in range(10) if color != GRAY)


def _random_connected_cells(
    rng: np.random.Generator, h: int, w: int, count: int
) -> set[tuple[int, int]]:
    start = (int(rng.integers(h)), int(rng.integers(w)))
    cells = {start}
    while len(cells) < count:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_connected_cells(rng, h, w, count)
        pick = frontier[int(rng.integers(len(frontier)))]
        cells.add(pick)
    return cells


def _make_sprite(
    rng: np.random.Generator, bg: int
) -> tuple[np.ndarray, tuple[int, int]] | None:
    h = int(rng.integers(_SPRITE_H_RANGE[0], _SPRITE_H_RANGE[1] + 1))
    w = int(rng.integers(_SPRITE_W_RANGE[0], _SPRITE_W_RANGE[1] + 1))
    max_cells = min(_SPRITE_CELLS_RANGE[1], h * w)
    min_cells = min(_SPRITE_CELLS_RANGE[0], max_cells)
    count = int(rng.integers(min_cells, max_cells + 1))
    cells = _random_connected_cells(rng, h, w, count)

    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) < 2 or len(cols) < 2:
        return None

    neighborful = []
    for r, c in cells:
        if any((nr, nc) in cells for nr, nc in _neighbors4(r, c)):
            neighborful.append((r, c))
    if not neighborful:
        return None

    anchor = neighborful[int(rng.integers(len(neighborful)))]
    body = sorted(cell for cell in cells if cell != anchor)
    if len(body) < 2:
        return None

    usable = [color for color in _PALETTE if color != bg]
    color_count = int(rng.integers(2, min(4, len(body)) + 1))
    chosen = list(map(int, rng.choice(usable, size=color_count, replace=False)))
    colors = chosen.copy()
    while len(colors) < len(body):
        colors.append(chosen[int(rng.integers(len(chosen)))])
    rng.shuffle(colors)

    patch = np.full((h, w), int(bg), dtype=int)
    for (r, c), color in zip(body, colors):
        patch[r, c] = int(color)
    patch[anchor] = GRAY
    return patch, anchor


def _anchor_fits_full(
    anchor_pos: tuple[int, int],
    local_anchor: tuple[int, int],
    patch_shape: tuple[int, int],
    grid_shape: tuple[int, int],
) -> bool:
    ar, ac = local_anchor
    h, w = patch_shape
    r, c = anchor_pos
    return ar <= r <= grid_shape[0] - h + ar and ac <= c <= grid_shape[1] - w + ac


def _source_cells(
    patch: np.ndarray, bg: int, source_origin: tuple[int, int]
) -> set[tuple[int, int]]:
    return {
        (source_origin[0] + int(r), source_origin[1] + int(c))
        for r, c in np.argwhere(patch != bg)
    }


def _trajectory_info(
    patch: np.ndarray,
    bg: int,
    source_origin: tuple[int, int],
    anchor_local: tuple[int, int],
    offset: tuple[int, int],
    axis: str,
    grid_shape: tuple[int, int],
) -> list[dict[str, object]] | None:
    source_anchor = (source_origin[0] + anchor_local[0], source_origin[1] + anchor_local[1])
    mirror_patch = _flip_patch(patch, axis)
    mirror_anchor = _mirrored_anchor(anchor_local, patch.shape, axis)
    total_cells = int(np.count_nonzero(patch != bg))

    occupied = set(_source_cells(patch, bg, source_origin))
    info: list[dict[str, object]] = []
    step = 1
    anchor_pos = (source_anchor[0] + offset[0], source_anchor[1] + offset[1])
    while True:
        step_patch = mirror_patch if step % 2 == 1 else patch
        step_anchor = mirror_anchor if step % 2 == 1 else anchor_local
        cells = _visible_cells_for_anchor(step_patch, bg, anchor_pos, step_anchor, grid_shape)
        if not cells:
            break
        visible = {(r, c) for r, c, _ in cells}
        if len(visible) < 2 or occupied & visible:
            return None
        occupied |= visible
        info.append(
            {
                "step": step,
                "anchor": anchor_pos,
                "visible_count": len(visible),
                "full": len(visible) == total_cells,
            }
        )
        if len(info) > 5:
            return None
        anchor_pos = (anchor_pos[0] + offset[0], anchor_pos[1] + offset[1])
        step += 1
    return info


def _render_input_clean(
    shape: tuple[int, int],
    bg: int,
    patch: np.ndarray,
    source_origin: tuple[int, int],
    target_anchor: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for r, c in np.argwhere(patch != bg):
        grid[source_origin[0] + int(r), source_origin[1] + int(c)] = int(patch[int(r), int(c)])
    grid[target_anchor] = GRAY
    return grid


def _signed_value(rng: np.random.Generator, magnitude: int) -> int:
    if magnitude == 0:
        return 0
    return int(rng.choice([-1, 1]) * magnitude)


def _sample_offset(rng: np.random.Generator, axis_goal: str) -> tuple[int, int, str]:
    if axis_goal == "lr":
        dc_mag = int(rng.integers(3, 7))
        dr_mag = int(rng.integers(0, dc_mag))
        return _signed_value(rng, dr_mag), _signed_value(rng, dc_mag), "lr"

    if bool(rng.integers(2)):
        mag = int(rng.integers(2, 6))
        return _signed_value(rng, mag), _signed_value(rng, mag), "ud"

    dr_mag = int(rng.integers(3, 7))
    dc_mag = int(rng.integers(0, dr_mag))
    return _signed_value(rng, dr_mag), _signed_value(rng, dc_mag), "ud"


def _candidate_targets(
    offset: tuple[int, int],
    patch: np.ndarray,
    anchor_local: tuple[int, int],
    axis: str,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    mirror_anchor = _mirrored_anchor(anchor_local, patch.shape, axis)
    targets: list[tuple[int, int]] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            a1 = (r, c)
            a0 = (r - offset[0], c - offset[1])
            a2 = (r + offset[0], c + offset[1])
            a3 = (r + 2 * offset[0], c + 2 * offset[1])
            if not _anchor_fits_full(a0, anchor_local, patch.shape, shape):
                continue
            if not _anchor_fits_full(a1, mirror_anchor, patch.shape, shape):
                continue
            if not _anchor_fits_full(a2, anchor_local, patch.shape, shape):
                continue
            if not _anchor_fits_full(a3, mirror_anchor, patch.shape, shape):
                continue
            targets.append(a1)
    return targets


def _target_is_isolated(
    target: tuple[int, int], source_cells: set[tuple[int, int]]
) -> bool:
    if target in source_cells:
        return False
    return all((nr, nc) not in source_cells for nr, nc in _neighbors4(*target))


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    patch: np.ndarray,
    source_origin: tuple[int, int],
    anchor_local: tuple[int, int],
    offset: tuple[int, int],
    axis: str,
    copy_info: list[dict[str, object]],
) -> bool:
    if int(np.count_nonzero(inp == GRAY)) != 2:
        return False
    comps = _find_components(inp != _background_color(inp))
    if len(comps) != 2:
        return False
    if len(copy_info) < 3:
        return False
    if not all(bool(copy_info[i]["full"]) for i in range(3)):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    base = np.array(inp, dtype=int, copy=True)
    for variant in ("original_only", "mirror_only", "wrong_axis"):
        alt = _render_orbit(base, patch, source_origin, anchor_local, offset, axis, variant=variant)
        if np.array_equal(alt, expected):
            return False
    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
    w = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
    bg = int(rng.choice(_PALETTE))
    want_clip = bool(rng.integers(2))

    sprite = _make_sprite(rng, bg)
    if sprite is None:
        return None
    patch, anchor_local = sprite

    for _ in range(150):
        axis_goal = "lr" if bool(rng.integers(2)) else "ud"
        dr, dc, axis = _sample_offset(rng, axis_goal)
        offset = (dr, dc)
        targets = _candidate_targets(offset, patch, anchor_local, axis, (h, w))
        if not targets:
            continue
        rng.shuffle(targets)
        for target_anchor in targets[:80]:
            source_anchor = (target_anchor[0] - dr, target_anchor[1] - dc)
            source_origin = (source_anchor[0] - anchor_local[0], source_anchor[1] - anchor_local[1])
            source_cells = _source_cells(patch, bg, source_origin)
            if not _target_is_isolated(target_anchor, source_cells):
                continue

            copy_info = _trajectory_info(
                patch, bg, source_origin, anchor_local, offset, axis, (h, w)
            )
            if copy_info is None or not (3 <= len(copy_info) <= 5):
                continue

            has_partial = any(not bool(item["full"]) for item in copy_info)
            if want_clip:
                if len(copy_info) < 4 or not has_partial or any(not bool(item["full"]) for item in copy_info[:3]):
                    continue
            else:
                if has_partial:
                    continue

            inp = _render_input_clean((h, w), bg, patch, source_origin, target_anchor)
            expected = _render_orbit(
                inp, patch, source_origin, anchor_local, offset, axis, variant="true"
            )
            if not _semantic_ok(
                inp, expected, patch, source_origin, anchor_local, offset, axis, copy_info
            ):
                continue
            return inp
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_TRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid alternating-mirror orbit input")
