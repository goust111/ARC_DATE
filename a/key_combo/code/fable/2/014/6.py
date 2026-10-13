from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_MAX_GEN_ATTEMPTS = 500
_SHAPE_ATTEMPTS = 200
_MAX_RED_OPTIONS = 36
_MAX_BLUE_OPTIONS = 48
_MAX_AXIS_TRIES = 10
_MAX_STEP_TRIES = 3


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.asarray(coords, dtype=int)))


def _normalize(coords: np.ndarray) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    return pts - pts.min(axis=0)


def _normalized_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    return _coords_key(_normalize(coords))


def _transform_local(coords: np.ndarray, name: str) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    rows = pts[:, 0]
    cols = pts[:, 1]
    if name == "id":
        out = np.stack([rows, cols], axis=1)
    elif name == "flip_h":
        out = np.stack([rows, -cols], axis=1)
    elif name == "flip_v":
        out = np.stack([-rows, cols], axis=1)
    elif name == "rot180":
        out = np.stack([-rows, -cols], axis=1)
    elif name == "transpose":
        out = np.stack([cols, rows], axis=1)
    elif name == "rot90":
        out = np.stack([cols, -rows], axis=1)
    elif name == "rot270":
        out = np.stack([-cols, rows], axis=1)
    elif name == "anti_transpose":
        out = np.stack([-cols, -rows], axis=1)
    else:
        raise ValueError(f"unknown transform: {name}")
    return _normalize(out)


def _d4_keys(coords: np.ndarray) -> set[tuple[tuple[int, int], ...]]:
    names = ("id", "flip_h", "flip_v", "rot180", "transpose", "rot90", "rot270", "anti_transpose")
    return {_coords_key(_transform_local(coords, name)) for name in names}


def _rotation_keys(coords: np.ndarray) -> set[tuple[tuple[int, int], ...]]:
    names = ("rot90", "rot180", "rot270")
    return {_coords_key(_transform_local(coords, name)) for name in names}


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _apply_glide(coords: np.ndarray, orient: str, axis2: int, step: int) -> np.ndarray:
    pts = np.asarray(coords, dtype=int)
    if orient == "h":
        out = np.empty_like(pts)
        out[:, 0] = int(axis2) - pts[:, 0]
        out[:, 1] = pts[:, 1] + int(step)
        return out
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] + int(step)
    out[:, 1] = int(axis2) - pts[:, 1]
    return out


def _in_bounds(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    pts = np.asarray(coords, dtype=int)
    return bool(
        (pts[:, 0] >= 0).all()
        and (pts[:, 0] < h).all()
        and (pts[:, 1] >= 0).all()
        and (pts[:, 1] < w).all()
    )


def _strictly_on_one_side(coords: np.ndarray, orient: str, axis2: int) -> bool:
    pts = np.asarray(coords, dtype=int)
    vals = 2 * pts[:, 0] if orient == "h" else 2 * pts[:, 1]
    return bool(vals.max() < axis2 or vals.min() > axis2)


def _cells(coords: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.asarray(coords, dtype=int)}


def _expanded(cells: set[tuple[int, int]], pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                out.add((int(r + dr), int(c + dc)))
    return out


def _coord_sets_separated(coord_sets: list[np.ndarray], pad: int = 1) -> bool:
    blocked: set[tuple[int, int]] = set()
    for coords in coord_sets:
        cur = _cells(coords)
        if blocked & cur:
            return False
        blocked |= _expanded(cur, pad=pad)
    return True


def _candidate_from_pair(src: np.ndarray, dst: np.ndarray, orient: str) -> dict[str, int | str] | None:
    sr0, sc0, sr1, sc1 = _bbox_coords(src)
    tr0, tc0, tr1, tc1 = _bbox_coords(dst)
    sh, sw = sr1 - sr0 + 1, sc1 - sc0 + 1
    th, tw = tr1 - tr0 + 1, tc1 - tc0 + 1
    if (sh, sw) != (th, tw):
        return None

    if orient == "h":
        step = tc0 - sc0
        if step <= 0:
            return None
        axis2_a = sr0 + tr1
        axis2_b = sr1 + tr0
        if axis2_a != axis2_b:
            return None
        axis2 = int(axis2_a)
    else:
        step = tr0 - sr0
        if step <= 0:
            return None
        axis2_a = sc0 + tc1
        axis2_b = sc1 + tc0
        if axis2_a != axis2_b:
            return None
        axis2 = int(axis2_a)

    moved = _apply_glide(src, orient, axis2, int(step))
    if _coords_key(moved) != _coords_key(dst):
        return None
    return {"orient": orient, "axis2": axis2, "step": int(step)}


def _infer_glide_from_reds(red_components: list[np.ndarray]) -> dict[str, int | str]:
    assert len(red_components) == 2, "expected exactly two red components"

    candidates: list[dict[str, int | str]] = []
    horiz = sorted(red_components, key=lambda pts: (_bbox_coords(pts)[1], _bbox_coords(pts)[0]))
    cand_h = _candidate_from_pair(horiz[0], horiz[1], "h")
    if cand_h is not None:
        candidates.append(cand_h)

    vert = sorted(red_components, key=lambda pts: (_bbox_coords(pts)[0], _bbox_coords(pts)[1]))
    cand_v = _candidate_from_pair(vert[0], vert[1], "v")
    if cand_v is not None:
        candidates.append(cand_v)

    assert len(candidates) == 1, "red demonstration pair must determine a unique glide reflection"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the blue object along the hidden glide reflection demonstrated by two red footprints.

    Input structure:
    - The background is black.
    - Exactly two red connected components have the same shape and form one demonstration pair under a hidden horizontal-axis or vertical-axis glide reflection.
    - Exactly one blue connected component is present, and its shape is different from the red one.
    - The demonstrated glide step is read in the grid reference frame: left-to-right for horizontal-axis glides and top-to-bottom for vertical-axis glides.

    Rule:
    - Infer the unique axis position and positive glide distance d that map the earlier red footprint to the later one by mirror-then-slide.
    - Keep both red footprints and the original blue footprint unchanged.
    - Repeatedly apply that same glide reflection to the current blue footprint and paint each fully in-bounds image blue.
    - Stop as soon as the next image would leave the grid.

    Key insight:
    - The red pair is not explained by a plain translation, rotation, or mirror; only mirror-plus-forward-slide fits, and the blue object must follow that same alternating-side orbit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    assert int(bg) == BG, "expected black background"

    red_components = _components_of_color(g, RED)
    blue_components = _components_of_color(g, BLUE)
    assert len(red_components) == 2, "expected exactly two red objects"
    assert len(blue_components) == 1, "expected exactly one blue object"

    glide = _infer_glide_from_reds(red_components)
    orient = str(glide["orient"])
    axis2 = int(glide["axis2"])
    step = int(glide["step"])

    out = g.copy()
    cur = np.asarray(blue_components[0], dtype=int)
    while True:
        nxt = _apply_glide(cur, orient, axis2, step)
        if not _in_bounds(nxt, g.shape):
            break
        out[nxt[:, 0], nxt[:, 1]] = BLUE
        cur = nxt
    return out


def _random_connected_mask(
    min_h: int,
    max_h: int,
    min_w: int,
    max_w: int,
    min_area: int,
    max_area: int,
    orient: str,
) -> np.ndarray:
    for _ in range(_SHAPE_ATTEMPTS):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        max_cells = min(int(max_area), int(h * w - 1))
        min_cells = min(int(min_area), max_cells)
        if min_cells < 4 or min_cells > max_cells:
            continue
        target = random.randint(min_cells, max_cells)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in _NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            weights = []
            picks = list(frontier)
            for nr, nc in picks:
                score = 1
                for dr, dc in _NEIGHBORS4:
                    if (nr + dr, nc + dc) in cells:
                        score += 2
                weights.append(score)
            chosen = random.choices(picks, weights=weights, k=1)[0]
            cells.add(chosen)
        if len(cells) != target:
            continue

        coords = np.asarray(sorted(cells), dtype=int)
        coords = _normalize(coords)
        hh = int(coords[:, 0].max()) + 1
        ww = int(coords[:, 1].max()) + 1
        if hh < 2 or ww < 2:
            continue
        mask = np.zeros((hh, ww), dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        if mask.sum() == hh * ww:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if orient == "h" and np.array_equal(mask, np.flipud(mask)):
            continue
        if orient == "v" and np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    raise RuntimeError("failed to sample a suitable connected shape")


def _shape_differs(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    key_b = _normalized_key(np.argwhere(mask_b))
    return key_b not in _d4_keys(np.argwhere(mask_a))


def _pure_translation(src: np.ndarray, dst: np.ndarray) -> bool:
    return _normalized_key(src) == _normalized_key(dst)


def _pure_rotation(src: np.ndarray, dst: np.ndarray) -> bool:
    dst_key = _normalized_key(dst)
    return dst_key in _rotation_keys(src)


def _pure_axis_mirror(src: np.ndarray, dst: np.ndarray) -> bool:
    sr0, sc0, sr1, sc1 = _bbox_coords(src)
    tr0, tc0, tr1, tc1 = _bbox_coords(dst)

    if sc0 == tc0 and sc1 == tc1:
        axis2_a = sr0 + tr1
        axis2_b = sr1 + tr0
        if axis2_a == axis2_b:
            mirrored = np.empty_like(src)
            mirrored[:, 0] = int(axis2_a) - src[:, 0]
            mirrored[:, 1] = src[:, 1]
            if _coords_key(mirrored) == _coords_key(dst):
                return True

    if sr0 == tr0 and sr1 == tr1:
        axis2_a = sc0 + tc1
        axis2_b = sc1 + tc0
        if axis2_a == axis2_b:
            mirrored = np.empty_like(src)
            mirrored[:, 0] = src[:, 0]
            mirrored[:, 1] = int(axis2_a) - src[:, 1]
            if _coords_key(mirrored) == _coords_key(dst):
                return True

    return False


def _paint(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _orbit_positions(
    mask: np.ndarray,
    grid_shape: tuple[int, int],
    orient: str,
    axis2: int,
    step: int,
    extra_steps: int,
    *,
    require_prev_oob: bool,
    require_next_oob: bool,
) -> list[list[np.ndarray]]:
    h, w = map(int, grid_shape)
    base = np.argwhere(mask).astype(int)
    mh = int(mask.shape[0])
    mw = int(mask.shape[1])
    options: list[list[np.ndarray]] = []

    for r0 in range(h - mh + 1):
        for c0 in range(w - mw + 1):
            cur = base + np.array([r0, c0], dtype=int)
            if not _strictly_on_one_side(cur, orient, axis2):
                continue
            if require_prev_oob and _in_bounds(_apply_glide(cur, orient, axis2, -step), grid_shape):
                continue

            orbit = [cur]
            ok = True
            for _ in range(extra_steps):
                nxt = _apply_glide(orbit[-1], orient, axis2, step)
                if not _in_bounds(nxt, grid_shape):
                    ok = False
                    break
                if not _strictly_on_one_side(nxt, orient, axis2):
                    ok = False
                    break
                orbit.append(nxt)
            if not ok:
                continue
            if require_next_oob and _in_bounds(_apply_glide(orbit[-1], orient, axis2, step), grid_shape):
                continue
            if not _coord_sets_separated(orbit, pad=1):
                continue
            options.append(orbit)

    random.shuffle(options)
    return options


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    orient = random.choice(["h", "v"])
    if orient == "h":
        height = random.randint(12, 22)
        width = random.randint(18, 29)
        axis_choices = list(range(1, 2 * height - 2))
    else:
        height = random.randint(18, 29)
        width = random.randint(12, 22)
        axis_choices = list(range(1, 2 * width - 2))

    random.shuffle(axis_choices)
    if len(axis_choices) > _MAX_AXIS_TRIES:
        axis_choices = axis_choices[:_MAX_AXIS_TRIES]

    red_mask = _random_connected_mask(2, 4, 2, 4, 4, 7, orient)
    blue_mask = _random_connected_mask(2, 4, 2, 4, 4, 7, orient)
    if not _shape_differs(red_mask, blue_mask):
        return None

    step_choices = [2, 3, 4, 5]
    random.shuffle(step_choices)
    step_choices = step_choices[:_MAX_STEP_TRIES]
    orbit_lengths = [2, 3, 4]
    random.shuffle(orbit_lengths)

    grid_shape = (height, width)
    red_key = _normalized_key(np.argwhere(red_mask))
    blue_key = _normalized_key(np.argwhere(blue_mask))
    if red_key == blue_key:
        return None

    for axis2 in axis_choices:
        for step in step_choices:
            red_orbits = _orbit_positions(
                red_mask,
                grid_shape,
                orient,
                axis2,
                step,
                1,
                require_prev_oob=True,
                require_next_oob=False,
            )
            if not red_orbits:
                continue
            if len(red_orbits) > _MAX_RED_OPTIONS:
                red_orbits = red_orbits[:_MAX_RED_OPTIONS]

            blue_options_by_len: dict[int, list[list[np.ndarray]]] = {}
            for blue_steps in orbit_lengths:
                blue_orbits = _orbit_positions(
                    blue_mask,
                    grid_shape,
                    orient,
                    axis2,
                    step,
                    blue_steps,
                    require_prev_oob=True,
                    require_next_oob=True,
                )
                if len(blue_orbits) > _MAX_BLUE_OPTIONS:
                    blue_orbits = blue_orbits[:_MAX_BLUE_OPTIONS]
                blue_options_by_len[blue_steps] = blue_orbits

            for red_orbit in red_orbits:
                red_src, red_dst = red_orbit
                if _pure_translation(red_src, red_dst):
                    continue
                if _pure_rotation(red_src, red_dst):
                    continue
                if _pure_axis_mirror(red_src, red_dst):
                    continue

                occupied = [red_src, red_dst]
                for blue_steps in orbit_lengths:
                    blue_orbits = blue_options_by_len[blue_steps]
                    if not blue_orbits:
                        continue
                    for blue_orbit in blue_orbits:
                        if not _coord_sets_separated(occupied + blue_orbit, pad=1):
                            continue

                        inp = np.full(grid_shape, BG, dtype=int)
                        _paint(inp, red_src, RED)
                        _paint(inp, red_dst, RED)
                        _paint(inp, blue_orbit[0], BLUE)

                        out = inp.copy()
                        for coords in blue_orbit[1:]:
                            _paint(out, coords, BLUE)

                        try:
                            red_components = _components_of_color(inp, RED)
                            inferred = _infer_glide_from_reds(red_components)
                        except AssertionError:
                            continue

                        if (
                            str(inferred["orient"]) != orient
                            or int(inferred["axis2"]) != int(axis2)
                            or int(inferred["step"]) != int(step)
                        ):
                            continue
                        if not np.array_equal(main(inp), out):
                            continue
                        if np.array_equal(inp, out):
                            continue
                        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, _out = built
        return inp
    raise RuntimeError("failed to generate a valid glide-reflection footprint task")
