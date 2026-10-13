import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(1, 10))
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRY_LIMIT = 400
GEN_MODES = ("none", "row", "col", "both")
MAX_FUTURE_STEPS = 5
_GEN_CALL_COUNT = 0


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
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
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS8:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_from_sprite(sprite: np.ndarray, bg: int = 0) -> np.ndarray:
    return np.argwhere(sprite != bg)


def _occupied_cells(sprite: np.ndarray, top: tuple[int, int], bg: int = 0) -> set[tuple[int, int]]:
    r0, c0 = map(int, top)
    return {
        (r0 + int(r), c0 + int(c))
        for r, c in np.argwhere(sprite != bg)
    }


def _buffered_cells(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _all_separated(occupancies: list[set[tuple[int, int]]]) -> bool:
    buffered: set[tuple[int, int]] = set()
    for occ in occupancies:
        if occ & buffered:
            return False
        buffered |= _buffered_cells(occ)
    return True


def _sprite_fits(grid_shape: tuple[int, int], sprite_shape: tuple[int, int], top: tuple[int, int]) -> bool:
    h, w = map(int, sprite_shape)
    gh, gw = map(int, grid_shape)
    r0, c0 = map(int, top)
    return 0 <= r0 and 0 <= c0 and r0 + h <= gh and c0 + w <= gw


def _paste(dst: np.ndarray, sprite: np.ndarray, top: tuple[int, int], *, bg: int = 0, color: int | None = None) -> None:
    r0, c0 = map(int, top)
    coords = np.argwhere(sprite != bg)
    if color is None:
        dst[r0 + coords[:, 0], c0 + coords[:, 1]] = sprite[coords[:, 0], coords[:, 1]]
    else:
        dst[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)


def _sprite_hits(sprite: np.ndarray, top: tuple[int, int], pos: tuple[int, int], bg: int = 0) -> bool:
    lr = int(pos[0] - top[0])
    lc = int(pos[1] - top[1])
    return 0 <= lr < sprite.shape[0] and 0 <= lc < sprite.shape[1] and int(sprite[lr, lc]) != int(bg)


def _advance(
    top: tuple[int, int],
    sprite: np.ndarray,
    step: tuple[int, int],
    grid_shape: tuple[int, int],
) -> tuple[tuple[int, int], np.ndarray, tuple[int, int], tuple[str, ...]]:
    dr, dc = map(int, step)
    next_sprite = sprite
    overflow_row = top[0] + dr < 0 or top[0] + dr + sprite.shape[0] > grid_shape[0]
    overflow_col = top[1] + dc < 0 or top[1] + dc + sprite.shape[1] > grid_shape[1]
    axes: list[str] = []
    if overflow_row:
        dr = -dr
        next_sprite = np.flipud(next_sprite)
        axes.append("row")
    if overflow_col:
        dc = -dc
        next_sprite = np.fliplr(next_sprite)
        axes.append("col")
    next_top = (int(top[0] + dr), int(top[1] + dc))
    assert _sprite_fits(grid_shape, next_sprite.shape, next_top)
    return next_top, next_sprite.copy(), (int(dr), int(dc)), tuple(axes)


def _candidate_partial_tops(
    full_sprite: np.ndarray,
    partial_cells: list[tuple[int, int, int]],
    grid_shape: tuple[int, int],
    bg: int = 0,
) -> list[tuple[int, int]]:
    full_cells = [(int(r), int(c), int(full_sprite[r, c])) for r, c in np.argwhere(full_sprite != bg)]
    color_counts = Counter(color for _, _, color in full_cells)
    anchor = min(partial_cells, key=lambda item: (color_counts[item[2]], item[0], item[1]))
    ar, ac, acol = anchor
    candidates: set[tuple[int, int]] = set()
    for lr, lc, color in full_cells:
        if color != acol:
            continue
        top = (int(ar - lr), int(ac - lc))
        if not _sprite_fits(grid_shape, full_sprite.shape, top):
            continue
        ok = True
        for rr, cc, pcolor in partial_cells:
            sr = rr - top[0]
            sc = cc - top[1]
            if not (0 <= sr < full_sprite.shape[0] and 0 <= sc < full_sprite.shape[1]):
                ok = False
                break
            if int(full_sprite[sr, sc]) != int(pcolor):
                ok = False
                break
        if ok:
            candidates.add(top)
    return sorted(candidates)


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    comps = _component_coords(g != bg)
    assert len(comps) >= 3, "expected a full sprite, a partial copy, and a marker pixel"

    areas = [int(len(comp)) for comp in comps]
    largest = max(areas)
    full_indices = [i for i, area in enumerate(areas) if area == largest]
    assert len(full_indices) == 1 and largest > 1, "expected one unique largest sprite"
    full_idx = full_indices[0]
    full_coords = comps[full_idx]
    full_box = _bbox(full_coords)
    full_sprite = _crop(g, full_box)
    full_top = (full_box[0], full_box[1])
    full_colors = {int(g[r, c]) for r, c in full_coords}

    marker_candidates: list[tuple[int, int, int, int]] = []
    for idx, comp in enumerate(comps):
        if idx == full_idx or len(comp) != 1:
            continue
        r, c = map(int, comp[0])
        color = int(g[r, c])
        if color not in full_colors:
            marker_candidates.append((idx, r, c, color))
    assert len(marker_candidates) == 1, "expected a unique marker pixel with a non-sprite color"
    marker_idx, mr, mc, marker_color = marker_candidates[0]

    partial_coords: list[tuple[int, int, int]] = []
    for idx, comp in enumerate(comps):
        if idx in (full_idx, marker_idx):
            continue
        for r, c in comp:
            partial_coords.append((int(r), int(c), int(g[r, c])))
    assert partial_coords, "expected visible pixels from the partial copy"

    candidate_tops = _candidate_partial_tops(full_sprite, partial_coords, g.shape, bg=bg)
    assert len(candidate_tops) == 1, "partial copy must imply a unique translation"
    partial_top = candidate_tops[0]
    step = (int(partial_top[0] - full_top[0]), int(partial_top[1] - full_top[1]))
    assert step != (0, 0), "expected a non-zero displacement vector"
    assert len(partial_coords) < int((full_sprite != bg).sum()), "partial copy must miss at least one pixel"

    return {
        "bg": int(bg),
        "full_sprite": full_sprite,
        "full_top": full_top,
        "partial_top": partial_top,
        "partial_cells": partial_coords,
        "marker_pos": (mr, mc),
        "marker_color": int(marker_color),
        "step": step,
    }


def _render(grid: np.ndarray, scene: dict) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = int(scene["bg"])
    marker_pos = tuple(scene["marker_pos"])
    marker_color = int(scene["marker_color"])
    current_top = tuple(scene["partial_top"])
    current_sprite = np.asarray(scene["full_sprite"], dtype=int).copy()
    step = tuple(scene["step"])

    limit = max(8, g.shape[0] * g.shape[1] * 2)
    for _ in range(limit):
        if _sprite_hits(current_sprite, current_top, marker_pos, bg=bg):
            _paste(out, current_sprite, current_top, bg=bg, color=marker_color)
            return out
        _paste(out, current_sprite, current_top, bg=bg)
        current_top, current_sprite, step, _ = _advance(current_top, current_sprite, step, g.shape)
    raise AssertionError("marker was never reached")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the partial sprite, then keep stamping it along the inferred vector with reflective mirroring until the marker is hit.

    Input structure:
    - A black background contains one complete multi-color sprite, one translated partial copy
      of the same sprite, and one isolated marker pixel in a different color.
    - The visible pixels of the partial copy keep their original sprite colors and preserve the
      same internal coordinates as the full sprite.
    - The sprite is asymmetric, so flips caused by boundary reflections are visually detectable.

    Rule:
    - Use the correspondence between the full sprite and the visible partial pixels to infer the
      unique displacement vector and to complete the partial copy.
    - Starting from that completed copy, stamp the full sprite repeatedly by the same vector.
    - If the next stamp would cross a grid boundary, reverse the offending vector component and
      mirror the sprite across that axis before continuing.
    - When a stamped copy covers the marker pixel, stop after that stamp and recolor the entire
      hit copy to the marker color.

    Key insight:
    - The partial copy is simultaneously a damaged sprite and a motion code: once its missing
      pixels are restored, the same translation drives a bouncing mirror-stamp trajectory.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    return _render(np.asarray(grid, dtype=int), scene)


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_connected_mask(height, width, area)
        chosen = random.choice(frontier)
        mask[chosen] = True
        cells.add(chosen)
    coords = np.argwhere(mask)
    box = _bbox(coords)
    return _crop(mask.astype(int), box).astype(bool)


def _make_sprite() -> np.ndarray:
    for _ in range(300):
        base_h = random.randint(3, 5)
        base_w = random.randint(3, 5)
        area = random.randint(5, min(7, base_h * base_w - 1))
        mask = _random_connected_mask(base_h, base_w, area)
        h, w = mask.shape
        if h < 3 or w < 3:
            continue
        if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
            continue
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        colors = random.sample(PALETTE, len(coords))
        random.shuffle(colors)
        sprite = np.zeros((h, w), dtype=int)
        for (r, c), color in zip(coords, colors):
            sprite[r, c] = int(color)
        if np.array_equal(sprite, np.flipud(sprite)) or np.array_equal(sprite, np.fliplr(sprite)):
            continue
        return sprite
    raise RuntimeError("failed to generate an asymmetric sprite")


def _choose_partial_subset(sprite: np.ndarray) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite != 0)]
    coord_set = set(coords)
    area = len(coords)
    min_keep = 3
    max_keep = max(min_keep, area - 2)
    target = random.randint(min_keep, max_keep)
    for _ in range(300):
        chosen = {random.choice(coords)}
        while len(chosen) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in chosen:
                for dr, dc in NEIGHBORS8:
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) in coord_set and (nr, nc) not in chosen:
                        frontier.append((nr, nc))
            if not frontier:
                break
            chosen.add(random.choice(frontier))
        if len(chosen) != target:
            continue
        part = np.zeros_like(sprite)
        for r, c in chosen:
            part[r, c] = sprite[r, c]
        if len(_component_coords(part != 0)) != 1:
            continue
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if len(rows) < 2 or len(cols) < 2:
            continue
        return part
    raise RuntimeError("failed to select a connected partial subset")


def _collect_states(
    sprite: np.ndarray,
    partial_top: tuple[int, int],
    step: tuple[int, int],
    grid_shape: tuple[int, int],
    max_future_steps: int,
) -> list[dict]:
    states = [
        {
            "top": tuple(partial_top),
            "sprite": sprite.copy(),
            "bounce_axes": (),
        }
    ]
    current_top = tuple(partial_top)
    current_sprite = sprite.copy()
    current_step = tuple(step)
    for _ in range(max_future_steps):
        current_top, current_sprite, current_step, axes = _advance(
            current_top,
            current_sprite,
            current_step,
            grid_shape,
        )
        states.append(
            {
                "top": tuple(current_top),
                "sprite": current_sprite.copy(),
                "bounce_axes": axes,
            }
        )
    return states


def _step_signature(states: list[dict], final_idx: int) -> tuple[bool, bool]:
    row = any("row" in state["bounce_axes"] for state in states[1 : final_idx + 1])
    col = any("col" in state["bounce_axes"] for state in states[1 : final_idx + 1])
    return row, col


def _has_prior_bounce(states: list[dict], final_idx: int) -> bool:
    return any(state["bounce_axes"] for state in states[1:final_idx])


def _sample_step(sprite_shape: tuple[int, int], mode: str) -> tuple[int, int]:
    h, w = map(int, sprite_shape)
    large_r = random.randint(h + 1, min(8, h + 3))
    large_c = random.randint(w + 1, min(8, w + 3))
    small_r = random.randint(1, 3)
    small_c = random.randint(1, 3)

    if mode == "row":
        dr = large_r
        dc = small_c
    elif mode == "col":
        dr = small_r
        dc = large_c
    elif mode == "both":
        dr = large_r
        dc = large_c
    else:
        if random.random() < 0.5:
            dr = large_r
            dc = small_c
        else:
            dr = small_r
            dc = large_c
    if random.random() < 0.5:
        dr = -dr
    if random.random() < 0.5:
        dc = -dc
    return int(dr), int(dc)


def _candidate_layouts(
    sprite: np.ndarray,
    step: tuple[int, int],
    grid_shape: tuple[int, int],
    mode: str,
) -> list[dict]:
    h, w = sprite.shape
    gh, gw = grid_shape
    candidates: list[dict] = []

    for pr in range(gh - h + 1):
        for pc in range(gw - w + 1):
            partial_top = (pr, pc)
            source_top = (pr - step[0], pc - step[1])
            if not _sprite_fits(grid_shape, sprite.shape, source_top):
                continue

            states = _collect_states(sprite, partial_top, step, grid_shape, MAX_FUTURE_STEPS)
            source_occ = _occupied_cells(sprite, source_top)
            partial_occ = _occupied_cells(states[0]["sprite"], states[0]["top"])
            if not _all_separated([source_occ, partial_occ]):
                continue

            final_options: list[int] = []
            for final_idx in range(1, len(states)):
                occs = [source_occ]
                occs.extend(_occupied_cells(state["sprite"], state["top"]) for state in states[: final_idx + 1])
                if not _all_separated(occs):
                    continue
                row_bounce, col_bounce = _step_signature(states, final_idx)
                if mode == "none":
                    if not row_bounce and not col_bounce:
                        final_options.append(final_idx)
                elif mode == "row":
                    if final_idx >= 2 and row_bounce and not col_bounce and _has_prior_bounce(states, final_idx):
                        final_options.append(final_idx)
                elif mode == "col":
                    if final_idx >= 2 and col_bounce and not row_bounce and _has_prior_bounce(states, final_idx):
                        final_options.append(final_idx)
                else:
                    if final_idx >= 2 and row_bounce and col_bounce and _has_prior_bounce(states, final_idx):
                        final_options.append(final_idx)

            if final_options:
                candidates.append(
                    {
                        "source_top": source_top,
                        "partial_top": partial_top,
                        "states": states,
                        "final_options": final_options,
                    }
                )
    return candidates


def _expected_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    scene = {
        "bg": 0,
        "full_sprite": meta["sprite"],
        "partial_top": meta["partial_top"],
        "marker_pos": meta["marker_pos"],
        "marker_color": meta["marker_color"],
        "step": meta["step"],
    }
    return _render(grid, scene)


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    parsed = _extract_scene(grid)
    if tuple(parsed["full_top"]) != tuple(meta["source_top"]):
        return False
    if tuple(parsed["partial_top"]) != tuple(meta["partial_top"]):
        return False
    if tuple(parsed["step"]) != tuple(meta["step"]):
        return False
    if tuple(parsed["marker_pos"]) != tuple(meta["marker_pos"]):
        return False
    if int(parsed["marker_color"]) != int(meta["marker_color"]):
        return False
    if len(_candidate_partial_tops(parsed["full_sprite"], parsed["partial_cells"], grid.shape)) != 1:
        return False

    expected = _expected_from_meta(grid, meta)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return False

    partial_complete = np.zeros_like(meta["sprite"])
    partial_complete[meta["partial_subset"] != 0] = 1
    if int((meta["sprite"] != 0).sum()) <= int(partial_complete.sum()):
        return False

    states = meta["states"]
    final_idx = int(meta["final_idx"])
    marker_pos = tuple(meta["marker_pos"])
    for idx, state in enumerate(states[:final_idx]):
        if _sprite_hits(state["sprite"], state["top"], marker_pos):
            return False
    if not _sprite_hits(states[final_idx]["sprite"], states[final_idx]["top"], marker_pos):
        return False

    row_bounce, col_bounce = _step_signature(states, final_idx)
    mode = meta["mode"]
    if mode == "none" and (row_bounce or col_bounce):
        return False
    if mode == "row" and (not row_bounce or col_bounce):
        return False
    if mode == "col" and (row_bounce or not col_bounce):
        return False
    if mode == "both" and (not row_bounce or not col_bounce):
        return False
    if mode != "none":
        if not _has_prior_bounce(states, final_idx):
            return False
        naive = np.array(grid, copy=True)
        _paste(naive, meta["sprite"], meta["partial_top"])
        if np.array_equal(naive, expected):
            return False

    final_occ = _occupied_cells(states[final_idx]["sprite"], states[final_idx]["top"])
    prior_occs = [_occupied_cells(meta["sprite"], meta["source_top"])]
    prior_occs.extend(_occupied_cells(state["sprite"], state["top"]) for state in states[:final_idx])
    if not _all_separated(prior_occs + [final_occ]):
        return False

    return True


def _gen_attempt(mode: str) -> np.ndarray | None:
    sprite = _make_sprite()
    step = _sample_step(sprite.shape, mode)
    h, w = sprite.shape
    gh = random.randint(max(18, h + 2 * abs(step[0]) + 4), 28)
    gw = random.randint(max(18, w + 2 * abs(step[1]) + 4), 28)

    layouts = _candidate_layouts(sprite, step, (gh, gw), mode)
    if not layouts:
        return None

    layout = random.choice(layouts)
    final_choices = layout["final_options"]
    if mode == "none":
        preferred = [idx for idx in final_choices if idx >= 2]
        final_idx = random.choice(preferred or final_choices)
    else:
        final_idx = random.choice(final_choices)

    partial_subset = _choose_partial_subset(sprite)
    states = layout["states"]
    final_state = states[final_idx]
    marker_cells = list(_occupied_cells(final_state["sprite"], final_state["top"]))
    if not marker_cells:
        return None

    sprite_colors = {int(v) for v in sprite.flatten() if int(v) != 0}
    marker_palette = [color for color in PALETTE if color not in sprite_colors]
    if not marker_palette:
        return None
    marker_color = random.choice(marker_palette)
    marker_pos = random.choice(marker_cells)

    grid = np.zeros((gh, gw), dtype=int)
    _paste(grid, sprite, layout["source_top"])
    _paste(grid, partial_subset, layout["partial_top"])
    grid[marker_pos] = int(marker_color)

    meta = {
        "mode": mode,
        "sprite": sprite,
        "step": tuple(step),
        "source_top": tuple(layout["source_top"]),
        "partial_top": tuple(layout["partial_top"]),
        "partial_subset": partial_subset,
        "states": states,
        "final_idx": int(final_idx),
        "marker_pos": tuple(marker_pos),
        "marker_color": int(marker_color),
    }
    if not _validate_case(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    preferred_mode = GEN_MODES[_GEN_CALL_COUNT % len(GEN_MODES)]
    _GEN_CALL_COUNT += 1

    mode_order = [preferred_mode]
    others = [mode for mode in GEN_MODES if mode != preferred_mode]
    random.shuffle(others)
    mode_order.extend(others)

    for _ in range(GEN_RETRY_LIMIT):
        for mode in mode_order:
            grid = _gen_attempt(mode)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid bouncing-mirror stamp input")
