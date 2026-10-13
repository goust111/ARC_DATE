from __future__ import annotations

import numpy as np


GREY = 5
SIDES = ("top", "right", "bottom", "left")
DIRS = [(-2, 0), (0, 2), (2, 0), (0, -2)]
START_DIR_BY_SIDE = {"top": 0, "right": 1, "bottom": 2, "left": 3}
PALETTE_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_BG = 0
GEN_MIN_INTERIOR = 9
GEN_MAX_INTERIOR = 27
GEN_MAX_MARKERS = 5
GEN_RETRIES = 400

_SIDE_QUEUE: list[str] = []
_SIDE_POS = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _side_cells(side: str, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(w)]
    if side == "bottom":
        return [(h - 1, c) for c in range(w)]
    if side == "left":
        return [(r, 0) for r in range(h)]
    if side == "right":
        return [(r, w - 1) for r in range(h)]
    raise ValueError(f"unknown side: {side}")


def _interior_spec(side: str, h: int, w: int) -> tuple[int, int, int, int]:
    if side == "top":
        return 1, 0, h - 1, w
    if side == "bottom":
        return 0, 0, h - 1, w
    if side == "left":
        return 0, 1, h, w - 1
    if side == "right":
        return 0, 0, h, w - 1
    raise ValueError(f"unknown side: {side}")


def _inside_node(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1] and r % 2 == 0 and c % 2 == 0


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = map(int, g.shape)

    bg = _get_bg(g)
    assert bg != GREY

    seed_coords = np.argwhere(g == GREY)
    assert seed_coords.shape == (1, 2), "expected exactly one grey seed"
    seed_abs = tuple(map(int, seed_coords[0]))

    candidates: list[tuple[str, list[tuple[int, int]]]] = []
    for side in SIDES:
        coords = [
            (r, c)
            for r, c in _side_cells(side, h, w)
            if int(g[r, c]) != bg and (r, c) != seed_abs
        ]
        if coords:
            candidates.append((side, coords))
    assert len(candidates) == 1, "expected markers on exactly one boundary side"
    side, palette_coords = candidates[0]

    if side in ("top", "bottom"):
        palette_coords.sort(key=lambda rc: rc[1])
    else:
        palette_coords.sort(key=lambda rc: rc[0])
    palette_colors = [int(g[r, c]) for r, c in palette_coords]
    assert len(set(palette_colors)) == len(palette_colors), "marker colors must be distinct"

    foreground = {(int(r), int(c)) for r, c in np.argwhere(g != bg)}
    assert foreground == set(palette_coords) | {seed_abs}, "input must contain only markers and one seed"
    assert seed_abs not in palette_coords, "seed must lie in the interior region"

    r0, c0, ih, iw = _interior_spec(side, h, w)
    seed_rel = (seed_abs[0] - r0, seed_abs[1] - c0)
    assert _inside_node((ih, iw), seed_rel[0], seed_rel[1]), "seed must lie on an interior lattice node"

    return {
        "grid": g,
        "bg": bg,
        "side": side,
        "palette_colors": palette_colors,
        "palette_coords": palette_coords,
        "seed_abs": seed_abs,
        "seed_rel": seed_rel,
        "offset": (r0, c0),
        "interior_shape": (ih, iw),
    }


def _trace_spiral(
    interior_shape: tuple[int, int],
    seed_rel: tuple[int, int],
    side: str,
    palette_colors: list[int],
) -> tuple[list[tuple[int, int, int]], dict]:
    assert palette_colors
    max_arms = 2 * (interior_shape[0] + interior_shape[1] + 4)
    dir_idx = START_DIR_BY_SIDE[side]
    r, c = seed_rel
    step_len = 1
    paints: list[tuple[int, int, int]] = []
    visible_arm_count = 0
    arm_colors: list[int] = []
    terminated = False

    for arm_idx in range(max_arms):
        color = int(palette_colors[arm_idx % len(palette_colors)])
        dr, dc = DIRS[dir_idx]
        drew_this_arm = False
        for _ in range(step_len):
            nr, nc = r + dr, c + dc
            if _inside_node(interior_shape, r, c) and _inside_node(interior_shape, nr, nc):
                paints.append(((r + nr) // 2, (c + nc) // 2, color))
                paints.append((nr, nc, color))
                drew_this_arm = True
            r, c = nr, nc
        if not drew_this_arm:
            terminated = True
            break
        visible_arm_count += 1
        arm_colors.append(color)
        dir_idx = (dir_idx + 1) % 4
        if arm_idx % 2 == 1:
            step_len += 1

    assert terminated, "spiral should terminate after a fully out-of-bounds arm"
    painted_cells = {(r0, c0) for r0, c0, _color in paints}
    return paints, {
        "visible_arms": visible_arm_count,
        "painted_cells": len(painted_cells),
        "used_colors": len(set(arm_colors)),
        "density": len(painted_cells) / float(interior_shape[0] * interior_shape[1]),
    }


def _render_from_parsed(parsed: dict) -> tuple[np.ndarray, dict]:
    out = parsed["grid"].copy()
    paints, stats = _trace_spiral(
        parsed["interior_shape"],
        parsed["seed_rel"],
        parsed["side"],
        parsed["palette_colors"],
    )
    r0, c0 = parsed["offset"]
    for rr, cc, color in paints:
        out[r0 + rr, c0 + cc] = color
    seed_r, seed_c = parsed["seed_abs"]
    out[seed_r, seed_c] = GREY
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """Draw a clipped square spiral on the doubled interior lattice, recoloring each arm by the boundary palette.

    Input structure
    - Exactly one boundary row or boundary column contains several distinct single-pixel markers.
    - The remaining interior region contains a single grey seed pixel on an even-even lattice node.
    - All other cells are background.

    Rule
    - Treat the interior as a doubled node lattice: even-even cells are nodes and the cells between adjacent nodes are edges.
    - Start from the seed, take the first arm toward the palette side, then keep turning clockwise with arm lengths 1, 1, 2, 2, 3, 3, ... measured in node steps.
    - Arm i paints every traversed edge cell and every reached node with the i-th palette color cyclically, but only when both edge endpoints stay inside the interior; the seed remains grey.
    - Stop as soon as one entire arm falls out of bounds and therefore paints nothing.

    Key insight
    - The boundary markers are both the ordered color palette and the direction cue for the spiral's first arm, while the actual drawing happens on a doubled lattice rather than a pixel-by-pixel curve.
    """
    parsed = _parse_grid(grid)
    out, _stats = _render_from_parsed(parsed)
    return out


def _next_side() -> str:
    global _SIDE_QUEUE, _SIDE_POS
    if _SIDE_POS >= len(_SIDE_QUEUE):
        _SIDE_QUEUE = np.random.permutation(SIDES).tolist()
        _SIDE_POS = 0
    side = _SIDE_QUEUE[_SIDE_POS]
    _SIDE_POS += 1
    return side


def _sample_shape(side: str) -> tuple[int, int]:
    if side in ("top", "bottom"):
        interior_h = int(np.random.randint(GEN_MIN_INTERIOR, GEN_MAX_INTERIOR + 1))
        interior_w = int(np.random.randint(GEN_MIN_INTERIOR, GEN_MAX_INTERIOR + 1))
        return interior_h + 1, interior_w
    interior_h = int(np.random.randint(GEN_MIN_INTERIOR, GEN_MAX_INTERIOR + 1))
    interior_w = int(np.random.randint(GEN_MIN_INTERIOR, GEN_MAX_INTERIOR + 1))
    return interior_h, interior_w + 1


def _sample_palette(side: str, h: int, w: int) -> tuple[list[tuple[int, int]], list[int]]:
    axis_len = w if side in ("top", "bottom") else h
    max_k = min(GEN_MAX_MARKERS, axis_len - 2, len(PALETTE_POOL))
    assert max_k >= 2
    k = int(np.random.randint(2, max_k + 1))
    positions = np.sort(np.random.choice(np.arange(1, axis_len - 1), size=k, replace=False)).astype(int)
    colors = np.random.choice(PALETTE_POOL, size=k, replace=False).astype(int).tolist()

    coords: list[tuple[int, int]] = []
    if side == "top":
        coords = [(0, int(c)) for c in positions]
    elif side == "bottom":
        coords = [(h - 1, int(c)) for c in positions]
    elif side == "left":
        coords = [(int(r), 0) for r in positions]
    elif side == "right":
        coords = [(int(r), w - 1) for r in positions]
    return coords, colors


def _candidate_seeds(interior_shape: tuple[int, int], side: str, k: int) -> list[tuple[tuple[int, int], dict]]:
    dummy_palette = list(range(k))
    candidates: list[tuple[tuple[int, int], dict]] = []
    for r in range(0, interior_shape[0], 2):
        for c in range(0, interior_shape[1], 2):
            _paints, stats = _trace_spiral(interior_shape, (r, c), side, dummy_palette)
            if stats["visible_arms"] < max(4, k):
                continue
            if stats["painted_cells"] < 10:
                continue
            if stats["density"] > 0.62:
                continue
            candidates.append(((r, c), stats))
    return candidates


def _choose_seed(interior_shape: tuple[int, int], side: str, k: int) -> tuple[int, int] | None:
    candidates = _candidate_seeds(interior_shape, side, k)
    if not candidates:
        return None

    if np.random.rand() < 0.5:
        ranked = sorted(
            candidates,
            key=lambda item: min(
                item[0][0],
                item[0][1],
                interior_shape[0] - 1 - item[0][0],
                interior_shape[1] - 1 - item[0][1],
            ),
        )
        pool = ranked[: max(1, len(ranked) // 2)]
    else:
        ranked = sorted(
            candidates,
            key=lambda item: min(
                item[0][0],
                item[0][1],
                interior_shape[0] - 1 - item[0][0],
                interior_shape[1] - 1 - item[0][1],
            ),
            reverse=True,
        )
        pool = ranked[: max(1, len(ranked) // 2)]
    seed_rel, _stats = pool[int(np.random.randint(len(pool)))]
    return seed_rel


def _build_input(
    h: int,
    w: int,
    side: str,
    palette_coords: list[tuple[int, int]],
    palette_colors: list[int],
    seed_rel: tuple[int, int],
) -> np.ndarray:
    grid = np.full((h, w), GEN_BG, dtype=int)
    for (r, c), color in zip(palette_coords, palette_colors, strict=True):
        grid[r, c] = int(color)
    r0, c0, _ih, _iw = _interior_spec(side, h, w)
    grid[r0 + seed_rel[0], c0 + seed_rel[1]] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    side = _next_side()
    h, w = _sample_shape(side)
    palette_coords, palette_colors = _sample_palette(side, h, w)
    interior_shape = _interior_spec(side, h, w)[2:]
    seed_rel = _choose_seed(interior_shape, side, len(palette_colors))
    if seed_rel is None:
        return None

    inp = _build_input(h, w, side, palette_coords, palette_colors, seed_rel)
    out = main(inp)

    added = int(np.count_nonzero(out != inp))
    inner = out.copy()
    if side == "top":
        inner = inner[1:, :]
    elif side == "bottom":
        inner = inner[:-1, :]
    elif side == "left":
        inner = inner[:, 1:]
    else:
        inner = inner[:, :-1]
    interior_colors = sorted(int(v) for v in np.unique(inner) if v not in {GEN_BG, GREY})

    if added < 10:
        return None
    if len(interior_colors) < min(2, len(palette_colors)):
        return None
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral-palette input")
