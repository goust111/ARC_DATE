from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds, neighbors4


GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 28)
PALETTE_LENGTHS = (3, 4, 5, 6)
PALETTE_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_ATTEMPTS = 400


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30

    bg = get_bg(g)
    assert bg == Color.BLACK, "expected black background"

    palette_candidates = []
    components = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    for comp in components:
        coords = comp["coords"]
        colors = g[coords[:, 0], coords[:, 1]]
        if np.any(colors == Color.GREY):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        if r0 != r1:
            continue
        if int(comp["area"]) != c1 - c0 + 1:
            continue
        if int(comp["area"]) < 3 or len(set(map(int, colors))) < 2:
            continue
        cols = sorted(int(c) for c in coords[:, 1])
        if cols != list(range(int(c0), int(c1) + 1)):
            continue
        palette = [int(g[int(r0), c]) for c in range(int(c0), int(c1) + 1)]
        palette_candidates.append((int(r0), int(c0), int(c1), palette))

    assert len(palette_candidates) == 1, "expected exactly one palette row"
    palette_row, palette_c0, palette_c1, palette = palette_candidates[0]
    palette_mask = np.zeros_like(g, dtype=bool)
    palette_mask[palette_row, palette_c0 : palette_c1 + 1] = True

    non_wall_colored = (g != bg) & (g != Color.GREY)
    seed_cells = np.argwhere(non_wall_colored & ~palette_mask)
    assert seed_cells.shape[0] == 1, "expected exactly one seed pixel"
    seed = tuple(map(int, seed_cells[0]))
    assert int(g[seed]) == palette[0], "seed color must match the first palette color"
    for nr, nc in neighbors4(*seed):
        if in_bounds(g.shape, nr, nc):
            assert int(g[nr, nc]) == bg, "seed must be isolated"

    wall_mask = g == Color.GREY
    for c in range(palette_c0, palette_c1 + 1):
        for nr, nc in neighbors4(palette_row, c):
            if in_bounds(g.shape, nr, nc):
                assert not wall_mask[nr, nc], "palette row must not touch walls"

    return {
        "grid": g,
        "bg": bg,
        "palette": palette,
        "palette_mask": palette_mask,
        "wall_mask": wall_mask,
        "seed": seed,
    }


def _geodesic_distances(shape: tuple[int, int], blocked: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    dist = np.full(shape, -1, dtype=int)
    queue = deque([seed])
    dist[seed] = 0
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(shape, nr, nc):
                continue
            if blocked[nr, nc] or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            queue.append((nr, nc))
    return dist


def _render_from_rules(grid: np.ndarray) -> tuple[np.ndarray, dict, np.ndarray]:
    scene = _parse_scene(grid)
    blocked = scene["wall_mask"] | scene["palette_mask"]
    blocked[scene["seed"]] = False
    dist = _geodesic_distances(scene["grid"].shape, blocked, scene["seed"])

    out = scene["grid"].copy()
    palette = scene["palette"]
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            if scene["wall_mask"][r, c] or scene["palette_mask"][r, c] or dist[r, c] < 0:
                continue
            out[r, c] = palette[int(dist[r, c]) % len(palette)]
    return out, scene, dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every seed-reachable corridor cell by cyclic geodesic distance from the seed using the palette row.

    Input structure:
    - The black background contains grey wall segments or blocks, one horizontal multi-color palette row, and one isolated seed pixel.
    - The seed color matches the leftmost palette color, and the palette row does not touch any wall.
    - Walls divide the black space into corridors, and some background cells may be disconnected from the seed.

    Rule:
    - Treat grey walls and the palette row as impassable, then run a 4-neighbor shortest-path flood from the seed.
    - Keep the seed at palette index 0, and color each reachable background cell at distance d with the palette color at index d modulo the palette length.
    - Leave the walls and palette row unchanged, and keep seed-unreachable background cells black.

    Key insight:
    - The visible rings are not based on straight-line distance: they follow corridor length around walls, while the palette row only supplies the repeating color cycle.
    """
    out, _scene, _dist = _render_from_rules(grid)
    return out


def _reserve_clear(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, *, pad: int = 0) -> bool:
    h, w = reserved.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve_box(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, *, pad: int = 0) -> None:
    h, w = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0 : r1 + 1, c0] = Color.GREY
    grid[r0 : r1 + 1, c1] = Color.GREY
    grid[r0, c0 : c1 + 1] = Color.GREY
    grid[r1, c0 : c1 + 1] = Color.GREY


def _side_of_barrier(pos: tuple[int, int], orientation: str, barrier: int) -> int:
    if orientation == "vertical":
        return -1 if pos[1] < barrier else 1
    return -1 if pos[0] < barrier else 1


def _place_gate_wall(
    grid: np.ndarray,
    reserved: np.ndarray,
    play_r0: int,
    play_r1: int,
    play_c0: int,
    play_c1: int,
) -> tuple[str, int, tuple[int, int], tuple[int, int], int]:
    orientation = random.choice(("vertical", "horizontal"))
    gap_size = random.choice((1, 2))

    if orientation == "vertical":
        barrier = random.randint(play_c0 + 3, play_c1 - 3)
        gap_near_top = random.choice((True, False))
        if gap_near_top:
            gap_r0 = random.randint(play_r0 + 1, min(play_r0 + 3, play_r1 - gap_size))
            seed_r = random.randint(max(play_r0, play_r1 - 4), play_r1)
        else:
            gap_r0 = random.randint(max(play_r0 + 1, play_r1 - 3 - gap_size + 1), play_r1 - gap_size)
            seed_r = random.randint(play_r0, min(play_r0 + 4, play_r1))
        gap = (gap_r0, gap_r0 + gap_size - 1)
        for r in range(play_r0, play_r1 + 1):
            if gap[0] <= r <= gap[1]:
                continue
            grid[r, barrier] = Color.GREY
        _reserve_box(reserved, play_r0, barrier, play_r1, barrier, pad=1)

        seed_side = random.choice((-1, 1))
        if seed_side < 0:
            seed_c = random.randint(play_c0, barrier - 2)
        else:
            seed_c = random.randint(barrier + 2, play_c1)
        seed = (seed_r, seed_c)
    else:
        barrier = random.randint(play_r0 + 3, play_r1 - 3)
        gap_near_left = random.choice((True, False))
        if gap_near_left:
            gap_c0 = random.randint(play_c0 + 1, min(play_c0 + 3, play_c1 - gap_size))
            seed_c = random.randint(max(play_c0, play_c1 - 4), play_c1)
        else:
            gap_c0 = random.randint(max(play_c0 + 1, play_c1 - 3 - gap_size + 1), play_c1 - gap_size)
            seed_c = random.randint(play_c0, min(play_c0 + 4, play_c1))
        gap = (gap_c0, gap_c0 + gap_size - 1)
        for c in range(play_c0, play_c1 + 1):
            if gap[0] <= c <= gap[1]:
                continue
            grid[barrier, c] = Color.GREY
        _reserve_box(reserved, barrier, play_c0, barrier, play_c1, pad=1)

        seed_side = random.choice((-1, 1))
        if seed_side < 0:
            seed_r = random.randint(play_r0, barrier - 2)
        else:
            seed_r = random.randint(barrier + 2, play_r1)
        seed = (seed_r, seed_c)

    return orientation, barrier, gap, seed, seed_side


def _place_unreachable_frames(
    grid: np.ndarray,
    reserved: np.ndarray,
    play_r0: int,
    play_r1: int,
    play_c0: int,
    play_c1: int,
    seed: tuple[int, int],
) -> bool:
    frame_count = random.choice((1, 1, 2))
    placed = 0
    for _ in range(60):
        if placed >= frame_count:
            return True
        h = random.randint(5, 8)
        w = random.randint(5, 8)
        r0 = random.randint(play_r0, play_r1 - h + 1)
        c0 = random.randint(play_c0, play_c1 - w + 1)
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if r0 < seed[0] < r1 and c0 < seed[1] < c1:
            continue
        if not _reserve_clear(reserved, r0, c0, r1, c1, pad=1):
            continue
        _draw_frame(grid, r0, c0, r1, c1)
        _reserve_box(reserved, r0, c0, r1, c1, pad=1)
        placed += 1
    return placed >= 1


def _place_extra_walls(
    grid: np.ndarray,
    reserved: np.ndarray,
    play_r0: int,
    play_r1: int,
    play_c0: int,
    play_c1: int,
) -> None:
    extra_count = random.randint(1, 3)
    for _ in range(extra_count):
        for _attempt in range(40):
            kind = random.choice(("bar", "bar", "block"))
            if kind == "bar":
                if random.choice((True, False)):
                    h = random.randint(3, min(7, play_r1 - play_r0 + 1))
                    r0 = random.randint(play_r0, play_r1 - h + 1)
                    c0 = random.randint(play_c0, play_c1)
                    r1, c1 = r0 + h - 1, c0
                else:
                    w = random.randint(3, min(7, play_c1 - play_c0 + 1))
                    r0 = random.randint(play_r0, play_r1)
                    c0 = random.randint(play_c0, play_c1 - w + 1)
                    r1, c1 = r0, c0 + w - 1
            else:
                h = random.randint(2, 4)
                w = random.randint(2, 4)
                r0 = random.randint(play_r0, play_r1 - h + 1)
                c0 = random.randint(play_c0, play_c1 - w + 1)
                r1, c1 = r0 + h - 1, c0 + w - 1

            if not _reserve_clear(reserved, r0, c0, r1, c1, pad=1):
                continue
            grid[r0 : r1 + 1, c0 : c1 + 1] = Color.GREY
            _reserve_box(reserved, r0, c0, r1, c1, pad=1)
            break


def _detour_strength(dist: np.ndarray, seed: tuple[int, int]) -> int:
    best = 0
    sr, sc = seed
    for r, c in np.argwhere(dist >= 0):
        manhattan = abs(int(r) - sr) + abs(int(c) - sc)
        best = max(best, int(dist[r, c]) - manhattan)
    return best


def _validate_semantics(inp: np.ndarray, out: np.ndarray, *, require_features: bool) -> None:
    expected, scene, dist = _render_from_rules(inp)
    assert np.array_equal(out, expected), "output does not follow the task rules"
    assert np.array_equal(out[scene["wall_mask"]], inp[scene["wall_mask"]]), "walls changed"
    assert np.array_equal(out[scene["palette_mask"]], inp[scene["palette_mask"]]), "palette row changed"

    bg = scene["bg"]
    palette = scene["palette"]
    reachable_background = 0
    unreachable_background = 0
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if scene["wall_mask"][r, c] or scene["palette_mask"][r, c]:
                continue
            if dist[r, c] < 0:
                if (r, c) != scene["seed"]:
                    assert int(inp[r, c]) == bg and int(out[r, c]) == bg
                    unreachable_background += 1
                continue
            assert int(out[r, c]) == palette[int(dist[r, c]) % len(palette)]
            if int(inp[r, c]) == bg:
                reachable_background += 1

    if require_features:
        assert reachable_background >= 25, "reachable region is too small"
        assert unreachable_background >= 6, "missing disconnected black background"
        assert int(dist.max()) >= len(palette) + 4, "cycle depth is too shallow"
        assert _detour_strength(dist, scene["seed"]) >= 2, "missing geodesic detour against Manhattan distance"


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    palette_len = random.choice(PALETTE_LENGTHS)
    palette = random.sample(PALETTE_COLORS, palette_len)
    palette_row = random.choice((1, 2, h - 3, h - 2))
    palette_c0 = random.randint(1, w - palette_len - 2)
    grid[palette_row, palette_c0 : palette_c0 + palette_len] = palette
    _reserve_box(reserved, palette_row, palette_c0, palette_row, palette_c0 + palette_len - 1, pad=1)

    if palette_row <= 2:
        play_r0, play_r1 = palette_row + 2, h - 2
    else:
        play_r0, play_r1 = 1, palette_row - 2
    play_c0, play_c1 = 1, w - 2
    if play_r1 - play_r0 < 7 or play_c1 - play_c0 < 7:
        return None

    try:
        orientation, barrier, gap, seed, seed_side = _place_gate_wall(
            grid,
            reserved,
            play_r0,
            play_r1,
            play_c0,
            play_c1,
        )
    except ValueError:
        return None

    if not _reserve_clear(reserved, seed[0], seed[1], seed[0], seed[1], pad=1):
        return None
    grid[seed] = palette[0]
    _reserve_box(reserved, seed[0], seed[1], seed[0], seed[1], pad=1)

    if not _place_unreachable_frames(grid, reserved, play_r0, play_r1, play_c0, play_c1, seed):
        return None
    _place_extra_walls(grid, reserved, play_r0, play_r1, play_c0, play_c1)

    try:
        out = main(grid)
        _validate_semantics(grid, out, require_features=True)
    except Exception:
        return None

    scene = _parse_scene(grid)
    blocked = scene["wall_mask"] | scene["palette_mask"]
    blocked[scene["seed"]] = False
    dist = _geodesic_distances(grid.shape, blocked, scene["seed"])

    opposite_reachable = 0
    seed_side_now = _side_of_barrier(scene["seed"], orientation, barrier)
    for r, c in np.argwhere(dist >= 0):
        if (r, c) == scene["seed"]:
            continue
        if _side_of_barrier((int(r), int(c)), orientation, barrier) != seed_side_now:
            opposite_reachable += 1
    if opposite_reachable < 10:
        return None

    if orientation == "vertical":
        gap_line = slice(gap[0], gap[1] + 1)
        if not np.all(grid[gap_line, barrier] == Color.BLACK):
            return None
    else:
        gap_line = slice(gap[0], gap[1] + 1)
        if not np.all(grid[barrier, gap_line] == Color.BLACK):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid maze-ripple input")

