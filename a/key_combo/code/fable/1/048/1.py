from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
TEAL = 8
EMITTER_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _find_mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_platforms(grid: np.ndarray) -> tuple[list[dict[str, int]], np.ndarray]:
    h, w = grid.shape
    mask = np.isin(grid, (GREY, TEAL))
    platform_map = np.full((h, w), -1, dtype=int)
    platforms: list[dict[str, int]] = []

    for coords in sorted(_find_mask_components(mask), key=lambda arr: (int(arr[:, 0].min()), int(arr[:, 1].min()))):
        rows = sorted(set(map(int, coords[:, 0])))
        cols = sorted(map(int, coords[:, 1]))
        assert len(rows) == 1, "each platform must be a single horizontal segment"
        row = rows[0]
        c0 = cols[0]
        c1 = cols[-1]
        assert 0 < row < h - 1, "platforms must stay inside the grid"
        assert 0 < c0 and c1 < w - 1, "platforms need space to drop past either end"
        assert cols == list(range(c0, c1 + 1)), "platform cells must be contiguous"

        teal_cols = [int(c) for r, c in coords if int(grid[int(r), int(c)]) == TEAL]
        assert len(teal_cols) == 1, "each platform needs exactly one teal guide endpoint"
        teal_col = teal_cols[0]
        assert teal_col in (c0, c1), "the teal guide must sit on a platform end"
        guide = -1 if teal_col == c0 else 1

        pid = len(platforms)
        for c in range(c0, c1 + 1):
            platform_map[row, c] = pid
        platforms.append({"row": row, "c0": c0, "c1": c1, "guide": guide, "uses": 0})

    return platforms, platform_map


def _emitters(grid: np.ndarray) -> list[tuple[int, int]]:
    cols = [(c, int(grid[0, c])) for c in range(grid.shape[1]) if int(grid[0, c]) != BG]
    assert cols, "expected at least one top-row emitter"
    colors = [color for _c, color in cols]
    assert len(colors) == len(set(colors)), "emitters must have distinct colors"
    assert all(color not in (GREY, TEAL) for color in colors), "emitters must not use platform colors"
    return cols


def _solve(grid: np.ndarray, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(np.min(g)) >= 0
    assert int(np.max(g)) <= 9

    h, w = g.shape
    emitters = _emitters(g)
    platforms, platform_map = _extract_platforms(g)

    allowed = np.zeros((h, w), dtype=bool)
    for c, _color in emitters:
        allowed[0, c] = True
    allowed |= platform_map >= 0
    assert bool(np.all((g == BG) | allowed)), "unexpected foreground outside emitters and platforms"

    out = g.copy()
    trail_mask = np.zeros((h, w), dtype=bool)
    meta = {"platform_use_counts": [0] * len(platforms), "beams": []}

    for col, color in emitters:
        r = 0
        c = int(col)
        beam_meta = {"col": c, "color": int(color), "platforms": [], "absorbed": False, "reached_bottom": False}
        for _step in range(h * w * 4):
            nr = r + 1
            if nr >= h:
                beam_meta["reached_bottom"] = True
                break

            pid = int(platform_map[nr, c])
            if pid >= 0:
                platform = platforms[pid]
                direction = int(platform["guide"])
                end_col = int(platform["c0"] if direction < 0 else platform["c1"])
                beam_meta["platforms"].append(pid)
                platform["uses"] += 1
                meta["platform_use_counts"][pid] += 1

                for cc in range(c, end_col + direction, direction):
                    out[nr, cc] = color

                platform["guide"] *= -1
                exit_c = end_col + direction
                assert 0 <= exit_c < w, "platform exits must stay inside the grid"
                assert int(platform_map[nr, exit_c]) == -1, "platforms must not touch each other"
                if trail_mask[nr, exit_c]:
                    beam_meta["absorbed"] = True
                    break
                assert int(out[nr, exit_c]) == BG, "platform exits must lead into background"
                out[nr, exit_c] = color
                trail_mask[nr, exit_c] = True
                r, c = nr, exit_c
                if r == h - 1:
                    beam_meta["reached_bottom"] = True
                    break
                continue

            if trail_mask[nr, c]:
                beam_meta["absorbed"] = True
                break

            assert int(out[nr, c]) == BG, "vertical beam travel should only pass through background"
            out[nr, c] = color
            trail_mask[nr, c] = True
            r = nr
            if r == h - 1:
                beam_meta["reached_bottom"] = True
                break
        else:
            raise AssertionError("beam simulation did not terminate")

        meta["beams"].append(beam_meta)

    for platform in platforms:
        end_col = int(platform["c0"] if int(platform["guide"]) < 0 else platform["c1"])
        out[int(platform["row"]), end_col] = TEAL

    if return_meta:
        return out, meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate top-row beams that fall, slide across toggling switch platforms, and stop on earlier trails.

    Input structure:
    - The background is black.
    - The top row contains several distinct-colored emitter pixels.
    - Inside the grid are disjoint one-cell-thick horizontal gray platforms, each with exactly one teal endpoint marking its current guide side.

    Rule:
    - Process emitters from left to right.
    - Each beam starts below its emitter, paints the background cells it falls through, and when it lands on a platform it slides across that platform toward the current teal endpoint.
    - After crossing that endpoint, the beam resumes vertical falling from the cell just beyond the platform end, and the platform immediately flips its teal endpoint to the opposite side for later beams.
    - If a beam would enter any previously drawn non-platform trail cell, it stops one cell earlier; after all beams finish, keep every trail and show each platform's final teal endpoint.

    Key insight:
    - Platforms are reusable stateful switches: earlier beams both change future routing by flipping platform endpoints and create absorbing trails that later beams must stop at.
    """
    return _solve(grid)


def _draw_platform(grid: np.ndarray, row: int, c0: int, c1: int, guide: int) -> None:
    grid[row, c0 : c1 + 1] = GREY
    grid[row, c0 if guide < 0 else c1] = TEAL


def _basic_case() -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(10, 18)
    max_len = min(6, w - 5)
    if max_len < 4:
        return None

    length = random.randint(4, max_len)
    c0_hi = w - length - 3
    if c0_hi < 1:
        return None
    c0 = random.randint(1, c0_hi)
    c1 = c0 + length - 1
    row = random.randint(3, min(h - 4, 7))
    guide = random.choice((-1, 1))

    e1, e2 = sorted(random.sample(list(range(c0, c1 + 1)), 2))
    e3 = c1 + 1
    emitter_cols = [e1, e2, e3]

    extra_choices = [c for c in range(e3 + 2, w) if c not in emitter_cols]
    if extra_choices and random.random() < 0.65:
        emitter_cols.append(random.choice(extra_choices))

    colors = random.sample(EMITTER_COLORS, len(emitter_cols))
    grid = np.zeros((h, w), dtype=int)
    _draw_platform(grid, row, c0, c1, guide)
    for col, color in zip(sorted(emitter_cols), colors):
        grid[0, col] = color
    return grid


def _chain_case() -> np.ndarray | None:
    h = random.randint(15, 24)
    w = random.randint(15, 24)
    len1 = random.randint(3, 5)
    len2 = random.randint(4, 5)
    c0_hi = w - (len1 + len2 + 3)
    if c0_hi < 1:
        return None

    p1_c0 = random.randint(1, c0_hi)
    p1_c1 = p1_c0 + len1 - 1
    exit_right = p1_c1 + 1
    p1_row = random.randint(3, 6)
    p1_guide = random.choice((-1, 1))

    p2_c0 = exit_right - 1
    p2_c1 = p2_c0 + len2 - 1
    if p2_c0 <= 0 or p2_c1 >= w - 1:
        return None
    p2_row_lo = p1_row + 4
    p2_row_hi = h - 4
    if p2_row_lo > p2_row_hi:
        return None
    p2_row = random.randint(p2_row_lo, p2_row_hi)
    p2_guide = random.choice((-1, 1))

    e1, e2 = sorted(random.sample(list(range(p1_c0, p1_c1 + 1)), 2))
    e3 = exit_right
    later_p2_cols = [c for c in range(exit_right + 1, p2_c1 + 1)]
    if not later_p2_cols:
        return None
    e4 = random.choice(later_p2_cols)
    emitter_cols = [e1, e2, e3, e4]

    extra_choices = [c for c in range(p2_c1 + 2, w) if c not in emitter_cols]
    if extra_choices and random.random() < 0.55:
        emitter_cols.append(random.choice(extra_choices))

    colors = random.sample(EMITTER_COLORS, len(emitter_cols))
    grid = np.zeros((h, w), dtype=int)
    _draw_platform(grid, p1_row, p1_c0, p1_c1, p1_guide)
    _draw_platform(grid, p2_row, p2_c0, p2_c1, p2_guide)
    for col, color in zip(sorted(emitter_cols), colors):
        grid[0, col] = color
    return grid


def _valid_generated(grid: np.ndarray, mode: str) -> bool:
    try:
        out, meta = _solve(grid, return_meta=True)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    if not any(beam["platforms"] for beam in meta["beams"]):
        return False
    if not any(beam["absorbed"] for beam in meta["beams"]):
        return False
    if not any(beam["reached_bottom"] for beam in meta["beams"]):
        return False
    if max(meta["platform_use_counts"], default=0) < 2:
        return False
    if mode == "chain" and sum(count >= 2 for count in meta["platform_use_counts"]) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        mode = random.choices(("basic", "chain"), weights=(5, 4), k=1)[0]
        grid = _basic_case() if mode == "basic" else _chain_case()
        if grid is not None and _valid_generated(grid, mode):
            return grid
    raise RuntimeError("failed to generate a valid switch-platform beam puzzle")
