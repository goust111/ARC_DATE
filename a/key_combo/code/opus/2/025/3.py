from __future__ import annotations

import numpy as np


BACKGROUND = 0
FILL_COLOR = 8
PALETTE = (1, 2, 3, 4, 5, 6, 7, 9)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    if connectivity == 4:
        nbrs = CARDINALS
    else:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in nbrs:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _side_coords(h: int, w: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(w)]
    if side == "bottom":
        return [(h - 1, c) for c in range(w)]
    if side == "left":
        return [(r, 0) for r in range(h)]
    if side == "right":
        return [(r, w - 1) for r in range(h)]
    raise ValueError(side)


def _center_gap(coords: list[tuple[int, int]], gap_len: int) -> list[tuple[int, int]]:
    n = len(coords)
    if gap_len == 2:
        start = (n - gap_len) // 2
        if start == 0:
            start = 1
        if start + gap_len == n:
            start -= 1
        return coords[start : start + gap_len]

    mid = (n - 1) / 2.0
    best = sorted(range(1, n - 1), key=lambda idx: (abs(idx - mid), idx))[0]
    return [coords[best]]


def _corner_gap(coords: list[tuple[int, int]], which: str) -> list[tuple[int, int]]:
    if which == "start":
        return coords[:2]
    if which == "end":
        return coords[-2:]
    raise ValueError(which)


def _build_cup(
    h: int,
    w: int,
    color: int,
    side: str,
    gap_mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    cup = np.zeros((h, w), dtype=int)
    cup[0, :] = color
    cup[-1, :] = color
    cup[:, 0] = color
    cup[:, -1] = color

    side_cells = _side_coords(h, w, side)
    if gap_mode == "center":
        max_gap = 2 if len(side_cells) >= 5 else 1
        gap_len = int(rng.integers(1, max_gap + 1))
        gap = _center_gap(side_cells, gap_len)
    elif gap_mode == "corner_start":
        gap = _corner_gap(side_cells, "start")
    elif gap_mode == "corner_end":
        gap = _corner_gap(side_cells, "end")
    else:
        raise ValueError(gap_mode)

    for r, c in gap:
        cup[r, c] = 0
    return cup, gap


def _analyze_cup_patch(patch: np.ndarray, color: int, bg: int) -> dict:
    arr = np.asarray(patch, dtype=int)
    h, w = arr.shape
    assert h >= 4 and w >= 4, "cups must be at least 4x4"

    missing_border: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            on_border = r in (0, h - 1) or c in (0, w - 1)
            if on_border:
                if int(arr[r, c]) != int(color):
                    missing_border.add((r, c))
            else:
                assert int(arr[r, c]) == int(bg), "cup interior must be background in the input"

    assert missing_border, "each cup must have an opening"

    candidates: list[tuple[int, int, str, list[tuple[int, int]]]] = []
    side_rank = {"top": 3, "left": 2, "right": 1, "bottom": 0}
    for side in ("top", "left", "right", "bottom"):
        coords = _side_coords(h, w, side)
        idxs = [i for i, coord in enumerate(coords) if coord in missing_border]
        if not idxs or len(idxs) > 2:
            continue
        if idxs[-1] - idxs[0] + 1 != len(idxs):
            continue
        gap = coords[idxs[0] : idxs[-1] + 1]
        if missing_border.issubset(set(gap)):
            candidates.append((len(gap), side_rank[side], side, list(gap)))

    assert candidates, "could not infer the cup opening"
    candidates.sort(reverse=True)
    _gap_len, _rank, side, gap = candidates[0]
    return {
        "side": side,
        "gap": gap,
        "h": h,
        "w": w,
    }


def _extract_cups(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    cups: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4):
        patch = crop_bbox(g, comp["bbox"])
        info = _analyze_cup_patch(patch, int(comp["color"]), int(bg))
        cups.append(
            {
                "color": int(comp["color"]),
                "bbox": tuple(map(int, comp["bbox"])),
                "coords": comp["coords"],
                "patch": patch,
                "side": info["side"],
                "gap": info["gap"],
                "h": info["h"],
                "w": info["w"],
            }
        )
    cups.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))
    return cups, int(bg)


def _stack_cups(cups: list[dict], shape: tuple[int, int], bg: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    h_grid, w_grid = map(int, shape)
    stacked = np.full((h_grid, w_grid), int(bg), dtype=int)
    positions: list[tuple[int, int]] = []
    row = 0
    col = 0
    for cup in cups:
        patch = np.asarray(cup["patch"], dtype=int)
        h, w = patch.shape
        assert row + h <= h_grid and col + w <= w_grid, "diagonal chain does not fit"
        view = stacked[row : row + h, col : col + w]
        mask = patch != int(bg)
        view[mask] = patch[mask]
        positions.append((row, col))
        row += h - 1
        col += w - 1
    return stacked, positions


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    cups, bg = _extract_cups(g)
    assert 3 <= len(cups) <= 5, "expected 3 to 5 cups"

    left_edges = [int(cup["bbox"][1]) for cup in cups]
    assert len(set(left_edges)) == len(left_edges), "expected unique left edges"

    bottoms = {int(cup["bbox"][2]) for cup in cups}
    assert len(bottoms) == 1, "expected a shared bottom row"

    stacked, positions = _stack_cups(cups, g.shape, bg)
    out = stacked.copy()

    alive: list[bool] = []
    gap_abs: list[list[tuple[int, int]]] = []
    for cup, (r0, c0) in zip(cups, positions):
        cells = [(r0 + dr, c0 + dc) for dr, dc in cup["gap"]]
        gap_abs.append(cells)
        alive.append(all(int(stacked[r, c]) == int(bg) for r, c in cells))

    h_grid, w_grid = g.shape
    for cup, (r0, c0), cells, is_alive in zip(cups, positions, gap_abs, alive):
        if not is_alive:
            continue

        for r in range(r0 + 1, r0 + cup["h"] - 1):
            for c in range(c0 + 1, c0 + cup["w"] - 1):
                if int(stacked[r, c]) == int(bg):
                    out[r, c] = FILL_COLOR

        dr, dc = SIDE_TO_DIR[cup["side"]]
        for start_r, start_c in cells:
            rr = start_r
            cc = start_c
            while 0 <= rr < h_grid and 0 <= cc < w_grid and int(stacked[rr, cc]) == int(bg):
                out[rr, cc] = FILL_COLOR
                rr += dr
                cc += dc

    if return_meta:
        return out, {
            "bg": bg,
            "cups": cups,
            "positions": positions,
            "stacked": stacked,
            "gap_abs": gap_abs,
            "alive": alive,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack bottom-aligned cup outlines into a diagonal chain, then fill and ray-cast only through openings that remain uncovered.

    Input structure:
    - The background is black and the foreground consists of 3 to 5 monochrome cup outlines with distinct left edges.
    - Each cup is a thin rectangular outline with one missing border run that defines an opening on the top, left, or right side.
    - The cups are separated in the input, share a common bottom row, and each cup uses a single non-background color.

    Rule:
    - Sort the cups by their left edge, clear the canvas, and restack them into a diagonal chain from the upper-left, placing each next bbox at the previous bbox's lower-right corner.
    - Later cup pixels overwrite earlier cup pixels only where the later cup actually has outline pixels; black gap/interior cells stay transparent.
    - After stacking, a cup is alive only if every one of its original gap cells is still black; alive cups get cyan interior fill plus cyan rays emitted outward from each gap cell until a cup wall or the grid edge stops the ray.

    Key insight:
    - The output depends on a four-step causal chain: left-to-right ordering determines the diagonal stack, the stack determines which openings survive, only surviving openings unlock fill, and the same surviving openings also determine the ray sources and directions.
    """
    return _solve(grid, return_meta=False)


def _choose_roles(n: int, rng: np.random.Generator) -> list[dict]:
    roles = [{"side": None, "gap_mode": None} for _ in range(n)]

    dead_prev_idx = int(rng.integers(1, n))
    roles[dead_prev_idx] = {
        "side": str(rng.choice(["top", "left"])),
        "gap_mode": "corner_start",
    }

    choices_top = [i for i in range(1, n) if i != dead_prev_idx]
    alive_top_idx = int(rng.choice(choices_top if choices_top else [0]))
    roles[alive_top_idx] = {"side": "top", "gap_mode": "center"}

    remaining = [i for i in range(n) if roles[i]["side"] is None]
    alive_side_idx = int(rng.choice(remaining))
    if alive_side_idx == 0:
        side = "right"
    else:
        side = str(rng.choice(["left", "right"]))
    roles[alive_side_idx] = {"side": side, "gap_mode": "center"}

    if n >= 4 and rng.random() < 0.45:
        extra_dead = [i for i in range(n - 1) if roles[i]["side"] is None]
        if extra_dead:
            idx = int(rng.choice(extra_dead))
            roles[idx] = {"side": "right", "gap_mode": "corner_end"}

    for i in range(n):
        if roles[i]["side"] is None:
            side = str(rng.choice(["top", "left", "right"]))
            if i == 0 and side == "left":
                side = "right"
            roles[i] = {"side": side, "gap_mode": "center"}
    return roles


def _sample_layout(
    n: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], int, int] | None:
    widths = [int(rng.integers(4, 7)) for _ in range(n)]
    heights = [int(rng.integers(4, 7)) for _ in range(n)]

    chain_h = 1 + sum(h - 1 for h in heights)
    chain_w = 1 + sum(w - 1 for w in widths)
    input_w = sum(widths) + (n - 1) + 2
    min_h = max(18, max(heights) + 8, chain_h)
    min_w = max(chain_w, input_w)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = int(rng.integers(min_h, min(30, min_h + 2) + 1))
    grid_w = int(rng.integers(min_w, min(30, min_w + 2) + 1))
    return widths, heights, grid_h, grid_w


def _render_input(
    widths: list[int],
    heights: list[int],
    roles: list[dict],
    colors: list[int],
    grid_h: int,
    grid_w: int,
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    left = 1
    bottom = grid_h - 1
    for w, h, role, color in zip(widths, heights, roles, colors):
        cup, _gap = _build_cup(h, w, color, role["side"], role["gap_mode"], rng)
        top = bottom - h + 1
        grid[top : top + h, left : left + w] = cup
        left += w + 1
    return grid


def _semantic_validate(grid: np.ndarray) -> bool:
    inp = np.asarray(grid, dtype=int)
    out, meta = _solve(inp, return_meta=True)

    cups = meta["cups"]
    positions = meta["positions"]
    stacked = meta["stacked"]
    bg = int(meta["bg"])
    alive = list(meta["alive"])
    gap_abs = meta["gap_abs"]

    assert 3 <= len(cups) <= 5
    assert any(alive) and not all(alive), "need both live and blocked openings"
    assert any(cup["side"] == "top" for cup, flag in zip(cups, alive) if flag), "need at least one live top opening"
    assert any(cup["side"] in ("left", "right") for cup, flag in zip(cups, alive) if flag), "need at least one live side opening"
    assert not np.array_equal(out, stacked), "output must differ from the stacked chain"
    assert np.any(out == FILL_COLOR), "output must add cyan fill or rays"

    for cup, (r0, c0), cells, flag in zip(cups, positions, gap_abs, alive):
        if flag:
            assert all(int(stacked[r, c]) == bg for r, c in cells), "live openings must stay uncovered"
            for r, c in cells:
                assert int(out[r, c]) == FILL_COLOR, "live gap cells must become cyan"
            filled = out[r0 + 1 : r0 + cup["h"] - 1, c0 + 1 : c0 + cup["w"] - 1]
            assert np.any(filled == FILL_COLOR), "live cups must get interior fill"
        else:
            assert any(int(stacked[r, c]) != bg for r, c in cells), "dead openings must be geometrically blocked"
            filled = out[r0 + 1 : r0 + cup["h"] - 1, c0 + 1 : c0 + cup["w"] - 1]
            assert not np.any(filled == FILL_COLOR), "dead cups must stay unfilled"

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(3, 6))
    layout = _sample_layout(n, rng)
    if layout is None:
        return None
    widths, heights, grid_h, grid_w = layout
    roles = _choose_roles(n, rng)
    colors = list(map(int, rng.choice(np.array(PALETTE, dtype=int), size=n, replace=False)))

    inp = _render_input(widths, heights, roles, colors, grid_h, grid_w, rng)
    try:
        if _semantic_validate(inp):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-cup-chain input")
