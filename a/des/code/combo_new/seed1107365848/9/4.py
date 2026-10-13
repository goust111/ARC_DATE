from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
MIX = 3
GREY = 5

BORDER_COLOR_POOL = (1, 2, 4, 6, 7, 8, 9)
BRANCHES = ("a_only", "b_only", "both", "none")

GRID_H_BOUNDS = (14, 24)
GRID_W_BOUNDS = (14, 24)
RECT_SIZE_BOUNDS = (2, 5)
EXTRA_OBJECT_BOUNDS = (0, 2)
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 4-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _majority(values: np.ndarray) -> int:
    return int(Counter(int(v) for v in values).most_common(1)[0][0])


def _infer_border_colors(grid: np.ndarray) -> tuple[int, int]:
    top_bottom = np.concatenate((grid[0, 1:-1], grid[-1, 1:-1]))
    left_right = np.concatenate((grid[1:-1, 0], grid[1:-1, -1]))
    return _majority(top_bottom), _majority(left_right)


def _is_rect_component(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _infer_object_color(inner: np.ndarray, exclude: set[int]) -> int | None:
    candidates = [int(v) for v in np.unique(inner) if int(v) not in exclude]
    if not candidates:
        return None

    rect_like: list[tuple[int, int, int]] = []
    for color in candidates:
        comps = _components(inner == color)
        if comps and all(_is_rect_component(comp) for comp in comps):
            rect_like.append((len(comps), -int(np.sum(inner == color)), color))
    if rect_like:
        rect_like.sort(reverse=True)
        return int(rect_like[0][2])

    return min(candidates, key=lambda color: int(np.sum(inner == color)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay border-driven stripe patterns onto grey rectangles.

    Input structure:
    - The grid has a 1-cell rectangular border: top and bottom edges share color A, left and right edges share color B, and the corner cells may be either border color.
    - Inside the border, the background is black and the foreground objects are solid grey rectangles.
    - Each rectangle may touch the A border, the B border, both kinds of border, or neither.

    Rule:
    - A rectangle touching only the A border gets horizontal stripes: odd local rows stay grey and even local rows become color A.
    - A rectangle touching only the B border gets vertical stripes: odd local columns stay grey and even local columns become color B.
    - A rectangle touching both border colors becomes a checkerboard: even local parity cells stay grey and odd local parity cells become green.
    - A rectangle touching neither border stays unchanged.

    Key insight:
    - The two border colors act as directional signals, so edge contact decides whether an object receives horizontal stripes, vertical stripes, or their checkerboard combination.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        return out

    h, w = g.shape
    a_color, b_color = _infer_border_colors(g)
    obj_color = _infer_object_color(g[1:-1, 1:-1], {a_color, b_color})
    if obj_color is None:
        return out

    for comp in _components(g == obj_color):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue

        touches_a = r0 == 1 or r1 == h - 2
        touches_b = c0 == 1 or c1 == w - 2
        if not touches_a and not touches_b:
            continue

        for r, c in comp:
            local_r = r - r0
            local_c = c - c0
            if touches_a and touches_b:
                out[r, c] = obj_color if (local_r + local_c) % 2 == 0 else MIX
            elif touches_a:
                out[r, c] = obj_color if local_r % 2 == 0 else a_color
            else:
                out[r, c] = obj_color if local_c % 2 == 0 else b_color

    return out


def _rect_conflicts(
    candidate: tuple[int, int, int, int],
    placed: list[tuple[int, int, int, int, str]],
) -> bool:
    r0, c0, hh, ww = candidate
    r1 = r0 + hh - 1
    c1 = c0 + ww - 1

    for or0, oc0, oh, ow, _kind in placed:
        or1 = or0 + oh - 1
        oc1 = oc0 + ow - 1
        separated = (
            r1 + 1 < or0
            or or1 + 1 < r0
            or c1 + 1 < oc0
            or oc1 + 1 < c0
        )
        if not separated:
            return True
    return False


def _candidate_positions(
    kind: str,
    h: int,
    w: int,
    hh: int,
    ww: int,
) -> list[tuple[int, int]]:
    row_free = range(2, h - hh - 1)
    col_free = range(2, w - ww - 1)
    positions: list[tuple[int, int]] = []

    if kind == "a_only":
        for c0 in col_free:
            positions.append((1, c0))
            positions.append((h - 1 - hh, c0))
    elif kind == "b_only":
        for r0 in row_free:
            positions.append((r0, 1))
            positions.append((r0, w - 1 - ww))
    elif kind == "both":
        positions.extend(
            [
                (1, 1),
                (1, w - 1 - ww),
                (h - 1 - hh, 1),
                (h - 1 - hh, w - 1 - ww),
            ]
        )
    elif kind == "none":
        for r0 in row_free:
            for c0 in col_free:
                positions.append((r0, c0))

    uniq = []
    seen = set()
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            uniq.append(pos)
    return uniq


def _place_object(
    rng: random.Random,
    h: int,
    w: int,
    kind: str,
    placed: list[tuple[int, int, int, int, str]],
) -> bool:
    hh = rng.randint(*RECT_SIZE_BOUNDS)
    ww = rng.randint(*RECT_SIZE_BOUNDS)
    candidates = _candidate_positions(kind, h, w, hh, ww)
    rng.shuffle(candidates)

    for r0, c0 in candidates:
        if not _rect_conflicts((r0, c0, hh, ww), placed):
            placed.append((r0, c0, hh, ww, kind))
            return True
    return False


def _render_input(
    h: int,
    w: int,
    a_color: int,
    b_color: int,
    rects: list[tuple[int, int, int, int, str]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = a_color
    grid[-1, :] = a_color
    grid[:, 0] = b_color
    grid[:, -1] = b_color

    for corner in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
        grid[corner] = rng.choice((a_color, b_color))

    for r0, c0, hh, ww, _kind in rects:
        grid[r0 : r0 + hh, c0 : c0 + ww] = GREY
    return grid


def _render_output(
    inp: np.ndarray,
    a_color: int,
    b_color: int,
    rects: list[tuple[int, int, int, int, str]],
) -> np.ndarray:
    out = inp.copy()
    for r0, c0, hh, ww, kind in rects:
        for dr in range(hh):
            for dc in range(ww):
                rr = r0 + dr
                cc = c0 + dc
                if kind == "a_only":
                    out[rr, cc] = GREY if dr % 2 == 0 else a_color
                elif kind == "b_only":
                    out[rr, cc] = GREY if dc % 2 == 0 else b_color
                elif kind == "both":
                    out[rr, cc] = GREY if (dr + dc) % 2 == 0 else MIX
                else:
                    out[rr, cc] = GREY
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_BOUNDS)
    w = rng.randint(*GRID_W_BOUNDS)
    a_color, b_color = rng.sample(BORDER_COLOR_POOL, 2)

    rects: list[tuple[int, int, int, int, str]] = []
    for kind in ("both", "a_only", "b_only", "none"):
        if not _place_object(rng, h, w, kind, rects):
            return None

    extra_count = rng.randint(*EXTRA_OBJECT_BOUNDS)
    extra_branches = list(BRANCHES)
    rng.shuffle(extra_branches)
    for kind in extra_branches[:extra_count]:
        _place_object(rng, h, w, kind, rects)

    inp = _render_input(h, w, a_color, b_color, rects, rng)
    expected = _render_output(inp, a_color, b_color, rects)

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
