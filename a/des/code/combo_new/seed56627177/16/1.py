import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
FG = 2
SYMMETRIES = ("vertical", "horizontal", "central", "rot90")
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    mask = grid != bg
    r0, c0, r1, c1 = _bbox(mask)
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not bool(mask.any()):
        return None
    r0, c0, r1, c1 = _bbox(mask)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = np.zeros_like(mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    if np.array_equal(mask, expected):
        return r0, c0, r1, c1
    return None


def _find_unique_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        bbox = _frame_bbox(grid == color)
        if bbox is not None:
            candidates.append((color, bbox))
    assert len(candidates) == 1, "expected a unique one-pixel frame"
    return candidates[0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _overlay(dst: np.ndarray, src: np.ndarray, bg: int) -> np.ndarray:
    out = dst.copy()
    mask = src != bg
    conflict = mask & (out != bg) & (out != src)
    assert not bool(conflict.any())
    out[mask] = src[mask]
    return out


def _symmetry_scores(grid: np.ndarray) -> dict[str, int]:
    scores = {
        "vertical": int(np.count_nonzero(grid != np.fliplr(grid))),
        "horizontal": int(np.count_nonzero(grid != np.flipud(grid))),
        "central": int(np.count_nonzero(grid != np.rot90(grid, 2))),
        "rot90": 10**9,
    }
    if grid.shape[0] == grid.shape[1]:
        scores["rot90"] = int(np.count_nonzero(grid != np.rot90(grid, 1)))
    return scores


def _infer_symmetry(grid: np.ndarray) -> str:
    order = ("rot90", "vertical", "horizontal", "central")
    scores = _symmetry_scores(grid)
    best = min(scores[name] for name in order)
    for name in order:
        if scores[name] == best:
            return name
    raise AssertionError("unreachable")


def _complete_sprite(seed: np.ndarray, symmetry: str, bg: int) -> np.ndarray:
    out = seed.copy()
    if symmetry == "vertical":
        return _overlay(out, np.fliplr(seed), bg)
    if symmetry == "horizontal":
        return _overlay(out, np.flipud(seed), bg)
    if symmetry == "central":
        return _overlay(out, np.rot90(seed, 2), bg)
    assert symmetry == "rot90"
    assert seed.shape[0] == seed.shape[1], "90-degree symmetry needs a square seed bbox"
    for k in (1, 2, 3):
        out = _overlay(out, np.rot90(seed, k), bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the symmetry encoded by fragments inside a frame, then complete the outside seed sprite with that symmetry.

    Input structure:
    - The background is black and one non-background color forms a single-pixel-thick rectangular frame.
    - Inside the frame, several same-colored fragments collectively exhibit one symmetry type: vertical mirror, horizontal mirror, 180-degree point symmetry, or 90-degree rotational symmetry.
    - Outside the frame, a separate connected seed sprite uses another color arrangement and is not yet symmetric.

    Rule:
    - Identify the frame and inspect only its interior to infer which symmetry type the inner fragments obey.
    - Extract the outside seed sprite, use its bounding-box center as the symmetry center, and add the reflected or rotated copies needed to close that symmetry.
    - Remove the frame and the inner fragments; return only the completed outside sprite, cropped to its bounding box on a black background.

    Key insight:
    - The framed pattern is a control code for the transformation type; it is not the object to finish, but an instruction telling how to complete the different sprite outside the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    frame_color, (fr0, fc0, fr1, fc1) = _find_unique_frame(g, bg)

    inner = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
    assert inner.size > 0 and bool((inner != bg).any())
    symmetry = _infer_symmetry(inner)

    outside_mask = (g != bg) & (g != frame_color)
    outside_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    comps = _components(outside_mask)
    assert len(comps) == 1, "expected a single outside seed sprite"

    pts = np.asarray(comps[0], dtype=int)
    sr0, sc0 = pts.min(axis=0)
    sr1, sc1 = pts.max(axis=0)
    seed = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    seed[seed == frame_color] = bg

    completed = _complete_sprite(seed, symmetry, bg)
    return _crop_non_bg(completed, bg)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _random_connected_mask(
    h: int,
    w: int,
    *,
    min_cells: int,
    max_cells: int,
    touch_all_sides: bool,
) -> np.ndarray | None:
    for _ in range(200):
        target = random.randint(min_cells, max_cells)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(mask[nr, nc]):
                    frontier.add((nr, nc))

        add_frontier(*start)
        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if bool(mask[cell]):
                continue
            mask[cell] = True
            add_frontier(*cell)

        if int(mask.sum()) != target:
            continue
        if touch_all_sides and not _touches_all_sides(mask):
            continue
        return mask
    return None


def _mask_to_patch(mask: np.ndarray, color: int = FG) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _make_asymmetric_patch(h: int, w: int) -> np.ndarray | None:
    total = h * w
    lo = max(3, total // 3)
    hi = min(total - 1, (2 * total) // 3)
    if lo > hi:
        return None
    for _ in range(200):
        mask = _random_connected_mask(h, w, min_cells=lo, max_cells=hi, touch_all_sides=False)
        if mask is None:
            continue
        patch = _mask_to_patch(mask)
        scores = _symmetry_scores(patch)
        if scores["vertical"] == 0 or scores["horizontal"] == 0 or scores["central"] == 0:
            continue
        if h == w and scores["rot90"] == 0:
            continue
        return patch
    return None


def _make_code_pattern(symmetry: str, inner_size: int) -> np.ndarray | None:
    assert inner_size % 2 == 1
    q = inner_size // 2
    for _ in range(200):
        mh = random.randint(2, min(4, q))
        mw = random.randint(2, min(4, q))
        motif = _make_asymmetric_patch(mh, mw)
        if motif is None:
            continue
        r0 = random.randint(0, q - mh)
        c0 = random.randint(0, q - mw)
        base = np.full((inner_size, inner_size), BG, dtype=int)
        base[r0 : r0 + mh, c0 : c0 + mw] = motif
        if symmetry == "vertical":
            code = _overlay(base, np.fliplr(base), BG)
        elif symmetry == "horizontal":
            code = _overlay(base, np.flipud(base), BG)
        elif symmetry == "central":
            code = _overlay(base, np.rot90(base, 2), BG)
        else:
            code = base.copy()
            for k in (1, 2, 3):
                code = _overlay(code, np.rot90(base, k), BG)
        scores = _symmetry_scores(code)
        if symmetry != _infer_symmetry(code):
            continue
        if symmetry == "rot90":
            if scores["rot90"] != 0 or scores["vertical"] == 0 or scores["horizontal"] == 0:
                continue
        else:
            if scores[symmetry] != 0:
                continue
            if any(scores[name] == 0 for name in SYMMETRIES if name != symmetry):
                continue
        return code
    return None


def _make_seed_patch(symmetry: str) -> np.ndarray | None:
    for _ in range(200):
        if symmetry == "rot90":
            side = random.randint(4, 6)
            h = w = side
        else:
            h = random.randint(4, 6)
            w = random.randint(4, 6)
        total = h * w
        lo = max(4, total // 3)
        hi = min(total - 1, (2 * total) // 3)
        if lo > hi:
            continue
        mask = _random_connected_mask(h, w, min_cells=lo, max_cells=hi, touch_all_sides=True)
        if mask is None:
            continue
        patch = _mask_to_patch(mask)
        if _symmetry_scores(patch)[symmetry] == 0:
            continue
        completed = _complete_sprite(patch, symmetry, BG)
        if int(np.count_nonzero(completed != BG)) == int(np.count_nonzero(patch != BG)):
            continue
        return patch
    return None


def _choose_layout(frame_outer: int, seed_h: int, seed_w: int) -> tuple[int, int, int, int, int, int] | None:
    for _ in range(100):
        side = random.choice(("left", "right", "top", "bottom"))
        top = random.randint(1, 3)
        bottom = random.randint(1, 3)
        left = random.randint(1, 3)
        right = random.randint(1, 3)
        gap = random.randint(2, 4)
        if side in ("left", "right"):
            base_h = max(frame_outer, seed_h)
            extra_h_cap = max(0, min(4, 30 - (base_h + top + bottom)))
            extra_h = random.randint(0, extra_h_cap)
            h = base_h + top + bottom + extra_h
            w = frame_outer + seed_w + gap + left + right
            if max(h, w) > 30:
                continue
            frame_top = top + random.randint(0, h - top - bottom - frame_outer)
            seed_top = top + random.randint(0, h - top - bottom - seed_h)
            if side == "left":
                seed_left = left
                frame_left = seed_left + seed_w + gap
            else:
                frame_left = left
                seed_left = frame_left + frame_outer + gap
        else:
            base_w = max(frame_outer, seed_w)
            extra_w_cap = max(0, min(4, 30 - (base_w + left + right)))
            extra_w = random.randint(0, extra_w_cap)
            w = base_w + left + right + extra_w
            h = frame_outer + seed_h + gap + top + bottom
            if max(h, w) > 30:
                continue
            frame_left = left + random.randint(0, w - left - right - frame_outer)
            seed_left = left + random.randint(0, w - left - right - seed_w)
            if side == "top":
                seed_top = top
                frame_top = seed_top + seed_h + gap
            else:
                frame_top = top
                seed_top = frame_top + frame_outer + gap
        return h, w, frame_top, frame_left, seed_top, seed_left
    return None


def _gen_attempt(symmetry: str) -> np.ndarray | None:
    inner_size = random.choice((7, 9, 11, 13))
    code = _make_code_pattern(symmetry, inner_size)
    seed = _make_seed_patch(symmetry)
    if code is None or seed is None:
        return None

    frame_outer = inner_size + 2
    seed_h, seed_w = seed.shape
    layout = _choose_layout(frame_outer, seed_h, seed_w)
    if layout is None:
        return None
    h, w, frame_top, frame_left, seed_top, seed_left = layout

    grid = np.full((h, w), BG, dtype=int)
    ft, fl = frame_top, frame_left
    fb, fr = ft + frame_outer - 1, fl + frame_outer - 1
    grid[ft, fl : fr + 1] = FRAME
    grid[fb, fl : fr + 1] = FRAME
    grid[ft : fb + 1, fl] = FRAME
    grid[ft : fb + 1, fr] = FRAME

    inner = grid[ft + 1 : fb, fl + 1 : fr]
    inner[code != BG] = code[code != BG]

    seed_dst = grid[seed_top : seed_top + seed_h, seed_left : seed_left + seed_w]
    seed_dst[seed != BG] = seed[seed != BG]

    expected = _crop_non_bg(_complete_sprite(seed, symmetry, BG), BG)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    symmetry = random.choice(SYMMETRIES)
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(symmetry)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-code ARC input")
