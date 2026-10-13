from __future__ import annotations

import numpy as np


DIR_OPPOSITE = {"right": "left", "left": "right", "down": "up", "up": "down"}
BG = 8
COLOR_POOL = np.array([0, 1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
GRID_MAX = 30
MAX_GENERATION_ATTEMPTS = 300
_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _is_jagged_profile(profile: list[int]) -> bool:
    if len(profile) < 2:
        return False
    levels = sorted(set(int(v) for v in profile))
    if len(levels) != 2 or levels[1] - levels[0] != 1:
        return False
    return all(int(profile[i]) != int(profile[i - 1]) for i in range(1, len(profile)))


def _row_runs(mask: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(mask[r]) for r in range(mask.shape[0])]


def _col_runs(mask: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(mask[:, c]) for c in range(mask.shape[1])]


def _contiguous(vals: np.ndarray) -> bool:
    return bool(vals.size > 0 and np.array_equal(vals, np.arange(int(vals[0]), int(vals[-1]) + 1)))


def _analyze_piece(mask: np.ndarray) -> tuple[str, tuple[int, ...]]:
    h, w = mask.shape
    candidates: list[tuple[str, tuple[int, ...]]] = []

    right_profile: list[int] = []
    ok = True
    for cols in _row_runs(mask):
        if not _contiguous(cols) or int(cols[0]) != 0:
            ok = False
            break
        right_profile.append(int(cols.size))
    if ok and _is_jagged_profile(right_profile):
        candidates.append(("right", tuple(right_profile)))

    left_profile: list[int] = []
    ok = True
    for cols in _row_runs(mask):
        if not _contiguous(cols) or int(cols[-1]) != w - 1:
            ok = False
            break
        left_profile.append(int(cols.size))
    if ok and _is_jagged_profile(left_profile):
        candidates.append(("left", tuple(left_profile)))

    down_profile: list[int] = []
    ok = True
    for rows in _col_runs(mask):
        if not _contiguous(rows) or int(rows[0]) != 0:
            ok = False
            break
        down_profile.append(int(rows.size))
    if ok and _is_jagged_profile(down_profile):
        candidates.append(("down", tuple(down_profile)))

    up_profile: list[int] = []
    ok = True
    for rows in _col_runs(mask):
        if not _contiguous(rows) or int(rows[-1]) != h - 1:
            ok = False
            break
        up_profile.append(int(rows.size))
    if ok and _is_jagged_profile(up_profile):
        candidates.append(("up", tuple(up_profile)))

    if len(candidates) != 1:
        raise ValueError("piece must have exactly one jagged side")
    return candidates[0]


def _compatible(a: dict, b: dict) -> bool:
    if DIR_OPPOSITE[a["orientation"]] != b["orientation"]:
        return False
    if len(a["profile"]) != len(b["profile"]):
        return False
    seam_sum = int(a["profile"][0] + b["profile"][0])
    return all(int(x) + int(y) == seam_sum for x, y in zip(a["profile"], b["profile"]))


def _solve_matching(pieces: list[dict]) -> list[tuple[int, int]]:
    n = len(pieces)
    mates = {i: [j for j in range(i + 1, n) if _compatible(pieces[i], pieces[j])] for i in range(n)}
    solutions: list[list[tuple[int, int]]] = []
    used = [False] * n
    current: list[tuple[int, int]] = []

    def dfs() -> None:
        if len(solutions) > 1:
            return
        try:
            i = next(idx for idx in range(n) if not used[idx])
        except StopIteration:
            solutions.append(current.copy())
            return
        used[i] = True
        for j in mates[i]:
            if used[j]:
                continue
            used[j] = True
            current.append((i, j))
            dfs()
            current.pop()
            used[j] = False
        used[i] = False

    dfs()
    if not solutions:
        raise ValueError("no valid pairing")
    return solutions[0]


def _render_pair(a: dict, b: dict, bg: int) -> np.ndarray:
    if a["orientation"] == "right":
        left_piece, right_piece = a, b
    elif a["orientation"] == "left":
        left_piece, right_piece = b, a
    elif a["orientation"] == "down":
        top_piece, bottom_piece = a, b
        total_h = int(top_piece["profile"][0] + bottom_piece["profile"][0])
        width = len(top_piece["profile"])
        rect = np.full((total_h, width), bg, dtype=int)
        for c, h1 in enumerate(top_piece["profile"]):
            rect[: int(h1), c] = int(top_piece["color"])
        for c, h2 in enumerate(bottom_piece["profile"]):
            rect[total_h - int(h2) :, c] = int(bottom_piece["color"])
        return rect
    else:
        top_piece, bottom_piece = b, a
        total_h = int(top_piece["profile"][0] + bottom_piece["profile"][0])
        width = len(top_piece["profile"])
        rect = np.full((total_h, width), bg, dtype=int)
        for c, h1 in enumerate(top_piece["profile"]):
            rect[: int(h1), c] = int(top_piece["color"])
        for c, h2 in enumerate(bottom_piece["profile"]):
            rect[total_h - int(h2) :, c] = int(bottom_piece["color"])
        return rect

    height = len(left_piece["profile"])
    total_w = int(left_piece["profile"][0] + right_piece["profile"][0])
    rect = np.full((height, total_w), bg, dtype=int)
    for r, w1 in enumerate(left_piece["profile"]):
        rect[r, : int(w1)] = int(left_piece["color"])
    for r, w2 in enumerate(right_piece["profile"]):
        rect[r, total_w - int(w2) :] = int(right_piece["color"])
    return rect


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair interlocking jagged-edge fragments, merge each pair into a rectangle,
    then sort the rectangles by area and bottom-pack them with 1-column gaps.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    pieces = _find_components(g, bg)
    if len(pieces) < 4 or len(pieces) > 8 or len(pieces) % 2 != 0:
        raise ValueError("expected 4-8 monochrome pieces")

    for piece in pieces:
        orientation, profile = _analyze_piece(piece["mask"])
        piece["orientation"] = orientation
        piece["profile"] = profile

    pairings = _solve_matching(pieces)
    merged: list[tuple[int, tuple[int, int], np.ndarray]] = []
    for i, j in pairings:
        rect = _render_pair(pieces[i], pieces[j], bg)
        anchor = (
            min(int(pieces[i]["bbox"][0]), int(pieces[j]["bbox"][0])),
            min(int(pieces[i]["bbox"][1]), int(pieces[j]["bbox"][1])),
        )
        merged.append((int(rect.size), anchor, rect))

    merged.sort(key=lambda item: (item[0], item[1][0], item[1][1]))
    out_h = max(rect.shape[0] for _, _, rect in merged)
    out_w = sum(rect.shape[1] for _, _, rect in merged) + len(merged) - 1
    out = np.full((out_h, out_w), bg, dtype=int)
    c0 = 0
    for _area, _anchor, rect in merged:
        h, w = rect.shape
        out[out_h - h :, c0 : c0 + w] = rect
        c0 += w + 1
    return out


def _make_lr_pair(rect_h: int, rect_w: int, split: int, start: int, colors: tuple[int, int], bg: int) -> dict:
    bits = np.array([(start + i) % 2 for i in range(rect_h)], dtype=int)
    left_profile = split + bits
    right_profile = rect_w - left_profile

    left = np.full((rect_h, split + 1), bg, dtype=int)
    for r, width in enumerate(left_profile):
        left[r, : int(width)] = int(colors[0])

    right = np.full((rect_h, rect_w - split), bg, dtype=int)
    for r, width in enumerate(right_profile):
        right[r, -int(width) :] = int(colors[1])

    return {
        "kind": "lr",
        "rect_h": rect_h,
        "rect_w": rect_w,
        "area": rect_h * rect_w,
        "pieces": [left, right],
        "merged": _render_pair(
            {"orientation": "right", "profile": tuple(int(v) for v in left_profile), "color": int(colors[0])},
            {"orientation": "left", "profile": tuple(int(v) for v in right_profile), "color": int(colors[1])},
            bg,
        ),
    }


def _make_ud_pair(rect_h: int, rect_w: int, split: int, start: int, colors: tuple[int, int], bg: int) -> dict:
    bits = np.array([(start + i) % 2 for i in range(rect_w)], dtype=int)
    top_profile = split + bits
    bottom_profile = rect_h - top_profile

    top = np.full((split + 1, rect_w), bg, dtype=int)
    for c, height in enumerate(top_profile):
        top[: int(height), c] = int(colors[0])

    bottom = np.full((rect_h - split, rect_w), bg, dtype=int)
    for c, height in enumerate(bottom_profile):
        bottom[-int(height) :, c] = int(colors[1])

    return {
        "kind": "ud",
        "rect_h": rect_h,
        "rect_w": rect_w,
        "area": rect_h * rect_w,
        "pieces": [top, bottom],
        "merged": _render_pair(
            {"orientation": "down", "profile": tuple(int(v) for v in top_profile), "color": int(colors[0])},
            {"orientation": "up", "profile": tuple(int(v) for v in bottom_profile), "color": int(colors[1])},
            bg,
        ),
    }


def _min_rect_width(kind: str) -> int:
    return 5 if kind == "lr" else 4


def _sample_pairs(bg: int) -> list[dict] | None:
    n_pairs = int(_RNG.integers(2, 5))
    kinds = ["lr" if int(_RNG.integers(0, 2)) == 0 else "ud" for _ in range(n_pairs)]
    width_budget = GRID_MAX - (n_pairs - 1)
    colors = [int(c) for c in _RNG.choice(COLOR_POOL, size=2 * n_pairs, replace=False)]

    pairs: list[dict] = []
    used_areas: set[int] = set()
    used_lr_heights: set[int] = set()
    used_ud_widths: set[int] = set()
    used_width = 0

    for i, kind in enumerate(kinds):
        future_min = sum(_min_rect_width(next_kind) for next_kind in kinds[i + 1 :])
        max_rect_w = min(8, width_budget - used_width - future_min)
        min_rect_w = _min_rect_width(kind)
        if max_rect_w < min_rect_w:
            return None

        made = None
        for _ in range(100):
            rect_w = int(_RNG.integers(min_rect_w, max_rect_w + 1))
            start = int(_RNG.integers(0, 2))
            pair_colors = (colors[2 * i], colors[2 * i + 1])
            if kind == "lr":
                rect_h = int(_RNG.integers(4, 9))
                if rect_h in used_lr_heights:
                    continue
                area = rect_h * rect_w
                if area in used_areas:
                    continue
                split = int(_RNG.integers(2, rect_w - 2))
                made = _make_lr_pair(rect_h, rect_w, split, start, pair_colors, bg)
                used_lr_heights.add(rect_h)
            else:
                if rect_w in used_ud_widths:
                    continue
                rect_h = int(_RNG.integers(5, 10))
                area = rect_h * rect_w
                if area in used_areas:
                    continue
                split = int(_RNG.integers(2, rect_h - 2))
                made = _make_ud_pair(rect_h, rect_w, split, start, pair_colors, bg)
                used_ud_widths.add(rect_w)

            used_areas.add(area)
            used_width += rect_w
            pairs.append(made)
            break

        if made is None:
            return None

    return pairs


def _expected_output(pairs: list[dict], bg: int) -> np.ndarray:
    rects = sorted((pair["merged"] for pair in pairs), key=lambda rect: int(rect.size))
    out_h = max(rect.shape[0] for rect in rects)
    out_w = sum(rect.shape[1] for rect in rects) + len(rects) - 1
    out = np.full((out_h, out_w), bg, dtype=int)
    c0 = 0
    for rect in rects:
        h, w = rect.shape
        out[out_h - h :, c0 : c0 + w] = rect
        c0 += w + 1
    return out


def _place_pieces(pieces: list[np.ndarray], bg: int) -> np.ndarray | None:
    max_h = max(piece.shape[0] for piece in pieces)
    max_w = max(piece.shape[1] for piece in pieces)
    grid_h = int(_RNG.integers(max(18, max_h + 3), GRID_MAX + 1))
    grid_w = int(_RNG.integers(max(18, max_w + 3), GRID_MAX + 1))
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    order = [int(i) for i in _RNG.permutation(len(pieces))]
    positions: list[tuple[int, int] | None] = [None] * len(pieces)
    for idx in order:
        piece = pieces[idx]
        h, w = piece.shape
        if h + 2 > grid_h or w + 2 > grid_w:
            return None
        placed = False
        for _ in range(300):
            r0 = int(_RNG.integers(1, grid_h - h))
            c0 = int(_RNG.integers(1, grid_w - w))
            rr0, rr1 = r0 - 1, r0 + h + 1
            cc0, cc1 = c0 - 1, c0 + w + 1
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            positions[idx] = (r0, c0)
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    for piece, pos in zip(pieces, positions):
        if pos is None:
            return None
        r0, c0 = pos
        mask = piece != bg
        grid[r0 : r0 + piece.shape[0], c0 : c0 + piece.shape[1]][mask] = piece[mask]
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = BG
    pairs = _sample_pairs(bg)
    if pairs is None:
        return None

    expected = _expected_output(pairs, bg)
    if expected.shape[0] > GRID_MAX or expected.shape[1] > GRID_MAX:
        return None

    pieces = [piece for pair in pairs for piece in pair["pieces"]]
    _RNG.shuffle(pieces)
    grid = _place_pieces(pieces, bg)
    if grid is None:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
