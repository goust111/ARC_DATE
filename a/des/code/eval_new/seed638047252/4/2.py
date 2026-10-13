from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GREY = 5
ACCENT = 2
BODY_PALETTE = [1, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
PLACE_RETRIES = 80
PATTERN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _mode_int(values: list[int]) -> int:
    counts = Counter(int(v) for v in values)
    best_n = max(counts.values())
    return min(v for v, n in counts.items() if n == best_n)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == int(bg):
                continue
            q = deque([(int(r0), int(c0))])
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            comps.append(
                {
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "area": len(coords),
                }
            )
    return comps


def _patch_from_component(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return patch


def _is_rect_outline(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 2:
        return False
    if int(mask.sum()) != 2 * (h + w) - 4:
        return False
    if not mask[0, :].all() or not mask[-1, :].all():
        return False
    if not mask[:, 0].all() or not mask[:, -1].all():
        return False
    if h > 2 and w > 2 and bool(mask[1:-1, 1:-1].any()):
        return False
    return True


def _find_reference_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, tuple[int, int], int]] = []
    for comp in _find_components(g, bg):
        patch = _patch_from_component(g, comp, bg)
        vals = sorted({int(v) for v in patch[patch != int(bg)]})
        if len(vals) != 1:
            continue
        if not _is_rect_outline(patch != int(bg)):
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        touches_corner = (r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)
        if not touches_corner:
            continue
        out_shape = (int(r1 - r0 + 1), int(c1 - c0 + 1))
        bbox_area = out_shape[0] * out_shape[1]
        candidates.append((bbox_area, out_shape, int(vals[0])))
    assert candidates, "expected one grey reference frame in a corner"
    _area, out_shape, color = max(candidates, key=lambda x: x[0])
    return out_shape, color


def _analyze_piece_patch(patch: np.ndarray, bg: int) -> dict:
    mask = np.asarray(patch != int(bg), dtype=bool)
    ph, pw = map(int, mask.shape)

    best: tuple[int, int, int, int, int] | None = None
    for core_top in range(ph):
        for core_left in range(pw):
            for core_bottom in range(core_top + 2, ph):
                for core_right in range(core_left + 2, pw):
                    core_h = int(core_bottom - core_top + 1)
                    core_w = int(core_right - core_left + 1)
                    area = int(core_h * core_w)

                    if not bool(mask[core_top, core_left]):
                        continue
                    if not bool(mask[core_top, core_right]):
                        continue
                    if not bool(mask[core_bottom, core_left]):
                        continue
                    if not bool(mask[core_bottom, core_right]):
                        continue
                    if core_h > 2 and core_w > 2 and not bool(mask[core_top + 1 : core_bottom, core_left + 1 : core_right].all()):
                        continue

                    ok = True
                    for r in range(ph):
                        for c in range(pw):
                            inside = core_top <= r <= core_bottom and core_left <= c <= core_right
                            if bool(mask[r, c]):
                                if inside:
                                    continue
                                on_side = (
                                    (r == core_top - 1 and core_left <= c <= core_right)
                                    or (r == core_bottom + 1 and core_left <= c <= core_right)
                                    or (c == core_left - 1 and core_top <= r <= core_bottom)
                                    or (c == core_right + 1 and core_top <= r <= core_bottom)
                                )
                                if not on_side:
                                    ok = False
                                    break
                            elif inside and core_top < r < core_bottom and core_left < c < core_right:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        continue

                    cand = (area, -core_top, -core_left, core_bottom, core_right)
                    if best is None or cand > best:
                        best = cand

    assert best is not None, "failed to infer a legal core rectangle"
    _area, neg_top, neg_left, core_bottom, core_right = best
    core_top = int(-neg_top)
    core_left = int(-neg_left)
    core_h = int(core_bottom - core_top + 1)
    core_w = int(core_right - core_left + 1)
    assert core_h >= 3 and core_w >= 3

    def side_sig(side: str) -> tuple[int, ...]:
        vals: list[int] = []
        if side in ("left", "right"):
            for rr in range(core_h):
                r = core_top + rr
                if side == "left":
                    outside = core_left - 1
                    boundary = core_left
                else:
                    outside = core_right + 1
                    boundary = core_right
                if 0 <= outside < mask.shape[1] and bool(mask[r, outside]):
                    vals.append(1)
                elif not bool(mask[r, boundary]):
                    vals.append(-1)
                else:
                    vals.append(0)
        else:
            for cc in range(core_w):
                c = core_left + cc
                if side == "top":
                    outside = core_top - 1
                    boundary = core_top
                else:
                    outside = core_bottom + 1
                    boundary = core_bottom
                if 0 <= outside < mask.shape[0] and bool(mask[outside, c]):
                    vals.append(1)
                elif not bool(mask[boundary, c]):
                    vals.append(-1)
                else:
                    vals.append(0)
        return tuple(vals)

    return {
        "patch": np.asarray(patch, dtype=int),
        "core_top": int(core_top),
        "core_left": int(core_left),
        "core_h": int(core_h),
        "core_w": int(core_w),
        "sig": {
            "left": side_sig("left"),
            "right": side_sig("right"),
            "top": side_sig("top"),
            "bottom": side_sig("bottom"),
        },
    }


def _is_flat(sig: tuple[int, ...]) -> bool:
    return all(int(v) == 0 for v in sig)


def _assemble_from_core_positions(
    pieces: list[dict],
    positions: list[tuple[int, int]],
    out_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    out = np.full(out_shape, int(bg), dtype=int)
    out_h, out_w = map(int, out_shape)
    for piece, (core_r, core_c) in zip(pieces, positions):
        patch = piece["patch"]
        start_r = int(core_r - piece["core_top"])
        start_c = int(core_c - piece["core_left"])
        for rr, cc in np.argwhere(patch != int(bg)):
            gr = int(start_r + rr)
            gc = int(start_c + cc)
            assert 0 <= gr < out_h and 0 <= gc < out_w
            val = int(patch[int(rr), int(cc)])
            if int(out[gr, gc]) not in (int(bg), int(val)):
                raise AssertionError("assembled pieces overlap inconsistently")
            out[gr, gc] = int(val)
    assert not bool((out == int(bg)).any()), "assembled rectangle must be solid"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered jigsaw-like fragments into the original full rectangle.

    Input structure:
    - bg = black. Several disconnected fragments are scattered on the canvas.
    - Each fragment is a mostly solid rectangle with single-pixel edge tabs in a second color
      and matching bg notches cut out of some edge cells.
    - A monochrome grey rectangular reference frame sits in one corner; its bbox size equals
      the output size.

    Rule:
    - Read each fragment's hidden core rectangle and its four edge signatures, where every edge
      position is classified as tab, notch, or flat.
    - Two fragments touch exactly when one edge signature is the positionwise complement of the
      opposite edge on another fragment; tabs must land in the other piece's notches.
    - Use all complementary matches to recover the fragments' relative translations, then place
      them without rotation or reflection into the reference frame's size.

    Key insight: The colored tabs and black notches are not decoration; they are a signed edge code
    that simultaneously identifies adjacency and fixes the exact offset between neighboring pieces.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(int(v) for v in g.flatten()).most_common(1)[0][0])
    out_shape, ref_color = _find_reference_frame(g, bg)

    pieces: list[dict] = []
    for comp in _find_components(g, bg):
        patch = _patch_from_component(g, comp, bg)
        vals = sorted({int(v) for v in patch[patch != int(bg)]})
        if len(vals) == 1 and int(vals[0]) == int(ref_color) and _is_rect_outline(patch != int(bg)):
            continue
        pieces.append(_analyze_piece_patch(patch, bg))

    assert pieces, "expected puzzle fragments"

    opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    matches: list[dict[str, int]] = []
    for i, piece in enumerate(pieces):
        side_map: dict[str, int] = {}
        for side, sig in piece["sig"].items():
            if _is_flat(sig):
                continue
            want = tuple(-int(v) for v in sig)
            need_len = len(sig)
            cand = [
                j
                for j, other in enumerate(pieces)
                if j != i and len(other["sig"][opposite[side]]) == need_len and other["sig"][opposite[side]] == want
            ]
            assert len(cand) == 1, f"ambiguous match for piece {i} side {side}"
            side_map[side] = int(cand[0])
        matches.append(side_map)

    positions: dict[int, tuple[int, int]] = {0: (0, 0)}
    q = deque([0])
    while q:
        i = q.popleft()
        r, c = positions[i]
        for side, j in matches[i].items():
            if side == "right":
                nr, nc = int(r), int(c + pieces[i]["core_w"])
            elif side == "left":
                nr, nc = int(r), int(c - pieces[j]["core_w"])
            elif side == "bottom":
                nr, nc = int(r + pieces[i]["core_h"]), int(c)
            else:
                nr, nc = int(r - pieces[j]["core_h"]), int(c)
            if j in positions:
                assert positions[j] == (nr, nc), "inconsistent placement graph"
            else:
                positions[j] = (nr, nc)
                q.append(j)
    assert len(positions) == len(pieces), "fragment graph must be connected"

    min_r = min(r for r, _ in positions.values())
    min_c = min(c for _, c in positions.values())
    ordered_positions: list[tuple[int, int]] = []
    for i in range(len(pieces)):
        r, c = positions[i]
        ordered_positions.append((int(r - min_r), int(c - min_c)))

    max_r = max(r + pieces[i]["core_h"] for i, (r, _c) in enumerate(ordered_positions))
    max_c = max(c + pieces[i]["core_w"] for i, (_r, c) in enumerate(ordered_positions))
    assert (int(max_r), int(max_c)) == tuple(map(int, out_shape)), "reference frame size mismatch"

    for i, piece in enumerate(pieces):
        r, c = ordered_positions[i]
        if _is_flat(piece["sig"]["top"]):
            assert int(r) == 0
        if _is_flat(piece["sig"]["left"]):
            assert int(c) == 0
        if _is_flat(piece["sig"]["bottom"]):
            assert int(r + piece["core_h"]) == int(out_shape[0])
        if _is_flat(piece["sig"]["right"]):
            assert int(c + piece["core_w"]) == int(out_shape[1])

    return _assemble_from_core_positions(pieces, ordered_positions, out_shape, bg)


def _sample_layout(rng: random.Random) -> tuple[list[int], list[int]]:
    n_rows, n_cols = rng.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
    row_heights = [rng.randint(5, 6) for _ in range(n_rows)]
    col_widths = [rng.randint(5, 6) for _ in range(n_cols)]
    return row_heights, col_widths


def _sample_pattern(length: int, used: set[tuple[int, ...]], rng: random.Random) -> tuple[int, ...]:
    assert length >= 5
    inner_len = length - 2
    for _ in range(PATTERN_RETRIES):
        inner = [rng.choice([-1, 0, 1]) for _ in range(inner_len)]
        if all(v == 0 for v in inner):
            continue
        if all(v != 0 for v in inner):
            continue
        seq = tuple([0] + inner + [0])
        if seq in used:
            continue
        used.add(seq)
        return seq
    raise RuntimeError("failed to sample a unique edge pattern")


def _make_piece_patch(
    core_h: int,
    core_w: int,
    left_sig: tuple[int, ...],
    right_sig: tuple[int, ...],
    top_sig: tuple[int, ...],
    bottom_sig: tuple[int, ...],
    body_color: int,
    accent_color: int,
    bg: int,
) -> np.ndarray:
    arr = np.full((core_h + 2, core_w + 2), int(bg), dtype=int)
    arr[1 : 1 + core_h, 1 : 1 + core_w] = int(body_color)

    for rr, v in enumerate(left_sig):
        if int(v) == -1:
            arr[1 + rr, 1] = int(bg)
        elif int(v) == 1:
            arr[1 + rr, 0] = int(accent_color)
    for rr, v in enumerate(right_sig):
        if int(v) == -1:
            arr[1 + rr, core_w] = int(bg)
        elif int(v) == 1:
            arr[1 + rr, core_w + 1] = int(accent_color)
    for cc, v in enumerate(top_sig):
        if int(v) == -1:
            arr[1, 1 + cc] = int(bg)
        elif int(v) == 1:
            arr[0, 1 + cc] = int(accent_color)
    for cc, v in enumerate(bottom_sig):
        if int(v) == -1:
            arr[core_h, 1 + cc] = int(bg)
        elif int(v) == 1:
            arr[core_h + 1, 1 + cc] = int(accent_color)

    pts = np.argwhere(arr != int(bg))
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _color_grid(row_heights: list[int], col_widths: list[int], rng: random.Random) -> list[list[int]]:
    colors: list[list[int]] = []
    for i, _h in enumerate(row_heights):
        row: list[int] = []
        for j, _w in enumerate(col_widths):
            forbidden = set()
            if i > 0:
                forbidden.add(colors[i - 1][j])
            if j > 0:
                forbidden.add(row[j - 1])
            choices = [c for c in BODY_PALETTE if c not in forbidden]
            row.append(int(rng.choice(choices)))
        colors.append(row)
    return colors


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)

    def gap1(x0: int, x1: int, y0: int, y1: int) -> int:
        if y1 < x0:
            return int(x0 - y1)
        if x1 < y0:
            return int(y0 - x1)
        return 0

    return gap1(ar0, ar1, br0, br1) + gap1(ac0, ac1, bc0, bc1)


def _draw_outline(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = int(color)
    grid[r0 + h - 1, c0 : c0 + w] = int(color)
    grid[r0 : r0 + h, c0] = int(color)
    grid[r0 : r0 + h, c0 + w - 1] = int(color)


def _place_pieces(
    pieces: list[dict],
    out_shape: tuple[int, int],
    rng: random.Random,
) -> np.ndarray | None:
    out_h, out_w = map(int, out_shape)
    min_h = max(out_h + 8, 24)
    min_w = max(out_w + 8, 24)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    grid = np.full((grid_h, grid_w), int(BG), dtype=int)

    corner = rng.choice(["tl", "tr", "bl", "br"])
    frame_r = 0 if "t" in corner else grid_h - out_h
    frame_c = 0 if "l" in corner else grid_w - out_w
    _draw_outline(grid, int(frame_r), int(frame_c), int(out_h), int(out_w), int(GREY))

    blocked: list[tuple[int, int, int, int]] = [
        (int(frame_r), int(frame_c), int(frame_r + out_h - 1), int(frame_c + out_w - 1))
    ]
    order = list(range(len(pieces)))
    rng.shuffle(order)

    for idx in order:
        patch = pieces[idx]["patch"]
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = rng.randint(0, grid_h - ph)
            c0 = rng.randint(0, grid_w - pw)
            cand = (int(r0), int(c0), int(r0 + ph - 1), int(c0 + pw - 1))
            if any(_rect_gap(cand, rect) <= 1 for rect in blocked):
                continue
            blocked.append(cand)
            for rr, cc in np.argwhere(patch != int(BG)):
                grid[int(r0 + rr), int(c0 + cc)] = int(patch[int(rr), int(cc)])
            placed = True
            break
        if not placed:
            return None

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_heights, col_widths = _sample_layout(rng)
    n_rows = len(row_heights)
    n_cols = len(col_widths)
    out_h = int(sum(row_heights))
    out_w = int(sum(col_widths))

    used_vertical: dict[int, set[tuple[int, ...]]] = {}
    used_horizontal: dict[int, set[tuple[int, ...]]] = {}
    vertical_patterns: dict[tuple[int, int], tuple[int, ...]] = {}
    horizontal_patterns: dict[tuple[int, int], tuple[int, ...]] = {}

    for i, h in enumerate(row_heights):
        used_vertical.setdefault(int(h), set())
        for j in range(n_cols - 1):
            vertical_patterns[(i, j)] = _sample_pattern(int(h), used_vertical[int(h)], rng)
    for i in range(n_rows - 1):
        for j, w in enumerate(col_widths):
            used_horizontal.setdefault(int(w), set())
            horizontal_patterns[(i, j)] = _sample_pattern(int(w), used_horizontal[int(w)], rng)

    body_colors = _color_grid(row_heights, col_widths, rng)

    pieces: list[dict] = []
    core_positions: list[tuple[int, int]] = []
    row_offsets = np.cumsum([0] + row_heights[:-1]).tolist()
    col_offsets = np.cumsum([0] + col_widths[:-1]).tolist()

    for i in range(n_rows):
        for j in range(n_cols):
            left_sig = tuple(0 for _ in range(row_heights[i])) if j == 0 else tuple(
                -int(v) for v in vertical_patterns[(i, j - 1)]
            )
            right_sig = tuple(0 for _ in range(row_heights[i])) if j == n_cols - 1 else vertical_patterns[(i, j)]
            top_sig = tuple(0 for _ in range(col_widths[j])) if i == 0 else tuple(
                -int(v) for v in horizontal_patterns[(i - 1, j)]
            )
            bottom_sig = tuple(0 for _ in range(col_widths[j])) if i == n_rows - 1 else horizontal_patterns[(i, j)]

            patch = _make_piece_patch(
                int(row_heights[i]),
                int(col_widths[j]),
                left_sig,
                right_sig,
                top_sig,
                bottom_sig,
                int(body_colors[i][j]),
                int(ACCENT),
                int(BG),
            )
            piece = _analyze_piece_patch(patch, BG)
            pieces.append(piece)
            core_positions.append((int(row_offsets[i]), int(col_offsets[j])))

    expected = _assemble_from_core_positions(pieces, core_positions, (out_h, out_w), BG)
    grid = _place_pieces(pieces, (out_h, out_w), rng)
    if grid is None:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragment-assembly input")
