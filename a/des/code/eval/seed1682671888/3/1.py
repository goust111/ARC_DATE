from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


SIDES = ("top", "right", "bottom", "left")
LAYOUTS = [(1, 3), (3, 1), (2, 2), (1, 4), (4, 1), (2, 3), (3, 2)]
MAX_GEN_ATTEMPTS = 400


def _rotate_cw(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, -int(k) % 4)


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _find_band(line: np.ndarray, bg: int) -> int:
    nz = [i for i, v in enumerate(map(int, line)) if v != bg]
    assert nz, "missing border band"
    start, end = min(nz), max(nz)
    assert nz == list(range(start, end + 1)), "band must be contiguous"
    color = int(line[start])
    assert all(int(v) == color for v in line[start : end + 1]), "band must be monochrome"
    return color


def _band_colors(grid: np.ndarray, bg: int) -> list[int]:
    top = _find_band(grid[0, :], bg)
    right = _find_band(grid[:, -1], bg)
    bottom = _find_band(grid[-1, :], bg)
    left = _find_band(grid[:, 0], bg)
    colors = [int(top), int(right), int(bottom), int(left)]
    assert len(set(colors)) == 4, "band colors must be distinct"
    return colors


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _is_border_touching(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _framed_inner(sub: np.ndarray, bg: int) -> tuple[np.ndarray, int] | None:
    h, w = sub.shape
    if h != w or h < 5:
        return None
    border = np.concatenate((sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]))
    colors = set(map(int, border))
    if len(colors) != 1:
        return None
    frame = int(next(iter(colors)))
    if frame == bg:
        return None
    inner = sub[1:-1, 1:-1]
    if inner.size == 0 or bool((inner == bg).any()) or bool((inner == frame).any()):
        return None
    if inner.shape[0] != inner.shape[1]:
        return None
    return np.array(inner, dtype=int), frame


def _rare_color(patch: np.ndarray) -> int:
    counts = Counter(map(int, patch.reshape(-1)))
    assert len(counts) == 2, "each piece must use exactly two non-background colors"
    rare, rare_count = min(counts.items(), key=lambda kv: kv[1])
    assert rare_count < max(counts.values()), "rare color must be unique"
    return int(rare)


def _edge_signatures(patch: np.ndarray, rare: int) -> dict[str, tuple[int, ...]]:
    return {
        "top": tuple(int(v) for v in (patch[0, 1:-1] == rare)),
        "right": tuple(int(v) for v in (patch[1:-1, -1] == rare)),
        "bottom": tuple(int(v) for v in (patch[-1, 1:-1] == rare)),
        "left": tuple(int(v) for v in (patch[1:-1, 0] == rare)),
    }


def _complement(sig: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(1 - int(v) for v in sig)


def _extract_restored_tiles(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    side_colors = _band_colors(g, bg)
    side_of_color = {int(color): idx for idx, color in enumerate(side_colors)}

    comps = _components4(g != bg)
    inner_tiles: list[np.ndarray] = []
    seam_color = None
    base_side = None

    for comp in comps:
        if _is_border_touching(comp, h, w):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        sub = np.array(g[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        framed = _framed_inner(sub, bg)
        if framed is not None:
            inner, frame = framed
            assert seam_color is None, "expected exactly one framed base piece"
            seam_color = int(frame)
            base_rare = _rare_color(inner)
            assert base_rare in side_of_color, "base rare color must map to a border band"
            base_side = int(side_of_color[base_rare])
            inner_tiles.append(inner)
        else:
            assert sub.shape[0] == sub.shape[1], "pieces must be square"
            assert not bool((sub == bg).any()), "plain pieces must be dense"
            inner_tiles.append(sub)

    assert seam_color is not None and base_side is not None, "missing framed base piece"
    assert inner_tiles, "missing interior pieces"

    size = int(inner_tiles[0].shape[0])
    tiles: list[dict] = []
    for patch in inner_tiles:
        assert patch.shape == (size, size), "all pieces must share one square size"
        rare = _rare_color(patch)
        assert rare in side_of_color, "rare color must be one of the band colors"
        rot = (int(side_of_color[rare]) - int(base_side)) % 4
        restored = np.array(_rotate_cw(patch, rot), dtype=int)
        restored_rare = _rare_color(restored)
        tiles.append(
            {
                "patch": restored,
                "rare": int(restored_rare),
                "edges": _edge_signatures(restored, restored_rare),
            }
        )
    return tiles, int(seam_color)


def _fits_position(
    tile: dict,
    r: int,
    c: int,
    rows: int,
    cols: int,
    placed: dict[tuple[int, int], int],
    tiles: list[dict],
    zero_sig: tuple[int, ...],
) -> bool:
    edges = tile["edges"]

    if r == 0:
        if edges["top"] != zero_sig:
            return False
    else:
        if edges["top"] == zero_sig:
            return False
        up = placed.get((r - 1, c))
        if up is not None and edges["top"] != _complement(tiles[up]["edges"]["bottom"]):
            return False

    if r == rows - 1:
        if edges["bottom"] != zero_sig:
            return False
    else:
        if edges["bottom"] == zero_sig:
            return False
        down = placed.get((r + 1, c))
        if down is not None and edges["bottom"] != _complement(tiles[down]["edges"]["top"]):
            return False

    if c == 0:
        if edges["left"] != zero_sig:
            return False
    else:
        if edges["left"] == zero_sig:
            return False
        left = placed.get((r, c - 1))
        if left is not None and edges["left"] != _complement(tiles[left]["edges"]["right"]):
            return False

    if c == cols - 1:
        if edges["right"] != zero_sig:
            return False
    else:
        if edges["right"] == zero_sig:
            return False
        right = placed.get((r, c + 1))
        if right is not None and edges["right"] != _complement(tiles[right]["edges"]["left"]):
            return False

    return True


def _solve_layout(tiles: list[dict]) -> tuple[int, int, list[list[int]]]:
    n = len(tiles)
    zero_sig = tuple(0 for _ in tiles[0]["edges"]["top"])
    solutions: list[tuple[int, int, list[list[int]]]] = []

    for rows in range(1, n + 1):
        if n % rows != 0:
            continue
        cols = n // rows
        positions = [(r, c) for r in range(rows) for c in range(cols)]
        placed: dict[tuple[int, int], int] = {}
        used: set[int] = set()

        def dfs() -> None:
            if len(solutions) >= 2:
                return
            if len(placed) == len(positions):
                layout = [[-1 for _ in range(cols)] for _ in range(rows)]
                for (rr, cc), idx in placed.items():
                    layout[rr][cc] = idx
                solutions.append((rows, cols, layout))
                return

            best_pos = None
            best_cands = None
            for rr, cc in positions:
                if (rr, cc) in placed:
                    continue
                cands = [
                    idx
                    for idx in range(n)
                    if idx not in used and _fits_position(tiles[idx], rr, cc, rows, cols, placed, tiles, zero_sig)
                ]
                if not cands:
                    return
                if best_cands is None or len(cands) < len(best_cands):
                    best_pos = (rr, cc)
                    best_cands = cands
            assert best_pos is not None and best_cands is not None

            rr, cc = best_pos
            for idx in best_cands:
                placed[(rr, cc)] = int(idx)
                used.add(int(idx))
                dfs()
                used.remove(int(idx))
                del placed[(rr, cc)]
                if len(solutions) >= 2:
                    return

        dfs()

    assert len(solutions) == 1, "piece assembly must be unique"
    return solutions[0]


def _assemble_output(tiles: list[dict], seam_color: int, rows: int, cols: int, layout: list[list[int]]) -> np.ndarray:
    size = int(tiles[0]["patch"].shape[0])
    out = np.full((rows * size + (rows - 1), cols * size + (cols - 1)), int(seam_color), dtype=int)
    for r in range(rows):
        for c in range(cols):
            idx = int(layout[r][c])
            r0 = r * (size + 1)
            c0 = c * (size + 1)
            out[r0 : r0 + size, c0 : c0 + size] = tiles[idx]["patch"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble rotated square fragments by decoding their rotation from border bands,
    then pack them into the unique rectangle whose touching edges are bitwise-complementary.
    The seams between packed fragments are filled with the special frame color.
    """
    tiles, seam_color = _extract_restored_tiles(np.asarray(grid, dtype=int))
    rows, cols, layout = _solve_layout(tiles)
    return _assemble_output(tiles, seam_color, rows, cols, layout)


def _pattern_pool(length: int) -> list[tuple[int, ...]]:
    pairs: dict[tuple[int, ...], tuple[int, ...]] = {}
    for bits_int in range(1, (1 << length) - 1):
        bits = tuple((bits_int >> i) & 1 for i in range(length))
        comp = _complement(bits)
        key = min(bits, comp)
        pairs[key] = key
    return list(pairs.values())


def _build_tile(
    size: int,
    body_color: int,
    rare_color: int,
    edges: dict[str, tuple[int, ...]],
    core_pos: tuple[int, int],
) -> np.ndarray:
    tile = np.full((size, size), int(body_color), dtype=int)
    cr, cc = core_pos
    tile[int(cr), int(cc)] = int(rare_color)
    for i, bit in enumerate(edges["top"], start=1):
        if bit:
            tile[0, i] = int(rare_color)
    for i, bit in enumerate(edges["bottom"], start=1):
        if bit:
            tile[-1, i] = int(rare_color)
    for i, bit in enumerate(edges["left"], start=1):
        if bit:
            tile[i, 0] = int(rare_color)
    for i, bit in enumerate(edges["right"], start=1):
        if bit:
            tile[i, -1] = int(rare_color)
    return tile


def _make_expected_output(
    rows: int,
    cols: int,
    size: int,
    body_color: int,
    band_colors: list[int],
    seam_color: int,
    base_side: int,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    zero_sig = tuple(0 for _ in range(size - 2))
    edges_by_pos: dict[tuple[int, int], dict[str, tuple[int, ...]]] = {
        (r, c): {side: zero_sig for side in SIDES} for r in range(rows) for c in range(cols)
    }

    needed = rows * (cols - 1) + cols * (rows - 1)
    pool = _pattern_pool(size - 2)
    assert len(pool) >= needed, "not enough unique connector patterns"
    random.shuffle(pool)
    pool_idx = 0

    for r in range(rows):
        for c in range(cols - 1):
            pat = pool[pool_idx]
            pool_idx += 1
            if random.random() < 0.5:
                left_pat = pat
                right_pat = _complement(pat)
            else:
                left_pat = _complement(pat)
                right_pat = pat
            edges_by_pos[(r, c)]["right"] = left_pat
            edges_by_pos[(r, c + 1)]["left"] = right_pat

    for r in range(rows - 1):
        for c in range(cols):
            pat = pool[pool_idx]
            pool_idx += 1
            if random.random() < 0.5:
                top_pat = pat
                bottom_pat = _complement(pat)
            else:
                top_pat = _complement(pat)
                bottom_pat = pat
            edges_by_pos[(r, c)]["bottom"] = top_pat
            edges_by_pos[(r + 1, c)]["top"] = bottom_pat

    piece_count = rows * cols
    deltas = []
    while len(deltas) < piece_count:
        chunk = [0, 1, 2, 3]
        random.shuffle(chunk)
        deltas.extend(chunk)
    deltas = deltas[:piece_count]
    if 0 not in deltas:
        deltas[random.randrange(piece_count)] = 0

    base_piece = random.randrange(piece_count)
    if deltas[base_piece] != 0:
        swap = deltas.index(0)
        deltas[swap], deltas[base_piece] = deltas[base_piece], deltas[swap]

    core_choices = [
        (size // 2 - 1, size // 2 - 1),
        (size // 2 - 1, size // 2),
        (size // 2, size // 2 - 1),
        (size // 2, size // 2),
    ]

    tiles: list[np.ndarray] = []
    input_patches: list[np.ndarray] = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            delta = int(deltas[idx])
            rare_color = int(band_colors[(base_side + delta) % 4])
            core_pos = random.choice(core_choices)
            canonical = _build_tile(size, body_color, rare_color, edges_by_pos[(r, c)], core_pos)
            tiles.append(canonical)
            input_patches.append(np.array(_rotate_cw(canonical, -delta), dtype=int))
            idx += 1

    layout = [[r * cols + c for c in range(cols)] for r in range(rows)]
    expected = _assemble_output(
        [{"patch": tile, "edges": _edge_signatures(tile, _rare_color(tile))} for tile in tiles],
        seam_color,
        rows,
        cols,
        layout,
    )
    return expected, input_patches, base_piece


def _place_component(grid: np.ndarray, occ: np.ndarray, comp: np.ndarray) -> bool:
    h, w = grid.shape
    ch, cw = comp.shape
    if h - ch - 3 < 2 or w - cw - 3 < 2:
        return False
    for _ in range(300):
        r0 = random.randint(2, h - ch - 3)
        c0 = random.randint(2, w - cw - 3)
        rr0 = r0 - 1
        cc0 = c0 - 1
        rr1 = r0 + ch + 1
        cc1 = c0 + cw + 1
        if bool(occ[rr0:rr1, cc0:cc1].any()):
            continue
        grid[r0 : r0 + ch, c0 : c0 + cw] = comp
        occ[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _draw_bands(grid: np.ndarray, bg: int, colors: list[int]) -> None:
    h, w = grid.shape
    top_len = random.randint(2, max(2, w - 4))
    top_start = random.randint(1, w - top_len - 1)
    grid[0, top_start : top_start + top_len] = int(colors[0])

    right_len = random.randint(2, max(2, h - 4))
    right_start = random.randint(1, h - right_len - 1)
    grid[right_start : right_start + right_len, -1] = int(colors[1])

    bottom_len = random.randint(2, max(2, w - 4))
    bottom_start = random.randint(1, w - bottom_len - 1)
    grid[-1, bottom_start : bottom_start + bottom_len] = int(colors[2])

    left_len = random.randint(2, max(2, h - 4))
    left_start = random.randint(1, h - left_len - 1)
    grid[left_start : left_start + left_len, 0] = int(colors[3])

    assert _band_colors(grid, bg) == [int(c) for c in colors]


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(LAYOUTS)
    size = random.choice([6, 7])
    palette = random.sample(list(range(10)), 7)
    bg, body_color, seam_color = map(int, palette[:3])
    band_colors = [int(v) for v in palette[3:7]]
    base_side = random.randrange(4)

    expected, patches, base_piece = _make_expected_output(rows, cols, size, body_color, band_colors, seam_color, base_side)

    max_piece = size + 2
    min_h = max(24, rows * size + 8, cols * size + 8, max_piece + 6)
    min_w = max(24, cols * size + 8, rows * size + 8, max_piece + 6)
    h = random.randint(min_h, min_h + 8)
    w = random.randint(min_w, min_w + 8)
    grid = np.full((h, w), int(bg), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    framed = np.full((size + 2, size + 2), int(seam_color), dtype=int)
    framed[1:-1, 1:-1] = patches[base_piece]
    pieces = [framed] + [patches[i] for i in range(len(patches)) if i != base_piece]
    random.shuffle(pieces)
    for comp in pieces:
        if not _place_component(grid, occ, comp):
            return None

    _draw_bands(grid, bg, band_colors)

    if Counter(map(int, grid.reshape(-1))).most_common(1)[0][0] != bg:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
