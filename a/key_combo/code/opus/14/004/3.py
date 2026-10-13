from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
PINK = 6
PALETTE = [1, 2, 3, 4, 5, 7, 8, 9]
MAX_ATTEMPTS = 400


def _segments(length: int, separators: set[int]) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    start = None
    for i in range(length):
        if i in separators:
            if start is not None:
                parts.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        parts.append((start, length))
    return parts


def _marker_origin(block: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(block == PINK)
    if coords.shape[0] != 4:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if (int(r1 - r0 + 1), int(c1 - c0 + 1)) != (2, 2):
        return None
    if not np.all(block[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == PINK):
        return None
    return int(r0), int(c0)


def _native_color(block: np.ndarray) -> int:
    flat = block.ravel()
    counts = Counter(int(v) for v in flat if int(v) not in (BLACK, PINK))
    assert counts, "every block must contain a non-black, non-pink foreground color"
    best_count = max(counts.values())
    candidates = [color for color, count in counts.items() if count == best_count]
    if len(candidates) == 1:
        return candidates[0]

    first_idx = {}
    for idx, value in enumerate(flat):
        color = int(value)
        if color in candidates and color not in first_idx:
            first_idx[color] = idx
    return min(candidates, key=lambda color: first_idx[color])


def _analyze_grid(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[dict], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    sep_rows = {r for r in range(g.shape[0]) if np.all(g[r, :] == BLACK)}
    sep_cols = {c for c in range(g.shape[1]) if np.all(g[:, c] == BLACK)}
    row_blocks = _segments(g.shape[0], sep_rows)
    col_blocks = _segments(g.shape[1], sep_cols)
    assert row_blocks and col_blocks, "grid must contain at least one logical block"

    infos: list[dict] = []
    anchors: list[tuple[int, int]] = []
    for br, (r0, r1) in enumerate(row_blocks):
        for bc, (c0, c1) in enumerate(col_blocks):
            block = g[r0:r1, c0:c1]
            marker = _marker_origin(block)
            info = {
                "br": br,
                "bc": bc,
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "fg": _native_color(block),
                "marker": marker,
            }
            infos.append(info)
            if marker is not None:
                anchors.append((br, bc))

    assert len(anchors) == 1, "expected exactly one anchor block with a 2x2 pink marker"
    return row_blocks, col_blocks, infos, anchors[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Invert odd-distance logical blocks after separator-line coordinate compression.

    Input structure:
    - A black-background grid is partitioned by full black separator rows and full black separator columns.
    - The non-separator cells form a logical grid of rectangular blocks with uneven physical sizes.
    - Each block contains one visible foreground color on black, and exactly one block also contains a solid 2x2 pink marker.

    Rule:
    - Detect the full black separator rows and columns, then compress them away conceptually to recover block coordinates.
    - Find the block containing the 2x2 pink marker and use that block as the anchor.
    - For each block, compute Manhattan distance in block coordinates to the anchor block; even distances keep the block unchanged, while odd distances invert black with that block's native foreground color.
    - Separator lines remain black, and the pink marker itself is preserved.

    Key insight:
    - The inversion condition depends on Manhattan distance between logical blocks, not on raw pixel position or block size, so the separator lines must be removed conceptually before measuring parity.
    """
    g = np.asarray(grid, dtype=int)
    row_blocks, col_blocks, infos, anchor = _analyze_grid(g)
    ar, ac = anchor

    out = g.copy()
    for info in infos:
        distance = abs(int(info["br"]) - ar) + abs(int(info["bc"]) - ac)
        if distance % 2 == 0:
            continue

        r0, r1 = int(info["r0"]), int(info["r1"])
        c0, c1 = int(info["c0"]), int(info["c1"])
        fg = int(info["fg"])
        block = g[r0:r1, c0:c1]
        out_block = out[r0:r1, c0:c1]

        out_block[block == BLACK] = fg
        out_block[block == fg] = BLACK
        out_block[block == PINK] = PINK

    assert len(row_blocks) * len(col_blocks) == len(infos)
    return out


def _choose_sizes() -> list[int]:
    count = random.randint(3, 5)
    return [random.randint(2, 5) for _ in range(count)]


def _ranges_from_sizes(sizes: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for i, size in enumerate(sizes):
        ranges.append((start, start + size))
        start += size
        if i != len(sizes) - 1:
            start += 1
    return ranges


def _parity_counts(nr: int, nc: int, ar: int, ac: int) -> tuple[int, int]:
    odd = 0
    even_non_anchor = 0
    for br in range(nr):
        for bc in range(nc):
            d = abs(br - ar) + abs(bc - ac)
            if d % 2 == 1:
                odd += 1
            elif d != 0:
                even_non_anchor += 1
    return odd, even_non_anchor


def _grow_connected(h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_mask(h: int, w: int) -> np.ndarray:
    total = h * w
    for _ in range(80):
        kind = random.choice(["rect", "l", "plus", "diag", "frame", "bars", "grow"])
        mask = np.zeros((h, w), dtype=bool)

        if kind == "rect":
            ph = random.randint(1, h)
            pw = random.randint(1, w)
            if ph == h and pw == w:
                continue
            r0 = random.randint(0, h - ph)
            c0 = random.randint(0, w - pw)
            mask[r0 : r0 + ph, c0 : c0 + pw] = True

        elif kind == "l":
            r_edge = 0 if random.random() < 0.5 else h - 1
            c_edge = 0 if random.random() < 0.5 else w - 1
            mask[r_edge, :] = True
            mask[:, c_edge] = True
            if h >= 4 and random.random() < 0.5:
                mask[1 if r_edge == 0 else h - 2, : random.randint(1, w)] = True
            if w >= 4 and random.random() < 0.5:
                mask[: random.randint(1, h), 1 if c_edge == 0 else w - 2] = True

        elif kind == "plus":
            rr = random.randrange(h)
            cc = random.randrange(w)
            mask[rr, :] = True
            mask[:, cc] = True
            if random.random() < 0.4 and h >= 3:
                mask[max(0, rr - 1) : min(h, rr + 2), cc] = True
            if random.random() < 0.4 and w >= 3:
                mask[rr, max(0, cc - 1) : min(w, cc + 2)] = True

        elif kind == "diag":
            if random.random() < 0.5:
                for r in range(h):
                    c = min(w - 1, int(round(r * (w - 1) / max(1, h - 1))))
                    mask[r, c] = True
            else:
                for r in range(h):
                    c = (w - 1) - min(w - 1, int(round(r * (w - 1) / max(1, h - 1))))
                    mask[r, c] = True
            if random.random() < 0.35:
                mask = mask | np.fliplr(mask)

        elif kind == "frame":
            if h < 3 or w < 3:
                continue
            mask[0, :] = True
            mask[-1, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            if random.random() < 0.4:
                side = random.choice(["top", "bottom", "left", "right"])
                if side == "top":
                    mask[0, random.randrange(w)] = False
                elif side == "bottom":
                    mask[-1, random.randrange(w)] = False
                elif side == "left":
                    mask[random.randrange(h), 0] = False
                else:
                    mask[random.randrange(h), -1] = False

        elif kind == "bars":
            rr = random.randrange(h)
            cc = random.randrange(w)
            mask[rr, :] = True
            mask[:, cc] = True
            if h >= 3:
                r1 = random.randrange(h)
                c0 = random.randint(0, w - 1)
                c1 = random.randint(c0, w - 1)
                mask[r1, c0 : c1 + 1] = True

        else:
            target = random.randint(max(1, total // 4), max(1, (2 * total) // 3))
            mask = _grow_connected(h, w, target)

        count = int(mask.sum())
        if 0 < count < total:
            return mask

    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    return mask if h > 1 else np.array([[True, False][:w]], dtype=bool)


def _choose_anchor(rows: list[int], cols: list[int]) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for ar, h in enumerate(rows):
        for ac, w in enumerate(cols):
            if h < 4 or w < 4:
                continue
            odd, even_non_anchor = _parity_counts(len(rows), len(cols), ar, ac)
            if odd >= 3 and even_non_anchor >= 3:
                candidates.append((ar, ac))
    if not candidates:
        return None
    return random.choice(candidates)


def _marker_position(h: int, w: int) -> tuple[int, int]:
    row_choices = list(range(h - 1))
    col_choices = list(range(w - 1))
    if h >= 5 and random.random() < 0.7:
        row_choices = list(range(1, h - 2))
    if w >= 5 and random.random() < 0.7:
        col_choices = list(range(1, w - 2))
    return random.choice(row_choices), random.choice(col_choices)


def _render_block(h: int, w: int, color: int, anchor: bool) -> tuple[np.ndarray, dict] | None:
    for _ in range(100):
        block = np.zeros((h, w), dtype=int)
        marker = None
        if anchor:
            mr, mc = _marker_position(h, w)
            marker = (mr, mc)
            block[mr : mr + 2, mc : mc + 2] = PINK

        mask = _random_mask(h, w)
        if marker is not None:
            mr, mc = marker
            mask[mr : mr + 2, mc : mc + 2] = False

        fg_count = int(mask.sum())
        black_count = h * w - fg_count - (4 if marker is not None else 0)
        if fg_count <= 0 or black_count <= 0:
            continue

        block[mask] = color
        return block, {"color": color, "marker": marker}
    return None


def _repair_full_black_rows(grid: np.ndarray, metas: list[dict], row_ranges: list[tuple[int, int]]) -> bool:
    for r in range(grid.shape[0]):
        if any(r0 <= r < r1 for r0, r1 in row_ranges):
            if np.any(grid[r, :] != BLACK):
                continue
            candidates = [meta for meta in metas if meta["r0"] <= r < meta["r1"]]
            random.shuffle(candidates)
            placed = False
            for meta in candidates:
                sub = grid[meta["r0"] : meta["r1"], meta["c0"] : meta["c1"]]
                if int(np.count_nonzero(sub == BLACK)) <= 1:
                    continue
                options = [c for c in range(meta["c0"], meta["c1"]) if grid[r, c] == BLACK]
                if not options:
                    continue
                c = random.choice(options)
                grid[r, c] = meta["color"]
                placed = True
                break
            if not placed:
                return False
    return True


def _repair_full_black_cols(grid: np.ndarray, metas: list[dict], col_ranges: list[tuple[int, int]]) -> bool:
    for c in range(grid.shape[1]):
        if any(c0 <= c < c1 for c0, c1 in col_ranges):
            if np.any(grid[:, c] != BLACK):
                continue
            candidates = [meta for meta in metas if meta["c0"] <= c < meta["c1"]]
            random.shuffle(candidates)
            placed = False
            for meta in candidates:
                sub = grid[meta["r0"] : meta["r1"], meta["c0"] : meta["c1"]]
                if int(np.count_nonzero(sub == BLACK)) <= 1:
                    continue
                options = [r for r in range(meta["r0"], meta["r1"]) if grid[r, c] == BLACK]
                if not options:
                    continue
                r = random.choice(options)
                grid[r, c] = meta["color"]
                placed = True
                break
            if not placed:
                return False
    return True


def _validate_generated(grid: np.ndarray, anchor: tuple[int, int], row_sizes: list[int], col_sizes: list[int]) -> bool:
    try:
        row_blocks, col_blocks, infos, found_anchor = _analyze_grid(grid)
    except Exception:
        return False

    if found_anchor != anchor:
        return False
    if [r1 - r0 for r0, r1 in row_blocks] != row_sizes:
        return False
    if [c1 - c0 for c0, c1 in col_blocks] != col_sizes:
        return False

    odd, even_non_anchor = _parity_counts(len(row_sizes), len(col_sizes), *anchor)
    if odd < 3 or even_non_anchor < 3:
        return False

    for info in infos:
        block = grid[info["r0"] : info["r1"], info["c0"] : info["c1"]]
        fg = int(info["fg"])
        if int(np.count_nonzero(block == fg)) == 0:
            return False
        if int(np.count_nonzero(block == BLACK)) == 0:
            return False
        if info["marker"] is not None and int(np.count_nonzero(block == PINK)) != 4:
            return False

    for r0, r1 in row_blocks:
        for r in range(r0, r1):
            if not np.any(grid[r, :] != BLACK):
                return False
    for c0, c1 in col_blocks:
        for c in range(c0, c1):
            if not np.any(grid[:, c] != BLACK):
                return False

    try:
        out = main(grid)
    except Exception:
        return False
    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    row_sizes = _choose_sizes()
    col_sizes = _choose_sizes()
    anchor = _choose_anchor(row_sizes, col_sizes)
    if anchor is None:
        return None

    row_ranges = _ranges_from_sizes(row_sizes)
    col_ranges = _ranges_from_sizes(col_sizes)
    grid = np.zeros((row_ranges[-1][1], col_ranges[-1][1]), dtype=int)

    metas: list[dict] = []
    used_colors: dict[tuple[int, int], int] = {}
    for br, (r0, r1) in enumerate(row_ranges):
        for bc, (c0, c1) in enumerate(col_ranges):
            neighbor_colors = {
                used_colors[(br - 1, bc)]
                for key in [(br - 1, bc)]
                if key in used_colors
            }
            if (br, bc - 1) in used_colors:
                neighbor_colors.add(used_colors[(br, bc - 1)])
            choices = [color for color in PALETTE if color not in neighbor_colors] or PALETTE
            color = random.choice(choices)
            rendered = _render_block(r1 - r0, c1 - c0, color, anchor == (br, bc))
            if rendered is None:
                return None
            block, block_meta = rendered
            grid[r0:r1, c0:c1] = block
            used_colors[(br, bc)] = color
            metas.append(
                {
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "color": color,
                    "marker": block_meta["marker"],
                }
            )

    if not _repair_full_black_rows(grid, metas, row_ranges):
        return None
    if not _repair_full_black_cols(grid, metas, col_ranges):
        return None

    if not _validate_generated(grid, anchor, row_sizes, col_sizes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None and max(grid.shape) <= 30:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
