from __future__ import annotations

from collections import defaultdict

import numpy as np


BG_COLORS = (0, 5)
PAIR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
TEXTURES = ("solid", "horizontal", "vertical")
MAX_ATTEMPTS = 400
MIN_BAND_HEIGHT = 5
MIN_COL_GAP = 3


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_checkerboard_colors(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    colors = [int(v) for v in np.unique(grid)]
    best: tuple[int, int, int] | None = None
    for i, a in enumerate(colors):
        for b in colors[i + 1 :]:
            for phase in (0, 1):
                matched = 0
                mismatched = 0
                for r in range(h):
                    for c in range(w):
                        v = int(grid[r, c])
                        if v != a and v != b:
                            continue
                        expected = a if ((r + c + phase) & 1) == 0 else b
                        if v == expected:
                            matched += 1
                        else:
                            mismatched += 1
                if mismatched == 0:
                    cand = (matched, a, b)
                    if best is None or cand > best:
                        best = cand
    if best is None:
        raise AssertionError("failed to identify checkerboard background")
    _matched, a, b = best
    return int(a), int(b)


def _classify_block(mask: np.ndarray) -> tuple[str, int, int] | None:
    h, w = mask.shape
    if (h, w) == (3, 3):
        if bool(mask.all()):
            return "solid", 1, 1
        return None
    if (h, w) != (4, 4):
        return None
    if bool(mask.all()):
        return "solid", 1, 1

    border = mask.copy()
    border[1:3, 1:3] = False
    if not bool(mask[border].all()):
        return None

    interior = mask[1:3, 1:3]
    if bool(interior[0].all()) and not bool(interior[1].any()):
        return "horizontal", 1, 1
    if not bool(interior[0].any()) and bool(interior[1].all()):
        return "horizontal", 2, 1
    if bool(interior[:, 0].all()) and not bool(interior[:, 1].any()):
        return "vertical", 1, 1
    if not bool(interior[:, 0].any()) and bool(interior[:, 1].all()):
        return "vertical", 1, 2
    return None


def _detect_blocks(grid: np.ndarray) -> dict[int, list[dict]]:
    bg_a, bg_b = _find_checkerboard_colors(grid)
    blocks_by_color: dict[int, list[dict]] = defaultdict(list)
    for color in (int(v) for v in np.unique(grid) if int(v) not in (bg_a, bg_b)):
        for comp in _components(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                patch[r - r0, c - c0] = True
            info = _classify_block(patch)
            if info is None:
                raise AssertionError(f"invalid block shape for color {color}")
            texture, anchor_r, anchor_c = info
            blocks_by_color[color].append(
                {
                    "color": int(color),
                    "texture": texture,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "anchor": (int(r0 + anchor_r), int(c0 + anchor_c)),
                }
            )

    for color, blocks in blocks_by_color.items():
        if len(blocks) != 2:
            raise AssertionError(f"color {color} must appear in exactly two blocks")
        if blocks[0]["texture"] != blocks[1]["texture"]:
            raise AssertionError(f"paired blocks for color {color} must share one texture")
    return blocks_by_color


def _ordered_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    tr, tc = end
    col_step = 1 if tc >= sc else -1
    horiz = [(int(sr), int(c)) for c in range(sc, tc + col_step, col_step)]
    row_step = 1 if tr >= sr else -1
    vert = [(int(r), int(tc)) for r in range(sr, tr + row_step, row_step)]
    return horiz + vert[1:]


def _bridge_cells(
    left: dict,
    right: dict,
    *,
    reserve_full_dashed: bool,
) -> set[tuple[int, int]]:
    sr, sc = map(int, left["anchor"])
    tr, tc = map(int, right["anchor"])
    texture = str(left["texture"])
    if texture == "vertical":
        cells: set[tuple[int, int]] = set()
        col_step = 1 if tc >= sc else -1
        for c in range(sc, tc + col_step, col_step):
            for rr in (sr - 1, sr, sr + 1):
                cells.add((int(rr), int(c)))
        row_step = 1 if tr >= sr else -1
        for r in range(sr, tr + row_step, row_step):
            for cc in (tc - 1, tc, tc + 1):
                cells.add((int(r), int(cc)))
        return cells

    path = _ordered_path((sr, sc), (tr, tc))
    if texture == "horizontal" and not reserve_full_dashed:
        return {path[i] for i in range(0, len(path), 2)}
    return set(path)


def _paint_bridge(out: np.ndarray, left: dict, right: dict) -> None:
    h, w = out.shape
    color = int(left["color"])
    for r, c in _bridge_cells(left, right, reserve_full_dashed=False):
        if 0 <= r < h and 0 <= c < w:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color framed blocks and draw an L bridge whose line style is
    encoded by the block texture.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    blocks_by_color = _detect_blocks(g)
    for blocks in blocks_by_color.values():
        left, right = sorted(
            blocks,
            key=lambda block: (int(block["anchor"][1]), int(block["anchor"][0])),
        )
        _paint_bridge(out, left, right)
    return out


def _checkerboard(h: int, w: int, phase: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    parity = (rows + cols + int(phase)) & 1
    return np.where(parity == 0, BG_COLORS[0], BG_COLORS[1]).astype(int)


def _make_block_patch(texture: str, size: int, variant: int) -> tuple[np.ndarray, tuple[int, int]]:
    if texture == "solid":
        patch = np.ones((size, size), dtype=bool)
        return patch, ((size - 1) // 2, (size - 1) // 2)

    patch = np.zeros((4, 4), dtype=bool)
    patch[0, :] = True
    patch[-1, :] = True
    patch[:, 0] = True
    patch[:, -1] = True
    if texture == "horizontal":
        inner_row = 1 + int(variant)
        patch[inner_row, 1:3] = True
        return patch, (inner_row, 1)
    if texture == "vertical":
        inner_col = 1 + int(variant)
        patch[1:3, inner_col] = True
        return patch, (1, inner_col)
    raise AssertionError("unknown texture")


def _place_block(grid: np.ndarray, block: dict) -> None:
    r0, c0, r1, c1 = map(int, block["bbox"])
    patch = block["mask"]
    color = int(block["color"])
    region = grid[r0 : r1 + 1, c0 : c1 + 1]
    region[patch] = color


def _sample_textures(pair_count: int, rng: np.random.Generator) -> list[str]:
    if pair_count == 2:
        items = list(TEXTURES)
        rng.shuffle(items)
        return items[:2]
    if pair_count == 3:
        items = list(TEXTURES)
        rng.shuffle(items)
        return items
    items = list(TEXTURES) + [TEXTURES[int(rng.integers(0, len(TEXTURES)))]]
    rng.shuffle(items)
    return items


def _band_layout(h: int, pair_count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    min_h = pair_count * MIN_BAND_HEIGHT + (pair_count - 1) + 2
    if h < min_h:
        raise AssertionError("grid too short for requested pair count")

    top = 1
    bottom = 1
    heights = [MIN_BAND_HEIGHT] * pair_count
    gaps = [1] * (pair_count - 1)
    extra = h - min_h
    slots = [("top", 0), ("bottom", 0)]
    slots.extend(("band", i) for i in range(pair_count))
    slots.extend(("gap", i) for i in range(pair_count - 1))
    for _ in range(extra):
        kind, idx = slots[int(rng.integers(0, len(slots)))]
        if kind == "top":
            top += 1
        elif kind == "bottom":
            bottom += 1
        elif kind == "band":
            heights[idx] += 1
        else:
            gaps[idx] += 1

    bands: list[tuple[int, int]] = []
    cursor = top
    for i, height in enumerate(heights):
        bands.append((int(cursor), int(cursor + height - 1)))
        cursor += height
        if i < len(gaps):
            cursor += gaps[i]
    if cursor + bottom != h:
        raise AssertionError("band layout failed to tile the grid")
    return bands


def _expected_output(inp: np.ndarray, pairs: list[tuple[dict, dict]]) -> np.ndarray:
    out = inp.copy()
    for left, right in pairs:
        _paint_bridge(out, left, right)
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(2, 5))
    min_h = pair_count * MIN_BAND_HEIGHT + (pair_count - 1) + 2
    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(18, 31))
    grid = _checkerboard(h, w, int(rng.integers(0, 2)))
    bands = _band_layout(h, pair_count, rng)
    colors = [int(v) for v in rng.choice(np.array(PAIR_COLORS, dtype=int), size=pair_count, replace=False)]
    textures = _sample_textures(pair_count, rng)

    pairs: list[tuple[dict, dict]] = []
    for band, color, texture in zip(bands, colors, textures):
        if texture == "solid":
            size_left = int(rng.choice(np.array([3, 4], dtype=int)))
            size_right = int(rng.choice(np.array([3, 4], dtype=int)))
            variant = 0
        else:
            size_left = size_right = 4
            variant = int(rng.integers(0, 2))

        left_mask, left_anchor = _make_block_patch(texture, size_left, variant)
        right_mask, right_anchor = _make_block_patch(texture, size_right, variant)

        band_top, band_bottom = band
        left_rows = list(range(band_top, band_bottom - size_left + 2))
        right_rows = list(range(band_top, band_bottom - size_right + 2))
        if not left_rows or not right_rows:
            return None
        left_row = int(rng.choice(np.array(left_rows, dtype=int)))
        if len(right_rows) > 1 and rng.random() < 0.7:
            choices = [row for row in right_rows if row != left_row]
            right_row = int(rng.choice(np.array(choices or right_rows, dtype=int)))
        else:
            right_row = int(rng.choice(np.array(right_rows, dtype=int)))

        left_col_max = w - size_left - size_right - MIN_COL_GAP - 1
        if left_col_max < 1:
            return None
        left_col = int(rng.integers(1, left_col_max + 1))
        right_col_min = left_col + size_left + MIN_COL_GAP
        right_col_max = w - size_right - 1
        if right_col_min > right_col_max:
            return None
        right_col = int(rng.integers(right_col_min, right_col_max + 1))

        left = {
            "color": color,
            "texture": texture,
            "mask": left_mask,
            "bbox": (
                int(left_row),
                int(left_col),
                int(left_row + size_left - 1),
                int(left_col + size_left - 1),
            ),
            "anchor": (int(left_row + left_anchor[0]), int(left_col + left_anchor[1])),
        }
        right = {
            "color": color,
            "texture": texture,
            "mask": right_mask,
            "bbox": (
                int(right_row),
                int(right_col),
                int(right_row + size_right - 1),
                int(right_col + size_right - 1),
            ),
            "anchor": (int(right_row + right_anchor[0]), int(right_col + right_anchor[1])),
        }

        reservation = _bridge_cells(left, right, reserve_full_dashed=True)
        r0a, c0a, r1a, c1a = map(int, left["bbox"])
        r0b, c0b, r1b, c1b = map(int, right["bbox"])
        reservation.update(
            (r, c)
            for r in range(r0a, r1a + 1)
            for c in range(c0a, c1a + 1)
        )
        reservation.update(
            (r, c)
            for r in range(r0b, r1b + 1)
            for c in range(c0b, c1b + 1)
        )
        if any(not (0 <= r < h and 0 <= c < w) for r, c in reservation):
            return None
        pairs.append((left, right))

    inp = grid.copy()
    for left, right in pairs:
        _place_block(inp, left)
        _place_block(inp, right)

    expected = _expected_output(inp, pairs)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_ATTEMPTS} attempts")
