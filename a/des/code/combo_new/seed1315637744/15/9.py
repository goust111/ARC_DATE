from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400
PATTERN_ATTEMPTS = 200
LAYOUT_ATTEMPTS = 200
MIN_SIDE = 5
MAX_SIDE = 8
COPY_COUNT = 4


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return comps


def _inverse_rot_coord(r: int, c: int, n: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return int(r), int(c)
    if k == 1:
        return int(c), int(n - 1 - r)
    if k == 2:
        return int(n - 1 - r), int(n - 1 - c)
    return int(n - 1 - c), int(r)


def _forward_rot_coord(r: int, c: int, n: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return int(r), int(c)
    if k == 1:
        return int(n - 1 - c), int(r)
    if k == 2:
        return int(n - 1 - r), int(n - 1 - c)
    return int(c), int(n - 1 - r)


def _extract_blocks(grid: np.ndarray, bg: int) -> list[dict]:
    blocks = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        blocks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "area": int(comp["area"]),
            }
        )
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare each rotated copy against the left reference pattern and keep only the corrected tampered cells.

    Input structure:
    - The background is black.
    - One square multi-color reference pattern sits on the left.
    - Several separate square copies of the same pattern appear on the right; each copy may be rotated by 0, 90, 180, or 270 degrees.
    - Every right-side copy differs from the matching rotated reference in exactly one pixel.

    Rule:
    - Find the leftmost square foreground block and treat it as the reference pattern.
    - For every other square block, test all four rotations of the reference and identify the unique rotation for which exactly one cell disagrees.
    - Locate that mismatching cell in the copy, rotate its coordinates back into the reference orientation, and read the correct color from the reference.
    - Return a square output the same size as the reference, filled with background except at those recovered reference coordinates.

    Key insight:
    - Each copy is almost a rotated reference, so the correct orientation is the one with a single mismatch; undoing that rotation maps the tampered cell back into the canonical reference coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BG

    blocks = _extract_blocks(g, bg)
    assert len(blocks) >= 2
    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    ref = blocks[0]["patch"]
    n = int(ref.shape[0])
    assert ref.shape == (n, n)
    out = np.full((n, n), bg, dtype=int)

    for block in blocks[1:]:
        patch = block["patch"]
        assert patch.shape == ref.shape
        diffs = [np.argwhere(patch != np.rot90(ref, k)) for k in range(4)]
        valid = [k for k, cells in enumerate(diffs) if len(cells) == 1]
        assert len(valid) == 1
        k = int(valid[0])
        r, c = map(int, diffs[k][0])
        rr, cc = _inverse_rot_coord(r, c, n, k)
        assert out[rr, cc] == bg
        out[rr, cc] = int(ref[rr, cc])

    return out


def _rand_comp(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    pieces = []
    prev = 0
    for cut in cuts + [total]:
        pieces.append(int(cut - prev))
        prev = cut
    return pieces


def _paint_blocky_pattern(n: int, colors: list[int]) -> np.ndarray:
    grid = np.empty((n, n), dtype=int)
    row_sizes = _rand_comp(n, random.randint(2, min(4, n)))
    col_sizes = _rand_comp(n, random.randint(2, min(4, n)))

    r0 = 0
    for rh in row_sizes:
        c0 = 0
        for cw in col_sizes:
            grid[r0 : r0 + rh, c0 : c0 + cw] = random.choice(colors)
            c0 += cw
        r0 += rh

    for _ in range(random.randint(3, 6)):
        color = random.choice(colors)
        kind = random.choice(("hband", "vband", "rect", "diag", "corner"))
        if kind == "hband":
            row = random.randrange(n)
            c0 = random.randrange(n)
            c1 = random.randrange(c0, n)
            grid[row, c0 : c1 + 1] = color
        elif kind == "vband":
            col = random.randrange(n)
            r0 = random.randrange(n)
            r1 = random.randrange(r0, n)
            grid[r0 : r1 + 1, col] = color
        elif kind == "rect":
            r0 = random.randrange(n)
            c0 = random.randrange(n)
            r1 = random.randrange(r0, n)
            c1 = random.randrange(c0, n)
            grid[r0 : r1 + 1, c0 : c1 + 1] = color
        elif kind == "diag":
            if random.choice((True, False)):
                offset = random.randrange(-n + 1, n)
                for r in range(n):
                    c = r + offset
                    if 0 <= c < n:
                        grid[r, c] = color
            else:
                total = random.randrange(0, 2 * n - 1)
                for r in range(n):
                    c = total - r
                    if 0 <= c < n:
                        grid[r, c] = color
        else:
            height = random.randint(1, max(1, n // 2))
            width = random.randint(1, max(1, n // 2))
            corner = random.choice(("tl", "tr", "bl", "br"))
            if corner == "tl":
                grid[:height, :width] = color
            elif corner == "tr":
                grid[:height, n - width :] = color
            elif corner == "bl":
                grid[n - height :, :width] = color
            else:
                grid[n - height :, n - width :] = color

    return grid


def _good_reference(ref: np.ndarray) -> bool:
    if len(set(map(int, ref.ravel()))) < 4:
        return False
    for k in (1, 2, 3):
        if np.array_equal(ref, np.rot90(ref, k)):
            return False
    return True


def _make_reference(n: int) -> np.ndarray:
    for _ in range(PATTERN_ATTEMPTS):
        color_count = random.randint(4, min(7, len(PALETTE)))
        colors = random.sample(PALETTE, color_count)
        ref = _paint_blocky_pattern(n, colors)
        if _good_reference(ref):
            return ref
    raise RuntimeError("failed to generate a rotationally asymmetric reference")


def _make_copy(ref: np.ndarray, rotation: int, tampered_ref_coord: tuple[int, int]) -> np.ndarray:
    n = int(ref.shape[0])
    copy = np.rot90(ref, rotation).copy()
    rr, cc = map(int, tampered_ref_coord)
    cr, cc2 = _forward_rot_coord(rr, cc, n, rotation)
    correct = int(copy[cr, cc2])
    options = [color for color in PALETTE if color != correct]
    copy[cr, cc2] = random.choice(options)
    return copy


def _copy_is_unambiguous(ref: np.ndarray, copy: np.ndarray) -> bool:
    counts = [int(np.sum(copy != np.rot90(ref, k))) for k in range(4)]
    return counts.count(1) == 1


def _layout_dims(n: int) -> tuple[int, int, int, int, list[tuple[int, int]]] | None:
    for _ in range(LAYOUT_ATTEMPTS):
        col_gap = random.randint(1, 2)
        row_gap = random.randint(1, 3)
        copy_w = 2 * n + col_gap
        copy_h = 2 * n + row_gap

        min_width = 1 + n + 1 + copy_w + 1
        min_height = 1 + copy_h + 1
        if min_width > 30 or min_height > 30:
            continue

        width = random.randint(min_width, 30)
        height = random.randint(max(min_height, n + 1), 30)

        max_ref_left = width - n - 1 - copy_w - 1
        if max_ref_left < 1:
            continue
        ref_left = random.randint(1, max_ref_left)

        min_copy_left = ref_left + n + 1
        max_copy_left = width - copy_w - 1
        if min_copy_left > max_copy_left:
            continue
        copy_left = random.randint(min_copy_left, max_copy_left)

        copy_top = random.randint(1, height - copy_h - 1)
        ref_top = random.randint(0, height - n)
        positions = [
            (copy_top, copy_left),
            (copy_top, copy_left + n + col_gap),
            (copy_top + n + row_gap, copy_left),
            (copy_top + n + row_gap, copy_left + n + col_gap),
        ]
        return height, width, ref_top, ref_left, positions
    return None


def _render_grid(ref: np.ndarray, copies: list[np.ndarray], positions: list[tuple[int, int]], height: int, width: int, ref_top: int, ref_left: int) -> np.ndarray:
    n = int(ref.shape[0])
    grid = np.full((height, width), BG, dtype=int)
    grid[ref_top : ref_top + n, ref_left : ref_left + n] = ref
    for patch, (r0, c0) in zip(copies, positions):
        grid[r0 : r0 + n, c0 : c0 + n] = patch
    return grid


def _expected_output(ref: np.ndarray, tampered: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(ref.shape, BG, dtype=int)
    for r, c in tampered:
        out[r, c] = int(ref[r, c])
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(MIN_SIDE, MAX_SIDE)
    ref = _make_reference(n)

    tampered = random.sample([(r, c) for r in range(n) for c in range(n)], COPY_COUNT)
    rotations = list(range(4))
    random.shuffle(rotations)

    layout = _layout_dims(n)
    if layout is None:
        return None
    height, width, ref_top, ref_left, positions = layout

    random.shuffle(positions)
    copies = []
    for rotation, coord in zip(rotations, tampered):
        patch = _make_copy(ref, rotation, coord)
        if not _copy_is_unambiguous(ref, patch):
            return None
        copies.append(patch)

    grid = _render_grid(ref, copies, positions, height, width, ref_top, ref_left)
    expected = _expected_output(ref, tampered)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
