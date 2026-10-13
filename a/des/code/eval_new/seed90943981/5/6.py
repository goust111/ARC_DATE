from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

INPUT_COLORS = np.array([BLUE, RED, GREEN, MAGENTA, ORANGE, AZURE, MAROON], dtype=int)

GEN_RETRIES = 300
FRAME_PATTERN_RETRIES = 200
LAYOUT_RETRIES = 80
OUTPUT_GAP = 1
INPUT_GAP_RANGE = (2, 5)
INPUT_PAD_RANGE = (0, 3)
MAX_SIZE = 30


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _mix_colors(a: int, b: int) -> int:
    if a == BLACK:
        return int(b)
    if b == BLACK:
        return int(a)
    if a == b:
        return int(a)
    return YELLOW


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _side_counts(interior: np.ndarray) -> tuple[int, int, int, int]:
    h, w = interior.shape
    left = int(np.count_nonzero(interior[:, : w // 2] != BLACK))
    right = int(np.count_nonzero(interior[:, w // 2 :] != BLACK))
    top = int(np.count_nonzero(interior[: h // 2, :] != BLACK))
    bottom = int(np.count_nonzero(interior[h // 2 :, :] != BLACK))
    return left, right, top, bottom


def _infer_axis(interior: np.ndarray) -> str:
    h, w = interior.shape
    left, right, top, bottom = _side_counts(interior)
    if w % 2 == 0 and h % 2 == 1:
        return "vertical"
    if h % 2 == 0 and w % 2 == 1:
        return "horizontal"
    if h % 2 == 0 and w % 2 == 0:
        diff_v = abs(left - right)
        diff_h = abs(top - bottom)
        if diff_v > diff_h:
            return "vertical"
        if diff_h > diff_v:
            return "horizontal"
        return "vertical"
    raise AssertionError("frame interior must be foldable along at least one axis")


def _fold_interior(interior: np.ndarray, axis: str) -> np.ndarray:
    h, w = interior.shape
    if axis == "vertical":
        half = w // 2
        left = interior[:, :half]
        right = interior[:, half:]
        left_count = int(np.count_nonzero(left != BLACK))
        right_count = int(np.count_nonzero(right != BLACK))
        if left_count >= right_count:
            base = left
            folded = np.fliplr(right)
        else:
            base = right
            folded = np.fliplr(left)
    else:
        half = h // 2
        top = interior[:half, :]
        bottom = interior[half:, :]
        top_count = int(np.count_nonzero(top != BLACK))
        bottom_count = int(np.count_nonzero(bottom != BLACK))
        if top_count >= bottom_count:
            base = top
            folded = np.flipud(bottom)
        else:
            base = bottom
            folded = np.flipud(top)

    out = np.zeros_like(base)
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            out[r, c] = _mix_colors(int(base[r, c]), int(folded[r, c]))
    return out


def _fold_frame(interior: np.ndarray) -> np.ndarray:
    folded = _fold_interior(interior, _infer_axis(interior))
    out = np.full((folded.shape[0] + 2, folded.shape[1] + 2), BLACK, dtype=int)
    out[0, :] = GREY
    out[-1, :] = GREY
    out[:, 0] = GREY
    out[:, -1] = GREY
    out[1:-1, 1:-1] = folded
    return out


def _extract_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _components(grid == GREY):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if len(comp) != int(border.sum()):
            continue
        if not bool(np.all(patch[border] == GREY)):
            continue
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "interior": patch[1:-1, 1:-1].copy(),
            }
        )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each grey rectangular frame along its inferred axis and merge overlapping colors.

    Input structure:
    - The background is black.
    - Several grey 1-pixel rectangular frames appear on the grid; each frame encloses a black interior with scattered colored pixels.
    - Every frame interior has at least one foldable even dimension, and the colored pixels are distributed asymmetrically across the intended fold axis.

    Rule:
    - Detect each grey rectangular frame and inspect only its interior.
    - Infer the fold axis per frame: even-width/odd-height interiors fold vertically, odd-width/even-height interiors fold horizontally, and even-by-even interiors use the axis whose two sides have the larger non-black count difference.
    - Keep the more populated side fixed, mirror the less populated side onto it, and combine paired pixels with this rule: empty+color keeps the color, same+same keeps that color, different colors become yellow.
    - Rebuild each folded result with a grey border, then sort frames by input reading order and pack the folded frames left-to-right with one black column between them.

    Key insight:
    - The solver must first infer which axis is valid for each frame from parity and asymmetry, then apply a directed fold where side population decides which half stays in place, and finally handle the nontrivial color-mixing outcome when two different colors collide.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    if not frames:
        return g.copy()

    patches = [_fold_frame(np.asarray(frame["interior"], dtype=int)) for frame in frames]
    out_h = max(p.shape[0] for p in patches)
    out_w = sum(p.shape[1] for p in patches) + OUTPUT_GAP * (len(patches) - 1)
    out = np.full((out_h, out_w), BLACK, dtype=int)

    col = 0
    for patch in patches:
        _paste(out, patch, 0, col)
        col += patch.shape[1] + OUTPUT_GAP
    return out


def _sample_dims(mode: str, rng: np.random.Generator) -> tuple[int, int]:
    odd = np.array([3, 5, 7], dtype=int)
    even = np.array([4, 6, 8], dtype=int)
    if mode == "v_parity":
        return int(rng.choice(odd)), int(rng.choice(even))
    if mode == "h_parity":
        return int(rng.choice(even)), int(rng.choice(odd))
    if mode in {"both_even_v", "both_even_h"}:
        return int(rng.choice(even)), int(rng.choice(even))
    raise AssertionError("unknown mode")


def _assemble_interior(
    base_half: np.ndarray,
    other_half: np.ndarray,
    axis: str,
    base_side: str,
    full_shape: tuple[int, int],
) -> np.ndarray:
    h, w = full_shape
    out = np.zeros((h, w), dtype=int)
    if axis == "vertical":
        half = w // 2
        if base_side == "left":
            out[:, :half] = base_half
            out[:, half:] = np.fliplr(other_half)
        else:
            out[:, :half] = np.fliplr(other_half)
            out[:, half:] = base_half
    else:
        half = h // 2
        if base_side == "top":
            out[:half, :] = base_half
            out[half:, :] = np.flipud(other_half)
        else:
            out[:half, :] = np.flipud(other_half)
            out[half:, :] = base_half
    return out


def _make_interior(
    mode: str,
    rng: np.random.Generator,
    *,
    force_diff_overlap: bool,
    force_same_overlap: bool,
) -> np.ndarray | None:
    axis = "vertical" if mode in {"v_parity", "both_even_v"} else "horizontal"
    for _ in range(FRAME_PATTERN_RETRIES):
        h, w = _sample_dims(mode, rng)
        pair_shape = (h, w // 2) if axis == "vertical" else (h // 2, w)
        pair_area = int(pair_shape[0] * pair_shape[1])
        if pair_area < 4:
            continue

        max_overlap = 2 if pair_area >= 10 else 1
        n_both_diff = 1 if force_diff_overlap else int(rng.integers(0, max_overlap + 1))
        max_same_room = pair_area - n_both_diff
        if max_same_room <= 0:
            continue
        n_both_same = 1 if force_same_overlap else int(rng.integers(0, min(2, max_same_room) + 1))
        overlap = n_both_diff + n_both_same
        if overlap >= pair_area:
            continue

        max_other_only = min(2, pair_area - overlap - 1)
        if max_other_only < 0:
            continue
        n_other_only = int(rng.integers(0, max_other_only + 1))
        if overlap + n_other_only == 0:
            n_other_only = 1

        min_margin = 2 if mode in {"both_even_v", "both_even_h"} else 1
        min_base_only = n_other_only + min_margin
        max_base_only = min(pair_area - overlap - n_other_only, n_other_only + 4)
        if max_base_only < min_base_only:
            continue
        n_base_only = int(rng.integers(min_base_only, max_base_only + 1))

        total_used = overlap + n_other_only + n_base_only
        if total_used > pair_area:
            continue

        positions = [(r, c) for r in range(pair_shape[0]) for c in range(pair_shape[1])]
        rng.shuffle(positions)

        diff_pos = positions[:n_both_diff]
        same_pos = positions[n_both_diff : n_both_diff + n_both_same]
        other_only_pos = positions[n_both_diff + n_both_same : n_both_diff + n_both_same + n_other_only]
        base_only_pos = positions[
            n_both_diff + n_both_same + n_other_only : n_both_diff + n_both_same + n_other_only + n_base_only
        ]

        palette_size = int(rng.integers(2, 5))
        palette = np.asarray(rng.choice(INPUT_COLORS, size=palette_size, replace=False), dtype=int)
        base = np.zeros(pair_shape, dtype=int)
        other = np.zeros(pair_shape, dtype=int)

        for r, c in base_only_pos:
            base[r, c] = int(rng.choice(palette))
        for r, c in other_only_pos:
            other[r, c] = int(rng.choice(palette))
        for r, c in same_pos:
            col = int(rng.choice(palette))
            base[r, c] = col
            other[r, c] = col
        for r, c in diff_pos:
            cols = rng.choice(palette, size=2, replace=False)
            base[r, c] = int(cols[0])
            other[r, c] = int(cols[1])

        if axis == "vertical":
            base_side = str(rng.choice(np.array(["left", "right"], dtype=object)))
        else:
            base_side = str(rng.choice(np.array(["top", "bottom"], dtype=object)))
        interior = _assemble_interior(base, other, axis, base_side, (h, w))

        observed_axis = _infer_axis(interior)
        if axis == "vertical" and observed_axis != "vertical":
            continue
        if axis == "horizontal" and observed_axis != "horizontal":
            continue

        left, right, top, bottom = _side_counts(interior)
        if axis == "vertical" and left == right:
            continue
        if axis == "horizontal" and top == bottom:
            continue
        if mode == "both_even_v" and abs(left - right) <= abs(top - bottom):
            continue
        if mode == "both_even_h" and abs(top - bottom) <= abs(left - right):
            continue

        folded = _fold_interior(interior, axis)
        if force_diff_overlap and not bool(np.any(folded == YELLOW)):
            continue
        if force_same_overlap:
            overlap_same_seen = False
            for r in range(base.shape[0]):
                for c in range(base.shape[1]):
                    if int(base[r, c]) != BLACK and int(base[r, c]) == int(other[r, c]):
                        overlap_same_seen = True
                        break
                if overlap_same_seen:
                    break
            if not overlap_same_seen:
                continue
        if int(np.count_nonzero(interior != BLACK)) < 4:
            continue
        return interior
    return None


def _frame_patch(interior: np.ndarray) -> np.ndarray:
    patch = np.full((interior.shape[0] + 2, interior.shape[1] + 2), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY
    patch[1:-1, 1:-1] = interior
    return patch


def _build_input_grid(patches: list[np.ndarray], rng: np.random.Generator) -> np.ndarray | None:
    order = [patches[i] for i in rng.permutation(len(patches))]

    groupings: list[list[list[np.ndarray]]] = []
    one_row = [order]
    groupings.append(one_row)
    groupings.append([order[:1], order[1:]])
    groupings.append([order[:2], order[2:]])
    rng.shuffle(groupings)

    for groups in groupings:
        for _ in range(LAYOUT_RETRIES):
            row_widths: list[int] = []
            row_heights: list[int] = []
            row_gaps: list[list[int]] = []
            for group in groups:
                if not group:
                    break
                gaps = [int(rng.integers(*INPUT_GAP_RANGE)) for _ in range(max(0, len(group) - 1))]
                width = sum(p.shape[1] for p in group) + sum(gaps)
                height = max(p.shape[0] for p in group)
                row_widths.append(int(width))
                row_heights.append(int(height))
                row_gaps.append(gaps)
            else:
                pad_top = int(rng.integers(*INPUT_PAD_RANGE))
                pad_bottom = int(rng.integers(*INPUT_PAD_RANGE))
                pad_left = int(rng.integers(*INPUT_PAD_RANGE))
                pad_right = int(rng.integers(*INPUT_PAD_RANGE))
                row_gap = int(rng.integers(*INPUT_GAP_RANGE)) if len(groups) == 2 else 0
                extra_width = int(rng.integers(0, 3))

                total_h = pad_top + sum(row_heights) + row_gap + pad_bottom
                total_w = pad_left + max(row_widths) + pad_right + extra_width
                if total_h > MAX_SIZE or total_w > MAX_SIZE:
                    continue

                canvas = np.full((total_h, total_w), BLACK, dtype=int)
                row_r = pad_top
                for group, width, height, gaps in zip(groups, row_widths, row_heights, row_gaps, strict=False):
                    slack = total_w - pad_left - pad_right - width
                    row_c = pad_left + int(rng.integers(0, slack + 1))
                    cursor = row_c
                    for idx, patch in enumerate(group):
                        _paste(canvas, patch, row_r, cursor)
                        cursor += patch.shape[1]
                        if idx < len(gaps):
                            cursor += gaps[idx]
                    row_r += height + row_gap
                    row_gap = 0
                return canvas
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    both_even_mode = str(rng.choice(np.array(["both_even_v", "both_even_h"], dtype=object)))
    modes = ["v_parity", "h_parity", both_even_mode]

    interiors: list[np.ndarray] = []
    for idx, mode in enumerate(modes):
        interior = _make_interior(
            mode,
            rng,
            force_diff_overlap=(idx == 0),
            force_same_overlap=(idx == 1),
        )
        if interior is None:
            return None
        interiors.append(interior)

    patches = [_frame_patch(interior) for interior in interiors]
    grid = _build_input_grid(patches, rng)
    if grid is None:
        return None

    out = main(grid)
    if grid.ndim != 2 or out.ndim != 2:
        return None
    if max(grid.shape) > MAX_SIZE or max(out.shape) > MAX_SIZE:
        return None
    if not bool(np.any(out == YELLOW)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid folding-frame input")
