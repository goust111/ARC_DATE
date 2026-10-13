from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

import numpy as np


def _load_arc_utils() -> Any:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "new" / "solvers" / "arc_utils.py"
        if candidate.is_file():
            return SourceFileLoader("arc_utils_dynamic", str(candidate)).load_module()
    raise ImportError("failed to load arc_utils.py from any parent directory")


_arc = _load_arc_utils()
get_bg = _arc.get_bg
bbox = _arc.bbox
paste = _arc.paste


def _iter_blockings(size: int) -> list[int]:
    return [k for k in range(1, size + 1) if size % k == 0]


def _infer_missing_color_from_period(
    colors_row_major: list[int | None], cols_per_row: int
) -> tuple[int, int] | None:
    n = len(colors_row_major)
    try:
        missing_idx = colors_row_major.index(None)
    except ValueError:
        return None

    best: tuple[int, int, int, int] | None = None
    for period in range(1, n):
        residue_color: list[int | None] = [None] * period
        ok = True
        for idx, color in enumerate(colors_row_major):
            if color is None:
                continue
            r = idx % period
            prev = residue_color[r]
            if prev is None:
                residue_color[r] = color
            elif prev != color:
                ok = False
                break

        if not ok:
            continue

        residue = missing_idx % period
        missing_color = residue_color[residue]
        if missing_color is None:
            continue

        same_residue_known = sum(
            1 for i, color in enumerate(colors_row_major) if color is not None and i % period == residue
        )
        if same_residue_known < 2:
            continue

        score = (1 if cols_per_row % period != 0 else 0, -period, period, int(missing_color))
        if best is None or score > best:
            best = score

    if best is None:
        return None

    return best[2], best[3]


def _analyze_grid_layout(g: np.ndarray) -> tuple[int, int, int, int, np.ndarray, int, int]:
    assert g.ndim == 2
    h, w = map(int, g.shape)
    bg = int(get_bg(g))
    assert bbox(g != bg) is not None, "expected non-background content"

    best: tuple[tuple[int, int, int], int, int, int, np.ndarray, int, int] | None = None

    for bh in _iter_blockings(h):
        for bw in _iter_blockings(w):
            rows = h // bh
            cols = w // bw
            if rows * cols < 2:
                continue

            seq: list[int | None] = []
            template: np.ndarray | None = None
            missing_idx = -1
            intact_blocks = 0
            valid = True

            for bi in range(rows * cols):
                r = bi // cols
                c = bi % cols
                block = g[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw]
                block_colors = np.setdiff1d(np.unique(block), [bg])

                if block_colors.size == 0:
                    if missing_idx != -1:
                        valid = False
                        break
                    missing_idx = bi
                    seq.append(None)
                    continue

                if block_colors.size != 1:
                    valid = False
                    break

                color = int(block_colors[0])
                mask = block != bg
                if not np.any(mask):
                    valid = False
                    break

                if template is None:
                    template = mask.copy()
                    if template.shape != (bh, bw):
                        valid = False
                        break
                    if template.all() or not template.any():
                        valid = False
                        break
                elif not np.array_equal(template, mask):
                    valid = False
                    break

                seq.append(color)
                intact_blocks += 1

            if not valid or missing_idx == -1 or template is None:
                continue

            inferred = _infer_missing_color_from_period(seq, cols)
            if inferred is None:
                continue
            period, missing_color = inferred

            score = (
                1 if cols % period != 0 else 0,
                -period,
                template.sum(),
                intact_blocks,
            )
            if best is None or score > best[0]:
                best = (score, bh, bw, cols, template, missing_idx, missing_color)

    assert best is not None, "cannot infer a unique block tiling"
    _, bh, bw, cols, template, missing_idx, missing_color = best
    rows = h // bh
    return bh, bw, rows, cols, template.astype(bool), missing_idx, missing_color


def main(grid: np.ndarray) -> np.ndarray:
    """Reconstruct one erased block using the shared template and row-major periodic color sequence.

    Input structure:
    - The grid is tiled by equal-sized rectangular blocks.
    - Every non-erased block is one colorized copy of one binary template (template on a background).
    - Exactly one block is fully erased to the global background color.
    - Block paint colors in row-major order follow a 1D periodic sequence.

    Rule:
    - Read the shared template from any intact block and locate the block whose pixels are all background.
    - Infer the period of the row-major paint-color sequence from intact blocks and recover the missing block’s color.
    - Repaint the missing block with the shared template recolored with the inferred color.

    Key insight:
    - The spatial layout is a 2D wrapping of a 1D periodic color string; the shared template gives shape
      and the period gives the unique missing color in that string.
    """
    g = np.asarray(grid, dtype=int)
    bh, bw, rows, cols, template, missing_idx, missing_color = _analyze_grid_layout(g)
    out = g.copy()

    missing_r, missing_c = divmod(missing_idx, cols)
    patch = np.full((bh, bw), int(get_bg(g)), dtype=int)
    patch[template] = missing_color
    paste(out, patch, missing_r * bh, missing_c * bw)

    assert 0 <= rows <= out.shape[0] // bh
    return out


# ---------------- Generator ----------------

BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def _random_template_mask(rng: np.random.Generator, bh: int, bw: int) -> np.ndarray:
    style = rng.integers(3)
    mask = np.zeros((bh, bw), dtype=bool)

    if style == 0:
        # Random walk, connected stroke.
        length = int(max(2, rng.integers(2, bh * bw)))
        r = int(rng.integers(bh))
        c = int(rng.integers(bw))
        mask[r, c] = True
        filled = 1
        for _ in range(length - 1):
            candidates = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < bh and 0 <= cc < bw and not mask[rr, cc]:
                    candidates.append((rr, cc))
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            mask[r, c] = True
            filled += 1
            if filled >= length:
                break
    elif style == 1:
        # Two-axis bar-cross motif.
        r = int(rng.integers(bh))
        c = int(rng.integers(bw))
        mask[r, :] = True
        mask[:, c] = True
    else:
        # Blocky blobs plus one corner anchor.
        r0 = int(rng.integers(max(1, bh // 2), bh + 1))
        c0 = int(rng.integers(max(1, bw // 2), bw + 1))
        rr0 = int(rng.integers(0, bh - r0 + 1))
        cc0 = int(rng.integers(0, bw - c0 + 1))
        mask[rr0 : rr0 + r0, cc0 : cc0 + c0] = True
        rr1 = int(rng.integers(0, bh))
        cc1 = int(rng.integers(0, bw))
        mask[rr1, cc1] = True

    if not mask.any() or mask.all():
        rr, cc = int(rng.integers(bh)), int(rng.integers(bw))
        rr2, cc2 = int((rr + 1) % bh), int((cc + 1) % bw)
        mask[rr, cc] = True
        mask[rr2, cc2] = True
    return mask


def _build_candidate(rng: np.random.Generator) -> np.ndarray | None:
    block_h = int(rng.integers(2, 8))
    block_w = int(rng.integers(2, 8))
    rows = int(rng.integers(2, 7))
    cols = int(rng.integers(2, 7))
    h = rows * block_h
    w = cols * block_w
    if max(h, w) > 30:
        return None
    if min(rows, cols) < 2:
        return None

    count = rows * cols
    valid_periods = [p for p in range(2, count) if cols % p != 0]
    if not valid_periods:
        return None

    if rng.random() < 0.7:
        valid_periods = [p for p in valid_periods if p <= 5 or p == count - 1]
    period = int(rng.choice(valid_periods))

    # Ensure at least one residue class has multiple known samples so the missing block
    # is forced by periodicity.
    residue_counts = [0] * period
    for idx in range(count):
        residue_counts[idx % period] += 1

    if all(v == 1 for v in residue_counts):
        return None

    template = _random_template_mask(rng, block_h, block_w)

    # Sample the period sequence and randomly erase one slot.
    period_colors: list[int] = []
    while len(set(period_colors)) < 2:
        period_colors = [int(rng.choice(PALETTE)) for _ in range(period)]
    colors = [period_colors[i % period] for i in range(count)]
    missing_idx = int(rng.integers(count))
    missing_residue = missing_idx % period
    if residue_counts[missing_residue] == 1:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for bi in range(count):
        if bi == missing_idx:
            continue
        r = bi // cols
        c = bi % cols
        block_color = int(colors[bi])
        block = np.full((block_h, block_w), BG, dtype=int)
        block[template] = block_color
        paste(grid, block, r * block_h, c * block_w)

    try:
        out = main(grid)
    except Exception:
        return None

    # Avoid degenerate and duplicate-edge cases where main cannot be distinguished.
    if not np.array_equal(out, grid) and np.any(out != grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 4")
