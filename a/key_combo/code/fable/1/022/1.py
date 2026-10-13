from __future__ import annotations

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _most_common(values: np.ndarray) -> int:
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _find_divider(grid: np.ndarray) -> tuple[str, int, int]:
    h, w = grid.shape
    row_hits: list[tuple[str, int, int]] = []
    col_hits: list[tuple[str, int, int]] = []

    for r in range(h):
        if np.all(grid[r, :] == grid[r, 0]):
            color = int(grid[r, 0])
            if int(np.count_nonzero(grid == color)) == w:
                row_hits.append(("row", r, color))

    for c in range(w):
        if np.all(grid[:, c] == grid[0, c]):
            color = int(grid[0, c])
            if int(np.count_nonzero(grid == color)) == h:
                col_hits.append(("col", c, color))

    hits = row_hits + col_hits
    if len(hits) != 1:
        raise ValueError("expected exactly one full divider line")
    return hits[0]


def _parse_panels(grid: np.ndarray) -> dict:
    orient, idx, divider_color = _find_divider(grid)
    if orient == "row":
        top = grid[:idx, :].copy()
        bottom = grid[idx + 1 :, :].copy()
        if top.size < bottom.size:
            seed_panel, mask_panel, seed_side = top, bottom, "top"
        else:
            seed_panel, mask_panel, seed_side = bottom, top, "bottom"
    else:
        left = grid[:, :idx].copy()
        right = grid[:, idx + 1 :].copy()
        if left.size < right.size:
            seed_panel, mask_panel, seed_side = left, right, "left"
        else:
            seed_panel, mask_panel, seed_side = right, left, "right"

    shared = set(map(int, np.unique(seed_panel))) & set(map(int, np.unique(mask_panel)))
    if len(shared) == 1:
        bg = int(next(iter(shared)))
    else:
        bg = _most_common(np.concatenate([seed_panel.ravel(), mask_panel.ravel()]))

    seed_colors = sorted(set(map(int, np.unique(seed_panel))) - {bg})
    mask_colors = sorted(set(map(int, np.unique(mask_panel))) - {bg})
    if len(seed_colors) != 1 or len(mask_colors) != 1:
        raise ValueError("expected one seed color and one mask color")

    return {
        "divider_orient": orient,
        "divider_index": idx,
        "divider_color": int(divider_color),
        "seed_panel": seed_panel,
        "mask_panel": mask_panel,
        "seed_side": seed_side,
        "bg": int(bg),
        "seed_color": int(seed_colors[0]),
        "mask_color": int(mask_colors[0]),
    }


def _mask_symmetry(mask: np.ndarray) -> str:
    mirror_lr = np.array_equal(mask, np.fliplr(mask))
    mirror_ud = np.array_equal(mask, np.flipud(mask))
    point = np.array_equal(mask, np.rot90(mask, 2))

    if point and not mirror_lr and not mirror_ud:
        return "point"
    if mirror_lr and not mirror_ud and not point:
        return "mirror_lr"
    if mirror_ud and not mirror_lr and not point:
        return "mirror_ud"
    raise ValueError("mask must have exactly one supported symmetry type")


def _validate_no_self_pair(mask: np.ndarray, symmetry: str) -> None:
    h, w = mask.shape
    if symmetry == "mirror_lr" and (w % 2 == 1) and bool(mask[:, w // 2].any()):
        raise ValueError("mirror-lr mask cannot occupy the symmetry axis")
    if symmetry == "mirror_ud" and (h % 2 == 1) and bool(mask[h // 2, :].any()):
        raise ValueError("mirror-ud mask cannot occupy the symmetry axis")
    if symmetry == "point" and (h % 2 == 1) and (w % 2 == 1) and bool(mask[h // 2, w // 2]):
        raise ValueError("point-symmetric mask cannot occupy the center cell")


def _patch_for_cell(
    seed_panel: np.ndarray,
    symmetry: str,
    seed_side: str,
    cell_r: int,
    cell_c: int,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    mh, mw = mask_shape
    if symmetry == "mirror_lr":
        original = seed_side == "right" and cell_c >= mw // 2
        original = original or (seed_side != "right" and cell_c < mw // 2)
        return seed_panel if original else np.fliplr(seed_panel)
    if symmetry == "mirror_ud":
        original = seed_side == "bottom" and cell_r >= mh // 2
        original = original or (seed_side != "bottom" and cell_r < mh // 2)
        return seed_panel if original else np.flipud(seed_panel)

    if seed_side in {"left", "right"}:
        original = seed_side == "right" and cell_c >= mw // 2
        original = original or (seed_side != "right" and cell_c < mw // 2)
    else:
        original = seed_side == "bottom" and cell_r >= mh // 2
        original = original or (seed_side != "bottom" and cell_r < mh // 2)
    return seed_panel if original else np.rot90(seed_panel, 2)


def _expand(seed_panel: np.ndarray, mask: np.ndarray, bg: int, symmetry: str, seed_side: str) -> np.ndarray:
    sh, sw = seed_panel.shape
    mh, mw = mask.shape
    out = np.full((sh * mh, sw * mw), int(bg), dtype=int)
    for r in range(mh):
        for c in range(mw):
            if not mask[r, c]:
                continue
            patch = _patch_for_cell(seed_panel, symmetry, seed_side, r, c, mask.shape)
            _paste(out, patch, r * sh, c * sw)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a symmetric placement mask into seed-sized blocks while orienting each stamp to preserve the mask symmetry.

    Input structure:
    - A single-color full row or full column divider splits the grid into a small seed panel and a larger mask panel.
    - The small panel is a monochrome asymmetric seed sprite drawn inside its own rectangular patch with background cells included.
    - The large panel is a monochrome placement mask on the same background; it is either mirror-symmetric across the axis parallel to the divider or point-symmetric.

    Rule:
    - Remove the divider and discard the seed panel. Each cell of the large mask becomes one output block whose size matches the seed panel.
    - Background mask cells map to all-background blocks.
    - Foreground mask cells receive seed copies: the half of the mask nearest the seed keeps the shown seed orientation, while the symmetric partner receives the matching mirror copy or 180-degree rotation so that the full expanded output has the same symmetry type as the mask.

    Key insight:
    - The copies cannot all share one orientation: first identify whether the mask symmetry is mirror or point symmetry, then orient each paired stamp so the enlarged picture inherits that same global symmetry.
    """
    g = np.asarray(grid, dtype=int)
    parsed = _parse_panels(g)
    seed_panel = parsed["seed_panel"]
    mask_panel = parsed["mask_panel"]
    bg = int(parsed["bg"])

    seed_mask = seed_panel != bg
    if not seed_mask.any() or seed_mask.all():
        raise ValueError("seed panel must contain both background and foreground")

    mask = mask_panel != bg
    symmetry = _mask_symmetry(mask)
    _validate_no_self_pair(mask, symmetry)
    return _expand(seed_panel, mask, bg, symmetry, parsed["seed_side"])


def _seed_ok(mask: np.ndarray) -> bool:
    if not mask.any() or mask.all():
        return False
    if not _is_connected(mask):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.rot90(mask, 2)):
        return False
    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if r1 == r0 or c1 == c0:
        return False
    return int(mask.sum()) >= 3


def _random_seed_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    for _ in range(400):
        mask = np.zeros((h, w), dtype=bool)
        area = int(rng.integers(3, h * w))
        picks = rng.choice(h * w, size=area, replace=False)
        mask.flat[picks] = True
        if _seed_ok(mask):
            return mask
    raise RuntimeError("failed to generate asymmetric seed")


def _mirror_mask_ok(mask: np.ndarray, axis: str) -> bool:
    if not mask.any() or mask.all():
        return False
    if not _is_connected(mask):
        return False
    if axis == "lr":
        if not np.array_equal(mask, np.fliplr(mask)):
            return False
        if np.array_equal(mask, np.flipud(mask)):
            return False
    else:
        if not np.array_equal(mask, np.flipud(mask)):
            return False
        if np.array_equal(mask, np.fliplr(mask)):
            return False
    return not np.array_equal(mask, np.rot90(mask, 2))


def _point_mask_ok(mask: np.ndarray) -> bool:
    if not mask.any() or mask.all():
        return False
    if not _is_connected(mask):
        return False
    if not np.array_equal(mask, np.rot90(mask, 2)):
        return False
    if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
        return False
    return int(mask.sum()) >= 4


def _random_mirror_mask(rng: np.random.Generator, h: int, w: int, axis: str) -> np.ndarray:
    if axis == "lr":
        half_shape = (h, w // 2)
    else:
        half_shape = (h // 2, w)

    for _ in range(500):
        half = np.zeros(half_shape, dtype=bool)
        area = int(rng.integers(1, half.size + 1))
        picks = rng.choice(half.size, size=area, replace=False)
        half.flat[picks] = True
        if axis == "lr":
            mask = np.concatenate([half, np.fliplr(half)], axis=1)
        else:
            mask = np.concatenate([half, np.flipud(half)], axis=0)
        if _mirror_mask_ok(mask, axis):
            return mask
    raise RuntimeError("failed to generate mirror-symmetric mask")


def _random_point_mask(
    rng: np.random.Generator,
    h: int,
    w: int,
    orientation: str,
) -> np.ndarray:
    if orientation == "vertical":
        reps_shape = (h, w // 2)
        reps_to_full = lambda base: np.concatenate([base, np.rot90(base, 2)], axis=1)
    else:
        reps_shape = (h // 2, w)
        reps_to_full = lambda base: np.concatenate([base, np.rot90(base, 2)], axis=0)

    for _ in range(500):
        base = np.zeros(reps_shape, dtype=bool)
        area = int(rng.integers(2, reps_shape[0] * reps_shape[1] + 1))
        picks = rng.choice(reps_shape[0] * reps_shape[1], size=area, replace=False)
        base.flat[picks] = True
        mask = reps_to_full(base)
        if _point_mask_ok(mask):
            return mask
    raise RuntimeError("failed to generate point-symmetric mask")


def _paint_mask(shape: tuple[int, int], fg_mask: np.ndarray, bg: int, color: int) -> np.ndarray:
    panel = np.full(shape, int(bg), dtype=int)
    panel[fg_mask] = int(color)
    return panel


def _block_grid(output: np.ndarray, block_shape: tuple[int, int]) -> np.ndarray:
    bh, bw = block_shape
    oh, ow = output.shape
    assert oh % bh == 0 and ow % bw == 0
    return output.reshape(oh // bh, bh, ow // bw, bw).transpose(0, 2, 1, 3)


def _semantic_check(inp: np.ndarray) -> bool:
    try:
        parsed = _parse_panels(inp)
        out = main(inp)
    except Exception:
        return False

    seed_panel = parsed["seed_panel"]
    mask_panel = parsed["mask_panel"]
    bg = int(parsed["bg"])
    seed_mask = seed_panel != bg
    if not _seed_ok(seed_mask):
        return False

    mask = mask_panel != bg
    try:
        symmetry = _mask_symmetry(mask)
        _validate_no_self_pair(mask, symmetry)
    except Exception:
        return False

    sh, sw = seed_panel.shape
    mh, mw = mask.shape
    if out.shape != (sh * mh, sw * mw):
        return False

    blocks = _block_grid(out, seed_panel.shape)
    for r in range(mh):
        for c in range(mw):
            block = blocks[r, c]
            if not mask[r, c]:
                if not np.all(block == bg):
                    return False
                continue
            want = _patch_for_cell(seed_panel, symmetry, parsed["seed_side"], r, c, mask.shape)
            if not np.array_equal(block, want):
                return False

    if symmetry == "mirror_lr" and not np.array_equal(out, np.fliplr(out)):
        return False
    if symmetry == "mirror_ud" and not np.array_equal(out, np.flipud(out)):
        return False
    if symmetry == "point" and not np.array_equal(out, np.rot90(out, 2)):
        return False
    return True


def _make_candidate(rng: np.random.Generator) -> np.ndarray:
    layout = "vertical" if int(rng.integers(0, 2)) == 0 else "horizontal"
    symmetry_kind = "mirror" if int(rng.integers(0, 2)) == 0 else "point"
    colors = rng.choice(10, size=4, replace=False)
    bg, divider_color, seed_color, mask_color = map(int, colors)

    if layout == "vertical":
        shared_h = int(rng.integers(3, 6))
        seed_w = int(rng.integers(2, 5))
        max_mask_w = min(8, 30 // seed_w)
        mask_w_choices = [w for w in range(max(seed_w + 1, 4), max_mask_w + 1) if w % 2 == 0]
        if not mask_w_choices:
            raise RuntimeError("no valid mask width")
        mask_w = int(mask_w_choices[int(rng.integers(0, len(mask_w_choices)))])
        seed_mask = _random_seed_mask(rng, shared_h, seed_w)
        if symmetry_kind == "mirror":
            mask = _random_mirror_mask(rng, shared_h, mask_w, "lr")
        else:
            mask = _random_point_mask(rng, shared_h, mask_w, "vertical")
        seed_panel = _paint_mask((shared_h, seed_w), seed_mask, bg, seed_color)
        mask_panel = _paint_mask((shared_h, mask_w), mask, bg, mask_color)
        if int(rng.integers(0, 2)) == 0:
            return np.concatenate(
                [seed_panel, np.full((shared_h, 1), divider_color, dtype=int), mask_panel],
                axis=1,
            )
        return np.concatenate(
            [mask_panel, np.full((shared_h, 1), divider_color, dtype=int), seed_panel],
            axis=1,
        )

    shared_w = int(rng.integers(3, 6))
    seed_h = int(rng.integers(2, 5))
    max_mask_h = min(8, 30 // seed_h)
    mask_h_choices = [h for h in range(max(seed_h + 1, 4), max_mask_h + 1) if h % 2 == 0]
    if not mask_h_choices:
        raise RuntimeError("no valid mask height")
    mask_h = int(mask_h_choices[int(rng.integers(0, len(mask_h_choices)))])
    seed_mask = _random_seed_mask(rng, seed_h, shared_w)
    if symmetry_kind == "mirror":
        mask = _random_mirror_mask(rng, mask_h, shared_w, "ud")
    else:
        mask = _random_point_mask(rng, mask_h, shared_w, "horizontal")
    seed_panel = _paint_mask((seed_h, shared_w), seed_mask, bg, seed_color)
    mask_panel = _paint_mask((mask_h, shared_w), mask, bg, mask_color)
    if int(rng.integers(0, 2)) == 0:
        return np.concatenate(
            [seed_panel, np.full((1, shared_w), divider_color, dtype=int), mask_panel],
            axis=0,
        )
    return np.concatenate(
        [mask_panel, np.full((1, shared_w), divider_color, dtype=int), seed_panel],
        axis=0,
    )


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        try:
            grid = _make_candidate(rng)
        except RuntimeError:
            continue
        if max(grid.shape) > 30:
            continue
        if _semantic_check(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
