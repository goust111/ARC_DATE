from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, bbox_h, bbox_w, colors_present, crop_bbox, get_bg


GRID_MAX = 30
GEN_ATTEMPTS = 400
PALETTE = list(range(10))


def _is_singleton_color(grid: np.ndarray, color: int) -> bool:
    coords = np.argwhere(grid == int(color))
    if len(coords) == 0:
        return False
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == int(color):
                return False
    return True


def _reconstruct_region(
    template: np.ndarray,
    region_box: tuple[int, int, int, int],
    anchor_box: tuple[int, int, int, int],
) -> np.ndarray:
    rr0, cc0, rr1, cc1 = region_box
    tr0, tc0, _tr1, _tc1 = anchor_box
    th, tw = template.shape
    rows = (np.arange(rr0, rr1 + 1) - int(tr0)) % int(th)
    cols = (np.arange(cc0, cc1 + 1) - int(tc0)) % int(tw)
    return template[np.ix_(rows, cols)]


def _guide_candidates(grid: np.ndarray, bg: int) -> list[int]:
    candidates = []
    for color in colors_present(grid, exclude={bg}):
        mask = grid == int(color)
        if int(mask.sum()) != 4:
            continue
        if not _is_singleton_color(grid, int(color)):
            continue
        box = bbox(mask)
        if box is None or bbox_h(box) < 3 or bbox_w(box) < 3:
            continue
        candidates.append(int(color))
    if candidates:
        return candidates
    return colors_present(grid, exclude={bg})


def _solve_with_guide(grid: np.ndarray, guide_color: int, bg: int):
    guide_mask = grid == int(guide_color)
    guide_box = bbox(guide_mask)
    if guide_box is None:
        return None

    g2 = np.array(grid, dtype=int).copy()
    g2[guide_mask] = int(bg)
    region_mask = g2 != int(bg)
    region_box = bbox(region_mask)
    if region_box is None:
        return None

    gh = bbox_h(guide_box)
    gw = bbox_w(guide_box)
    rh = bbox_h(region_box)
    rw = bbox_w(region_box)
    rr0, cc0, rr1, cc1 = region_box
    gr0, gc0, gr1, gc1 = guide_box

    if not (rr0 <= gr0 <= gr1 <= rr1 and cc0 <= gc0 <= gc1 <= cc1):
        return None
    if rh <= gh and rw <= gw:
        return None

    template = crop_bbox(g2, guide_box)
    if not np.any(template != int(bg)):
        return None

    # The sparse guide marks background-only positions of the designated unit cell.
    local_guides = np.argwhere(guide_mask[gr0 : gr1 + 1, gc0 : gc1 + 1])
    if len(local_guides) != 4:
        return None
    for r, c in local_guides:
        if int(template[int(r), int(c)]) != int(bg):
            return None

    recon = _reconstruct_region(template, region_box, guide_box)
    observed = crop_bbox(g2, region_box)
    mismatches = int(np.sum(recon != observed))

    # Wrong candidates typically imply a much worse periodic fit.
    penalty = mismatches
    if np.all(recon == int(bg)):
        penalty += 10_000

    out = np.array(grid, dtype=int).copy()
    out[guide_mask] = int(bg)
    out[rr0 : rr1 + 1, cc0 : cc1 + 1] = recon
    return penalty, out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Repair a corrupted periodic mosaic by retiling it from the guide-designated clean template cell.

    Input structure:
    - A background canvas contains one rectangular periodic pattern region built from repeated cell-sized template windows.
    - Most visible template instances have been wholly replaced by a structurally similar impostor version, and extra pixel noise corrupts scattered cells.
    - Four sparse guide-mask pixels of one color sit on background positions whose bounding box selects one complete clean true template window.

    Rule:
    - Identify the guide mask, erase those guide pixels to background, and take the guide bbox crop as the designated clean template window.
    - Find the full periodic region and use the designated window's global position as the phase anchor.
    - Retile the entire pattern region with that clean template, thereby removing impostor windows and pixel noise; leave everything outside the region unchanged except for erasing the guide pixels.

    Key insight:
    - Neither per-position voting nor choosing the most frequent window works, because impostor windows outnumber true ones; the only reliable template is the one uniquely designated by the sparse guide bbox.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    best = None
    for guide_color in _guide_candidates(g, bg):
        solved = _solve_with_guide(g, int(guide_color), bg)
        if solved is None:
            continue
        score, out = solved
        key = (int(score), int(guide_color))
        if best is None or key < best[0]:
            best = (key, out)

    assert best is not None, "failed to identify the designated guide template"
    return best[1]


def _has_proper_period(arr: np.ndarray) -> bool:
    h, w = arr.shape
    for ph in range(1, h):
        if h % ph == 0 and np.array_equal(arr, np.tile(arr[:ph, :], (h // ph, 1))):
            return True
    for pw in range(1, w):
        if w % pw == 0 and np.array_equal(arr, np.tile(arr[:, :pw], (1, w // pw))):
            return True
    return False


def _make_true_template(
    rng: random.Random, th: int, tw: int, bg: int, tile_colors: list[int]
) -> np.ndarray:
    for _ in range(300):
        tile = np.full((th, tw), int(bg), dtype=int)

        # Force non-bg touches on all four sides while keeping the corners empty for the guide mask.
        mandatory = {
            (0, rng.randrange(1, tw - 1)),
            (th - 1, rng.randrange(1, tw - 1)),
            (rng.randrange(1, th - 1), 0),
            (rng.randrange(1, th - 1), tw - 1),
        }
        cells = [
            (r, c)
            for r in range(th)
            for c in range(tw)
            if (r, c) not in {(0, 0), (0, tw - 1), (th - 1, 0), (th - 1, tw - 1)}
        ]
        rng.shuffle(cells)
        extra = rng.randint(max(4, (th * tw) // 4), max(5, (th * tw) // 2))
        chosen = set(mandatory)
        for cell in cells:
            if len(chosen) >= len(mandatory) + extra:
                break
            chosen.add(cell)

        used_colors = tile_colors[: rng.randint(2, min(4, len(tile_colors)))]
        rng.shuffle(used_colors)
        for idx, (r, c) in enumerate(sorted(chosen)):
            tile[r, c] = int(used_colors[idx % len(used_colors)])

        if len(set(int(v) for v in tile.flat if int(v) != int(bg))) < 2:
            continue
        if _has_proper_period(tile):
            continue
        return tile
    raise AssertionError("failed to generate a non-degenerate true template")


def _side_has_foreground(tile: np.ndarray, bg: int) -> bool:
    return (
        np.any(tile[0, :] != int(bg))
        and np.any(tile[-1, :] != int(bg))
        and np.any(tile[:, 0] != int(bg))
        and np.any(tile[:, -1] != int(bg))
    )


def _make_impostor_template(
    rng: random.Random, true_tile: np.ndarray, bg: int, tile_colors: list[int]
) -> np.ndarray:
    th, tw = true_tile.shape
    mutable = [
        (r, c)
        for r in range(th)
        for c in range(tw)
        if (r, c) not in {(0, 0), (0, tw - 1), (th - 1, 0), (th - 1, tw - 1)}
    ]
    palette = [int(bg)] + [int(c) for c in tile_colors]
    for _ in range(300):
        impostor = true_tile.copy()
        rng.shuffle(mutable)
        change_count = rng.randint(3, max(4, min(len(mutable), (th * tw) // 2)))
        changed = 0
        for r, c in mutable:
            old = int(impostor[r, c])
            choices = [col for col in palette if int(col) != old]
            new = int(rng.choice(choices))
            impostor[r, c] = new
            if int(new) != old:
                changed += 1
            if changed >= change_count:
                break

        if np.array_equal(impostor, true_tile):
            continue
        if np.sum(impostor != true_tile) < 3:
            continue
        if not _side_has_foreground(impostor, bg):
            continue
        if _has_proper_period(impostor):
            continue
        return impostor
    raise AssertionError("failed to generate a non-degenerate impostor template")


def _visible_full_indices(start: int, end: int, period: int) -> list[int]:
    return [
        idx
        for idx in range((end + period - 1) // period)
        if idx * period >= start and (idx + 1) * period <= end
    ]


def _tile_plane(patterns: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = crop_box
    return patterns[r0 : r1 + 1, c0 : c1 + 1].copy()


def _clean_expected(
    true_tile: np.ndarray,
    region_box: tuple[int, int, int, int],
    guide_box: tuple[int, int, int, int],
    canvas_shape: tuple[int, int],
    bg: int,
) -> np.ndarray:
    out = np.full(canvas_shape, int(bg), dtype=int)
    clean_region = _reconstruct_region(true_tile, region_box, guide_box)
    rr0, cc0, rr1, cc1 = region_box
    out[rr0 : rr1 + 1, cc0 : cc1 + 1] = clean_region
    return out


def _build_case(rng: random.Random):
    bg = rng.choice(PALETTE)
    guide_color = rng.choice([c for c in PALETTE if c != bg])
    usable = [c for c in PALETTE if c not in {bg, guide_color}]
    rng.shuffle(usable)

    th = rng.randint(4, 6)
    tw = rng.randint(4, 6)
    cells_r = rng.randint(4, 6)
    cells_c = rng.randint(4, 6)
    plane_h = cells_r * th
    plane_w = cells_c * tw

    top_trim = rng.randint(0, th - 1)
    left_trim = rng.randint(0, tw - 1)
    bottom_trim = rng.randint(0, th - 1)
    right_trim = rng.randint(0, tw - 1)
    crop_h = plane_h - top_trim - bottom_trim
    crop_w = plane_w - left_trim - right_trim
    if crop_h < th * 2 + 1 or crop_w < tw * 2 + 1:
        return None

    full_rows = _visible_full_indices(top_trim, plane_h - bottom_trim, th)
    full_cols = _visible_full_indices(left_trim, plane_w - right_trim, tw)
    if len(full_rows) < 2 or len(full_cols) < 2:
        return None

    tile_palette = usable[: rng.randint(3, min(6, len(usable)))]
    true_tile = _make_true_template(rng, th, tw, bg, tile_palette)
    if not _side_has_foreground(true_tile, bg):
        return None
    impostor_tile = _make_impostor_template(rng, true_tile, bg, tile_palette)

    visible_cells = []
    for ir in range(cells_r):
        for ic in range(cells_c):
            cell_r0 = ir * th
            cell_c0 = ic * tw
            cell_r1 = cell_r0 + th - 1
            cell_c1 = cell_c0 + tw - 1
            if cell_r1 < top_trim or cell_c1 < left_trim:
                continue
            if cell_r0 >= plane_h - bottom_trim or cell_c0 >= plane_w - right_trim:
                continue
            visible_cells.append((ir, ic))
    if len(visible_cells) < 6:
        return None

    selected_ir = rng.choice(full_rows)
    selected_ic = rng.choice(full_cols)
    selected_cell = (selected_ir, selected_ic)

    true_count = rng.randint(1, max(1, len(visible_cells) // 3))
    if true_count >= len(visible_cells) - true_count:
        true_count = max(1, (len(visible_cells) - 1) // 2)
    other_cells = [cell for cell in visible_cells if cell != selected_cell]
    rng.shuffle(other_cells)
    true_cells = {selected_cell, *other_cells[: max(0, true_count - 1)]}
    if len(visible_cells) - len(true_cells) <= len(true_cells):
        return None

    plane = np.full((plane_h, plane_w), int(bg), dtype=int)
    for ir in range(cells_r):
        for ic in range(cells_c):
            tile = true_tile if (ir, ic) in true_cells else impostor_tile
            r0 = ir * th
            c0 = ic * tw
            plane[r0 : r0 + th, c0 : c0 + tw] = tile

    clean_crop = _tile_plane(
        plane,
        (top_trim, left_trim, top_trim + crop_h - 1, left_trim + crop_w - 1),
    )
    if not (
        np.any(clean_crop[0, :] != int(bg))
        and np.any(clean_crop[-1, :] != int(bg))
        and np.any(clean_crop[:, 0] != int(bg))
        and np.any(clean_crop[:, -1] != int(bg))
    ):
        return None

    canvas_h = crop_h + rng.randint(2, 6)
    canvas_w = crop_w + rng.randint(2, 6)
    if max(canvas_h, canvas_w) > GRID_MAX:
        return None

    offset_r = rng.randint(1, canvas_h - crop_h - 1)
    offset_c = rng.randint(1, canvas_w - crop_w - 1)
    region_box = (offset_r, offset_c, offset_r + crop_h - 1, offset_c + crop_w - 1)

    input_grid = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    input_grid[offset_r : offset_r + crop_h, offset_c : offset_c + crop_w] = clean_crop

    # Replace most visible cells with the impostor, then add sparse noise.
    for ir, ic in visible_cells:
        tile = true_tile if (ir, ic) in true_cells else impostor_tile
        src_r0 = ir * th
        src_c0 = ic * tw
        dst_r0 = offset_r + src_r0 - top_trim
        dst_c0 = offset_c + src_c0 - left_trim
        rr0 = max(dst_r0, offset_r)
        cc0 = max(dst_c0, offset_c)
        rr1 = min(dst_r0 + th - 1, offset_r + crop_h - 1)
        cc1 = min(dst_c0 + tw - 1, offset_c + crop_w - 1)
        if rr0 > rr1 or cc0 > cc1:
            continue
        tile_r0 = rr0 - dst_r0
        tile_c0 = cc0 - dst_c0
        patch = tile[tile_r0 : tile_r0 + (rr1 - rr0 + 1), tile_c0 : tile_c0 + (cc1 - cc0 + 1)]
        input_grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = patch

    guide_r0 = offset_r + selected_ir * th - top_trim
    guide_c0 = offset_c + selected_ic * tw - left_trim
    guide_box = (guide_r0, guide_c0, guide_r0 + th - 1, guide_c0 + tw - 1)

    forbidden = {
        (guide_r0, guide_c0),
        (guide_r0, guide_c0 + tw - 1),
        (guide_r0 + th - 1, guide_c0),
        (guide_r0 + th - 1, guide_c0 + tw - 1),
    }

    noise_palette = [c for c in usable if c != guide_color]
    region_cells = [
        (r, c)
        for r in range(region_box[0], region_box[2] + 1)
        for c in range(region_box[1], region_box[3] + 1)
        if not (guide_r0 <= r <= guide_box[2] and guide_c0 <= c <= guide_box[3])
    ]
    rng.shuffle(region_cells)
    noise_count = rng.randint(max(1, crop_h * crop_w // 20), max(2, crop_h * crop_w // 10))
    for r, c in region_cells[:noise_count]:
        old = int(input_grid[r, c])
        choices = [col for col in noise_palette if int(col) != old]
        if choices:
            input_grid[r, c] = int(rng.choice(choices))

    for r, c in forbidden:
        if int(input_grid[r, c]) != int(bg):
            return None
        input_grid[r, c] = int(guide_color)

    expected = _clean_expected(true_tile, region_box, guide_box, input_grid.shape, bg)

    meta = {
        "bg": int(bg),
        "guide_color": int(guide_color),
        "region_box": region_box,
        "guide_box": guide_box,
        "true_tile": true_tile,
        "impostor_tile": impostor_tile,
        "visible_cells": visible_cells,
        "true_cells": true_cells,
    }
    return input_grid, expected, meta


def _semantic_checks(inp: np.ndarray, out: np.ndarray, expected: np.ndarray, meta: dict) -> None:
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= GRID_MAX and max(out.shape) <= GRID_MAX
    assert np.array_equal(out, expected), "main() output mismatches generator ground truth"

    rr0, cc0, rr1, cc1 = meta["region_box"]
    gr0, gc0, gr1, gc1 = meta["guide_box"]
    bg = int(meta["bg"])
    guide_color = int(meta["guide_color"])

    assert np.all(out[inp == guide_color] == bg), "guide pixels must be erased in the output"
    assert np.any(inp[rr0 : rr1 + 1, cc0 : cc1 + 1] != out[rr0 : rr1 + 1, cc0 : cc1 + 1]), "region should need repair"
    assert np.array_equal(
        out[gr0 : gr1 + 1, gc0 : gc1 + 1], meta["true_tile"]
    ), "designated template window should be recovered exactly"
    assert len(meta["visible_cells"]) - len(meta["true_cells"]) > len(meta["true_cells"]), "impostors must be majority"


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(GEN_ATTEMPTS):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        built = _build_case(rng)
        if built is None:
            continue
        inp, expected, meta = built
        try:
            actual = main(inp)
            _semantic_checks(inp, actual, expected, meta)
        except AssertionError:
            continue
        return inp, expected, meta
    raise AssertionError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    inp, _expected, _meta = _generate_case()
    return inp

