from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _ensure_arc_utils_on_path() -> None:
    try:
        import arc_utils  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "new" / "solvers"
        if candidate.is_dir():
            sys.path.append(str(candidate))
            return
    raise ModuleNotFoundError("Could not locate new/solvers/arc_utils.py")


_ensure_arc_utils_on_path()

from arc_utils import Color, bbox, crop_bbox, find_components, label_mask, paste  # noqa: E402


BG = Color.BLACK
BLUE = Color.BLUE
RED = Color.RED
FRAME = Color.GRAY
SIDE_ORDER = ("top", "bottom", "left", "right")
INNER_DIMS = [(3, 4), (3, 5), (3, 6), (4, 4), (4, 5), (5, 3), (6, 3)]
MARGIN_RANGE = (3, 8)
MAX_GENERATE_ATTEMPTS = 400
PARTITION_ATTEMPTS = 250
_GEN_CALL_INDEX = 0


def _normalize_coords(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    mins = coords.min(axis=0)
    return frozenset((int(r - mins[0]), int(c - mins[1])) for r, c in coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    box = bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    lab, n = label_mask(np.asarray(mask, dtype=bool), connectivity=4)
    return bool(mask.any()) and n == 1


def _is_irregular(mask: np.ndarray) -> bool:
    h, w = mask.shape
    return int(mask.sum()) < h * w and h >= 2 and w >= 2


def _gap_details(frame_bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> dict:
    r0, c0, r1, c1 = map(int, frame_bbox)
    gr, gc = map(int, gap)
    if gr == r0:
        return {
            "side": "top",
            "inner": (gr + 1, gc),
            "outward": (-1, 0),
            "inner_local": (0, gc - (c0 + 1)),
        }
    if gr == r1:
        return {
            "side": "bottom",
            "inner": (gr - 1, gc),
            "outward": (1, 0),
            "inner_local": (r1 - 1 - (r0 + 1), gc - (c0 + 1)),
        }
    if gc == c0:
        return {
            "side": "left",
            "inner": (gr, gc + 1),
            "outward": (0, -1),
            "inner_local": (gr - (r0 + 1), 0),
        }
    if gc == c1:
        return {
            "side": "right",
            "inner": (gr, gc - 1),
            "outward": (0, 1),
            "inner_local": (gr - (r0 + 1), c1 - 1 - (c0 + 1)),
        }
    raise AssertionError("gap must lie on the frame border")


def _find_frame(grid: np.ndarray) -> dict:
    frames = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME],
        return_format="dict",
    )
    assert len(frames) == 1
    frame = frames[0]
    r0, c0, r1, c1 = map(int, frame["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 5 and w >= 5

    missing: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border:
                if int(grid[r, c]) != FRAME:
                    missing.append((r, c))
            else:
                assert int(grid[r, c]) == BG

    assert len(missing) == 1
    gap = missing[0]
    assert gap not in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    assert int(frame["area"]) == (2 * h + 2 * w - 4) - 1

    details = _gap_details((r0, c0, r1, c1), gap)
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1
    return {
        "bbox": (r0, c0, r1, c1),
        "inner_bbox": (inner_r0, inner_c0, inner_r1, inner_c1),
        "gap": gap,
        "side": details["side"],
        "inner_cell": details["inner"],
        "inner_local": details["inner_local"],
        "outward": details["outward"],
        "inner_shape": (inner_r1 - inner_r0 + 1, inner_c1 - inner_c0 + 1),
    }


def _extract_sprite(grid: np.ndarray, color: int, frame_bbox: tuple[int, int, int, int]) -> dict:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[color],
        return_format="dict",
    )
    assert len(comps) == 1
    comp = comps[0]
    r0, c0, r1, c1 = map(int, comp["bbox"])
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    assert r1 < fr0 or r0 > fr1 or c1 < fc0 or c0 > fc1
    sprite = crop_bbox(grid, comp["bbox"])
    mask = sprite == int(color)
    assert _is_connected(mask)
    return {
        "color": int(color),
        "coords": comp["coords"],
        "bbox": (r0, c0, r1, c1),
        "sprite": sprite,
        "mask": mask,
        "area": int(comp["area"]),
    }


def _find_packing_solutions(
    blue_mask: np.ndarray,
    red_mask: np.ndarray,
    cavity_shape: tuple[int, int],
) -> list[dict]:
    blue_mask = np.asarray(blue_mask, dtype=bool)
    red_mask = np.asarray(red_mask, dtype=bool)
    inner_h, inner_w = map(int, cavity_shape)
    bh, bw = blue_mask.shape
    rh, rw = red_mask.shape

    blue_coords = np.argwhere(blue_mask)
    red_coords = np.argwhere(red_mask)
    if blue_coords.size == 0 or red_coords.size == 0:
        return []
    if int(blue_coords.shape[0] + red_coords.shape[0]) != inner_h * inner_w:
        return []
    if bh > inner_h or bw > inner_w or rh > inner_h or rw > inner_w:
        return []

    red_sig = _normalize_coords(red_coords)
    solutions: list[dict] = []
    for br in range(inner_h - bh + 1):
        for bc in range(inner_w - bw + 1):
            blue_occ = np.zeros((inner_h, inner_w), dtype=bool)
            blue_occ[br : br + bh, bc : bc + bw] = blue_mask
            remainder = ~blue_occ
            rem_coords = np.argwhere(remainder)
            if int(rem_coords.shape[0]) != int(red_coords.shape[0]):
                continue
            rem_box = bbox(remainder)
            assert rem_box is not None
            rr0, rc0, rr1, rc1 = map(int, rem_box)
            if (rr1 - rr0 + 1, rc1 - rc0 + 1) != (rh, rw):
                continue
            if _normalize_coords(rem_coords) != red_sig:
                continue

            assembled = np.full((inner_h, inner_w), RED, dtype=int)
            assembled[blue_occ] = BLUE
            solutions.append(
                {
                    "blue_top_left": (br, bc),
                    "red_top_left": (rr0, rc0),
                    "assembled": assembled,
                }
            )
    return solutions


def _solve_instance(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    frame = _find_frame(g)
    blue = _extract_sprite(g, BLUE, frame["bbox"])
    red = _extract_sprite(g, RED, frame["bbox"])

    solutions = _find_packing_solutions(blue["mask"], red["mask"], frame["inner_shape"])
    assert len(solutions) == 1
    solution = solutions[0]

    out = g.copy()
    for sprite in (blue, red):
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    ir0, ic0, ir1, ic1 = frame["inner_bbox"]
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = solution["assembled"]

    spill_r, spill_c = frame["inner_cell"]
    spill_color = int(out[spill_r, spill_c])
    assert spill_color in (BLUE, RED)

    dr, dc = frame["outward"]
    r, c = frame["gap"]
    while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
        assert int(out[r, c]) == BG
        out[r, c] = spill_color
        r += dr
        c += dc

    meta = {
        "frame": frame,
        "blue": blue,
        "red": red,
        "spill_color": spill_color,
        "solutions": solutions,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble two complementary sprites into a gapped frame and spill the gap-adjacent color outward.

    Input structure
    - The grid has a black background, one gray one-pixel-thick rectangular frame, and exactly one missing non-corner frame pixel.
    - The frame interior is empty black, while a blue sprite and a red sprite sit elsewhere outside the frame.
    - Cropping the two sprites to their own bboxes yields complementary shapes that can be translated, without rotation or overlap, to fill the frame interior exactly.

    Rule
    - Crop the blue and red sprites to their minimal bboxes.
    - Find the unique translation placement that packs the two cropped sprites into a solid rectangle whose size matches the frame interior, then place that packed rectangle inside the frame.
    - Read the color of the first interior cell behind the frame gap, color the gap and the straight outward ray from it with that same color, and erase the original sprites outside the frame back to black.

    Key insight
    - The spill color is not marked directly: it is only revealed after solving the unique complementary packing inside the frame, because the packed rectangle decides which sprite reaches the gap-adjacent interior cell.
    """
    out, _meta = _solve_instance(grid)
    return out


def _neighbor_cells(cell: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    h, w = shape
    result = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            result.append((rr, cc))
    return result


def _random_connected_mask(
    shape: tuple[int, int],
    target_area: int,
    rng: np.random.Generator,
    *,
    force_in: set[tuple[int, int]] | None = None,
    force_out: set[tuple[int, int]] | None = None,
) -> np.ndarray | None:
    h, w = map(int, shape)
    cells = {(r, c) for r in range(h) for c in range(w)}
    force_in = set() if force_in is None else set(force_in)
    force_out = set() if force_out is None else set(force_out)
    if force_in & force_out:
        return None
    if len(force_in) > target_area:
        return None
    if force_in - cells or force_out - cells:
        return None

    if not force_in:
        start = (int(rng.integers(h)), int(rng.integers(w)))
        chosen = {start}
    else:
        chosen = set(force_in)

    while len(chosen) < target_area:
        frontier = set()
        for cell in chosen:
            for nbr in _neighbor_cells(cell, shape):
                if nbr not in chosen and nbr not in force_out:
                    frontier.add(nbr)
        if not frontier:
            return None
        options = tuple(frontier)
        chosen.add(options[int(rng.integers(len(options)))])

    mask = np.zeros((h, w), dtype=bool)
    for r, c in chosen:
        mask[r, c] = True
    return mask


def _gap_choice(side: str, inner_shape: tuple[int, int], idx: int) -> tuple[tuple[int, int], tuple[int, int]]:
    inner_h, inner_w = map(int, inner_shape)
    if side == "top":
        return (0, idx), (0, idx)
    if side == "bottom":
        return (inner_h - 1, idx), (inner_h - 1, idx)
    if side == "left":
        return (idx, 0), (idx, 0)
    if side == "right":
        return (idx, inner_w - 1), (idx, inner_w - 1)
    raise AssertionError("bad side")


def _sample_partition(
    inner_shape: tuple[int, int],
    gap_local: tuple[int, int],
    target_spill_color: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    inner_h, inner_w = map(int, inner_shape)
    area = inner_h * inner_w
    blue_sizes = [size for size in range(max(5, area - 10), min(10, area - 5) + 1)]
    if not blue_sizes:
        return None

    force_in = {gap_local} if target_spill_color == BLUE else set()
    force_out = {gap_local} if target_spill_color == RED else set()

    for _ in range(PARTITION_ATTEMPTS):
        blue_area = int(blue_sizes[int(rng.integers(len(blue_sizes)))])
        blue_mask = _random_connected_mask(
            inner_shape,
            blue_area,
            rng,
            force_in=force_in,
            force_out=force_out,
        )
        if blue_mask is None:
            continue
        red_mask = ~blue_mask
        if not _is_connected(red_mask):
            continue

        blue_crop = _crop_mask(blue_mask)
        red_crop = _crop_mask(red_mask)
        if not (_is_irregular(blue_crop) and _is_irregular(red_crop)):
            continue

        solutions = _find_packing_solutions(blue_crop, red_crop, inner_shape)
        if len(solutions) != 1:
            continue
        if int(solutions[0]["assembled"][gap_local]) != int(target_spill_color):
            continue
        return blue_mask, red_mask, blue_crop, red_crop
    return None


def _rect_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _ray_cells(frame_bbox: tuple[int, int, int, int], gap: tuple[int, int], outward: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r, c = map(int, gap)
    dr, dc = map(int, outward)
    cells = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _valid_positions(
    sprite_mask: np.ndarray,
    grid_shape: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
    ray: set[tuple[int, int]],
    extra_forbidden: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    h, w = map(int, sprite_mask.shape)
    grid_h, grid_w = map(int, grid_shape)
    coords = np.argwhere(sprite_mask)
    positions: list[tuple[int, int]] = []
    for r0 in range(grid_h - h + 1):
        for c0 in range(grid_w - w + 1):
            rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _rect_close(rect, frame_bbox, gap=1):
                continue
            if any(_rect_close(rect, other, gap=1) for other in extra_forbidden):
                continue
            blocked = False
            for rr, cc in coords:
                if (r0 + int(rr), c0 + int(cc)) in ray:
                    blocked = True
                    break
            if not blocked:
                positions.append((r0, c0))
    return positions


def _expected_output(
    grid: np.ndarray,
    frame: dict,
    blue_coords: np.ndarray,
    red_coords: np.ndarray,
    assembled: np.ndarray,
    spill_color: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[blue_coords[:, 0], blue_coords[:, 1]] = BG
    out[red_coords[:, 0], red_coords[:, 1]] = BG
    ir0, ic0, ir1, ic1 = frame["inner_bbox"]
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = assembled
    dr, dc = frame["outward"]
    r, c = frame["gap"]
    while 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
        out[r, c] = int(spill_color)
        r += dr
        c += dc
    return out


def _gen_attempt(rng: np.random.Generator, side_hint: str, spill_hint: int) -> np.ndarray | None:
    inner_h, inner_w = INNER_DIMS[int(rng.integers(len(INNER_DIMS)))]
    side = side_hint
    if side in ("top", "bottom"):
        gap_idx = int(rng.integers(inner_w))
    else:
        gap_idx = int(rng.integers(inner_h))
    gap_local, spill_local = _gap_choice(side, (inner_h, inner_w), gap_idx)
    target_spill_color = int(spill_hint)

    partition = _sample_partition((inner_h, inner_w), spill_local, target_spill_color, rng)
    if partition is None:
        return None
    blue_mask, red_mask, blue_crop, red_crop = partition

    top = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    bottom = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    left = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    right = int(rng.integers(MARGIN_RANGE[0], MARGIN_RANGE[1] + 1))
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    grid_h = top + frame_h + bottom
    grid_w = left + frame_w + right

    frame_bbox = (top, left, top + frame_h - 1, left + frame_w - 1)
    inner_bbox = (top + 1, left + 1, top + inner_h, left + inner_w)
    if side == "top":
        gap = (top, left + 1 + gap_idx)
    elif side == "bottom":
        gap = (top + frame_h - 1, left + 1 + gap_idx)
    elif side == "left":
        gap = (top + 1 + gap_idx, left)
    else:
        gap = (top + 1 + gap_idx, left + frame_w - 1)
    frame = {
        "bbox": frame_bbox,
        "inner_bbox": inner_bbox,
        "gap": gap,
        "side": side,
        "inner_cell": (top + 1 + spill_local[0], left + 1 + spill_local[1]),
        "inner_local": spill_local,
        "outward": _gap_details(frame_bbox, gap)["outward"],
        "inner_shape": (inner_h, inner_w),
    }

    ray = set(_ray_cells(frame_bbox, gap, frame["outward"], (grid_h, grid_w)))
    blue_positions = _valid_positions(blue_crop, (grid_h, grid_w), frame_bbox, ray, [])
    if not blue_positions:
        return None
    blue_top = blue_positions[int(rng.integers(len(blue_positions)))]
    blue_rect = (
        blue_top[0],
        blue_top[1],
        blue_top[0] + blue_crop.shape[0] - 1,
        blue_top[1] + blue_crop.shape[1] - 1,
    )

    red_positions = _valid_positions(red_crop, (grid_h, grid_w), frame_bbox, ray, [blue_rect])
    if not red_positions:
        return None
    red_top = red_positions[int(rng.integers(len(red_positions)))]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME
    grid[gap] = BG

    blue_sprite = np.where(blue_crop, BLUE, BG)
    red_sprite = np.where(red_crop, RED, BG)
    paste(grid, blue_sprite, blue_top[0], blue_top[1], transparent=BG)
    paste(grid, red_sprite, red_top[0], red_top[1], transparent=BG)

    blue_global = np.argwhere(grid == BLUE)
    red_global = np.argwhere(grid == RED)
    if len(blue_global) != int(blue_crop.sum()) or len(red_global) != int(red_crop.sum()):
        return None

    if any(tuple(cell) in ray for cell in blue_global) or any(tuple(cell) in ray for cell in red_global):
        return None

    assembled = np.full((inner_h, inner_w), RED, dtype=int)
    assembled[blue_mask] = BLUE
    spill_color = int(assembled[spill_local])
    if spill_color != target_spill_color:
        return None

    expected = _expected_output(grid, frame, blue_global, red_global, assembled, spill_color)
    actual, meta = _solve_instance(grid)
    if not np.array_equal(actual, expected):
        return None
    if len(meta["solutions"]) != 1 or meta["spill_color"] != spill_color:
        return None
    if meta["frame"]["side"] != side:
        return None
    if meta["frame"]["inner_local"] != spill_local:
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX

    rng = np.random.default_rng()
    side_hint = SIDE_ORDER[_GEN_CALL_INDEX % len(SIDE_ORDER)]
    spill_hint = BLUE if _GEN_CALL_INDEX % 2 == 0 else RED
    _GEN_CALL_INDEX += 1

    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng, side_hint, spill_hint)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-frame instance")
