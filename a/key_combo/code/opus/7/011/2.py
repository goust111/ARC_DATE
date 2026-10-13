import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, label_mask


PALETTE_COLORS = (Color.ORANGE, Color.TEAL, Color.GREY)
INPUT_COLOR_POOL = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK)
GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACE_RETRIES = 200


def _parse_palette(last_row: np.ndarray) -> list[int]:
    nz = np.flatnonzero(last_row != Color.BLACK)
    assert len(nz) == 3, "expected exactly three palette pixels"
    assert int(nz[-1] - nz[0]) == 2, "palette pixels must be contiguous"
    palette = [int(x) for x in last_row[nz[0] : nz[0] + 3]]
    assert palette == list(PALETTE_COLORS), f"unexpected palette colors: {palette}"
    return palette


def _hole_count_from_mask(mask: np.ndarray) -> int:
    bg_mask = ~mask.astype(bool)
    labels, n = label_mask(bg_mask, connectivity=4)
    holes = 0
    h, w = mask.shape
    for idx in range(1, n + 1):
        pts = np.argwhere(labels == idx)
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        holes += 1
    return int(holes)


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = np.asarray(comp["coords"], dtype=int)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _extract_specs(grid: np.ndarray) -> list[dict]:
    body = np.asarray(grid[:-1], dtype=int)
    comps = find_components(body, bg=Color.BLACK, connectivity=4, monochromatic=True, return_format="dict")
    specs = []
    for comp in comps:
        bbox = comp["bbox"]
        subgrid = crop_bbox(body, bbox)
        hole_count = _hole_count_from_mask(subgrid == int(comp["color"]))
        specs.append(
            {
                "color": int(comp["color"]),
                "bbox": bbox,
                "coords": np.asarray(comp["coords"], dtype=int),
                "area": int(comp["area"]),
                "holes": int(hole_count),
            }
        )
    return specs


def _build_output(hole_counts: list[int], palette: list[int]) -> np.ndarray:
    ordered = sorted([int(x) for x in hole_counts], reverse=True)
    n = len(ordered) * 2 - 1
    out = np.full((n, n), Color.BLACK, dtype=int)
    for depth, holes in enumerate(ordered):
        color = int(palette[holes % len(palette)])
        out[depth : n - depth, depth : n - depth] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Compress the shapes into nested square rings ordered by hole count and recolored by a palette cycle.

    Input structure:
    - The background is black (0).
    - Above the bottom row there are 4 or 5 disjoint monochromatic 4-connected shapes, each with a distinct color.
    - The bottom row contains exactly one contiguous 3-pixel palette strip with colors orange, teal, grey from left to right.

    Rule:
    - Ignore the bottom palette row when detecting shapes.
    - Count the number of enclosed black 4-connected hole regions inside each shape.
    - Sort the shapes by hole count from largest to smallest and create a (2K-1)x(2K-1) nested-square summary.
    - The ring for a shape with hole count C uses the palette color at index (C mod 3).

    Key insight:
    - The same hole count is used twice: first to choose the nesting order, then again to choose the ring color through the palette cycle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2
    palette = _parse_palette(g[-1])
    specs = _extract_specs(g)
    assert 4 <= len(specs) <= 5
    counts = [spec["holes"] for spec in specs]
    assert len(set(counts)) == len(counts)
    return _build_output(counts, palette)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _choose_holes(base_h: int, base_w: int, hole_count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, base_h - 1) for c in range(1, base_w - 1)]
    random.shuffle(cells)
    chosen: list[tuple[int, int]] = []

    def backtrack(start: int) -> bool:
        if len(chosen) == hole_count:
            return True
        for idx in range(start, len(cells)):
            cell = cells[idx]
            if any(abs(cell[0] - rr) + abs(cell[1] - cc) == 1 for rr, cc in chosen):
                continue
            chosen.append(cell)
            if backtrack(idx + 1):
                return True
            chosen.pop()
        return False

    if backtrack(0):
        return chosen[:]
    return None


def _bbox_area(mask: np.ndarray) -> int:
    rows, cols = np.where(mask)
    return int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _exterior_zeros(mask: np.ndarray) -> np.ndarray:
    labels, n = label_mask(~mask.astype(bool), connectivity=4)
    exterior = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    for idx in range(1, n + 1):
        pts = np.argwhere(labels == idx)
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            exterior[labels == idx] = True
    return exterior


def _grow_irregular(mask: np.ndarray, hole_count: int, min_add: int, max_add: int) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    target = random.randint(min_add, max_add)
    added = 0
    for _ in range(40):
        if added >= target or out.sum() >= 30:
            break
        work = np.pad(out, 1)
        exterior = _exterior_zeros(work)
        candidates = []
        for r in range(work.shape[0]):
            for c in range(work.shape[1]):
                if work[r, c] or not exterior[r, c]:
                    continue
                if (
                    (r > 0 and work[r - 1, c])
                    or (r + 1 < work.shape[0] and work[r + 1, c])
                    or (c > 0 and work[r, c - 1])
                    or (c + 1 < work.shape[1] and work[r, c + 1])
                ):
                    candidates.append((r, c))
        random.shuffle(candidates)
        advanced = False
        for r, c in candidates:
            trial = work.copy()
            trial[r, c] = True
            trial = _crop_mask(trial)
            if int(trial.sum()) > 30:
                continue
            if _hole_count_from_mask(trial) != hole_count:
                continue
            labels, n = label_mask(trial, connectivity=4)
            if n != 1:
                continue
            out = trial
            added += 1
            advanced = True
            break
        if not advanced:
            break
    return out


def _make_zero_hole_shape() -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        canvas = np.zeros((7, 7), dtype=bool)
        start = (3, 3)
        canvas[start] = True
        cells = {start}
        target = random.randint(8, 16)
        while len(cells) < target:
            frontier = []
            for r, c in list(cells):
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 7 and 0 <= nc < 7 and not canvas[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            canvas[nr, nc] = True
            cells.add((nr, nc))
        mask = _crop_mask(canvas)
        if _hole_count_from_mask(mask) != 0:
            continue
        if int(mask.sum()) < 8 or int(mask.sum()) > 30:
            continue
        if int(mask.sum()) == _bbox_area(mask):
            continue
        labels, n = label_mask(mask, connectivity=4)
        if n != 1:
            continue
        return _random_transform(mask)
    raise RuntimeError("failed to generate a zero-hole shape")


def _make_holey_shape(hole_count: int) -> np.ndarray:
    if hole_count == 0:
        return _make_zero_hole_shape()
    dims = [(h, w) for h in range(3, 7) for w in range(3, 7) if h * w - hole_count <= 28]
    random.shuffle(dims)
    for _ in range(SHAPE_RETRIES):
        for base_h, base_w in dims:
            holes = _choose_holes(base_h, base_w, hole_count)
            if holes is None:
                continue
            mask = np.ones((base_h, base_w), dtype=bool)
            for r, c in holes:
                mask[r, c] = False
            if _hole_count_from_mask(mask) != hole_count:
                continue
            mask = _grow_irregular(mask, hole_count, min_add=0, max_add=min(5, 30 - int(mask.sum())))
            mask = _random_transform(mask)
            if _hole_count_from_mask(mask) != hole_count:
                continue
            if int(mask.sum()) < 8 or int(mask.sum()) > 30:
                continue
            if int(mask.sum()) == _bbox_area(mask):
                continue
            labels, n = label_mask(mask, connectivity=4)
            if n != 1:
                continue
            return mask
    raise RuntimeError(f"failed to generate a {hole_count}-hole shape")


def _touches_8(coords_a: np.ndarray, coords_b: np.ndarray) -> bool:
    aset = {tuple(map(int, x)) for x in coords_a}
    for r, c in coords_b:
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (rr + dr, cc + dc) in aset:
                    return True
    return False


def _validate_semantics(grid: np.ndarray, intended_counts: list[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    palette = _parse_palette(g[-1])
    specs = _extract_specs(g)
    assert len(specs) == len(intended_counts)
    actual_counts = [spec["holes"] for spec in specs]
    assert sorted(actual_counts) == sorted(intended_counts)
    assert len(set(actual_counts)) == len(actual_counts)
    assert set(actual_counts).issubset(set(range(5)))
    assert {0, 1, 2}.issubset(set(actual_counts))
    for spec in specs:
        assert 8 <= spec["area"] <= 30
        mask = _component_mask(spec)
        if spec["holes"] == 0:
            assert int(mask.sum()) < _bbox_area(mask)
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            assert not _touches_8(specs[i]["coords"], specs[j]["coords"])
    area_order = [spec["holes"] for spec in sorted(specs, key=lambda spec: spec["area"], reverse=True)]
    hole_order = [spec["holes"] for spec in sorted(specs, key=lambda spec: spec["holes"], reverse=True)]
    assert area_order != hole_order
    out = _build_output(actual_counts, palette)
    assert out.shape == (len(specs) * 2 - 1, len(specs) * 2 - 1)
    assert set(np.unique(out)) == set(PALETTE_COLORS)
    return out


def _place_shapes(height: int, width: int, specs: list[dict]) -> np.ndarray | None:
    body_h = height - 1
    canvas = np.zeros((height, width), dtype=int)
    occupied = np.zeros((body_h, width), dtype=bool)
    place_order = sorted(specs, key=lambda spec: max(spec["mask"].shape), reverse=True)
    for spec in place_order:
        mask = spec["mask"]
        mh, mw = mask.shape
        choices = [(r, c) for r in range(body_h - mh + 1) for c in range(width - mw + 1)]
        random.shuffle(choices)
        placed = False
        for r, c in choices[:PLACE_RETRIES]:
            r0 = max(0, r - 1)
            c0 = max(0, c - 1)
            r1 = min(body_h, r + mh + 1)
            c1 = min(width, c + mw + 1)
            if occupied[r0:r1, c0:c1].any():
                continue
            region = canvas[r : r + mh, c : c + mw]
            region[mask] = int(spec["color"])
            occupied[r : r + mh, c : c + mw] |= mask
            placed = True
            break
        if not placed:
            return None
    start = random.randint(0, width - 3)
    canvas[-1, start : start + 3] = np.array(PALETTE_COLORS, dtype=int)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    count = random.choice((4, 5))
    if count == 5:
        hole_counts = [0, 1, 2, 3, 4]
    else:
        hole_counts = [0, 1, 2, random.choice((3, 4))]
    colors = list(INPUT_COLOR_POOL)
    random.shuffle(colors)
    specs = []
    try:
        for holes, color in zip(hole_counts, colors):
            mask = _make_holey_shape(holes)
            specs.append({"holes": int(holes), "color": int(color), "mask": mask, "area": int(mask.sum())})
    except RuntimeError:
        return None

    if len({spec["area"] for spec in specs}) != len(specs):
        return None
    area_rank = [spec["holes"] for spec in sorted(specs, key=lambda spec: spec["area"], reverse=True)]
    hole_rank = [spec["holes"] for spec in sorted(specs, key=lambda spec: spec["holes"], reverse=True)]
    if area_rank == hole_rank:
        return None

    height = random.randint(19, 25)
    width = random.randint(19, 25)
    if count == 5:
        height = max(height, 21)
        width = max(width, 21)
    grid = _place_shapes(height, width, specs)
    if grid is None:
        return None

    expected = _validate_semantics(grid, hole_counts)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
