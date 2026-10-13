import random
from collections import defaultdict, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2

GEN_RETRY_LIMIT = 500
GEN_GRID_MIN = 18
GEN_GRID_MAX = 24


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _count_holes(mask: np.ndarray) -> int:
    bg = ~mask
    h, w = bg.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _object_records(grid: np.ndarray) -> list[dict]:
    comps = _components(grid == RED)
    records = []
    for comp in comps:
        bbox = _bbox_from_coords(comp)
        r0, c0, r1, c1 = bbox
        crop = (grid[r0 : r1 + 1, c0 : c1 + 1] == RED)
        records.append(
            {
                "bbox": bbox,
                "crop": crop,
                "holes": _count_holes(crop),
                "area": int(crop.sum()),
            }
        )
    return records


def _assemble_output(records: list[dict]) -> np.ndarray:
    ordered = sorted(records, key=lambda rec: (rec["holes"], rec["bbox"][0]))
    out_h = max(rec["crop"].shape[0] for rec in ordered) + 2
    out_w = sum(rec["crop"].shape[1] for rec in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for idx, rec in enumerate(ordered):
        crop = rec["crop"]
        h, w = crop.shape
        top = (out_h - h) // 2
        color = BLUE if idx % 2 == 0 else RED
        out[top : top + h, col : col + w][crop] = color
        col += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort red objects by enclosed-hole count, recolor them alternately, and pack them into one centered row.

    Input structure:
    - The background is black.
    - There are several non-touching red 4-connected foreground objects scattered around the grid.
    - Each object may contain zero or more enclosed black 4-connected holes inside its own bounding box.

    Rule:
    - Detect every red 4-connected object and crop it to its bounding box.
    - Count, for each cropped object, how many black 4-connected background regions inside that crop do not touch the crop border.
    - Sort the objects by that hole count in ascending order; if two objects have the same hole count, break the tie by the row of the object's bounding-box top edge.
    - In the sorted sequence, recolor objects at positions 0, 2, 4, ... blue and recolor objects at positions 1, 3, 5, ... red.
    - Place the recolored cropped objects from left to right with one blank column between neighbors, vertically centered in the output, while leaving all hole pixels black.

    Key insight:
    - The input positions are distractors: the output order is determined by each object's internal topology, and the final color depends only on parity after that topological sort.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    records = _object_records(g)
    assert records, "expected red foreground objects"
    return _assemble_output(records)


def _random_polyomino(area: int) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((8, 8), dtype=bool)
        start = (4, 4)
        mask[start] = True
        cells = [start]
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            seen = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 8 and 0 <= nc < 8 and not mask[nr, nc] and (nr, nc) not in seen:
                        frontier.append((nr, nc))
                        seen.add((nr, nc))
            if not frontier:
                break
            chosen = random.choice(frontier)
            mask[chosen] = True
            cells.append(chosen)
        mask = _crop_mask(mask)
        if mask.sum() != area:
            continue
        if _count_holes(mask) != 0:
            continue
        h, w = mask.shape
        if h < 3 or w < 3:
            continue
        if max(h, w) > 5:
            continue
        return mask
    return None


def _chain_holes(k: int) -> np.ndarray:
    mask = np.ones((3, 2 * k + 1), dtype=bool)
    for i in range(k):
        mask[1, 2 * i + 1] = False
    return mask


def _mutate_same_topology(mask: np.ndarray, target_holes: int, min_area: int, max_area: int) -> np.ndarray:
    mask = np.pad(mask, 1)
    steps = random.randint(2, 6)
    for _ in range(steps):
        if random.random() < 0.7:
            candidates = []
            h, w = mask.shape
            for r in range(h):
                for c in range(w):
                    if mask[r, c]:
                        continue
                    if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                        candidates.append((r, c))
            random.shuffle(candidates)
            for r, c in candidates:
                trial = mask.copy()
                trial[r, c] = True
                cropped = _crop_mask(trial)
                area = int(cropped.sum())
                if area <= max_area and _count_holes(cropped) == target_holes:
                    mask = trial
                    break
        if random.random() < 0.8:
            candidates = list(map(tuple, np.argwhere(mask)))
            random.shuffle(candidates)
            for r, c in candidates:
                trial = mask.copy()
                trial[r, c] = False
                if not np.any(trial):
                    continue
                cropped = _crop_mask(trial)
                if len(_components(cropped)) != 1:
                    continue
                area = int(cropped.sum())
                if area >= min_area and _count_holes(cropped) == target_holes:
                    mask = trial
                    break

    cropped = _crop_mask(mask)
    if random.random() < 0.5:
        cropped = np.fliplr(cropped)
    if random.random() < 0.5:
        cropped = np.flipud(cropped)
    rotations = random.randint(0, 3)
    if rotations:
        cropped = np.rot90(cropped, rotations)
    return cropped.copy()


def _generate_shape(hole_count: int) -> np.ndarray | None:
    if hole_count == 0:
        area = random.randint(8, 12)
        return _random_polyomino(area)

    base = _chain_holes(hole_count)
    limits = {
        1: (8, 13),
        2: (11, 15),
        3: (16, 18),
    }
    min_area, max_area = limits[hole_count]
    for _ in range(120):
        shape = _mutate_same_topology(base, hole_count, min_area, max_area)
        if _count_holes(shape) != hole_count:
            continue
        if len(_components(shape)) != 1:
            continue
        area = int(shape.sum())
        if not (min_area <= area <= max_area):
            continue
        if max(shape.shape) > 7:
            continue
        return shape
    return None


def _place_shapes(shape_specs: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, list[dict]] | tuple[None, None]:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    padded_occ = np.zeros((h, w), dtype=bool)
    used_tops: set[int] = set()
    records: list[dict] = []

    indices = list(range(len(shape_specs)))
    random.shuffle(indices)
    duplicate_holes = None
    hole_counts = [hole_count for _, hole_count in shape_specs]
    counts = defaultdict(int)
    for hc in hole_counts:
        counts[hc] += 1
    for hc, count in counts.items():
        if count >= 2:
            duplicate_holes = hc
            break

    for idx in indices:
        shape, hole_count = shape_specs[idx]
        sh, sw = shape.shape
        candidates = []
        for r0 in range(1, h - sh):
            if r0 in used_tops:
                continue
            for c0 in range(1, w - sw):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r0 + sh + 1)
                cc1 = min(w, c0 + sw + 1)
                if padded_occ[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None, None
        random.shuffle(candidates)
        r0, c0 = candidates[0]
        used_tops.add(r0)
        grid[r0 : r0 + sh, c0 : c0 + sw][shape] = RED
        padded_occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True
        records.append(
            {
                "bbox": (r0, c0, r0 + sh - 1, c0 + sw - 1),
                "crop": shape.copy(),
                "holes": hole_count,
                "area": int(shape.sum()),
            }
        )

    if duplicate_holes is None:
        return None, None
    dup = [rec for rec in records if rec["holes"] == duplicate_holes]
    if len(dup) < 2:
        return None, None
    dup_row_order = [rec["bbox"] for rec in sorted(dup, key=lambda rec: rec["bbox"][0])]
    dup_col_order = [rec["bbox"] for rec in sorted(dup, key=lambda rec: rec["bbox"][1])]
    if dup_row_order == dup_col_order:
        return None, None

    hole_order = [rec["bbox"] for rec in sorted(records, key=lambda rec: (rec["holes"], rec["bbox"][0]))]
    col_order = [rec["bbox"] for rec in sorted(records, key=lambda rec: rec["bbox"][1])]
    row_order = [rec["bbox"] for rec in sorted(records, key=lambda rec: rec["bbox"][0])]
    if hole_order == col_order or hole_order == row_order:
        return None, None

    return grid, records


def _semantic_checks(grid: np.ndarray, records: list[dict], expected: np.ndarray) -> bool:
    if len(records) != 5:
        return False

    hole_counts = sorted(rec["holes"] for rec in records)
    if sorted(set(hole_counts)) != [0, 1, 2, 3]:
        return False
    if len(set(hole_counts)) == len(hole_counts):
        return False

    top_rows = [rec["bbox"][0] for rec in records]
    if len(set(top_rows)) != len(top_rows):
        return False

    comps = _object_records(grid)
    if len(comps) != len(records):
        return False
    if sorted(rec["holes"] for rec in comps) != hole_counts:
        return False

    if expected.ndim != 2 or max(expected.shape) > 30:
        return False
    if not np.any(expected == BLUE) or not np.any(expected == RED):
        return False

    ordered = sorted(records, key=lambda rec: (rec["holes"], rec["bbox"][0]))
    parity_colors = [BLUE if i % 2 == 0 else RED for i in range(len(ordered))]
    spans = []
    col = 0
    for rec, color in zip(ordered, parity_colors):
        crop = rec["crop"]
        h, w = crop.shape
        top = (expected.shape[0] - h) // 2
        patch = expected[top : top + h, col : col + w]
        if np.any(patch[crop] != color):
            return False
        if np.any(patch[~crop] != BG):
            return False
        spans.append((col, col + w - 1))
        col += w + 1
    for (_, end0), (start1, _) in zip(spans, spans[1:]):
        if start1 - end0 != 2:
            return False
    return True


def _generate_attempt() -> tuple[np.ndarray, list[dict]] | tuple[None, None]:
    hole_counts = [0, 1, 2, 3, random.choice([0, 1, 2, 3])]
    random.shuffle(hole_counts)

    shape_specs: list[tuple[np.ndarray, int]] = []
    for hole_count in hole_counts:
        shape = _generate_shape(hole_count)
        if shape is None:
            return None, None
        shape_specs.append((shape, hole_count))

    dummy_records = [
        {
            "bbox": (0, 0, shape.shape[0] - 1, shape.shape[1] - 1),
            "crop": shape,
            "holes": hole_count,
            "area": int(shape.sum()),
        }
        for shape, hole_count in shape_specs
    ]
    if max(_assemble_output(dummy_records).shape) > 30:
        return None, None

    return _place_shapes(shape_specs)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid, records = _generate_attempt()
        if grid is None or records is None:
            continue
        expected = _assemble_output(records)
        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        if not _semantic_checks(grid, records, expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid topology-sorting input")
