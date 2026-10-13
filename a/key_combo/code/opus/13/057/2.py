import random
from collections import deque

import numpy as np


BG = 0
BARRIER = 5
COLOR_FOR_ONE = 1
COLOR_FOR_TWO = 2
COLOR_FOR_MANY = 4
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_RETRIES = 500


def neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            count += 1
            queue = deque([(r, c)])
            labels[r, c] = count
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = count
                        queue.append((nr, nc))
    return labels, count


def component_coords(labels: np.ndarray, comp_id: int) -> np.ndarray:
    return np.argwhere(labels == int(comp_id))


def recolor_for_count(k: int) -> int:
    if int(k) == 1:
        return COLOR_FOR_ONE
    if int(k) == 2:
        return COLOR_FOR_TWO
    return COLOR_FOR_MANY


def analyze_barriers(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bg_labels, bg_count = label_mask(g == bg)
    fg_labels, fg_count = label_mask(g != bg)
    comps = []
    for comp_id in range(1, fg_count + 1):
        coords = component_coords(fg_labels, comp_id)
        adj_regions = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    reg = int(bg_labels[nr, nc])
                    if reg > 0:
                        adj_regions.add(reg)
        comps.append(
            {
                "id": comp_id,
                "coords": coords,
                "area": int(coords.shape[0]),
                "adjacent_regions": adj_regions,
                "k": int(len(adj_regions)),
            }
        )
    return {
        "bg": bg,
        "bg_labels": bg_labels,
        "bg_count": int(bg_count),
        "fg_labels": fg_labels,
        "components": comps,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each thin barrier by how many different background regions touch it.

    Input structure:
    - Black background cells form one or more 4-connected open spaces.
    - Foreground cells form several disconnected thin barrier components.
    - The barriers partition the background into 4-connected regions, while some
      barriers may also sit inside only one region as dead-end intrusions.

    Rule:
    - Label every 4-connected background region.
    - Label every 4-connected foreground barrier component.
    - For each barrier component, collect the distinct background-region labels
      appearing in the 4-neighborhood of any barrier pixel.
    - Recolor the whole component blue for 1 touched region, red for 2 touched
      regions, and yellow for 3 or more touched regions; keep the background unchanged.

    Key insight:
    - The decisive count is local to each barrier component: global region count
      is irrelevant if a given barrier only borders one or two of those regions.
    """
    g = np.asarray(grid, dtype=int)
    info = analyze_barriers(g)
    out = np.full_like(g, info["bg"])
    for comp in info["components"]:
        color = recolor_for_count(comp["k"])
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _ring_cells(h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                cells.append((r, c))
    return cells


def _spur_shape(rng: random.Random) -> list[tuple[int, int]]:
    kind = rng.choice(["h", "v", "l", "snake"])
    if kind == "h":
        length = rng.randint(3, 6)
        return [(0, c) for c in range(length)]
    if kind == "v":
        length = rng.randint(3, 6)
        return [(r, 0) for r in range(length)]
    if kind == "l":
        leg1 = rng.randint(2, 4)
        leg2 = rng.randint(2, 4)
        cells = [(0, c) for c in range(leg1)]
        cells.extend((r, leg1 - 1) for r in range(1, leg2))
        return cells
    length = rng.randint(4, 6)
    cells = [(0, 0)]
    r = c = 0
    horizontal = True
    for step in range(1, length):
        if horizontal:
            c += 1
        else:
            r += 1
        cells.append((r, c))
        if step < length - 1:
            horizontal = not horizontal
    return cells


def _normalize_shape(shape: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return sorted((r - min_r, c - min_c) for r, c in shape)


def _shape_bbox(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _apply_cells(grid: np.ndarray, cells: list[tuple[int, int]], value: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(value)


def _expanded_bbox_ok(grid: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> bool:
    h, w = grid.shape
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0 = max(0, min(rs) - pad)
    c0 = max(0, min(cs) - pad)
    r1 = min(h - 1, max(rs) + pad)
    c1 = min(w - 1, max(cs) + pad)
    target = set((int(r), int(c)) for r, c in cells)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if (rr, cc) in target:
                continue
            if int(grid[rr, cc]) != BG:
                return False
    return True


def _same_region(labels: np.ndarray, cells: list[tuple[int, int]]) -> int | None:
    region = None
    for r, c in cells:
        lab = int(labels[int(r), int(c)])
        if lab == 0:
            return None
        if region is None:
            region = lab
        elif lab != region:
            return None
    return region


def _try_place_shape(
    grid: np.ndarray,
    shape: list[tuple[int, int]],
    rng: random.Random,
    *,
    require_inner_bg: bool = False,
) -> bool:
    shape = _normalize_shape(shape)
    sh, sw = _shape_bbox(shape)
    if grid.shape[0] < sh + 2 or grid.shape[1] < sw + 2:
        return False

    positions = [(r0, c0) for r0 in range(1, grid.shape[0] - sh) for c0 in range(1, grid.shape[1] - sw)]
    rng.shuffle(positions)
    labels, _ = label_mask(grid == BG)
    shape_set = set(shape)
    for r0, c0 in positions:
        placed = [(r0 + r, c0 + c) for r, c in shape]
        if any(int(grid[r, c]) != BG for r, c in placed):
            continue
        if _same_region(labels, placed) is None:
            continue
        if not _expanded_bbox_ok(grid, placed, pad=1):
            continue
        if require_inner_bg:
            for r in range(r0, r0 + sh):
                for c in range(c0, c0 + sw):
                    if (r - r0, c - c0) not in shape_set and int(grid[r, c]) != BG:
                        break
                else:
                    continue
                break
            else:
                _apply_cells(grid, placed, BARRIER)
                return True
            continue
        _apply_cells(grid, placed, BARRIER)
        return True
    return False


def _random_transform(grid: np.ndarray, rng: random.Random) -> np.ndarray:
    out = grid.copy()
    if rng.random() < 0.5:
        out = np.flipud(out)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    if rng.random() < 0.5:
        out = np.rot90(out, int(rng.randint(1, 3)))
    return np.asarray(out, dtype=int)


def _valid_thin_barriers(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(grid[r : r + 2, c : c + 2] == BARRIER):
                return False
    return True


def _semantic_checks(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if set(np.unique(g)) - {BG, BARRIER}:
        return False
    if not _valid_thin_barriers(g):
        return False

    info = analyze_barriers(g)
    comps = info["components"]
    if not (4 <= len(comps) <= 6):
        return False
    if info["bg_count"] < 3:
        return False
    if not any(comp["k"] == 1 for comp in comps):
        return False
    if not any(comp["k"] == 2 for comp in comps):
        return False
    if not any(comp["k"] >= 3 for comp in comps):
        return False
    if not all(3 <= comp["area"] <= 16 for comp in comps):
        return False

    out = main(g)
    if out.shape != g.shape:
        return False
    if np.any(out[g == BG] != BG):
        return False
    output_colors = {int(x) for x in np.unique(out[g != BG])}
    if not {COLOR_FOR_ONE, COLOR_FOR_TWO, COLOR_FOR_MANY}.issubset(output_colors):
        return False
    for comp in comps:
        coords = comp["coords"]
        expected_color = recolor_for_count(comp["k"])
        values = {int(out[r, c]) for r, c in coords}
        if values != {expected_color}:
            return False
    return True


def _build_base_grid(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(9, 12)
    w = rng.randint(9, 13)
    grid = np.zeros((h, w), dtype=int)

    stem_col = rng.randint(2, min(3, w - 5))
    branch_row = rng.randint(2, h - 3)
    grid[:, stem_col] = BARRIER
    grid[branch_row, : stem_col + 1] = BARRIER

    target_components = rng.randint(4, 6)
    loop_count = 1 if target_components == 4 else rng.randint(1, 2)
    spur_count = target_components - 1 - loop_count
    if spur_count < 2:
        spur_count = 2
        loop_count = target_components - 1 - spur_count
    if loop_count < 1:
        return None

    for _ in range(loop_count):
        ring_h, ring_w = rng.choice([(3, 3), (3, 4), (4, 3)])
        if not _try_place_shape(grid, _ring_cells(ring_h, ring_w), rng, require_inner_bg=True):
            return None

    for _ in range(spur_count):
        placed = False
        for _attempt in range(25):
            if _try_place_shape(grid, _spur_shape(rng), rng, require_inner_bg=False):
                placed = True
                break
        if not placed:
            return None

    return _random_transform(grid, rng)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _build_base_grid(rng)
        if grid is None:
            continue
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid barrier-neighborhood sample")
