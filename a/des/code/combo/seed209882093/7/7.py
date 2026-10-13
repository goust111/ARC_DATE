from __future__ import annotations

import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLORS = tuple(range(1, 10))

# Generator limits.
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
OBJECT_COUNT_RANGE = (2, 5)
HOLE_VALUES = (0, 1, 2, 3)
HOLE_WEIGHTS = (3, 4, 2, 1)
MAX_GENERATE_ATTEMPTS = 500
MAX_SHAPE_ATTEMPTS = 200
MAX_PLACE_ATTEMPTS = 200


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    height, width = mask.shape
    start = tuple(int(x) for x in cells[0])
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, height, width):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _count_holes(mask: np.ndarray) -> int:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    holes = 0
    for r in range(height):
        for c in range(width):
            if mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            touches_boundary = r == 0 or c == 0 or r == height - 1 or c == width - 1
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr == 0 or nc == 0 or nr == height - 1 or nc == width - 1:
                        touches_boundary = True
                    stack.append((nr, nc))
            if not touches_boundary:
                holes += 1
    return holes


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros((height, width), dtype=bool)
    objects = []
    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            bbox = _bbox_from_coords(coords)
            objects.append({"color": color, "coords": coords, "bbox": bbox})
    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate each foreground object according to the number of enclosed black holes.

    Rule:
    - Find each monochrome 4-connected foreground object.
    - Count the black holes fully enclosed by that object inside its bbox.
    - If bbox width >= height, copy the object horizontally; otherwise vertically.
    - Produce hole_count + 1 copies with a 1-cell black gap between consecutive copies.
    - Clip anything that goes outside the original canvas.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = np.zeros_like(g)

    for obj in _find_objects(g):
        r0, c0, r1, c1 = obj["bbox"]
        box_h = int(r1 - r0 + 1)
        box_w = int(c1 - c0 + 1)

        mask = np.zeros((box_h, box_w), dtype=bool)
        for r, c in obj["coords"]:
            mask[r - r0, c - c0] = True

        copies = _count_holes(mask) + 1
        step_r, step_c = (0, box_w + 1) if box_w >= box_h else (box_h + 1, 0)

        for k in range(copies):
            dr = int(k * step_r)
            dc = int(k * step_c)
            for r, c in obj["coords"]:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width:
                    out[nr, nc] = int(obj["color"])

    return out


def _profile(length: int, cut_max: int) -> np.ndarray:
    prof = np.zeros(length, dtype=int)
    if cut_max <= 0 or length <= 2:
        return prof

    value = 0
    for i in range(1, length - 1):
        value = max(0, min(cut_max, value + random.choice((-1, 0, 1))))
        prof[i] = value

    if int(prof.max()) == 0:
        start = random.randint(1, length - 2)
        span = random.randint(1, max(1, length // 4))
        end = min(length - 1, start + span)
        prof[start:end] = 1
    return prof


def _make_horizontal_base(height: int, width: int, min_core: int) -> np.ndarray:
    cut_max = max(1, min(height - min_core, height // 3 + 1))
    top = _profile(width, cut_max)
    bottom = _profile(width, cut_max)
    top[0] = top[-1] = 0
    bottom[0] = bottom[-1] = 0

    for c in range(width):
        while height - int(top[c]) - int(bottom[c]) < min_core:
            if top[c] >= bottom[c] and top[c] > 0:
                top[c] -= 1
            elif bottom[c] > 0:
                bottom[c] -= 1
            else:
                break

    mask = np.zeros((height, width), dtype=bool)
    for c in range(width):
        r0 = int(top[c])
        r1 = int(height - bottom[c])
        mask[r0:r1, c] = True

    if width >= 6 and height >= 5:
        for side in ("left", "right"):
            if random.random() < 0.6:
                row0 = random.randint(1, height - 3)
                row1 = random.randint(row0 + 1, height - 2)
                col = 0 if side == "left" else width - 1
                mask[row0 : row1 + 1, col] = False
                mask[0, col] = True
                mask[-1, col] = True

    return mask


def _hole_candidates(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape
    candidates: list[tuple[int, int, int, int]] = []
    for hole_h, hole_w in ((1, 1), (1, 2), (2, 1), (2, 2)):
        if height < hole_h + 2 or width < hole_w + 2:
            continue
        for r in range(1, height - hole_h):
            for c in range(1, width - hole_w):
                block = mask[r : r + hole_h, c : c + hole_w]
                if not bool(np.all(block)):
                    continue
                ring = mask[r - 1 : r + hole_h + 1, c - 1 : c + hole_w + 1]
                if ring.shape != (hole_h + 2, hole_w + 2):
                    continue
                if bool(np.all(ring)):
                    candidates.append((r, c, hole_h, hole_w))
    return candidates


def _carve_holes(mask: np.ndarray, hole_count: int) -> np.ndarray | None:
    work = mask.copy()
    for _ in range(hole_count):
        candidates = _hole_candidates(work)
        if not candidates:
            return None
        r, c, hole_h, hole_w = random.choice(candidates)
        work[r : r + hole_h, c : c + hole_w] = False
    return work


def _make_object_mask(orientation: str, holes: int) -> np.ndarray | None:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        if orientation == "h":
            height = random.randint(4 if holes == 0 else 5, 7)
            width_min = max(height + 3, 8 + holes * 3)
            width_max = max(width_min, min(width_min + 6, 18))
            width = random.randint(width_min, width_max)
            mask = _make_horizontal_base(height, width, min_core=3 if holes > 0 else 2)
        else:
            width = random.randint(4 if holes == 0 else 5, 7)
            height_min = max(width + 3, 8 + holes * 3)
            height_max = max(height_min, min(height_min + 6, 18))
            height = random.randint(height_min, height_max)
            mask = _make_horizontal_base(width, height, min_core=3 if holes > 0 else 2).T

        if holes > 0:
            carved = _carve_holes(mask, holes)
            if carved is None:
                continue
            mask = carved

        mask = _crop_mask(mask)
        if not _is_connected(mask):
            continue
        if _count_holes(mask) != holes:
            continue
        if orientation == "h" and mask.shape[1] < mask.shape[0]:
            continue
        if orientation == "v" and mask.shape[0] <= mask.shape[1]:
            continue
        if int(mask.sum()) < 8:
            continue
        return mask
    return None


def _sample_hole_counts(n_objects: int) -> list[int]:
    counts = random.choices(HOLE_VALUES, weights=HOLE_WEIGHTS, k=n_objects)
    if all(count == 0 for count in counts):
        counts[random.randrange(n_objects)] = random.choice((1, 2, 3))
    if n_objects > 1 and all(count > 0 for count in counts) and random.random() < 0.7:
        counts[random.randrange(n_objects)] = 0
    return [int(x) for x in counts]


def _sample_orientations(n_objects: int) -> list[str]:
    orientations = [random.choice(("h", "v")) for _ in range(n_objects)]
    if n_objects > 1 and len(set(orientations)) == 1 and random.random() < 0.7:
        idx = random.randrange(n_objects)
        orientations[idx] = "v" if orientations[idx] == "h" else "h"
    return orientations


def _visible_bbox(spec: dict, top: int, left: int, grid_h: int, grid_w: int) -> tuple[int, int, int, int]:
    height, width = spec["mask"].shape
    copies = int(spec["copies"])
    if spec["axis"] == "h":
        total_w = copies * width + (copies - 1)
        return top, left, top + height - 1, min(grid_w - 1, left + total_w - 1)
    total_h = copies * height + (copies - 1)
    return top, left, min(grid_h - 1, top + total_h - 1), left + width - 1


def _placement_score(spec: dict) -> int:
    height, width = spec["mask"].shape
    copies = int(spec["copies"])
    if spec["axis"] == "h":
        return int(height * (copies * width + copies - 1))
    return int(width * (copies * height + copies - 1))


def _sample_anchor(spec: dict, grid_h: int, grid_w: int, clip: bool) -> tuple[int, int] | None:
    height, width = spec["mask"].shape
    copies = int(spec["copies"])
    total_h = copies * height + (copies - 1)
    total_w = copies * width + (copies - 1)

    if spec["axis"] == "h":
        if grid_h < height or grid_w < width:
            return None
        row_lo, row_hi = 0, grid_h - height
        if clip:
            col_lo = max(0, grid_w - total_w + 1)
            col_hi = grid_w - width
        else:
            col_lo = 0
            col_hi = grid_w - total_w
        if row_lo > row_hi or col_lo > col_hi:
            return None
        return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)

    if grid_h < height or grid_w < width:
        return None
    col_lo, col_hi = 0, grid_w - width
    if clip:
        row_lo = max(0, grid_h - total_h + 1)
        row_hi = grid_h - height
    else:
        row_lo = 0
        row_hi = grid_h - total_h
    if row_lo > row_hi or col_lo > col_hi:
        return None
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _bbox_is_free(reserved: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    height, width = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(height - 1, r1 + 1)
    cc1 = min(width - 1, c1 + 1)
    return not bool(np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _mark_bbox(reserved: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    height, width = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(height - 1, r1 + 1)
    cc1 = min(width - 1, c1 + 1)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _build_specs() -> list[dict] | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    hole_counts = _sample_hole_counts(n_objects)
    orientations = _sample_orientations(n_objects)
    colors = random.sample(COLORS, n_objects)

    clip_indices: set[int] = set()
    clip_candidates = [i for i, holes in enumerate(hole_counts) if holes > 0]
    if clip_candidates and random.random() < 0.45:
        clip_indices.add(random.choice(clip_candidates))

    specs = []
    for i in range(n_objects):
        mask = _make_object_mask(orientations[i], hole_counts[i])
        if mask is None:
            return None
        axis = "h" if mask.shape[1] >= mask.shape[0] else "v"
        specs.append(
            {
                "color": int(colors[i]),
                "mask": mask,
                "holes": int(hole_counts[i]),
                "copies": int(hole_counts[i] + 1),
                "axis": axis,
                "want_clip": i in clip_indices,
            }
        )

    specs.sort(key=_placement_score, reverse=True)
    return specs


def _place_specs(specs: list[dict], grid_h: int, grid_w: int) -> list[dict] | None:
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict] = []

    for spec in specs:
        modes = [False]
        if spec["copies"] > 1:
            candidate_modes = []
            if spec["want_clip"]:
                candidate_modes = [True, False]
            else:
                candidate_modes = [False, True]
            modes = []
            for clip in candidate_modes:
                if _sample_anchor(spec, grid_h, grid_w, clip) is not None and clip not in modes:
                    modes.append(clip)

        done = False
        for clip in modes:
            for _ in range(MAX_PLACE_ATTEMPTS):
                anchor = _sample_anchor(spec, grid_h, grid_w, clip)
                if anchor is None:
                    break
                top, left = anchor
                bbox = _visible_bbox(spec, top, left, grid_h, grid_w)
                if not _bbox_is_free(reserved, bbox):
                    continue
                _mark_bbox(reserved, bbox)
                placed.append(
                    {
                        "spec": spec,
                        "top": int(top),
                        "left": int(left),
                        "clip": bool(clip),
                    }
                )
                done = True
                break
            if done:
                break
        if not done:
            return None

    return placed


def _render_input(placements: list[dict], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for item in placements:
        spec = item["spec"]
        mask = spec["mask"]
        color = int(spec["color"])
        top = int(item["top"])
        left = int(item["left"])
        height, width = mask.shape
        for r in range(height):
            for c in range(width):
                if mask[r, c]:
                    grid[top + r, left + c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    specs = _build_specs()
    if specs is None:
        return None

    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    placements = _place_specs(specs, grid_h, grid_w)
    if placements is None:
        return None

    grid = _render_input(placements, grid_h, grid_w)

    if not _find_objects(grid):
        return None
    if not any(item["spec"]["holes"] > 0 for item in placements):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATE_ATTEMPTS} attempts")
