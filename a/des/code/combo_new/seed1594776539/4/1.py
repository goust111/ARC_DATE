import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_BG = 0
GEN_RETRIES = 400
SHAPE_RETRIES = 120
PLACE_RETRIES = 250
FRAME_COUNT_CHOICES = [1, 2, 3]
FRAME_COUNT_WEIGHTS = [1, 3, 2]
INTERIOR_SIDE_RANGE = (4, 7)
CANVAS_MIN_BY_COUNT = {1: 14, 2: 18, 3: 22}


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def bbox_of_coords(coords: np.ndarray) -> Tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def mask_from_coords(coords: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def components_mask(mask: np.ndarray) -> List[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def monochrome_components(grid: np.ndarray, bg: int) -> List[Dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[Dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            pts: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            bbox = bbox_of_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask_from_coords(coords, bbox),
                    "area": int(len(pts)),
                }
            )
    return comps


def is_connected(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    return len(components_mask(mask)) == 1


def mask_signature(mask: np.ndarray) -> Tuple[int, int, bytes]:
    arr = np.asarray(mask, dtype=np.uint8)
    return (int(arr.shape[0]), int(arr.shape[1]), arr.tobytes())


def canonical_signature(mask: np.ndarray) -> Tuple[int, int, bytes]:
    return min(mask_signature(np.rot90(mask, k)) for k in range(4))


def is_frame_component(comp: Dict[str, object], grid: np.ndarray, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]  # type: ignore[assignment]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    mask = comp["mask"]  # type: ignore[assignment]
    if mask.shape != border.shape or not np.array_equal(mask, border):
        return False
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    return bool(np.any(inner != bg))


def inside_box(coords: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(
        np.all((coords[:, 0] >= r0) & (coords[:, 0] <= r1) & (coords[:, 1] >= c0) & (coords[:, 1] <= c1))
    )


def solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = monochrome_components(g, bg)

    frame_comps = [comp for comp in comps if is_frame_component(comp, g, bg)]
    frame_ids = {id(comp) for comp in frame_comps}
    frame_comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))  # type: ignore[index]
    assert frame_comps, "expected at least one fenced rectangle"

    frames: List[Dict[str, object]] = []
    inner_boxes: List[Tuple[int, int, int, int]] = []
    for comp in frame_comps:
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[assignment]
        inner_box = (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1))
        inner_boxes.append(inner_box)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        key_mask = interior != bg
        hole_mask = ~key_mask
        assert bool(key_mask.any()) and bool(hole_mask.any()), "frame interior must contain key and hole"
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "hole_mask": hole_mask,
            }
        )

    candidates: List[Dict[str, object]] = []
    for comp in comps:
        if id(comp) in frame_ids:
            continue
        coords = comp["coords"]  # type: ignore[assignment]
        if any(inside_box(coords, box) for box in inner_boxes):
            continue
        candidates.append(comp)
    candidates.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))  # type: ignore[index]
    assert candidates, "expected outside candidate locks"

    options: List[List[Tuple[int, int]]] = []
    for frame in frames:
        matches: List[Tuple[int, int]] = []
        hole_mask = frame["hole_mask"]  # type: ignore[assignment]
        for ci, cand in enumerate(candidates):
            base_mask = cand["mask"]  # type: ignore[assignment]
            for rot in range(4):
                rotated = np.rot90(base_mask, rot)
                if rotated.shape == hole_mask.shape and np.array_equal(rotated, hole_mask):
                    matches.append((ci, rot))
        assert matches, "each frame must have at least one matching lock"
        options.append(sorted(set(matches)))

    order = sorted(
        range(len(frames)),
        key=lambda i: (len({cand_idx for cand_idx, _ in options[i]}), frames[i]["bbox"][0], frames[i]["bbox"][1]),
    )
    assignment: Dict[int, Tuple[int, int]] = {}
    used: set[int] = set()

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        fi = order[pos]
        for cand_idx, rot in sorted(
            options[fi],
            key=lambda item: (
                candidates[item[0]]["bbox"][0],  # type: ignore[index]
                candidates[item[0]]["bbox"][1],  # type: ignore[index]
                item[1],
            ),
        ):
            if cand_idx in used:
                continue
            used.add(cand_idx)
            assignment[fi] = (cand_idx, rot)
            if search(pos + 1):
                return True
            used.remove(cand_idx)
            del assignment[fi]
        return False

    assert search(0), "failed to assign distinct locks to all frames"

    out = g.copy()
    for fi, (cand_idx, _rot) in assignment.items():
        coords = candidates[cand_idx]["coords"]  # type: ignore[index]
        out[coords[:, 0], coords[:, 1]] = bg

    for fi, (cand_idx, _rot) in assignment.items():
        fr0, fc0, fr1, fc1 = frames[fi]["bbox"]  # type: ignore[assignment]
        hole_mask = frames[fi]["hole_mask"]  # type: ignore[assignment]
        color = int(candidates[cand_idx]["color"])
        region = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
        region[hole_mask] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced cavity with the outside lock whose rotated shape matches the key's negative space.

    Input structure:
    - The background is black and the grid contains one or more monochrome hollow rectangular fences.
    - Each fence encloses a smaller connected key shape that stays at least one cell away from the fence border.
    - Outside the fences, several disconnected monochrome candidate lock shapes appear on the black background.

    Rule:
    - For each fence, inspect the full interior rectangle obtained by removing the border and treat the existing key as occupied cells.
    - The remaining black cells form the hole shape that must be filled.
    - Find the outside candidate whose shape matches that hole after a rotation by 0, 90, 180, or 270 degrees, erase that candidate from the exterior, and copy its color into the hole.
    - Keep every fence and every original key unchanged.

    Key insight:
    - The needed lock is not defined by color but by the fenced key's complementary negative-space mask, compared against rotated outside pieces.
    """
    return solve_grid(grid)


def random_key_mask(ih: int, iw: int, rng: random.Random) -> Optional[np.ndarray]:
    inner_cells = [(r, c) for r in range(1, ih - 1) for c in range(1, iw - 1)]
    inner_area = len(inner_cells)
    if inner_area < 2:
        return None
    min_area = 2
    max_area = max(min_area, min(inner_area - 1, inner_area // 2 + 1))
    for _ in range(SHAPE_RETRIES):
        target = rng.randint(min_area, max_area)
        key = {rng.choice(inner_cells)}
        while len(key) < target:
            frontier = [
                cell
                for cell in inner_cells
                if cell not in key and any((cell[0] + dr, cell[1] + dc) in key for dr, dc in DIR4)
            ]
            rng.shuffle(frontier)
            placed = False
            for cell in frontier:
                trial = key | {cell}
                mask = np.zeros((ih, iw), dtype=bool)
                for r, c in trial:
                    mask[r, c] = True
                if is_connected(~mask):
                    key = trial
                    placed = True
                    break
            if not placed:
                break
        if len(key) != target:
            continue
        key_mask = np.zeros((ih, iw), dtype=bool)
        for r, c in key:
            key_mask[r, c] = True
        hole_mask = ~key_mask
        if not is_connected(key_mask) or not is_connected(hole_mask):
            continue
        if len({mask_signature(np.rot90(hole_mask, k)) for k in range(4)}) < 2:
            continue
        return key_mask
    return None


def choose_shapes(count: int, rng: random.Random) -> Optional[List[Dict[str, object]]]:
    shapes: List[Dict[str, object]] = []
    seen_signatures = set()
    for _ in range(count):
        found = None
        for _ in range(SHAPE_RETRIES):
            ih = rng.randint(*INTERIOR_SIDE_RANGE)
            iw = rng.randint(*INTERIOR_SIDE_RANGE)
            key_mask = random_key_mask(ih, iw, rng)
            if key_mask is None:
                continue
            hole_mask = ~key_mask
            sig = canonical_signature(hole_mask)
            if sig in seen_signatures:
                continue
            found = {"key_mask": key_mask, "hole_mask": hole_mask, "interior_shape": (ih, iw)}
            seen_signatures.add(sig)
            break
        if found is None:
            return None
        shapes.append(found)
    return shapes


def try_place_bbox(occupied: np.ndarray, height: int, width: int, rng: random.Random) -> Optional[Tuple[int, int]]:
    h, w = occupied.shape
    if height > h or width > w:
        return None
    for _ in range(PLACE_RETRIES):
        r0 = rng.randint(0, h - height)
        c0 = rng.randint(0, w - width)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + height + 1)
        cc1 = min(w, c0 + width + 1)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        occupied[r0 : r0 + height, c0 : c0 + width] = True
        return (r0, c0)
    return None


def build_expected(
    grid: np.ndarray,
    frames: Sequence[Dict[str, object]],
    candidates: Sequence[Dict[str, object]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for cand in candidates:
        coords = cand["coords"]  # type: ignore[assignment]
        out[coords[:, 0], coords[:, 1]] = GEN_BG
    for frame, cand in zip(frames, candidates):
        r0, c0 = frame["top_left"]  # type: ignore[assignment]
        hole_mask = frame["hole_mask"]  # type: ignore[assignment]
        ih, iw = hole_mask.shape
        region = out[r0 + 1 : r0 + 1 + ih, c0 + 1 : c0 + 1 + iw]
        region[hole_mask] = int(cand["color"])
    return out


def gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    count = rng.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    shape_specs = choose_shapes(count, rng)
    if shape_specs is None:
        return None

    color_pool = list(range(1, 10))
    rng.shuffle(color_pool)
    frame_color = color_pool[0]
    key_colors = color_pool[1 : 1 + count]
    lock_colors = color_pool[1 + count : 1 + 2 * count]

    candidate_masks = []
    for spec in shape_specs:
        rot = rng.randint(0, 3)
        candidate_masks.append(np.rot90(spec["hole_mask"], rot))
        spec["rotation"] = rot

    frame_boxes = []
    object_sizes = []
    for spec, cand_mask in zip(shape_specs, candidate_masks):
        ih, iw = spec["interior_shape"]  # type: ignore[assignment]
        frame_boxes.append((ih + 2, iw + 2))
        object_sizes.append((ih + 2, iw + 2))
        object_sizes.append(cand_mask.shape)

    max_h = max(h for h, _ in object_sizes)
    max_w = max(w for _, w in object_sizes)
    min_side = max(CANVAS_MIN_BY_COUNT[count], max_h + 2, max_w + 2)
    if min_side > 30:
        return None

    canvas_h = rng.randint(min_side, 30)
    canvas_w = rng.randint(min_side, 30)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)

    frame_positions: List[Tuple[int, int]] = []
    for fh, fw in frame_boxes:
        pos = try_place_bbox(occupied, fh, fw, rng)
        if pos is None:
            return None
        frame_positions.append(pos)

    candidate_positions: List[Tuple[int, int]] = []
    for cand_mask in candidate_masks:
        ch, cw = cand_mask.shape
        pos = try_place_bbox(occupied, ch, cw, rng)
        if pos is None:
            return None
        candidate_positions.append(pos)

    grid = np.full((canvas_h, canvas_w), GEN_BG, dtype=int)
    frames_meta: List[Dict[str, object]] = []
    candidates_meta: List[Dict[str, object]] = []

    for idx, spec in enumerate(shape_specs):
        top, left = frame_positions[idx]
        key_mask = spec["key_mask"]  # type: ignore[assignment]
        hole_mask = spec["hole_mask"]  # type: ignore[assignment]
        ih, iw = key_mask.shape
        bottom = top + ih + 1
        right = left + iw + 1

        grid[top, left : right + 1] = frame_color
        grid[bottom, left : right + 1] = frame_color
        grid[top : bottom + 1, left] = frame_color
        grid[top : bottom + 1, right] = frame_color

        interior = grid[top + 1 : bottom, left + 1 : right]
        interior[key_mask] = key_colors[idx]
        frames_meta.append(
            {
                "top_left": (top, left),
                "hole_mask": hole_mask,
                "key_color": key_colors[idx],
            }
        )

    for idx, cand_mask in enumerate(candidate_masks):
        top, left = candidate_positions[idx]
        ch, cw = cand_mask.shape
        grid[top : top + ch, left : left + cw][cand_mask] = lock_colors[idx]
        coords = np.argwhere(cand_mask)
        coords[:, 0] += top
        coords[:, 1] += left
        candidates_meta.append(
            {
                "color": lock_colors[idx],
                "coords": coords,
            }
        )

    expected = build_expected(grid, frames_meta, candidates_meta)
    actual = solve_grid(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
