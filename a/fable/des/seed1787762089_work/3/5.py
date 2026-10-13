import numpy as np


BG = 0
FRAME = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OUTWARD = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise ValueError("gap is not on bbox boundary")


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == FRAME):
        pts = np.asarray(comp, dtype=int)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 4 or w < 4:
            continue
        comp_set = set(comp)
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp):
            continue
        boundary = []
        missing = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r not in (r0, r1) and c not in (c0, c1):
                    continue
                if (r, c) in comp_set:
                    boundary.append((r, c))
                else:
                    missing.append((r, c))
        if len(missing) != 1:
            continue
        gap = missing[0]
        if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        if len(boundary) != 2 * h + 2 * w - 5:
            continue
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "pixels": boundary,
                "gap": gap,
                "side": _frame_gap_side(gap, (int(r0), int(c0), int(r1), int(c1))),
            }
        )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _ray_cells(grid: np.ndarray, frame: dict) -> list[tuple[int, int]]:
    h, w = grid.shape
    gr, gc = frame["gap"]
    dr, dc = OUTWARD[frame["side"]]
    cells: list[tuple[int, int]] = []
    r, c = gr + dr, gc + dc
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != BG:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _find_aligned_key(grid: np.ndarray, frame: dict) -> tuple[int, int, int] | None:
    gr, gc = frame["gap"]
    dr, dc = OUTWARD[frame["side"]]
    h, w = grid.shape
    r, c = gr + dr, gc + dc
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value == BG:
            r += dr
            c += dc
            continue
        if value == FRAME:
            return None
        return r, c, value
    return None


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int, int] | None], list[tuple[int, int]]]:
    frames = _extract_frames(grid)
    matches: list[tuple[int, int, int] | None] = []
    used_keys: set[tuple[int, int]] = set()
    for frame in frames:
        key = _find_aligned_key(grid, frame)
        if key is not None and (key[0], key[1]) not in used_keys:
            used_keys.add((key[0], key[1]))
            matches.append(key)
        else:
            matches.append(None)
    all_keys = [
        (int(r), int(c))
        for r, c in np.argwhere((grid != BG) & (grid != FRAME))
    ]
    unused = [pos for pos in all_keys if pos not in used_keys]
    return frames, matches, unused


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open aligned hollow rectangles with single-pixel keys.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _extract_frames(g)
    for frame in frames:
        key = _find_aligned_key(g, frame)
        r0, c0, r1, c1 = frame["bbox"]
        if key is None:
            for r, c in frame["pixels"]:
                out[r, c] = BG
            continue
        kr, kc, color = key
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        fill = out[r0 + 1 : r1, c0 + 1 : c1]
        fill[interior == BG] = color
        gr, gc = frame["gap"]
        out[gr, gc] = FRAME
        out[kr, kc] = BG
    return out


def _gap_coordinate(rng: np.random.Generator, bbox: tuple[int, int, int, int], side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side in ("top", "bottom"):
        c = int(rng.integers(c0 + 1, c1))
        return (r0, c) if side == "top" else (r1, c)
    r = int(rng.integers(r0 + 1, r1))
    return (r, c0) if side == "left" else (r, c1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[gap] = BG


def _place_frames(
    rng: np.random.Generator, h: int, w: int, matched_flags: list[bool]
) -> list[dict] | None:
    frames: list[dict] = []
    for matched in matched_flags:
        placed = False
        for _ in range(200):
            fh = int(rng.integers(4, 8))
            fw = int(rng.integers(4, 8))
            if h - fh - 2 <= 0 or w - fw - 2 <= 0:
                return None
            r0 = int(rng.integers(1, h - fh - 1))
            c0 = int(rng.integers(1, w - fw - 1))
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_overlap(bbox, other["bbox"], pad=1) for other in frames):
                continue
            side = str(rng.choice(["top", "bottom", "left", "right"]))
            gap = _gap_coordinate(rng, bbox, side)
            frames.append({"bbox": bbox, "gap": gap, "side": side, "matched": matched})
            placed = True
            break
        if not placed:
            return None
    return frames


def _isolated_bg_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG:
                continue
            ok = True
            for nr in range(max(0, r - 1), min(h, r + 2)):
                for nc in range(max(0, c - 1), min(w, c + 2)):
                    if (nr, nc) != (r, c) and int(grid[nr, nc]) != BG:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                cells.append((r, c))
    return cells


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 23))
    w = int(rng.integers(20, 27))
    num_frames = int(rng.integers(2, 5))
    num_matched = int(rng.integers(1, num_frames))
    matched_flags = [True] * num_matched + [False] * (num_frames - num_matched)
    rng.shuffle(matched_flags)

    min_keys = max(3, num_matched + 1)
    num_keys = int(rng.integers(min_keys, 7))
    num_extra_keys = num_keys - num_matched

    frames = _place_frames(rng, h, w, matched_flags)
    if frames is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for frame in frames:
        _paint_frame(grid, frame["bbox"], frame["gap"])

    scene_frames = _extract_frames(grid)
    if len(scene_frames) != num_frames:
        return None
    matched_by_bbox = {frame["bbox"]: bool(frame["matched"]) for frame in frames}

    ray_map = {frame["bbox"]: _ray_cells(grid, frame) for frame in scene_frames}
    all_ray_cells = {cell for cells in ray_map.values() for cell in cells}
    chosen_positions: dict[tuple[int, int, int, int], tuple[int, int]] = {}
    for frame in scene_frames:
        want_matched = matched_by_bbox[frame["bbox"]]
        if not want_matched:
            continue
        own_bbox = frame["bbox"]
        other_rays = set()
        for bbox, cells in ray_map.items():
            if bbox != own_bbox:
                other_rays.update(cells)
        choices = [cell for cell in ray_map[own_bbox] if cell not in other_rays]
        if not choices:
            return None
        chosen_positions[own_bbox] = choices[int(rng.integers(len(choices)))]

    used_colors = list(rng.choice(KEY_COLORS, size=num_keys, replace=False))
    rng.shuffle(used_colors)
    current_color = 0
    for frame in scene_frames:
        want_matched = matched_by_bbox[frame["bbox"]]
        if not want_matched:
            continue
        r, c = chosen_positions[frame["bbox"]]
        grid[r, c] = int(used_colors[current_color])
        current_color += 1

    safe_extra = [
        cell
        for cell in _isolated_bg_cells(grid)
        if cell not in all_ray_cells
    ]
    if len(safe_extra) < num_extra_keys:
        return None
    rng.shuffle(safe_extra)
    for i in range(num_extra_keys):
        r, c = safe_extra[i]
        grid[r, c] = int(used_colors[current_color])
        current_color += 1

    frames_found, matches, unused = _analyze_scene(grid)
    if len(frames_found) != num_frames:
        return None
    if sum(key is not None for key in matches) != num_matched:
        return None
    if len(unused) != num_extra_keys:
        return None
    if len(np.argwhere((grid != BG) & (grid != FRAME))) != num_keys:
        return None
    if not any(key is None for key in matches):
        return None
    if not unused:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
