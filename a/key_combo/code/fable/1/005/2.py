import random
from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    monochromatic: bool = True,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue
            color = int(grid[r0, c0])
            seen[r0, c0] = True
            q = deque([(r0, c0)])
            coords = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for nr, nc in neighbors4(r, c):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    if monochromatic and int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color if monochromatic else None,
                    "coords": coords_arr,
                    "bbox": bbox_coords(coords_arr),
                    "area": int(coords_arr.shape[0]),
                }
            )
    return out


def transformed_masks(mask: np.ndarray) -> list[tuple[bool, int, np.ndarray]]:
    result = []
    seen = set()
    base = mask.astype(bool)
    for flip_lr in (False, True):
        flipped = np.fliplr(base) if flip_lr else base
        for rot_k in range(4):
            cur = np.rot90(flipped, rot_k).copy()
            key = (cur.shape, cur.tobytes())
            if key in seen:
                continue
            seen.add(key)
            result.append((flip_lr, rot_k, cur))
    return result


def ray_indices(grid: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int]]:
    h, w = grid.shape
    rows = {}
    cols = {}
    for r in range(1, h - 1):
        left = int(grid[r, 0])
        right = int(grid[r, w - 1])
        if left != bg and left == right:
            rows[r] = left
    for c in range(1, w - 1):
        top = int(grid[0, c])
        bottom = int(grid[h - 1, c])
        if top != bg and top == bottom:
            cols[c] = top
    return rows, cols


def ray_footprint(shape: tuple[int, int], rows: dict[int, int], cols: dict[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if rows:
        mask[list(rows.keys()), :] = True
    if cols:
        mask[:, list(cols.keys())] = True
    return mask


def corner_score(bbox: tuple[int, int, int, int], h: int, w: int) -> int:
    r0, c0, r1, c1 = bbox
    return min(r0 + c0, r0 + (w - 1 - c1), (h - 1 - r1) + c0, (h - 1 - r1) + (w - 1 - c1))


def detect_legend(grid: np.ndarray, bg: int, ray_rows: dict[int, int], ray_cols: dict[int, int]) -> dict:
    h, w = grid.shape
    footprint = ray_footprint(grid.shape, ray_rows, ray_cols)
    candidates = []
    for comp in find_components(grid, bg=bg, monochromatic=True):
        coords = comp["coords"]
        if footprint[coords[:, 0], coords[:, 1]].any():
            continue
        bbox = comp["bbox"]
        patch = crop_bbox(grid, bbox)
        mask = patch == int(comp["color"])
        candidates.append(
            {
                "color": int(comp["color"]),
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "corner_score": corner_score(bbox, h, w),
            }
        )
    assert candidates, "expected a corner legend component"
    candidates.sort(key=lambda item: (item["corner_score"], item["bbox"][0], item["bbox"][1], item["coords"].shape[0]))
    return candidates[0]


def collect_solutions(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    ray_rows, ray_cols = ray_indices(g, bg)
    assert ray_rows and ray_cols, "expected both horizontal and vertical rays"
    footprint = ray_footprint(g.shape, ray_rows, ray_cols)
    break_mask = footprint & (g == bg)
    visible_ray_mask = footprint & (g != bg)
    assert break_mask.any(), "expected at least one ray break"

    legend = detect_legend(g, bg, ray_rows, ray_cols)
    sprite_mask = legend["mask"]
    blocked = (g != bg) & ~visible_ray_mask
    transforms = transformed_masks(sprite_mask)
    solutions = []
    seen = set()

    for flip_lr, rot_k, mask in transforms:
        coords = np.argwhere(mask)
        mh, mw = mask.shape
        for top in range(h - mh + 1):
            for left in range(w - mw + 1):
                rr = coords[:, 0] + top
                cc = coords[:, 1] + left
                if blocked[rr, cc].any():
                    continue
                placed = np.zeros((h, w), dtype=bool)
                placed[rr, cc] = True
                if visible_ray_mask[rr, cc].any():
                    continue
                if not np.array_equal(placed & footprint, break_mask):
                    continue
                key = placed.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                solutions.append(
                    {
                        "top": top,
                        "left": left,
                        "flip_lr": flip_lr,
                        "rot_k": rot_k,
                        "mask": mask,
                        "placed": placed,
                    }
                )

    return {
        "bg": bg,
        "ray_rows": ray_rows,
        "ray_cols": ray_cols,
        "footprint": footprint,
        "break_mask": break_mask,
        "visible_ray_mask": visible_ray_mask,
        "legend": legend,
        "solutions": solutions,
    }


def solve_instance(grid: np.ndarray) -> dict:
    info = collect_solutions(grid)
    assert len(info["solutions"]) == 1, f'expected unique hidden sprite, found {len(info["solutions"])}'
    info["solution"] = info["solutions"][0]
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reveal the invisible sprite that punched aligned gaps through several full-width and full-height rays.

    Input structure:
    - Black background with several one-cell-thick horizontal and vertical rays that span an entire row or column.
    - One corner contains a small monochrome legend sprite that does not touch any ray.
    - A transformed copy of that sprite is hidden elsewhere: only the ray cells covered by it disappear, leaving aligned gaps.

    Rule:
    - Detect the ray rows and columns from their surviving edge pixels, then collect all missing ray cells as the break pattern.
    - Find the unique placement and one of the eight rotate/flip poses of the legend sprite whose covered ray cells match every break and avoid every unbroken ray cell.
    - Erase the corner legend and draw the full sprite at that recovered position in the legend color.

    Key insight:
    - The target object is never visible directly; only its intersection with the rays is shown, so the legend must be matched in pose and translated to complete the unseen off-ray cells.
    """
    info = solve_instance(np.asarray(grid, dtype=int))
    out = np.asarray(grid, dtype=int).copy()
    legend = info["legend"]
    out[legend["coords"][:, 0], legend["coords"][:, 1]] = info["bg"]
    out[info["solution"]["placed"]] = legend["color"]
    return out


def random_connected_mask() -> np.ndarray:
    for _ in range(300):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        max_area = min(h * w - 2, 10)
        area = random.randint(5, max_area)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                for nr, nc in neighbors4(int(r), int(c)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True
        coords = np.argwhere(mask)
        bbox = bbox_coords(coords)
        mask = crop_bbox(mask.astype(int), bbox).astype(bool)
        coords = np.argwhere(mask)
        rows = {int(r) for r, _ in coords}
        cols = {int(c) for _, c in coords}
        if min(mask.shape) < 3:
            continue
        if len(rows) < 3 or len(cols) < 3:
            continue
        if len(transformed_masks(mask)) != 8:
            continue
        return mask
    raise RuntimeError("failed to build an asymmetric connected sprite")


def place_corner_legend(mask: np.ndarray, shape: tuple[int, int]) -> tuple[str, int, int]:
    h, w = shape
    mh, mw = mask.shape
    corner = random.choice(["tl", "tr", "bl", "br"])
    pad_r = random.randint(1, 2)
    pad_c = random.randint(1, 2)
    if corner == "tl":
        return corner, pad_r, pad_c
    if corner == "tr":
        return corner, pad_r, w - mw - pad_c
    if corner == "bl":
        return corner, h - mh - pad_r, pad_c
    return corner, h - mh - pad_r, w - mw - pad_c


def expanded_index_range(start: int, length: int, limit: int, gap: int = 1) -> set[int]:
    lo = max(0, start - gap)
    hi = min(limit - 1, start + length - 1 + gap)
    return set(range(lo, hi + 1))


def sample_hidden_layout(
    tmask: np.ndarray,
    grid_shape: tuple[int, int],
    forbidden_rows: set[int],
    forbidden_cols: set[int],
    legend_bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    h, w = grid_shape
    mh, mw = tmask.shape
    legend_r0, legend_c0, legend_r1, legend_c1 = legend_bbox
    for _ in range(300):
        top = random.randint(2, h - mh - 3)
        left = random.randint(2, w - mw - 3)
        rows = set(range(top, top + mh))
        cols = set(range(left, left + mw))
        if rows & forbidden_rows or cols & forbidden_cols:
            continue
        if not (top + mh - 1 < legend_r0 - 2 or top > legend_r1 + 2 or left + mw - 1 < legend_c0 - 2 or left > legend_c1 + 2):
            continue
        return top, left
    return None


def sample_ray_sets(
    placed: np.ndarray,
    grid_shape: tuple[int, int],
    forbidden_rows: set[int],
    forbidden_cols: set[int],
) -> tuple[set[int], set[int]] | None:
    h, w = grid_shape
    coords = np.argwhere(placed)
    occ_rows = sorted({int(r) for r, _ in coords})
    occ_cols = sorted({int(c) for _, c in coords})
    safe_rows = [r for r in range(1, h - 1) if r not in forbidden_rows]
    safe_cols = [c for c in range(1, w - 1) if c not in forbidden_cols]
    extra_row_pool = [r for r in safe_rows if r not in occ_rows]
    extra_col_pool = [c for c in safe_cols if c not in occ_cols]

    for _ in range(250):
        hit_rows = set(random.sample(occ_rows, random.randint(1, min(3, len(occ_rows)))))
        hit_cols = set(random.sample(occ_cols, random.randint(1, min(3, len(occ_cols)))))
        row_only = []
        col_only = []
        off_ray = []
        for r, c in coords:
            r = int(r)
            c = int(c)
            in_row = r in hit_rows
            in_col = c in hit_cols
            if in_row and not in_col:
                row_only.append((r, c))
            elif in_col and not in_row:
                col_only.append((r, c))
            elif not in_row and not in_col:
                off_ray.append((r, c))
        if not row_only or not col_only or not off_ray:
            continue

        extra_rows = set(random.sample(extra_row_pool, random.randint(0, min(2, len(extra_row_pool)))))
        extra_cols = set(random.sample(extra_col_pool, random.randint(0, min(2, len(extra_col_pool)))))
        ray_rows = hit_rows | extra_rows
        ray_cols = hit_cols | extra_cols
        if len(ray_rows) + len(ray_cols) < 3:
            continue
        return ray_rows, ray_cols
    return None


def build_input_from_instance(
    grid_shape: tuple[int, int],
    legend_mask: np.ndarray,
    legend_top: int,
    legend_left: int,
    placed: np.ndarray,
    ray_rows: set[int],
    ray_cols: set[int],
    legend_color: int,
    ray_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid_shape
    inp = np.zeros((h, w), dtype=int)
    for r in ray_rows:
        inp[r, :] = ray_color
    for c in ray_cols:
        inp[:, c] = ray_color

    ray_covered = placed & (ray_footprint(grid_shape, {r: ray_color for r in ray_rows}, {c: ray_color for c in ray_cols}))
    inp[ray_covered] = BG

    legend_rr, legend_cc = np.argwhere(legend_mask).T
    inp[legend_rr + legend_top, legend_cc + legend_left] = legend_color

    out = inp.copy()
    out[legend_rr + legend_top, legend_cc + legend_left] = BG
    out[placed] = legend_color
    return inp, out


def validate_generated_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    info = collect_solutions(inp)
    if len(info["solutions"]) != 1:
        return False
    info["solution"] = info["solutions"][0]
    placed = info["solution"]["placed"]
    footprint = info["footprint"]
    coords = np.argwhere(placed)
    row_hits = {(int(r), int(c)) for r, c in coords if footprint[int(r), int(c)] and int(r) in info["ray_rows"] and int(c) not in info["ray_cols"]}
    col_hits = {(int(r), int(c)) for r, c in coords if footprint[int(r), int(c)] and int(c) in info["ray_cols"] and int(r) not in info["ray_rows"]}
    off_ray = {(int(r), int(c)) for r, c in coords if not footprint[int(r), int(c)]}
    if not row_hits or not col_hits or not off_ray:
        return False
    if not np.array_equal(main(inp), expected):
        return False
    legend = info["legend"]
    if legend["coords"].shape[0] != int(legend["mask"].sum()):
        return False
    if np.array_equal(expected, inp):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    legend_mask = random_connected_mask()
    legend_color, ray_color = random.sample(PALETTE, 2)
    legend_corner, legend_top, legend_left = place_corner_legend(legend_mask, (h, w))
    legend_coords = np.argwhere(legend_mask)
    legend_bbox = (
        legend_top,
        legend_left,
        legend_top + legend_mask.shape[0] - 1,
        legend_left + legend_mask.shape[1] - 1,
    )
    forbidden_rows = expanded_index_range(legend_top, legend_mask.shape[0], h, gap=1)
    forbidden_cols = expanded_index_range(legend_left, legend_mask.shape[1], w, gap=1)

    poses = transformed_masks(legend_mask)
    random.shuffle(poses)
    for flip_lr, rot_k, tmask in poses:
        placed_pos = sample_hidden_layout(tmask, (h, w), forbidden_rows, forbidden_cols, legend_bbox)
        if placed_pos is None:
            continue
        top, left = placed_pos
        placed = np.zeros((h, w), dtype=bool)
        tcoords = np.argwhere(tmask)
        placed[tcoords[:, 0] + top, tcoords[:, 1] + left] = True
        ray_sets = sample_ray_sets(placed, (h, w), forbidden_rows, forbidden_cols)
        if ray_sets is None:
            continue
        ray_rows, ray_cols = ray_sets
        inp, expected = build_input_from_instance(
            (h, w),
            legend_mask,
            legend_top,
            legend_left,
            placed,
            ray_rows,
            ray_cols,
            legend_color,
            ray_color,
        )
        if not validate_generated_instance(inp, expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-sprite ray puzzle")
