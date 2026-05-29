# Combo variant of aa4ec2a5 (index 30)
# combo_spec: D11/0+D14/2
# Name: 对象可相互接触+相邻对象需相同洞状态

from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, find_components, get_bg


BG_COLOR = 4
BODY_COLOR = 1
SEED_COLOR = 3
OUTLINE_COLOR = 2
HOLE_COLOR = 6
HOLLOW_COLOR = 8


# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task could treat each blue 4-connected component as one object. That fails here:
#    a merged blue mass may contain several latent objects, each marked by one green seed cell.
#    In the key generated example, two seed-marked solid "C"-like shapes touch each other and jointly
#    surround a cavity. A one-pass component solver would misread that cavity as a true hole and recolor
#    the whole merged mass as hollow. The fused solver must instead split the merged foreground by seed-
#    guided geodesic growth, then test holes on the recovered objects, not on the union.
# b) Complexity calibration:
#    The generator always includes one touching solid pair whose union visually looks hollow, plus one or
#    two true hollow objects. This keeps the difference from the base task visible, while only using one
#    extra cue color (green) and two hole-status outcomes (solid vs hollow).
# c) Broken original assumptions:
#    - `find_components(... connectivity=4, monochromatic=True)` no longer returns one object per piece.
#    - Hole detection on a whole connected component can produce false positives from multi-object unions.
#    - Per-object processing is no longer purely local: after splitting, 8-neighbor-adjacent objects must
#      share one hole-status label, so a global adjacency pass is required.
# d) Placement legality:
#    The generator places prevalidated patches. Cells never overlap, every seed lies on its own object,
#    and different patches stay Chebyshev-separated so only the intended within-patch adjacencies exist.
# e) Visual observability:
#    The green seeds are human-visible separators. The touching solid pair encloses a cavity in the input
#    but stays blue in the output, while true hollow seeded objects turn teal/magenta. Across examples, a
#    human can see that "cavity in the union" is insufficient; the split-by-seed objects matter first.
# f) Visual cleanliness:
#    Only three input colors appear (yellow bg, blue body, green seed). Red outlines are still drawn only
#    on background, so the output remains readable even when objects touch.
# g) Inferability of color changes:
#    Green is structural only: each latent object has one seed, then all object cells collapse back to the
#    original blue/teal scheme after reasoning. Teal+magenta still correlate with genuine per-object holes.


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _interior_holes(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    holes: list[np.ndarray] = []
    for hole in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == sub.shape[0] - 1)
            | (hole[:, 1] == sub.shape[1] - 1)
        )
        if not bool(touches_edge):
            shifted = hole.copy()
            shifted[:, 0] += int(r0)
            shifted[:, 1] += int(c0)
            holes.append(shifted)
    return holes


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _foreground_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    return (grid == BODY_COLOR) | (grid == SEED_COLOR) | ((grid != bg) & (grid != OUTLINE_COLOR))


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_mask = (g == BODY_COLOR) | (g == SEED_COLOR)
    paint_grid = np.where(fg_mask, BODY_COLOR, bg)

    objects: list[dict] = []
    for comp in find_components(paint_grid, bg=bg, connectivity=4, monochromatic=False):
        comp_coords = comp["coords"]
        comp_mask = _coords_mask(g.shape, comp_coords)
        seed_coords = np.argwhere(comp_mask & (g == SEED_COLOR))
        if len(seed_coords) == 0:
            raise ValueError("every foreground component must contain at least one seed")

        labels = np.full(g.shape, -1, dtype=int)
        queue: deque[tuple[int, int, int]] = deque()
        seed_list = sorted((int(r), int(c)) for r, c in seed_coords)
        for idx, (r, c) in enumerate(seed_list):
            labels[r, c] = idx
            queue.append((r, c, idx))

        while queue:
            r, c, idx = queue.popleft()
            for nr, nc in _neighbors4(g.shape, r, c):
                if comp_mask[nr, nc] and labels[nr, nc] == -1:
                    labels[nr, nc] = idx
                    queue.append((nr, nc, idx))

        for idx, seed in enumerate(seed_list):
            coords = np.argwhere(labels == idx)
            if len(coords) == 0:
                raise ValueError("seed-based segmentation produced an empty object")
            mask = _coords_mask(g.shape, coords)
            holes = _interior_holes(mask, coords)
            objects.append(
                {
                    "coords": coords,
                    "mask": mask,
                    "seed": seed,
                    "holes": holes,
                    "raw_hollow": bool(holes),
                }
            )
    return objects


def _touches8(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    rows, cols = np.where(mask_a)
    h, w = mask_a.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            if np.any(mask_b[rr[ok], cc[ok]]):
                return True
    return False


def _consistent_hollow_flags(objects: list[dict]) -> list[bool]:
    n = len(objects)
    graph = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _touches8(objects[i]["mask"], objects[j]["mask"]):
                graph[i].add(j)
                graph[j].add(i)

    final_flags = [False] * n
    seen = [False] * n
    for start in range(n):
        if seen[start]:
            continue
        queue = [start]
        seen[start] = True
        group = []
        while queue:
            cur = queue.pop()
            group.append(cur)
            for nxt in graph[cur]:
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)

        shared_flag = any(objects[idx]["raw_hollow"] for idx in group)
        for idx in group:
            final_flags[idx] = shared_flag
    return final_flags


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split seed-marked touching foreground into latent objects, then outline each recovered object.

    Input structure:
    - A yellow background contains blue object cells, with exactly one green seed cell embedded in
      each latent object.
    - Several latent objects may touch in 4-connectivity, so one merged foreground component can
      hide multiple seed-defined objects; some merged solids jointly surround a cavity.

    Rule:
    - First partition every merged foreground component by geodesic growth from its green seeds.
    - Detect enclosed 4-connected holes on each recovered object, then force every 8-neighbor-
      adjacent object group to share one hole status.
    - Recolor hollow groups to teal (8), fill their genuine holes with magenta (6), and draw a
      one-cell 8-neighborhood red outline around each recovered object.

    Key insight:
    - The visible cavity of a touching union is not enough to declare a hole. The seeds define the
      true object split first; only the split objects participate in hole reasoning and in the
      adjacency-consistency pass.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = _extract_objects(g)
    final_hollow = _consistent_hollow_flags(objects)

    out = np.full_like(g, bg)
    outlines: list[np.ndarray] = []
    for obj, is_hollow in zip(objects, final_hollow):
        mask = obj["mask"]
        solid = mask.copy()
        if is_hollow:
            out[mask] = HOLLOW_COLOR
            for hole in obj["holes"]:
                out[hole[:, 0], hole[:, 1]] = HOLE_COLOR
                solid[hole[:, 0], hole[:, 1]] = True
        else:
            out[mask] = BODY_COLOR
        outlines.append(_outline_mask(solid))

    for outline in outlines:
        out[outline & (out == bg)] = OUTLINE_COLOR
    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 28)
GRID_W_RANGE = (18, 27)
BORDER_MIN = 1
PATCH_CLEARANCE = 3
PLACEMENT_RETRIES = 200
GEN_OUTER_RETRIES = 250


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _transform_patch(objects: list[dict], k: int, flip_lr: bool) -> list[dict]:
    transformed = []
    for obj in objects:
        mask = np.rot90(obj["mask"], k)
        seed_mask = np.zeros_like(obj["mask"], dtype=bool)
        sr, sc = obj["seed"]
        seed_mask[sr, sc] = True
        seed_mask = np.rot90(seed_mask, k)
        if flip_lr:
            mask = np.fliplr(mask)
            seed_mask = np.fliplr(seed_mask)
        seed = tuple(map(int, np.argwhere(seed_mask)[0]))
        transformed.append({"mask": mask, "seed": seed})
    return transformed


def _make_deceptive_solid_pair() -> list[dict]:
    wall = 2
    hole_h = random.randint(2, 4)
    hole_w = random.randint(2, 4)
    split = random.randint(1, hole_w - 1)
    h = hole_h + 2 * wall
    w = 2 * wall + hole_w

    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)

    left[:, :wall] = True
    left[:wall, : wall + split] = True
    left[h - wall :, : wall + split] = True

    right[:, wall + hole_w :] = True
    right[:wall, wall + split :] = True
    right[h - wall :, wall + split :] = True

    objects = [
        {"mask": left, "seed": (wall + hole_h // 2, wall // 2)},
        {"mask": right, "seed": (wall + hole_h // 2, w - wall // 2 - 1)},
    ]
    return _transform_patch(objects, random.randint(0, 3), random.random() < 0.5)


def _make_frame_object() -> dict:
    wall = random.choice([1, 1, 2])
    hole_h = random.randint(2, 4)
    hole_w = random.randint(2, 4)
    h = hole_h + 2 * wall
    w = hole_w + 2 * wall
    mask = np.ones((h, w), dtype=bool)
    mask[wall : wall + hole_h, wall : wall + hole_w] = False
    seed = (wall + hole_h // 2, wall // 2)
    return _transform_patch([{"mask": mask, "seed": seed}], random.randint(0, 3), random.random() < 0.5)[0]


def _make_diagonal_hollow_pair() -> list[dict]:
    obj_a = _make_frame_object()
    obj_b = _make_frame_object()
    ah, aw = obj_a["mask"].shape
    bh, bw = obj_b["mask"].shape
    patch_h = ah + bh
    patch_w = aw + bw

    mask_a = np.zeros((patch_h, patch_w), dtype=bool)
    mask_b = np.zeros((patch_h, patch_w), dtype=bool)
    mask_a[:ah, :aw] = obj_a["mask"]
    mask_b[ah:, aw:] = obj_b["mask"]

    seed_a = obj_a["seed"]
    seed_b = (ah + obj_b["seed"][0], aw + obj_b["seed"][1])
    objects = [{"mask": mask_a, "seed": seed_a}, {"mask": mask_b, "seed": seed_b}]
    return _transform_patch(objects, random.randint(0, 3), random.random() < 0.5)


def _union_mask(objects: list[dict]) -> np.ndarray:
    union = np.zeros_like(objects[0]["mask"], dtype=bool)
    for obj in objects:
        union |= obj["mask"]
    return union


def _coords_with_offset(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += int(top)
    coords[:, 1] += int(left)
    return coords


def _chebyshev_conflict(mask_a: np.ndarray, top_a: int, left_a: int, mask_b: np.ndarray, top_b: int, left_b: int) -> bool:
    coords_a = _coords_with_offset(mask_a, top_a, left_a)
    coords_b = _coords_with_offset(mask_b, top_b, left_b)
    for ra, ca in coords_a:
        if np.min(np.maximum(np.abs(coords_b[:, 0] - ra), np.abs(coords_b[:, 1] - ca))) < PATCH_CLEARANCE:
            return True
    return False


def _place_patches(height: int, width: int, patches: list[list[dict]]) -> list[tuple[int, int, list[dict]]]:
    unions = [_union_mask(patch) for patch in patches]
    order = sorted(range(len(patches)), key=lambda idx: -int(unions[idx].sum()))
    placed: list[tuple[int, int, list[dict]]] = []
    placed_unions: list[tuple[int, int, np.ndarray]] = []

    for idx in order:
        union = unions[idx]
        ph, pw = union.shape
        if ph > height - 2 * BORDER_MIN or pw > width - 2 * BORDER_MIN:
            raise ValueError("patch too large for the current canvas")

        success = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(BORDER_MIN, height - ph - BORDER_MIN)
            left = random.randint(BORDER_MIN, width - pw - BORDER_MIN)

            ok = True
            for old_top, old_left, old_union in placed_unions:
                if _chebyshev_conflict(union, top, left, old_union, old_top, old_left):
                    ok = False
                    break
            if ok:
                placed.append((top, left, patches[idx]))
                placed_unions.append((top, left, union))
                success = True
                break
        if not success:
            raise ValueError("failed to place patch")
    return placed


def _render_input(height: int, width: int, placed: list[tuple[int, int, list[dict]]]) -> tuple[np.ndarray, dict]:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    intended_by_seed: dict[tuple[int, int], dict] = {}
    for top, left, patch in placed:
        for obj in patch:
            mask = obj["mask"]
            coords = _coords_with_offset(mask, top, left)
            grid[coords[:, 0], coords[:, 1]] = BODY_COLOR
            seed = (top + obj["seed"][0], left + obj["seed"][1])
            grid[seed] = SEED_COLOR
            intended_by_seed[seed] = {"coords": coords}
    return grid, intended_by_seed


def _compare_with_intended(grid: np.ndarray, intended_by_seed: dict[tuple[int, int], dict]) -> bool:
    try:
        recovered = _extract_objects(grid)
    except Exception:
        return False
    if len(recovered) != len(intended_by_seed):
        return False

    for obj in recovered:
        seed = obj["seed"]
        if seed not in intended_by_seed:
            return False
        expected = intended_by_seed[seed]["coords"]
        got = obj["coords"]
        if expected.shape != got.shape:
            return False
        if not np.array_equal(expected[np.lexsort((expected[:, 1], expected[:, 0]))], got[np.lexsort((got[:, 1], got[:, 0]))]):
            return False
    return True


def _has_illusory_touching_cavity(grid: np.ndarray) -> bool:
    objects = _extract_objects(grid)
    raw_hollow = {obj["seed"]: obj["raw_hollow"] for obj in objects}
    by_seed = {obj["seed"]: obj for obj in objects}
    seeds = [obj["seed"] for obj in objects]

    graph = {seed: set() for seed in seeds}
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _touches8(objects[i]["mask"], objects[j]["mask"]):
                graph[objects[i]["seed"]].add(objects[j]["seed"])
                graph[objects[j]["seed"]].add(objects[i]["seed"])

    seen = set()
    for seed in seeds:
        if seed in seen:
            continue
        stack = [seed]
        seen.add(seed)
        group = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(group) < 2 or any(raw_hollow[s] for s in group):
            continue

        union = np.zeros_like(grid, dtype=bool)
        for s in group:
            union |= by_seed[s]["mask"]
        union_coords = np.argwhere(union)
        if _interior_holes(union, union_coords):
            return True
    return False


def _gen_attempt() -> np.ndarray:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    patches = [_make_deceptive_solid_pair()]
    if random.random() < 0.65:
        patches.append(_make_diagonal_hollow_pair())
    else:
        patches.append([_make_frame_object()])
        if random.random() < 0.5:
            patches.append([_make_frame_object()])

    total_objects = sum(len(patch) for patch in patches)
    hollow_candidates = 0
    for patch in patches[1:]:
        hollow_candidates += len(patch)
    if total_objects < 3 or total_objects > 4:
        raise ValueError("unexpected object count")
    if hollow_candidates < 1 or hollow_candidates > 2:
        raise ValueError("unexpected hollow count")

    placed = _place_patches(height, width, patches)
    grid, intended_by_seed = _render_input(height, width, placed)
    if not _compare_with_intended(grid, intended_by_seed):
        raise ValueError("seed split did not recover intended objects")
    if not _has_illusory_touching_cavity(grid):
        raise ValueError("fusion-defining fake cavity missing")

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        try:
            return _gen_attempt()
        except Exception:
            continue
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
