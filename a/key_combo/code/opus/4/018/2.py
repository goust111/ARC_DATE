from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_shared_arc_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load arc_utils from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_arc_utils = _load_arc_utils()
Color = _arc_utils.Color
bbox_coords = _arc_utils.bbox_coords
components_mask = _arc_utils.components_mask
find_components = _arc_utils.find_components
get_bg = _arc_utils.get_bg
paste = _arc_utils.paste


BG = Color.BLACK
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE]
OBJECT_COUNT_OPTIONS = (4, 5)
AREA_RANGE = (2, 5)
CANVAS_HEIGHT_RANGE = (12, 17)
CANVAS_WIDTH_RANGE = (11, 17)
CHAIN_PATTERNS = {
    4: [(2, 2), (3, 1)],
    5: [(3, 2), (2, 2, 1), (4, 1)],
}
GENERATION_RETRIES = 400


def _coords_from_patch(patch: np.ndarray) -> np.ndarray:
    return np.argwhere(patch != BG).astype(int)


def _build_patch(coords: set[tuple[int, int]], color: int) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    patch[arr[:, 0] - r0, arr[:, 1] - c0] = int(color)
    return patch


def _random_blob(color: int) -> np.ndarray:
    area = random.randint(*AREA_RANGE)
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < area:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        cells.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nxt = (r + dr, c + dc)
            if nxt not in cells:
                frontier.add(nxt)

    patch = _build_patch(cells, color)
    patch = np.rot90(patch, random.randrange(4))
    if random.random() < 0.5:
        patch = np.flipud(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return np.asarray(patch, dtype=int)


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([int(dr), int(dc)], dtype=int)


def _footprint(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, cell)) for cell in coords}


def _touches_any(coords: np.ndarray, footprint: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if (rr, cc) in footprint:
            return True
    return False


def _support_top(child_local: np.ndarray, left: int, parent_global: np.ndarray) -> int | None:
    parent_top_by_col: dict[int, int] = {}
    for r, c in parent_global:
        rr = int(r)
        cc = int(c)
        parent_top_by_col[cc] = min(rr, parent_top_by_col.get(cc, rr))

    child_bottom_by_col: dict[int, int] = {}
    for r, c in child_local:
        rr = int(r)
        cc = int(c) + int(left)
        child_bottom_by_col[cc] = max(rr, child_bottom_by_col.get(cc, rr))

    shared = sorted(parent_top_by_col.keys() & child_bottom_by_col.keys())
    if not shared:
        return None
    return min(parent_top_by_col[col] - 1 - child_bottom_by_col[col] for col in shared)


def _build_chain(patches: list[np.ndarray]) -> list[dict] | None:
    placed: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    first = patches[0]
    first_coords = _coords_from_patch(first)
    placed.append(
        {
            "patch": first,
            "color": int(first[first != BG][0]),
            "top": 0,
            "left": 0,
            "coords": first_coords,
        }
    )
    occupied |= _footprint(first_coords)

    for patch in patches[1:]:
        child_local = _coords_from_patch(patch)
        _parent_r0, parent_c0, _parent_r1, parent_c1 = bbox_coords(placed[-1]["coords"])
        _child_h, child_w = patch.shape
        candidates: list[tuple[int, int, np.ndarray]] = []
        for left in range(parent_c0 - child_w + 1, parent_c1 + 1):
            top = _support_top(child_local, left, placed[-1]["coords"])
            if top is None:
                continue
            child_global = _translate(child_local, top, left)
            if _touches_any(child_global, occupied):
                continue
            candidates.append((top, left, child_global))

        if not candidates:
            return None

        top, left, child_global = random.choice(candidates)
        placed.append(
            {
                "patch": patch,
                "color": int(patch[patch != BG][0]),
                "top": top,
                "left": left,
                "coords": child_global,
            }
        )
        occupied |= _footprint(child_global)

    min_r = min(int(obj["coords"][:, 0].min()) for obj in placed)
    min_c = min(int(obj["coords"][:, 1].min()) for obj in placed)
    for obj in placed:
        obj["coords"] = _translate(obj["coords"], -min_r, -min_c)
        obj["top"] = int(obj["coords"][:, 0].min())
        obj["left"] = int(obj["coords"][:, 1].min())
    return placed


def _occupancy(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    occ = np.full(shape, -1, dtype=int)
    for idx, obj in enumerate(objects):
        coords = obj["coords"]
        assert np.all(occ[coords[:, 0], coords[:, 1]] == -1), "objects must not overlap"
        occ[coords[:, 0], coords[:, 1]] = idx
    return occ


def _objects_that_can_fall(objects: list[dict], occ: np.ndarray) -> set[int]:
    h, _ = occ.shape
    movable = set(range(len(objects)))
    changed = True
    while changed:
        changed = False
        for idx in list(movable):
            coords = objects[idx]["coords"]
            shifted_rows = coords[:, 0] + 1
            if np.any(shifted_rows >= h):
                movable.remove(idx)
                changed = True
                continue

            labels = occ[shifted_rows, coords[:, 1]]
            blockers = {int(label) for label in labels if int(label) not in {-1, idx}}
            if any(blocker not in movable for blocker in blockers):
                movable.remove(idx)
                changed = True
    return movable


def _simulate_gravity(objects: list[dict], shape: tuple[int, int]) -> list[dict]:
    settled = []
    for obj in objects:
        settled.append(
            {
                "color": int(obj["color"]),
                "coords": np.asarray(obj["coords"], dtype=int).copy(),
                "initial_top": int(obj["initial_top"]),
            }
        )

    h, _ = shape
    for _step in range(h):
        occ = _occupancy(shape, settled)
        movable = _objects_that_can_fall(settled, occ)
        if not movable:
            break
        for idx in movable:
            settled[idx]["coords"][:, 0] += 1
    else:
        raise AssertionError("gravity did not settle within the grid height")

    for obj in settled:
        obj["final_top"] = int(obj["coords"][:, 0].min())
        obj["fall"] = int(obj["final_top"] - obj["initial_top"])
    return settled


def _rings_from_falls(objects: list[dict]) -> np.ndarray:
    falls = [int(obj["fall"]) for obj in objects]
    assert len(falls) == len(set(falls)), "fall distances must be unique"
    ordered = sorted(objects, key=lambda obj: int(obj["fall"]), reverse=True)
    layers = len(ordered)
    side = 2 * layers - (layers % 2)

    out = np.full((side, side), int(ordered[0]["color"]), dtype=int)
    for layer, obj in enumerate(ordered[1:], start=1):
        out[layer : side - layer, layer : side - layer] = int(obj["color"])
    return out


def _extract_objects(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert comps, "expected at least one foreground object"
    objects = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        objects.append(
            {
                "color": int(comp["color"]),
                "coords": coords.copy(),
                "initial_top": int(comp["bbox"][0]),
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate downward settling of the colored objects, then encode their fall distances as concentric square layers.

    Input structure:
    - The background is black.
    - The foreground contains 4 or 5 separated monochrome 4-connected objects in distinct colors.
    - Objects are small rigid shapes whose column ranges may overlap, so lower settled objects can block upper ones.

    Rule:
    - Apply synchronous downward gravity: during each step, every object that is not blocked by the bottom edge or by an already immovable object moves down by one row as a rigid body.
    - Record each object's fall distance as the change in its bounding-box top row from input to settled position.
    - Sort the object colors by decreasing fall distance, then fill one visible concentric square layer per object from outside to inside.
    - Use the smallest square that can display all layers: side `2k` for `k` even and `2k-1` for `k` odd.

    Key insight:
    - The output order is determined by interaction-aware fall distance, not by an object's initial height, final height, shape, or color identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    objects = _extract_objects(g)
    settled = _simulate_gravity(objects, g.shape)
    return _rings_from_falls(settled)


def _place_chains(chains: list[list[dict]]) -> tuple[int, int, list[dict]] | None:
    widths = []
    heights = []
    for chain in chains:
        max_r = max(int(obj["coords"][:, 0].max()) for obj in chain)
        max_c = max(int(obj["coords"][:, 1].max()) for obj in chain)
        heights.append(max_r + 1)
        widths.append(max_c + 1)

    base_gaps = [1] * max(0, len(chains) - 1)
    base_total = sum(widths) + sum(base_gaps)
    width_min = max(CANVAS_WIDTH_RANGE[0], base_total)
    width_max = CANVAS_WIDTH_RANGE[1]
    if width_min > width_max:
        return None

    k = sum(len(chain) for chain in chains)
    height_min = max(max(heights) + k - 1, CANVAS_HEIGHT_RANGE[0])
    height_max = CANVAS_HEIGHT_RANGE[1]
    if height_min > height_max:
        return None

    width = random.randint(width_min, width_max)
    height = random.randint(height_min, height_max)
    gaps = base_gaps[:]
    extra = width - base_total
    left_margin = random.randint(0, extra)
    extra -= left_margin
    for _ in range(extra):
        if gaps and random.random() < 0.75:
            gaps[random.randrange(len(gaps))] += 1
    cursor = left_margin

    final_objects: list[dict] = []
    for chain_idx, chain in enumerate(chains):
        chain_height = heights[chain_idx]
        top_offset = height - chain_height
        for depth, obj in enumerate(chain):
            coords = _translate(obj["coords"], top_offset, cursor)
            final_objects.append(
                {
                    "chain": chain_idx,
                    "depth": depth,
                    "color": int(obj["color"]),
                    "coords": coords,
                    "final_top": int(coords[:, 0].min()),
                }
            )
        cursor += widths[chain_idx]
        if chain_idx < len(chains) - 1:
            cursor += gaps[chain_idx]

    return height, width, final_objects


def _assign_lifts(final_objects: list[dict]) -> dict[int, int] | None:
    parents: dict[int, int | None] = {}
    by_chain: dict[int, list[int]] = {}
    for idx, obj in enumerate(final_objects):
        by_chain.setdefault(int(obj["chain"]), []).append(idx)
    for chain in by_chain.values():
        chain.sort(key=lambda idx: int(final_objects[idx]["depth"]))
        for depth, idx in enumerate(chain):
            parents[idx] = chain[depth - 1] if depth else None

    order = []
    chain_ids = list(by_chain)
    random.shuffle(chain_ids)
    for chain_id in chain_ids:
        order.extend(by_chain[chain_id])

    lifts: dict[int, int] = {}
    used: set[int] = set()

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        parent = parents[idx]
        lo = 0 if parent is None else lifts[parent] + 1
        hi = int(final_objects[idx]["final_top"])
        candidates = [lift for lift in range(lo, hi + 1) if lift not in used]
        random.shuffle(candidates)
        for lift in candidates:
            lifts[idx] = lift
            used.add(lift)
            if rec(pos + 1):
                return True
            used.remove(lift)
            del lifts[idx]
        return False

    return lifts if rec(0) else None


def _build_input_grid(
    height: int,
    width: int,
    final_objects: list[dict],
    lifts: dict[int, int],
) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    objects = []
    for idx, obj in enumerate(final_objects):
        coords = _translate(obj["coords"], -lifts[idx], 0)
        if coords[:, 0].min() < 0 or coords[:, 1].min() < 0:
            return None
        if coords[:, 0].max() >= height or coords[:, 1].max() >= width:
            return None
        if np.any(grid[coords[:, 0], coords[:, 1]] != BG):
            return None
        patch = np.full(
            (
                int(coords[:, 0].max()) - int(coords[:, 0].min()) + 1,
                int(coords[:, 1].max()) - int(coords[:, 1].min()) + 1,
            ),
            BG,
            dtype=int,
        )
        patch[coords[:, 0] - coords[:, 0].min(), coords[:, 1] - coords[:, 1].min()] = int(obj["color"])
        paste(grid, patch, int(coords[:, 0].min()), int(coords[:, 1].min()), transparent=BG)
        objects.append(
            {
                "color": int(obj["color"]),
                "coords": coords,
                "initial_top": int(coords[:, 0].min()),
            }
        )

    fg_components = components_mask(grid != BG, connectivity=4)
    if len(fg_components) != len(final_objects):
        return None

    simulated = _simulate_gravity(objects, grid.shape)
    by_color = {int(obj["color"]): obj for obj in simulated}
    for obj in final_objects:
        settled = by_color[int(obj["color"])]
        if _footprint(settled["coords"]) != _footprint(obj["coords"]):
            return None
    return grid


def _fall_less_than_free_drop(objects: list[dict], height: int) -> bool:
    for obj in objects:
        initial_bottom = int(obj["coords"][:, 0].max()) - int(obj["fall"])
        free_drop = height - 1 - initial_bottom
        if int(obj["fall"]) < int(free_drop):
            return True
    return False


def _has_inversion(objects: list[dict], key: str, relation: str) -> bool:
    for i in range(len(objects)):
        for j in range(len(objects)):
            if i == j:
                continue
            if int(objects[i]["fall"]) <= int(objects[j]["fall"]):
                continue
            if relation == "lower_starts_more" and int(objects[i][key]) > int(objects[j][key]):
                return True
            if relation == "higher_ends_more" and int(objects[i][key]) < int(objects[j][key]):
                return True
    return False


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(OBJECT_COUNT_OPTIONS)
    colors = random.sample(PALETTE, count)
    patches = [_random_blob(color) for color in colors]
    random.shuffle(patches)

    pattern = random.choice(CHAIN_PATTERNS[count])
    chains: list[list[dict]] = []
    cursor = 0
    for chain_len in pattern:
        built = _build_chain(patches[cursor : cursor + chain_len])
        if built is None:
            return None
        chains.append(built)
        cursor += chain_len

    random.shuffle(chains)
    placed = _place_chains(chains)
    if placed is None:
        return None
    height, width, final_objects = placed

    lifts = _assign_lifts(final_objects)
    if lifts is None:
        return None

    grid = _build_input_grid(height, width, final_objects, lifts)
    if grid is None:
        return None

    objects = _extract_objects(grid)
    if len(objects) != count:
        return None

    settled = _simulate_gravity(objects, grid.shape)
    if not _fall_less_than_free_drop(settled, height):
        return None
    if not _has_inversion(settled, "initial_top", "lower_starts_more"):
        return None
    if not _has_inversion(settled, "final_top", "higher_ends_more"):
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gravity-layer input")
