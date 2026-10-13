from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
WALL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_info(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(coords) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0, c1 - c0) != (1, 1):
        return None
    pts = set(coords)
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    missing = list(corners - pts)
    if len(missing) != 1:
        return None
    degrees = {}
    for r, c in pts:
        deg = 0
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in pts:
                deg += 1
        degrees[(r, c)] = deg
    elbows = [p for p, deg in degrees.items() if deg == 2]
    if len(elbows) != 1:
        return None
    elbow = elbows[0]
    miss = missing[0]
    dr = 1 if elbow[0] > miss[0] else -1
    dc = 1 if elbow[1] > miss[1] else -1
    return (dr, dc), elbow


def _is_solid_rect(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return False
    return bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)))


def _block_cells(block: dict) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(int(block["r0"]), int(block["r0"]) + int(block["h"]))
        for c in range(int(block["c0"]), int(block["c0"]) + int(block["w"]))
    ]


def _render_scene(shape: tuple[int, int], arrows: list[dict], blocks: list[dict], walls: set[tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in walls:
        out[r, c] = WALL
    for arrow in arrows:
        for r, c in arrow["coords"]:
            out[r, c] = int(arrow["color"])
    for block in blocks:
        out[int(block["r0"]) : int(block["r0"]) + int(block["h"]), int(block["c0"]) : int(block["c0"]) + int(block["w"])] = int(
            block["color"]
        )
    return out


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    assert bg == BG, "expected black background"

    arrows: list[dict] = []
    blocks: list[dict] = []
    walls = {(int(r), int(c)) for r, c in np.argwhere(g == WALL)}

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {BG, WALL}):
        for comp in _components_of_color(g, color):
            info = _arrow_info(comp)
            if info is not None:
                direction, elbow = info
                arrows.append(
                    {
                        "color": color,
                        "coords": tuple(sorted(comp)),
                        "dir": direction,
                        "elbow": elbow,
                    }
                )
                continue
            assert _is_solid_rect(g, comp, color), f"component color={color} is neither arrow nor solid block"
            r0, c0, r1, c1 = _bbox(comp)
            blocks.append(
                {
                    "id": len(blocks),
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                }
            )

    arrows.sort(key=lambda a: (a["elbow"][0], a["elbow"][1], a["color"]))
    return {"shape": g.shape, "arrows": arrows, "blocks": blocks, "walls": walls}


def _occupancy(blocks: list[dict]) -> dict[tuple[int, int], int]:
    occ: dict[tuple[int, int], int] = {}
    for block in blocks:
        for cell in _block_cells(block):
            assert cell not in occ, "blocks must not overlap"
            occ[cell] = int(block["id"])
    return occ


def _first_hit_block(arrow: dict, blocks_by_cell: dict[tuple[int, int], int], wall_cells: set[tuple[int, int]], arrow_cells: set[tuple[int, int]], shape: tuple[int, int]) -> int | None:
    h, w = shape
    dr, dc = arrow["dir"]
    r, c = arrow["elbow"]
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        if (r, c) in wall_cells or (r, c) in arrow_cells:
            return None
        hit = blocks_by_cell.get((r, c))
        if hit is not None:
            return int(hit)


def _simulate_push(blocks: list[dict], start_id: int, direction: tuple[int, int], wall_cells: set[tuple[int, int]], shape: tuple[int, int]) -> dict:
    h, w = shape
    dr, dc = direction
    occ = _occupancy(blocks)
    active = int(start_id)
    chain: list[int] = []
    steps_by_id: dict[int, int] = {}
    seen: set[int] = set()

    while True:
        assert active not in seen, "momentum chain looped"
        seen.add(active)
        moved_steps = 0
        block = blocks[active]
        while True:
            current_cells = _block_cells(block)
            next_cells = [(r + dr, c + dc) for r, c in current_cells]
            out_hit = any(nr < 0 or nr >= h or nc < 0 or nc >= w for nr, nc in next_cells)
            wall_hit = any((nr, nc) in wall_cells for nr, nc in next_cells if 0 <= nr < h and 0 <= nc < w)
            colliders = {
                int(occ[(nr, nc)])
                for nr, nc in next_cells
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in occ and int(occ[(nr, nc)]) != int(active)
            }
            assert not ((out_hit or wall_hit) and colliders), "ambiguous simultaneous stop and collision"
            if colliders:
                assert len(colliders) == 1, "a chain may only hit one block at once"
                chain.append(active)
                steps_by_id[active] = moved_steps
                active = next(iter(colliders))
                break
            if out_hit or wall_hit:
                chain.append(active)
                steps_by_id[active] = moved_steps
                return {"chain": chain, "steps": steps_by_id, "stop": "boundary" if out_hit else "wall"}

            for cell in current_cells:
                del occ[cell]
            block["r0"] += dr
            block["c0"] += dc
            for cell in _block_cells(block):
                assert cell not in occ, "moved block overlapped another block"
                occ[cell] = int(active)
            moved_steps += 1


def _simulate_scene(grid: np.ndarray, *, return_trace: bool = False) -> tuple[np.ndarray, list[dict]] | np.ndarray:
    scene = _parse_scene(grid)
    blocks = [dict(block) for block in scene["blocks"]]
    wall_cells = set(scene["walls"])
    arrow_cells = {cell for arrow in scene["arrows"] for cell in arrow["coords"]}
    blocks_by_cell = _occupancy(blocks)

    targets: list[tuple[dict, int]] = []
    targeted: set[int] = set()
    for arrow in scene["arrows"]:
        hit = _first_hit_block(arrow, blocks_by_cell, wall_cells, arrow_cells, scene["shape"])
        assert hit is not None, "every arrow must hit a unique movable block"
        assert hit not in targeted, "a movable block may be targeted by at most one ray"
        targeted.add(hit)
        targets.append((arrow, hit))

    traces: list[dict] = []
    for arrow, hit in targets:
        trace = _simulate_push(blocks, hit, arrow["dir"], wall_cells, scene["shape"])
        trace["arrow_color"] = int(arrow["color"])
        trace["target_color"] = int(scene["blocks"][hit]["color"])
        trace["chain_colors"] = [int(scene["blocks"][bid]["color"]) for bid in trace["chain"]]
        traces.append(trace)

    out = _render_scene(scene["shape"], scene["arrows"], blocks, wall_cells)
    if return_trace:
        traces.sort(key=lambda t: t["arrow_color"])
        return out, traces
    return out


def _simulate_without_transfer(grid: np.ndarray) -> np.ndarray:
    scene = _parse_scene(grid)
    blocks = [dict(block) for block in scene["blocks"]]
    wall_cells = set(scene["walls"])
    arrow_cells = {cell for arrow in scene["arrows"] for cell in arrow["coords"]}
    blocks_by_cell = _occupancy(blocks)

    targets: list[tuple[dict, int]] = []
    seen: set[int] = set()
    for arrow in scene["arrows"]:
        hit = _first_hit_block(arrow, blocks_by_cell, wall_cells, arrow_cells, scene["shape"])
        assert hit is not None and hit not in seen
        seen.add(hit)
        targets.append((arrow, hit))

    h, w = scene["shape"]
    for arrow, hit in targets:
        dr, dc = arrow["dir"]
        block = blocks[hit]
        while True:
            next_cells = [(r + dr, c + dc) for r, c in _block_cells(block)]
            if any(nr < 0 or nr >= h or nc < 0 or nc >= w for nr, nc in next_cells):
                break
            if any((nr, nc) in wall_cells for nr, nc in next_cells):
                break
            other_hit = False
            for nr, nc in next_cells:
                for other in blocks:
                    if other["id"] == hit:
                        continue
                    if int(other["r0"]) <= nr < int(other["r0"]) + int(other["h"]) and int(other["c0"]) <= nc < int(other["c0"]) + int(other["w"]):
                        other_hit = True
                        break
                if other_hit:
                    break
            if other_hit:
                break
            block["r0"] += dr
            block["c0"] += dc

    return _render_scene(scene["shape"], scene["arrows"], blocks, wall_cells)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the first ray-hit block along the arrow diagonal, transferring momentum through collisions.

    Input structure:
    - A black background contains grey wall pixels, several solid monochrome movable rectangles, and one or more monochrome 3-pixel L arrows in 2x2 boxes.
    - Each arrow's missing corner determines a diagonal direction; the elbow pixel is the ray source.
    - Every movable block is hit by at most one arrow ray, and different push chains do not cross.

    Rule:
    - Each arrow casts an invisible diagonal ray away from its missing corner, skipping empty cells until it reaches the first movable block.
    - The hit block slides one diagonal step at a time in that same direction until the next step would leave the grid or cover a grey wall pixel.
    - If its next step would instead overlap another movable block, the moving block stops immediately and the struck block takes over the motion from its current position, possibly passing momentum onward again.
    - Arrows, walls, and untouched blocks remain unchanged.

    Key insight:
    - The answer is fixed by a three-stage causal chain: ray selection chooses exactly one starting block, that block slides as far as allowed, and any collision hands the motion to the next block instead of letting the first one pass through.
    """
    return _simulate_scene(np.asarray(grid, dtype=int))


def _chain_patch(length: int, stop_kind: str, size: tuple[int, int], ray_gap: int, deltas: list[int], tail_steps: int, colors: list[int]) -> np.ndarray:
    h, w = size
    elbow = (1, 1)
    block_tops = [(1 + ray_gap, 1 + ray_gap)]
    for delta in deltas:
        pr, pc = block_tops[-1]
        block_tops.append((pr + delta, pc + delta))
    last_r, last_c = block_tops[-1]
    final_r = last_r + tail_steps
    final_c = last_c + tail_steps
    patch_h = final_r + h + (0 if stop_kind == "boundary" else 1)
    patch_w = final_c + w + (0 if stop_kind == "boundary" else 1)

    patch = np.full((patch_h, patch_w), BG, dtype=int)
    arrow_color = colors[0]
    patch[elbow[0], elbow[1]] = arrow_color
    patch[elbow[0] - 1, elbow[1]] = arrow_color
    patch[elbow[0], elbow[1] - 1] = arrow_color
    for (r0, c0), color in zip(block_tops, colors[1:]):
        patch[r0 : r0 + h, c0 : c0 + w] = color
    if stop_kind == "wall":
        patch[final_r + h, final_c + w] = WALL
    return patch


def _orient_patch(patch: np.ndarray, direction: tuple[int, int]) -> np.ndarray:
    dr, dc = direction
    out = patch
    if dr == -1:
        out = np.flipud(out)
    if dc == -1:
        out = np.fliplr(out)
    return out.copy()


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    region = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    assert not np.any(region[mask] != BG)
    region[mask] = patch[mask]


def _free_box(grid: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int = 1) -> bool:
    gh, gw = grid.shape
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(gh, r0 + h + gap)
    cc1 = min(gw, c0 + w + gap)
    return bool(np.all(grid[rr0:rr1, cc0:cc1] == BG))


def _sample_chain_specs() -> tuple[list[dict], int]:
    chain_counts = [2, 3]
    chain_count = random.choice(chain_counts)
    if chain_count == 2:
        lengths = random.choice([[1, 2], [2, 2], [1, 3]])
    else:
        lengths = random.choice([[1, 1, 2], [1, 2, 2]])
    random.shuffle(lengths)
    boundary_idx = random.randrange(chain_count) if random.random() < 0.55 else -1

    total_objects = chain_count + sum(lengths)
    assert total_objects <= len(PALETTE)
    colors = random.sample(PALETTE, total_objects)
    cursor = 0
    specs: list[dict] = []
    for i, length in enumerate(lengths):
        h = random.randint(1, 3)
        w = random.randint(1, 3)
        if random.random() < 0.45:
            h = w
        ray_gap = random.randint(2, 4)
        deltas = [random.randint(min(h, w) + 1, min(h, w) + 3) for _ in range(length - 1)]
        tail_steps = random.randint(1, 3)
        stop_kind = "boundary" if i == boundary_idx else "wall"
        used = 1 + length
        chain_colors = colors[cursor : cursor + used]
        cursor += used
        direction = random.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
        specs.append(
            {
                "length": length,
                "size": (h, w),
                "ray_gap": ray_gap,
                "deltas": deltas,
                "tail_steps": tail_steps,
                "stop_kind": stop_kind,
                "colors": chain_colors,
                "direction": direction,
            }
        )
    return specs, total_objects


def _generate_attempt() -> tuple[np.ndarray, dict] | None:
    chain_specs, used_color_count = _sample_chain_specs()
    patches = []
    for spec in chain_specs:
        patch = _chain_patch(
            spec["length"],
            spec["stop_kind"],
            spec["size"],
            spec["ray_gap"],
            spec["deltas"],
            spec["tail_steps"],
            spec["colors"],
        )
        oriented = _orient_patch(patch, spec["direction"])
        patches.append({"patch": oriented, "spec": spec})

    max_h = max(item["patch"].shape[0] for item in patches)
    max_w = max(item["patch"].shape[1] for item in patches)
    grid_h = random.randint(max(20, max_h + 2), 30)
    grid_w = random.randint(max(20, max_w + 2), 30)
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    placements: list[tuple[int, int, int, int]] = []
    meta_chains: list[dict] = []

    boundary_items = [item for item in patches if item["spec"]["stop_kind"] == "boundary"]
    wall_items = [item for item in patches if item["spec"]["stop_kind"] == "wall"]

    for item in boundary_items:
        ph, pw = item["patch"].shape
        dr, dc = item["spec"]["direction"]
        top = grid_h - ph if dr == 1 else 0
        left = grid_w - pw if dc == 1 else 0
        rect = (top, left, top + ph - 1, left + pw - 1)
        if any(not _rects_separated(rect, old, gap=1) for old in placements):
            return None
        _place_patch(grid, item["patch"], top, left)
        placements.append(rect)
        meta_chains.append({"spec": item["spec"], "top": top, "left": left})

    random.shuffle(wall_items)
    for item in wall_items:
        ph, pw = item["patch"].shape
        placed = False
        for _ in range(400):
            top = random.randint(0, grid_h - ph)
            left = random.randint(0, grid_w - pw)
            rect = (top, left, top + ph - 1, left + pw - 1)
            if any(not _rects_separated(rect, old, gap=1) for old in placements):
                continue
            _place_patch(grid, item["patch"], top, left)
            placements.append(rect)
            meta_chains.append({"spec": item["spec"], "top": top, "left": left})
            placed = True
            break
        if not placed:
            return None

    remaining_colors = [c for c in PALETTE if c not in {spec["colors"][0] for spec in chain_specs} and c not in {color for spec in chain_specs for color in spec["colors"][1:]}]
    distractor_color = None
    if remaining_colors and used_color_count <= 7 and random.random() < 0.8:
        dh = random.randint(1, 2)
        dw = random.randint(1, 2)
        color = remaining_colors[0]
        for _ in range(200):
            r0 = random.randint(0, grid_h - dh)
            c0 = random.randint(0, grid_w - dw)
            if not _free_box(grid, r0, c0, dh, dw, gap=1):
                continue
            grid[r0 : r0 + dh, c0 : c0 + dw] = color
            distractor_color = color
            break

    out, traces = _simulate_scene(grid, return_trace=True)
    if np.array_equal(grid, out):
        return None

    trace_by_arrow = {trace["arrow_color"]: trace for trace in traces}
    saw_transfer = False
    saw_direct = False
    stop_kinds = Counter()
    for chain in meta_chains:
        arrow_color = int(chain["spec"]["colors"][0])
        expected_colors = [int(v) for v in chain["spec"]["colors"][1:]]
        trace = trace_by_arrow.get(arrow_color)
        if trace is None:
            return None
        if trace["chain_colors"] != expected_colors:
            return None
        if trace["stop"] != chain["spec"]["stop_kind"]:
            return None
        if any(int(trace["steps"][bid]) <= 0 for bid in trace["chain"]):
            return None
        saw_transfer |= len(trace["chain"]) >= 2
        saw_direct |= len(trace["chain"]) == 1
        stop_kinds[trace["stop"]] += 1

    if not saw_transfer or not saw_direct:
        return None
    if len(stop_kinds) < 2:
        return None
    if distractor_color is not None:
        before = {(int(r), int(c)) for r, c in np.argwhere(grid == distractor_color)}
        after = {(int(r), int(c)) for r, c in np.argwhere(out == distractor_color)}
        if before != after:
            return None

    naive = _simulate_without_transfer(grid)
    if np.array_equal(naive, out):
        return None

    if not np.array_equal(main(grid), out):
        return None

    meta = {"traces": traces}
    return grid, meta


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _generate_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid ray-push momentum puzzle")

