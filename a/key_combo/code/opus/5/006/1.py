from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import (  # noqa: E402
    Color,
    bbox_coords,
    crop_bbox,
    find_components,
    get_bg,
    in_bounds,
    paste,
    random_free_location,
)


BG = Color.BLACK
RED = Color.RED
GREEN = Color.GREEN
YELLOW = Color.YELLOW
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORIENTATIONS = [(rot, flip) for rot in range(4) for flip in (False, True)]
GEN_RETRIES = 400


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    arr = np.zeros(
        (max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1), dtype=bool
    )
    for r, c in coords:
        arr[r, c] = True
    return arr


def _apply_orientation(arr: np.ndarray, orient: tuple[int, bool]) -> np.ndarray:
    rot, flip = orient
    out = np.rot90(arr, int(rot))
    if flip:
        out = np.fliplr(out)
    return out


def _transform_dir(direction: tuple[int, int], orient: tuple[int, bool]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    rot, flip = orient
    for _ in range(int(rot)):
        dr, dc = -dc, dr
    if flip:
        dc = -dc
    return int(dr), int(dc)


def _transform_point(
    shape: tuple[int, int], point: tuple[int, int], orient: tuple[int, bool]
) -> tuple[int, int]:
    marker = np.zeros(shape, dtype=bool)
    marker[point] = True
    out = _apply_orientation(marker, orient)
    rr, cc = np.argwhere(out)[0]
    return int(rr), int(cc)


def _orbit_keys(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {
        (tuple(v.shape), v.tobytes()) for v in (_apply_orientation(mask, o) for o in ORIENTATIONS)
    }


def _template_patch(mask: np.ndarray, port_cell: tuple[int, int], color: int) -> np.ndarray:
    dr, dc = TEMPLATE_SPECS_BY_KEY[(tuple(mask.shape), mask.tobytes())]["port_dir"]
    pr, pc = port_cell
    green = (pr + dr, pc + dc)
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    all_pts = cells + [green]
    r0 = min(r for r, _ in all_pts)
    c0 = min(c for _, c in all_pts)
    r1 = max(r for r, _ in all_pts)
    c1 = max(c for _, c in all_pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)
    patch[green[0] - r0, green[1] - c0] = GREEN
    return patch


def _upscale_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((k, k), dtype=int)).astype(bool)


def _dilate8(mask: np.ndarray, steps: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(int(steps)):
        nxt = out.copy()
        nxt[:-1, :] |= out[1:, :]
        nxt[1:, :] |= out[:-1, :]
        nxt[:, :-1] |= out[:, 1:]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:-1, :-1] |= out[1:, 1:]
        nxt[:-1, 1:] |= out[1:, :-1]
        nxt[1:, :-1] |= out[:-1, 1:]
        nxt[1:, 1:] |= out[:-1, :-1]
        out = nxt
    return out


def _run_lengths(mask: np.ndarray) -> list[int]:
    out: list[int] = []
    for seq in list(mask.astype(int)) + list(mask.astype(int).T):
        count = 0
        for v in seq:
            if int(v):
                count += 1
            elif count:
                out.append(int(count))
                count = 0
        if count:
            out.append(int(count))
    return out


def _infer_scale(masks: list[np.ndarray]) -> int:
    g = 0
    for mask in masks:
        for run in _run_lengths(mask):
            g = math.gcd(g, int(run))
    assert g >= 2
    return int(g)


def _downsample_scaled(mask: np.ndarray, k: int) -> np.ndarray:
    h, w = map(int, mask.shape)
    assert h % k == 0 and w % k == 0
    out = np.zeros((h // k, w // k), dtype=bool)
    for r in range(0, h, k):
        for c in range(0, w, k):
            block = mask[r : r + k, c : c + k]
            if bool(block.all()):
                out[r // k, c // k] = True
            else:
                assert not bool(block.any())
    return out


def _port_starts(
    bbox: tuple[int, int, int, int], port_cell: tuple[int, int], direction: tuple[int, int], k: int
) -> list[tuple[int, int]]:
    r0, c0, _r1, _c1 = map(int, bbox)
    pr, pc = map(int, port_cell)
    if direction == (-1, 0):
        row = r0 + pr * k - 1
        return [(int(row), int(c0 + pc * k + j)) for j in range(k)]
    if direction == (1, 0):
        row = r0 + pr * k + k
        return [(int(row), int(c0 + pc * k + j)) for j in range(k)]
    if direction == (0, -1):
        col = c0 + pc * k - 1
        return [(int(r0 + pr * k + j), int(col)) for j in range(k)]
    col = c0 + pc * k + k
    return [(int(r0 + pr * k + j), int(col)) for j in range(k)]


def _simulate_network(
    shape_infos: list[dict],
    shape: tuple[int, int],
    start_positions: list[tuple[int, int]],
    start_dir: tuple[int, int],
    *,
    blocker_mask: np.ndarray | None = None,
    use_base_dir: bool = False,
) -> tuple[set[tuple[int, int]], list[int]]:
    h, w = map(int, shape)
    blocker_mask = (
        np.zeros((h, w), dtype=bool) if blocker_mask is None else blocker_mask.astype(bool)
    )
    obj_id = np.full((h, w), -1, dtype=int)
    for idx, obj in enumerate(shape_infos):
        for r, c in obj["coords"]:
            obj_id[r, c] = int(idx)

    red_cells: set[tuple[int, int]] = set()
    active: set[int] = set()
    order: list[int] = []
    queue: list[int] = []

    def activate(idx: int) -> None:
        if idx in active:
            return
        active.add(int(idx))
        order.append(int(idx))
        queue.append(int(idx))

    def cast(starts: list[tuple[int, int]], direction: tuple[int, int]) -> None:
        positions = [(int(r), int(c)) for r, c in starts]
        dr, dc = map(int, direction)
        while True:
            inb = [(r, c) for r, c in positions if in_bounds((h, w), r, c)]
            if not inb:
                return
            hit_ids = {int(obj_id[r, c]) for r, c in inb if int(obj_id[r, c]) >= 0}
            blocked = [(r, c) for r, c in inb if int(obj_id[r, c]) < 0 and blocker_mask[r, c]]
            assert not (hit_ids and blocked)
            for r, c in inb:
                if int(obj_id[r, c]) < 0 and not blocker_mask[r, c]:
                    red_cells.add((int(r), int(c)))
            if hit_ids:
                assert len(hit_ids) == 1
                activate(next(iter(hit_ids)))
                return
            if blocked:
                return
            positions = [(int(r + dr), int(c + dc)) for r, c in positions]

    cast(start_positions, start_dir)
    while queue:
        idx = queue.pop(0)
        obj = shape_infos[int(idx)]
        direction = obj["base_dir"] if use_base_dir else obj["actual_dir"]
        starts = _port_starts(obj["bbox"], obj["port_cell_t"], direction, obj["k"])
        cast(starts, direction)
    return red_cells, order


RAW_TEMPLATE_SPECS = [
    {
        "coords": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)],
        "port_cell": (1, 0),
        "port_dir": (1, 0),
    },
    {
        "coords": [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1)],
        "port_cell": (1, 1),
        "port_dir": (1, 0),
    },
    {
        "coords": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)],
        "port_cell": (1, 1),
        "port_dir": (1, 0),
    },
    {
        "coords": [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3)],
        "port_cell": (1, 3),
        "port_dir": (1, 0),
    },
    {
        "coords": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)],
        "port_cell": (0, 0),
        "port_dir": (0, -1),
    },
]

TEMPLATE_LIBRARY: list[dict] = []
for spec in RAW_TEMPLATE_SPECS:
    mask = _mask_from_coords(spec["coords"])
    TEMPLATE_LIBRARY.append(
        {
            "mask": mask,
            "port_cell": tuple(map(int, spec["port_cell"])),
            "port_dir": tuple(map(int, spec["port_dir"])),
        }
    )

TEMPLATE_SPECS_BY_KEY = {
    (tuple(t["mask"].shape), t["mask"].tobytes()): t for t in TEMPLATE_LIBRARY
}

for idx, tmpl in enumerate(TEMPLATE_LIBRARY):
    assert len(_orbit_keys(tmpl["mask"])) == 8
    assert bool(tmpl["mask"][tmpl["port_cell"]])
    gr = tmpl["port_cell"][0] + tmpl["port_dir"][0]
    gc = tmpl["port_cell"][1] + tmpl["port_dir"][1]
    assert not in_bounds(tmpl["mask"].shape, gr, gc) or not bool(tmpl["mask"][gr, gc])
    for jdx in range(idx):
        assert _orbit_keys(tmpl["mask"]).isdisjoint(_orbit_keys(TEMPLATE_LIBRARY[jdx]["mask"]))


def _parse_start(grid: np.ndarray) -> dict:
    ys = np.argwhere(grid == YELLOW)
    assert ys.shape == (1, 2)
    yr, yc = map(int, ys[0])
    red_neighbors = []
    for dr, dc in DIRS:
        rr, cc = int(yr + dr), int(yc + dc)
        if in_bounds(grid.shape, rr, cc) and int(grid[rr, cc]) == RED:
            red_neighbors.append((int(rr), int(cc)))
    assert len(red_neighbors) == 1
    rr, cc = red_neighbors[0]
    red_comp = None
    for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[RED]):
        pts = {(int(r), int(c)) for r, c in comp["coords"]}
        if (rr, cc) in pts:
            red_comp = pts
            break
    assert red_comp is not None
    rows = {r for r, _ in red_comp}
    cols = {c for _, c in red_comp}
    assert len(rows) == 1 or len(cols) == 1
    direction = (int(rr - yr), int(cc - yc))
    dots = [int((r - yr) * direction[0] + (c - yc) * direction[1]) for r, c in red_comp]
    assert sorted(dots) == list(range(1, len(red_comp) + 1))
    tip = max(red_comp, key=lambda p: (p[0] - yr) * direction[0] + (p[1] - yc) * direction[1])
    return {
        "yellow": (int(yr), int(yc)),
        "segment": sorted(red_comp),
        "dir": (int(direction[0]), int(direction[1])),
        "tip": (int(tip[0]), int(tip[1])),
    }


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == BG
    start = _parse_start(g)

    colors = [c for c in range(10) if c not in {bg, RED, GREEN, YELLOW}]
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=colors)
    templates = [comp for comp in comps if 3 <= int(comp["area"]) <= 5]
    large = [comp for comp in comps if int(comp["area"]) >= 8]
    assert 3 <= len(templates) <= 4
    assert len(large) == len(templates)

    green_pts = {(int(r), int(c)) for r, c in np.argwhere(g == GREEN)}
    assert len(green_pts) == len(templates)

    t_infos = []
    for comp in templates:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp["coords"]:
            local[int(r) - r0, int(c) - c0] = True
        adj_green = []
        adj_port = []
        comp_set = {(int(r), int(c)) for r, c in comp["coords"]}
        for gr, gc in green_pts:
            nbs = [(gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)]
            touch = [p for p in nbs if p in comp_set]
            if touch:
                adj_green.append((int(gr), int(gc)))
                adj_port.extend(touch)
        assert len(adj_green) == 1
        assert len(set(adj_port)) == 1
        port_g = adj_green[0]
        port_cell_g = next(iter(set(adj_port)))
        port_dir = (int(port_g[0] - port_cell_g[0]), int(port_g[1] - port_cell_g[1]))
        assert port_dir in DIRS
        t_infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": [(int(r), int(c)) for r, c in comp["coords"]],
                "mask": local,
                "color": int(comp["color"]),
                "port_cell": (int(port_cell_g[0] - r0), int(port_cell_g[1] - c0)),
                "port_dir": port_dir,
            }
        )

    large_infos = []
    large_masks = []
    for comp in large:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp["coords"]:
            local[int(r) - r0, int(c) - c0] = True
        large_masks.append(local)
        large_infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": [(int(r), int(c)) for r, c in comp["coords"]],
                "mask": local,
                "color": int(comp["color"]),
            }
        )

    k = _infer_scale(large_masks)
    assert k in {2, 3}

    variants = []
    for tidx, tmpl in enumerate(t_infos):
        for orient in ORIENTATIONS:
            variants.append(
                {
                    "template_idx": int(tidx),
                    "orient": orient,
                    "mask": _apply_orientation(tmpl["mask"], orient),
                    "port_cell_t": _transform_point(tmpl["mask"].shape, tmpl["port_cell"], orient),
                    "actual_dir": _transform_dir(tmpl["port_dir"], orient),
                }
            )

    used_templates: set[int] = set()
    for obj in large_infos:
        down = _downsample_scaled(obj["mask"], k)
        cands = [v for v in variants if np.array_equal(v["mask"], down)]
        assert len(cands) == 1
        cand = cands[0]
        assert int(cand["template_idx"]) not in used_templates
        used_templates.add(int(cand["template_idx"]))
        tmpl = t_infos[int(cand["template_idx"])]
        obj.update(
            {
                "k": int(k),
                "template_idx": int(cand["template_idx"]),
                "orient": cand["orient"],
                "port_cell_t": cand["port_cell_t"],
                "actual_dir": cand["actual_dir"],
                "base_dir": tmpl["port_dir"],
            }
        )

    blocker = (g != bg) & (g != RED) & (g != YELLOW) & (g != GREEN)
    for obj in large_infos:
        for r, c in obj["coords"]:
            blocker[r, c] = False

    return {
        "bg": bg,
        "start": start,
        "templates": t_infos,
        "large": large_infos,
        "blocker_mask": blocker,
        "k": int(k),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate scaled large shapes through template-derived ports and repaint the reached network red.

    Input structure:
    - The background is black, with 3-4 small monochrome templates and 3-4 larger monochrome shapes.
    - Each small template is a 4-connected 3-5 pixel shape with one adjacent green pixel marking a port.
    - Each large shape is a rotated/flipped k-times-upscaled copy of exactly one template, and a yellow-plus-short-red start marker sits near the bottom edge.

    Rule:
    - Separate the small templates from the large shapes, infer the common upscale factor k, and downsample each large shape by k.
    - Match every downsampled large mask to exactly one template under the 8 dihedral orientations; the matched orientation transforms that template's port cell and port direction onto the large shape.
    - Extend the start ray until it first hits a large shape, recolor every activated large shape red, then keep emitting k-wide rays from each activated shape's transformed port until no new large shape is reached.
    - Preserve the original templates, green markers, and start marker; keep unreachable large shapes in their original colors; keep all emitted rays red.

    Key insight:
    - The large shapes do not show their ports directly: only template-to-large scale matching reveals which boundary block emits the next ray and in which direction.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_input(g)
    tip_r, tip_c = info["start"]["tip"]
    dr, dc = info["start"]["dir"]
    start_positions = [(int(tip_r + dr), int(tip_c + dc))]
    red_cells, order = _simulate_network(
        info["large"],
        g.shape,
        start_positions,
        info["start"]["dir"],
        blocker_mask=info["blocker_mask"],
        use_base_dir=False,
    )

    out = g.copy()
    for r, c in red_cells:
        if int(out[r, c]) == BG:
            out[r, c] = RED
    for idx in order:
        for r, c in info["large"][int(idx)]["coords"]:
            out[r, c] = RED
    return out


def _make_large_spec(template: dict, orient: tuple[int, bool], color: int, k: int) -> dict:
    oriented_mask = _apply_orientation(template["mask"], orient)
    port_cell_t = _transform_point(template["mask"].shape, template["port_cell"], orient)
    actual_dir = _transform_dir(template["port_dir"], orient)
    large_mask = _upscale_mask(oriented_mask, k)
    patch = np.full(large_mask.shape, BG, dtype=int)
    patch[large_mask] = int(color)
    return {
        "template": template,
        "orient": orient,
        "base_dir": template["port_dir"],
        "actual_dir": actual_dir,
        "port_cell_t": port_cell_t,
        "mask_small": oriented_mask,
        "mask": large_mask,
        "patch": patch,
        "color": int(color),
        "k": int(k),
    }


def _placed_shape(spec: dict, top_left: tuple[int, int]) -> dict:
    r0, c0 = map(int, top_left)
    coords = [(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(spec["mask"])]
    r1 = r0 + spec["mask"].shape[0] - 1
    c1 = c0 + spec["mask"].shape[1] - 1
    out = dict(spec)
    out.update({"bbox": (r0, c0, r1, c1), "coords": coords})
    return out


def _ray_first_hit_steps(
    occ: np.ndarray,
    patch_mask: np.ndarray,
    top_left: tuple[int, int],
    starts: list[tuple[int, int]],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> int | None:
    h, w = map(int, shape)
    r0, c0 = map(int, top_left)
    ph, pw = map(int, patch_mask.shape)

    def hits_candidate(r: int, c: int) -> bool:
        return r0 <= r < r0 + ph and c0 <= c < c0 + pw and bool(patch_mask[r - r0, c - c0])

    positions = [(int(r), int(c)) for r, c in starts]
    dr, dc = map(int, direction)
    steps = 0
    while True:
        inb = [(r, c) for r, c in positions if 0 <= r < h and 0 <= c < w]
        if not inb:
            return None
        occ_hit = any(bool(occ[r, c]) for r, c in inb)
        cand_hit = any(hits_candidate(r, c) for r, c in inb)
        if cand_hit:
            return None if occ_hit else int(steps)
        if occ_hit:
            return None
        positions = [(int(r + dr), int(c + dc)) for r, c in positions]
        steps += 1


def _find_target_placement(
    occ: np.ndarray,
    spec: dict,
    starts: list[tuple[int, int]],
    direction: tuple[int, int],
    shape: tuple[int, int],
    *,
    padding: int = 2,
    border: int = 1,
    min_steps: int = 3,
) -> tuple[int, int] | None:
    h, w = map(int, shape)
    ph, pw = map(int, spec["mask"].shape)
    blocked = _dilate8(occ, steps=padding)
    candidates = [
        (r, c)
        for r in range(border, h - border - ph + 1)
        for c in range(border, w - border - pw + 1)
    ]
    random.shuffle(candidates)
    scored = []
    for r0, c0 in candidates:
        if bool((blocked[r0 : r0 + ph, c0 : c0 + pw] & spec["mask"]).any()):
            continue
        steps = _ray_first_hit_steps(occ, spec["mask"], (r0, c0), starts, direction, shape)
        if steps is None or int(steps) < int(min_steps):
            continue
        scored.append((abs(int(steps) - 6), random.random(), int(r0), int(c0)))
    if not scored:
        return None
    scored.sort()
    _score, _noise, r0, c0 = scored[0]
    return int(r0), int(c0)


def _paint_mask(mask: np.ndarray, spec: dict, top_left: tuple[int, int]) -> None:
    r0, c0 = map(int, top_left)
    ph, pw = map(int, spec["mask"].shape)
    mask[r0 : r0 + ph, c0 : c0 + pw] |= spec["mask"]


def _choose_start_column(first_shape: dict, grid_shape: tuple[int, int], seg_len: int) -> int | None:
    h, _w = map(int, grid_shape)
    by_col: dict[int, int] = {}
    for r, c in first_shape["coords"]:
        by_col[c] = max(int(r), int(by_col.get(c, -10**9)))
    valid = [c for c, bottom in by_col.items() if bottom <= h - seg_len - 4]
    return random.choice(valid) if valid else None


def _start_patch(grid_shape: tuple[int, int], col: int, seg_len: int) -> tuple[np.ndarray, dict]:
    h, w = map(int, grid_shape)
    patch = np.full((h, w), BG, dtype=int)
    for rr in range(h - seg_len - 1, h - 1):
        patch[rr, col] = RED
    patch[h - 1, col] = YELLOW
    return patch, {
        "yellow": (int(h - 1), int(col)),
        "segment": [(int(rr), int(col)) for rr in range(h - seg_len - 1, h - 1)],
        "dir": (-1, 0),
        "tip": (int(h - seg_len - 1), int(col)),
    }


def _assemble_input(
    grid_shape: tuple[int, int],
    active_shapes: list[dict],
    inactive_shapes: list[dict],
    start_info: dict,
    template_patches: list[np.ndarray],
) -> np.ndarray:
    h, w = map(int, grid_shape)
    grid = np.full((h, w), BG, dtype=int)
    for shape in active_shapes + inactive_shapes:
        r0, c0, _r1, _c1 = shape["bbox"]
        paste(grid, shape["patch"], r0, c0, transparent=BG)
    for r, c in start_info["segment"]:
        grid[r, c] = RED
    yr, yc = start_info["yellow"]
    grid[yr, yc] = YELLOW
    occ = grid != BG
    for patch in template_patches:
        r0, c0 = random_free_location(
            occ.astype(int),
            patch,
            transparent=BG,
            border=1,
            padding=1,
        )
        paste(grid, patch, r0, c0, transparent=BG)
        occ = grid != BG
    return grid


def _instance_is_semantic(grid: np.ndarray) -> bool:
    info = _analyze_input(grid)
    start = info["start"]
    start_positions = [(int(start["tip"][0] + start["dir"][0]), int(start["tip"][1] + start["dir"][1]))]
    red_cells, order = _simulate_network(
        info["large"],
        grid.shape,
        start_positions,
        start["dir"],
        blocker_mask=info["blocker_mask"],
        use_base_dir=False,
    )
    if not (1 < len(order) < len(info["large"])):
        return False
    if not any(
        info["large"][idx]["actual_dir"] != info["large"][idx]["base_dir"]
        for idx in order[:-1]
    ):
        return False
    wrong_red, wrong_order = _simulate_network(
        info["large"],
        grid.shape,
        start_positions,
        start["dir"],
        blocker_mask=info["blocker_mask"],
        use_base_dir=True,
    )
    out = main(grid)
    wrong_out = grid.copy()
    for r, c in wrong_red:
        if int(wrong_out[r, c]) == BG:
            wrong_out[r, c] = RED
    for idx in wrong_order:
        for r, c in info["large"][int(idx)]["coords"]:
            wrong_out[r, c] = RED
    if np.array_equal(out, wrong_out):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(23, 29)
    w = random.randint(23, 29)
    grid_shape = (h, w)
    n_total = random.randint(3, 4)
    n_active = random.randint(2, n_total - 1)
    k = random.choice([2, 3])

    templates = random.sample(TEMPLATE_LIBRARY, n_total)
    small_colors = random.sample([1, 5, 6, 7, 8, 9], n_total)
    large_colors = random.sample([1, 5, 6, 7, 8, 9], n_total)
    if sum(int(a == b) for a, b in zip(small_colors, large_colors)) > 1:
        large_colors = large_colors[1:] + large_colors[:1]

    specs = []
    for idx, tmpl in enumerate(templates):
        specs.append(
            _make_large_spec(tmpl, random.choice(ORIENTATIONS), large_colors[idx], k)
        )

    if not any(
        specs[idx]["actual_dir"] != specs[idx]["base_dir"] for idx in range(n_active - 1)
    ):
        return None

    occ = np.zeros(grid_shape, dtype=bool)
    active_shapes: list[dict] = []
    inactive_specs = specs[n_active:]
    active_specs = specs[:n_active]

    first_spec = active_specs[0]
    first_candidates = [
        (r, c)
        for r in range(2, h - first_spec["mask"].shape[0] - 2)
        for c in range(2, w - first_spec["mask"].shape[1] - 2)
    ]
    random.shuffle(first_candidates)
    placed_first = None
    start_info = None
    seg_len = random.choice([2, 3])
    for r0, c0 in first_candidates[:250]:
        blocked = _dilate8(occ, steps=2)
        if bool(
            (
                blocked[
                    r0 : r0 + first_spec["mask"].shape[0],
                    c0 : c0 + first_spec["mask"].shape[1],
                ]
                & first_spec["mask"]
            ).any()
        ):
            continue
        candidate = _placed_shape(first_spec, (r0, c0))
        start_col = _choose_start_column(candidate, grid_shape, seg_len)
        if start_col is None:
            continue
        patch, start_info = _start_patch(grid_shape, start_col, seg_len)
        start_cells = {(int(r), int(start_col)) for r in range(h - seg_len - 1, h)}
        if any((r, c) in start_cells for r, c in candidate["coords"]):
            continue
        placed_first = candidate
        _paint_mask(occ, first_spec, (r0, c0))
        active_shapes.append(candidate)
        break
    if placed_first is None or start_info is None:
        return None

    for spec in active_specs[1:]:
        prev = active_shapes[-1]
        starts = _port_starts(prev["bbox"], prev["port_cell_t"], prev["actual_dir"], k)
        loc = _find_target_placement(occ, spec, starts, prev["actual_dir"], grid_shape)
        if loc is None:
            return None
        placed = _placed_shape(spec, loc)
        active_shapes.append(placed)
        _paint_mask(occ, spec, loc)

    start_positions = [(int(start_info["tip"][0] - 1), int(start_info["tip"][1]))]
    rays, order = _simulate_network(
        active_shapes,
        grid_shape,
        start_positions,
        start_info["dir"],
        blocker_mask=np.zeros(grid_shape, dtype=bool),
        use_base_dir=False,
    )
    if order != list(range(n_active)):
        return None

    reserve = occ.copy()
    for r, c in rays:
        reserve[r, c] = True
    for r, c in start_info["segment"]:
        reserve[r, c] = True
    reserve[start_info["yellow"]] = True

    inactive_shapes: list[dict] = []
    for spec in inactive_specs:
        canvas = _dilate8(reserve, steps=1).astype(int)
        try:
            r0, c0 = random_free_location(
                canvas,
                spec["patch"],
                transparent=BG,
                border=1,
                padding=1,
            )
        except ValueError:
            return None
        placed = _placed_shape(spec, (r0, c0))
        inactive_shapes.append(placed)
        _paint_mask(reserve, spec, (r0, c0))

    template_patches = [
        _template_patch(tmpl["mask"], tmpl["port_cell"], small_colors[idx])
        for idx, tmpl in enumerate(templates)
    ]
    grid = _assemble_input(grid_shape, active_shapes, inactive_shapes, start_info, template_patches)

    try:
        out = main(grid)
    except Exception:
        return None
    if not _instance_is_semantic(grid):
        return None
    if np.array_equal(out, grid):
        return None
    for shape in inactive_shapes:
        if all(int(out[r, c]) == RED for r, c in shape["coords"]):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to build a valid puzzle instance")
