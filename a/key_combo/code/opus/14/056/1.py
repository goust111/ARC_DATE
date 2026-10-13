from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


BG = 0
FRAME = 3
MARKER_COLORS = (1, 2, 4)
BLOCK_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
GAP_CELLS = (
    (0, 1), (0, 2),
    (1, 0), (2, 0),
    (1, 3), (2, 3),
    (3, 1), (3, 2),
)
OUTWARD_GAPS = {
    "tl": [(0, 1), (1, 0)],
    "tr": [(0, 2), (1, 3)],
    "bl": [(3, 1), (2, 0)],
    "br": [(3, 2), (2, 3)],
}
COLLISION_GAPS = {
    "tl": (3, 2),
    "tr": (3, 1),
    "bl": (0, 2),
    "br": (0, 1),
}
GEN_ATTEMPTS = 400
GEN_TEMPLATES = (
    {
        "name": "noncollision_three",
        "blocks": [((0, 12), (2, 0)), ((7, 7), (1, 0)), ((8, 0), (0, 2))],
        "collision": False,
        "marker_mode": "two_regions",
    },
    {
        "name": "noncollision_four",
        "blocks": [((0, 4), (0, 2)), ((0, 10), (1, 3)), ((6, 0), (1, 3)), ((6, 6), (1, 0))],
        "collision": False,
        "marker_mode": "two_regions",
    },
    {
        "name": "collision_four",
        "blocks": [((0, 1), (3, 2)), ((4, 8), (2, 0)), ((8, 0), (1, 3)), ((10, 10), (2, 0))],
        "collision": True,
        "marker_mode": "one_region",
    },
)


@dataclass
class Block:
    color: int
    frame_r: int
    frame_c: int
    gap: tuple[int, int]
    direction: tuple[int, int]

    @property
    def gap_abs(self) -> tuple[int, int]:
        return self.frame_r + self.gap[0], self.frame_c + self.gap[1]


@dataclass
class Ray:
    color: int
    start: tuple[int, int]
    direction: tuple[int, int]
    path: list[tuple[int, int]]
    active: bool = True


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def neighbors8(r: int, c: int):
    for dr, dc in DIR8:
        yield r + dr, c + dc


def components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def label_regions(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    regions: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] >= 0:
                continue
            idx = len(regions)
            q = deque([(r, c)])
            labels[r, c] = idx
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = idx
                        q.append((nr, nc))
            regions.append(np.array(cells, dtype=int))
    return labels, regions


def gap_direction(local_gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = local_gap
    assert local_gap in GAP_CELLS
    dr = -1 if gr <= 1 else 1
    dc = -1 if gc <= 1 else 1
    return dr, dc


def is_perpendicular(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] * b[0] + a[1] * b[1] == 0


def stamp_block(grid: np.ndarray, frame_r: int, frame_c: int, color: int, gap: tuple[int, int]) -> None:
    patch = np.full((4, 4), FRAME, dtype=int)
    patch[1:3, 1:3] = color
    patch[gap] = BG
    grid[frame_r:frame_r + 4, frame_c:frame_c + 4] = patch


def find_blocks(grid: np.ndarray) -> list[Block]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    blocks: list[Block] = []
    seen: set[tuple[int, int]] = set()
    border_cells = [(r, c) for r in range(4) for c in range(4) if r in (0, 3) or c in (0, 3)]
    for br in range(h - 1):
        for bc in range(w - 1):
            color = int(g[br, bc])
            if color in (BG, FRAME):
                continue
            if not (
                int(g[br, bc + 1]) == color
                and int(g[br + 1, bc]) == color
                and int(g[br + 1, bc + 1]) == color
            ):
                continue
            fr, fc = br - 1, bc - 1
            if fr < 0 or fc < 0 or fr + 4 > h or fc + 4 > w:
                continue
            gaps: list[tuple[int, int]] = []
            ok = True
            for lr, lc in border_cells:
                value = int(g[fr + lr, fc + lc])
                if (lr, lc) in GAP_CELLS:
                    if value == BG:
                        gaps.append((lr, lc))
                    elif value != FRAME:
                        ok = False
                        break
                elif value != FRAME:
                    ok = False
                    break
            if not ok or len(gaps) != 1:
                continue
            key = (fr, fc)
            if key in seen:
                continue
            seen.add(key)
            gap = gaps[0]
            blocks.append(Block(color=color, frame_r=fr, frame_c=fc, gap=gap, direction=gap_direction(gap)))
    blocks.sort(key=lambda b: (b.frame_r, b.frame_c))
    return blocks


def extract_markers(grid: np.ndarray, blocks: list[Block]) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    structure = np.zeros(g.shape, dtype=bool)
    for block in blocks:
        structure[block.frame_r:block.frame_r + 4, block.frame_c:block.frame_c + 4] = True
    markers: list[tuple[int, int, int]] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            value = int(g[r, c])
            if value == BG or structure[r, c]:
                continue
            markers.append((r, c, value))
    for r, c, _color in markers:
        for nr, nc in neighbors8(r, c):
            if in_bounds(g.shape, nr, nc) and int(g[nr, nc]) != BG and not structure[nr, nc] and (nr, nc) != (r, c):
                raise AssertionError("markers must be isolated single pixels")
        for nr, nc in neighbors8(r, c):
            if in_bounds(g.shape, nr, nc) and structure[nr, nc]:
                raise AssertionError("markers must not touch framed blocks")
    return markers


def structural_input(grid: np.ndarray, blocks: list[Block], markers: list[tuple[int, int, int]]) -> np.ndarray:
    base = np.asarray(grid, dtype=int).copy()
    for r, c, _color in markers:
        base[r, c] = BG
    return base


def simulate_rays(base: np.ndarray, blocks: list[Block]) -> tuple[np.ndarray, list[Ray], list[tuple[int, int]]]:
    ray_grid = np.full(base.shape, BG, dtype=int)
    rays = [Ray(color=b.color, start=b.gap_abs, direction=b.direction, path=[]) for b in blocks]
    collisions: list[tuple[int, int]] = []
    while True:
        proposals: dict[tuple[int, int], list[int]] = {}
        progressed = False
        for idx, ray in enumerate(rays):
            if not ray.active:
                continue
            if ray.path:
                nr = ray.path[-1][0] + ray.direction[0]
                nc = ray.path[-1][1] + ray.direction[1]
            else:
                nr, nc = ray.start
            if not in_bounds(base.shape, nr, nc):
                ray.active = False
                continue
            if int(base[nr, nc]) != BG or int(ray_grid[nr, nc]) != BG:
                ray.active = False
                continue
            proposals.setdefault((nr, nc), []).append(idx)
        if not proposals:
            break
        for cell, ids in proposals.items():
            if len(ids) > 1:
                dirs = [rays[idx].direction for idx in ids]
                if len(ids) == 2 and is_perpendicular(dirs[0], dirs[1]):
                    collisions.append(cell)
                else:
                    collisions.append(cell)
                for idx in ids:
                    rays[idx].active = False
                continue
            idx = ids[0]
            rays[idx].path.append(cell)
            ray_grid[cell] = rays[idx].color
            progressed = True
        if not progressed:
            break
    return ray_grid, rays, collisions


def vote_color(markers: list[tuple[int, int, int]]) -> int:
    counts = Counter(color for _r, _c, color in markers)
    best = max(counts.values())
    winners = {color for color, n in counts.items() if n == best}
    if len(winners) == 1:
        return next(iter(winners))
    return min(markers, key=lambda item: (item[0], item[1]))[2]


def fill_regions(structure: np.ndarray, markers: list[tuple[int, int, int]]) -> tuple[np.ndarray, list[dict]]:
    bg_mask = structure == BG
    labels, regions = label_regions(bg_mask)
    out = structure.copy()
    marker_by_region: dict[int, list[tuple[int, int, int]]] = {}
    for r, c, color in markers:
        if in_bounds(labels.shape, r, c) and labels[r, c] >= 0:
            marker_by_region.setdefault(int(labels[r, c]), []).append((r, c, color))
    info: list[dict] = []
    for idx, coords in enumerate(regions):
        rr, cc = coords[:, 0], coords[:, 1]
        region_markers = sorted(marker_by_region.get(idx, []), key=lambda item: (item[0], item[1]))
        fill = BG if not region_markers else vote_color(region_markers)
        if fill != BG:
            out[rr, cc] = fill
        info.append(
            {
                "coords": coords,
                "markers": region_markers,
                "fill": fill,
                "eligible": None,
            }
        )
    return out, info


def solve_with_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    blocks = find_blocks(g)
    assert 3 <= len(blocks) <= 4, "expected 3-4 framed blocks"
    markers = extract_markers(g, blocks)
    base = structural_input(g, blocks, markers)
    ray_grid, rays, collisions = simulate_rays(base, blocks)
    structure = base.copy()
    structure[ray_grid != BG] = ray_grid[ray_grid != BG]
    out, region_info = fill_regions(structure, markers)
    return {
        "blocks": blocks,
        "markers": markers,
        "base": base,
        "ray_grid": ray_grid,
        "rays": rays,
        "collisions": collisions,
        "structure": structure,
        "regions": region_info,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use gap-emitted diagonal rays to partition the background, then flood-fill each partition by its marker vote.

    Input structure:
    - The background is black.
    - There are 3-4 separated 4x4 green frames, each surrounding a 2x2 monochrome inner block and containing exactly one border gap.
    - Additional isolated one-cell marker pixels of several colors are scattered on background cells and do not act as structural walls.

    Rule:
    - Each frame emits one diagonal ray from its gap, colored like the inner 2x2 block; the gap position determines which of the four diagonal directions is the outward opposite of the block center.
    - Rays advance synchronously, painting background cells including the gap cell itself, and stop before the grid edge, any frame or inner block, or any previously painted ray cell; if two rays would enter the same cell in a 90-degree collision, both stop one step early and the collision cell stays black.
    - After all rays are drawn, treat frames, inner blocks, and rays as walls, find the 4-connected black regions, and fill each region with the majority marker color inside it; ties are broken by the topmost, then leftmost, marker in that region.
    - Marker pixels disappear into the fill, while frames, inner blocks, and ray lines remain unchanged.

    Key insight:
    - The markers only provide votes; the actual grouping is created by diagonal rays sealing frame gaps and carving the background into regions whose membership changes when a gap moves.
    """
    return solve_with_details(np.asarray(grid, dtype=int))["output"]


def template_extent(blocks: list[tuple[tuple[int, int], tuple[int, int]]]) -> tuple[int, int]:
    height = max(r for (r, _c), _gap in blocks) + 4
    width = max(c for (_r, c), _gap in blocks) + 4
    return height, width


def transform_template(
    blocks: list[tuple[tuple[int, int], tuple[int, int]]],
    flip_v: bool,
    flip_h: bool,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], tuple[int, int]]:
    height, width = template_extent(blocks)
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r, c), (gr, gc) in blocks:
        nr = height - 4 - r if flip_v else r
        nc = width - 4 - c if flip_h else c
        ngr = 3 - gr if flip_v else gr
        ngc = 3 - gc if flip_h else gc
        out.append(((nr, nc), (ngr, ngc)))
    out.sort(key=lambda item: (item[0][0], item[0][1]))
    return out, (height, width)


def eligible_marker_cells(base: np.ndarray, coords: np.ndarray) -> list[tuple[int, int]]:
    h, w = base.shape
    chosen: list[tuple[int, int]] = []
    for r, c in sorted(map(tuple, coords.tolist())):
        if r in (0, h - 1) or c in (0, w - 1):
            continue
        if any((not in_bounds(base.shape, nr, nc)) or int(base[nr, nc]) != BG for nr, nc in neighbors8(r, c)):
            continue
        chosen.append((r, c))
    return chosen


def pick_spaced_cells(
    candidates: list[tuple[int, int]],
    count: int,
    avoid: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    out: list[tuple[int, int]] = []
    avoid = [] if avoid is None else list(avoid)
    for cell in candidates:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 2 for r, c in out + avoid):
            out.append(cell)
            if len(out) == count:
                return out
    return None


def place_two_region_markers(
    rng: np.random.Generator,
    base: np.ndarray,
    regions: list[dict],
) -> list[tuple[int, int, int]] | None:
    tie_plan = [4, 2, 4, 2]
    majority_plans = [
        [4, 1, 1, 1, 2, 1],
        [1, 4, 4, 4, 2, 4, 1],
    ]
    majority_plan = list(majority_plans[int(rng.integers(len(majority_plans)))])
    enriched: list[tuple[int, list[tuple[int, int]], dict]] = []
    for info in regions:
        eligible = eligible_marker_cells(base, info["coords"])
        info["eligible"] = eligible
        enriched.append((len(eligible), eligible, info))
    enriched.sort(key=lambda item: item[0], reverse=True)
    if len(enriched) < 3:
        return None

    for tie_idx, (_n1, tie_eligible, tie_region) in enumerate(enriched):
        tie_cells = pick_spaced_cells(tie_eligible, len(tie_plan))
        if tie_cells is None:
            continue
        for maj_idx, (_n2, maj_eligible, maj_region) in enumerate(enriched):
            if maj_idx == tie_idx:
                continue
            maj_cells = pick_spaced_cells(maj_eligible, len(majority_plan), avoid=tie_cells)
            if maj_cells is None:
                continue
            markers: list[tuple[int, int, int]] = []
            for (r, c), color in zip(tie_cells, tie_plan):
                markers.append((r, c, color))
            for (r, c), color in zip(maj_cells, majority_plan):
                markers.append((r, c, color))
            return markers
    return None


def place_one_region_markers(
    rng: np.random.Generator,
    base: np.ndarray,
    regions: list[dict],
) -> list[tuple[int, int, int]] | None:
    plans = [
        [4, 1, 1, 1, 2, 1, 1, 2, 1, 1],
        [4, 2, 4, 2, 4, 2, 1, 4, 1, 4],
    ]
    plan = list(plans[int(rng.integers(len(plans)))])
    enriched: list[tuple[int, list[tuple[int, int]], dict]] = []
    for info in regions:
        eligible = eligible_marker_cells(base, info["coords"])
        info["eligible"] = eligible
        enriched.append((len(eligible), eligible, info))
    enriched.sort(key=lambda item: item[0], reverse=True)
    if len(enriched) < 3:
        return None
    cells = pick_spaced_cells(enriched[0][1], len(plan))
    if cells is None:
        return None
    return [(r, c, color) for (r, c), color in zip(cells, plan)]


def rayless_fill(base: np.ndarray, markers: list[tuple[int, int, int]]) -> np.ndarray:
    out, _ = fill_regions(base, markers)
    return out


def validate_generated(grid: np.ndarray, require_collision: bool, marker_mode: str) -> bool:
    details = solve_with_details(grid)
    markers = details["markers"]
    regions = details["regions"]
    rays = details["rays"]

    if not (10 <= len(markers) <= 15):
        return False
    if require_collision and not details["collisions"]:
        return False
    if len(regions) < 3:
        return False
    if not any(not region["markers"] for region in regions):
        return False
    if any(len(ray.path) < 2 for ray in rays):
        return False

    marker_regions = [region for region in regions if region["markers"]]
    if marker_mode == "two_regions":
        if len(marker_regions) != 2:
            return False
        fills = [region["fill"] for region in marker_regions]
        if len(set(fills)) < 2:
            return False
        tie_regions = []
        strict_regions = []
        for region in marker_regions:
            counts = Counter(color for _r, _c, color in region["markers"])
            ranked = counts.most_common()
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                tie_regions.append(region)
            else:
                strict_regions.append(region)
        if len(tie_regions) != 1 or len(strict_regions) != 1:
            return False
        if tie_regions[0]["fill"] != 4:
            return False
        if strict_regions[0]["markers"][0][2] == strict_regions[0]["fill"]:
            return False
    else:
        if len(marker_regions) != 1:
            return False
        if marker_regions[0]["markers"][0][2] == marker_regions[0]["fill"]:
            return False

    rayless = rayless_fill(details["base"], markers)
    if np.array_equal(rayless, details["output"]):
        return False

    _, before_regions = label_regions(details["base"] == BG)
    if len(before_regions) >= len(regions):
        return False

    for region in regions:
        if region["markers"]:
            region_cells = {tuple(cell) for cell in region["coords"].tolist()}
            if any((r, c) not in region_cells for r, c, _color in region["markers"]):
                return False
    return True


def gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    template = dict(GEN_TEMPLATES[int(rng.integers(len(GEN_TEMPLATES)))])
    transformed, (th, tw) = transform_template(
        list(template["blocks"]),
        flip_v=bool(rng.integers(2)),
        flip_h=bool(rng.integers(2)),
    )
    h = th + int(rng.integers(4, 9))
    w = tw + int(rng.integers(4, 9))
    off_r = int(rng.integers(1, h - th))
    off_c = int(rng.integers(1, w - tw))

    colors = list(rng.choice(BLOCK_COLORS, size=len(transformed), replace=False))
    grid = np.full((h, w), BG, dtype=int)
    for color, ((r, c), gap) in zip(colors, transformed):
        stamp_block(grid, off_r + r, off_c + c, int(color), gap)

    details = solve_with_details(grid)
    if template["collision"] and not details["collisions"]:
        return None
    if len(details["regions"]) < 3:
        return None

    if template["marker_mode"] == "two_regions":
        markers = place_two_region_markers(rng, details["base"], details["regions"])
    else:
        markers = place_one_region_markers(rng, details["base"], details["regions"])
    if markers is None:
        return None

    for r, c, color in markers:
        grid[r, c] = color

    if not validate_generated(grid, bool(template["collision"]), str(template["marker_mode"])):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-partition voting grid")
