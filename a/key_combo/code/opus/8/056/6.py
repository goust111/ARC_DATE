from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, neighbors4


BG = 0
GREY = 5
PATH = 6
REGION_COLORS = [1, 2, 3, 4, 7, 8, 9]
GEN_RETRIES = 400


def _find_hub_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    hubs: list[tuple[int, int]] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if np.all(grid[r0 : r0 + 3, c0 : c0 + 3] == PATH):
                hubs.append((r0, c0))
    assert len(hubs) == 1, "expected exactly one 3x3 magenta hub"
    r0, c0 = hubs[0]
    hub = np.zeros_like(grid, dtype=bool)
    hub[r0 : r0 + 3, c0 : c0 + 3] = True
    return hub


def _anchor_from_coords(coords: np.ndarray) -> tuple[int, int]:
    pts = sorted((int(r), int(c)) for r, c in coords)
    return pts[0]


def _path_regions(shape: tuple[int, int], path_mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    region_id = np.zeros(shape, dtype=int)
    regions: list[dict] = []
    for rid, coords in enumerate(components_mask(~path_mask, connectivity=4), start=1):
        anchor = _anchor_from_coords(coords)
        bbox = bbox_coords(coords)
        region_id[coords[:, 0], coords[:, 1]] = rid
        regions.append(
            {
                "id": rid,
                "coords": coords,
                "anchor": anchor,
                "bbox": bbox,
                "area": int(coords.shape[0]),
            }
        )
    return regions, region_id


def _branch_components(path_mask: np.ndarray, hub_mask: np.ndarray) -> list[dict]:
    branches: list[dict] = []
    for bid, coords in enumerate(components_mask(path_mask & (~hub_mask), connectivity=4), start=1):
        anchor = _anchor_from_coords(coords)
        branches.append({"id": bid, "coords": coords, "anchor": anchor})
    return sorted(branches, key=lambda item: item["anchor"])


def _region_seed_colors(grid: np.ndarray, regions: list[dict]) -> dict[int, int]:
    seeds: dict[int, int] = {}
    for region in regions:
        vals = [
            int(grid[int(r), int(c)])
            for r, c in region["coords"]
            if int(grid[int(r), int(c)]) not in {BG, GREY, PATH}
        ]
        counts = Counter(vals)
        assert counts, "each region must contain at least one real seed color"
        best = max(counts.values())
        winners = [color for color, count in counts.items() if count == best]
        assert len(winners) == 1, "region seed color must be uniquely determined"
        seeds[region["id"]] = int(winners[0])
    return seeds


def _branch_contact_counts(
    branch: dict, region_id: np.ndarray, region_meta: dict[int, dict]
) -> Counter[int]:
    h, w = region_id.shape
    counts: Counter[int] = Counter()
    for r0, c0 in branch["coords"]:
        r = int(r0)
        c = int(c0)
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                rid = int(region_id[nr, nc])
                if rid:
                    counts[rid] += 1
    assert counts, "each branch must touch at least one region"
    return counts


def _choose_region_by_contacts(counts: Counter[int], region_meta: dict[int, dict]) -> int:
    best = max(counts.values())
    candidates = [rid for rid, count in counts.items() if count == best]
    return min(candidates, key=lambda rid: region_meta[rid]["anchor"])


def _branch_adjacency_summary(path_mask: np.ndarray, hub_mask: np.ndarray) -> tuple[list[dict], list[dict], np.ndarray]:
    regions, region_id = _path_regions(path_mask.shape, path_mask)
    region_meta = {region["id"]: region for region in regions}
    branches = _branch_components(path_mask, hub_mask)
    for branch in branches:
        counts = _branch_contact_counts(branch, region_id, region_meta)
        top = max(counts.values())
        tied = sum(count == top for count in counts.values()) > 1
        branch["counts"] = counts
        branch["winner_region"] = _choose_region_by_contacts(counts, region_meta)
        branch["has_tie"] = tied
    return branches, regions, region_id


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    path_mask = g == PATH
    hub_mask = _find_hub_mask(g)
    branches, regions, region_id = _branch_adjacency_summary(path_mask, hub_mask)
    region_meta = {region["id"]: region for region in regions}
    seed_colors = _region_seed_colors(g, regions)

    out = g.copy()
    out[out == GREY] = BG

    hub_votes: list[tuple[int, tuple[int, int]]] = []
    h, w = g.shape
    for branch in branches:
        color = seed_colors[branch["winner_region"]]
        coords = branch["coords"]
        out[coords[:, 0], coords[:, 1]] = color
        touches_hub = False
        for r0, c0 in coords:
            r = int(r0)
            c = int(c0)
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and hub_mask[nr, nc]:
                    touches_hub = True
                    break
            if touches_hub:
                break
        if touches_hub:
            hub_votes.append((color, branch["anchor"]))

    assert hub_votes, "hub must have at least one adjacent branch"
    vote_counts = Counter(color for color, _ in hub_votes)
    best = max(vote_counts.values())
    tied_colors = [color for color, count in vote_counts.items() if count == best]
    color_anchor = {
        color: min(anchor for vote_color, anchor in hub_votes if vote_color == color)
        for color in tied_colors
    }
    center_color = min(tied_colors, key=lambda color: color_anchor[color])

    hub_coords = np.argwhere(hub_mask)
    r0, c0, r1, c1 = bbox_coords(hub_coords)
    out[(r0 + r1) // 2, (c0 + c1) // 2] = center_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise rectangular regions to recover their seed colors, then push those colors back onto the path network.

    Input structure:
    - Black background contains one magenta orthogonal path network with a single solid 3x3 hub.
    - The magenta network partitions the remaining cells into rectangular regions.
    - Each region is mostly one non-black, non-magenta, non-grey color, with scattered grey noise pixels.

    Rule:
    - Identify the 3x3 magenta hub and treat every other magenta connected component as one branch.
    - For each path-bounded region, ignore black, magenta, and grey cells; the unique majority among the remaining colors is that region's seed color.
    - Replace every grey noise cell with black.
    - Recolor each non-hub branch with the seed color of the adjacent region contributing the most orthogonal contacts; break branch ties by the adjacent region's top-left anchor.
    - Keep the hub magenta except for its center pixel, which takes the most common recolored branch color; break hub ties by the top-left anchor of the tied branches.

    Key insight:
    - Region colors are not read directly from raw pixels: the grey cells must be ignored first, and only then can those recovered seed colors be propagated back onto the network branches.
    """
    return _transform(grid)


def _sample_layout() -> dict[str, int]:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    hr = random.randint(6, h - 10)
    hc = random.randint(6, w - 10)
    top_choices = [hr - 2] * 2 + list(range(1, hr - 1))
    bottom_choices = [hr + 4] * 2 + list(range(hr + 4, h - 1))
    rt = random.choice(top_choices)
    rb = random.choice(bottom_choices)
    left_mid = random.random() < 0.65
    right_mid = random.random() < 0.65
    if not left_mid and not right_mid and random.random() < 0.75:
        if random.random() < 0.5:
            left_mid = True
        else:
            right_mid = True
    return {
        "h": h,
        "w": w,
        "hr": hr,
        "hc": hc,
        "rt": rt,
        "rb": rb,
        "left_mid": int(left_mid),
        "right_mid": int(right_mid),
    }


def _paint_layout(layout: dict[str, int]) -> np.ndarray:
    h, w = layout["h"], layout["w"]
    hr, hc, rt, rb = layout["hr"], layout["hc"], layout["rt"], layout["rb"]
    grid = np.full((h, w), BG, dtype=int)
    grid[hr : hr + 3, hc : hc + 3] = PATH
    grid[rt, :] = PATH
    grid[rt:hr, hc] = PATH
    grid[rt:hr, hc + 2] = PATH
    grid[rb, :] = PATH
    grid[hr + 3 : rb + 1, hc] = PATH
    grid[hr + 3 : rb + 1, hc + 2] = PATH
    if layout["left_mid"]:
        grid[hr + 1, :hc] = PATH
    if layout["right_mid"]:
        grid[hr + 1, hc + 3 :] = PATH
    return grid


def _rectangle_ok(region: dict) -> bool:
    r0, c0, r1, c1 = region["bbox"]
    return region["area"] == (r1 - r0 + 1) * (c1 - c0 + 1)


def _assign_region_colors(
    regions: list[dict], branches: list[dict], require_hub_tie: bool
) -> dict[int, int] | None:
    region_ids = [region["id"] for region in regions]
    colors: dict[int, int] = {}

    if require_hub_tie:
        if len(branches) % 2 != 0:
            return None
        ordered = sorted(branches, key=lambda branch: branch["anchor"])
        split = len(branches) // 2
        top_group = ordered[:split]
        bottom_group = ordered[split:]
        top_regions = {branch["winner_region"] for branch in top_group}
        bottom_regions = {branch["winner_region"] for branch in bottom_group}
        if top_regions & bottom_regions:
            return None
        color_a, color_b = random.sample(REGION_COLORS, 2)
        for rid in top_regions:
            colors[rid] = color_a
        for rid in bottom_regions:
            colors[rid] = color_b

    distinct_target = random.randint(3, min(5, len(REGION_COLORS)))
    palette = random.sample(REGION_COLORS, distinct_target)
    for rid in region_ids:
        if rid not in colors:
            colors[rid] = random.choice(palette)

    if len(set(colors.values())) < 3:
        free = [rid for rid in region_ids if rid not in colors or sum(v == colors[rid] for v in colors.values()) > 1]
        random.shuffle(free)
        while len(set(colors.values())) < 3 and free:
            colors[free.pop()] = random.choice(REGION_COLORS)

    winner_colors = [colors[branch["winner_region"]] for branch in branches]
    if len(set(winner_colors)) < 2:
        return None
    if require_hub_tie:
        counts = Counter(winner_colors)
        if sorted(counts.values(), reverse=True)[:2] != [2, 2]:
            return None
    return colors


def _fill_regions(
    grid: np.ndarray, regions: list[dict], region_colors: dict[int, int]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = grid.copy()
    grey_cells: list[tuple[int, int]] = []
    for region in regions:
        coords = region["coords"]
        area = int(coords.shape[0])
        max_noise = min(4, (area - 1) // 2)
        assert max_noise >= 1
        noise_lo = max(1, max_noise - 1)
        noise = random.randint(noise_lo, max_noise)
        color = region_colors[region["id"]]
        out[coords[:, 0], coords[:, 1]] = color
        noise_idx = random.sample(range(area), noise)
        for idx in noise_idx:
            r, c = coords[idx]
            out[int(r), int(c)] = GREY
            grey_cells.append((int(r), int(c)))
    return out, grey_cells


def _expected_from_metadata(
    inp: np.ndarray, hub_mask: np.ndarray, branches: list[dict], region_colors: dict[int, int]
) -> np.ndarray:
    out = inp.copy()
    out[out == GREY] = BG
    votes: list[tuple[int, tuple[int, int]]] = []
    for branch in branches:
        color = region_colors[branch["winner_region"]]
        coords = branch["coords"]
        out[coords[:, 0], coords[:, 1]] = color
        votes.append((color, branch["anchor"]))

    counts = Counter(color for color, _ in votes)
    best = max(counts.values())
    tied = [color for color, count in counts.items() if count == best]
    anchors = {color: min(anchor for vote_color, anchor in votes if vote_color == color) for color in tied}
    center_color = min(tied, key=lambda color: anchors[color])
    hub_coords = np.argwhere(hub_mask)
    r0, c0, r1, c1 = bbox_coords(hub_coords)
    out[(r0 + r1) // 2, (c0 + c1) // 2] = center_color
    return out


def _layout_semantics_ok(
    branches: list[dict], regions: list[dict], require_branch_tie: bool, require_hub_tie: bool
) -> bool:
    if len(branches) < 4 or len(regions) < 4:
        return False
    if not all(_rectangle_ok(region) for region in regions):
        return False
    if not all(region["area"] >= 3 for region in regions):
        return False
    if not any(len(branch["counts"]) >= 2 for branch in branches):
        return False
    if not any(not branch["has_tie"] for branch in branches):
        return False
    if require_branch_tie and not any(branch["has_tie"] for branch in branches):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    require_branch_tie = random.random() < 0.45
    require_hub_tie = random.random() < 0.35

    path_only = _paint_layout(layout)
    hub_mask = _find_hub_mask(path_only)
    branches, regions, _region_id = _branch_adjacency_summary(path_only == PATH, hub_mask)
    if not _layout_semantics_ok(branches, regions, require_branch_tie, require_hub_tie):
        return None

    region_colors = _assign_region_colors(regions, branches, require_hub_tie)
    if region_colors is None:
        return None

    inp, grey_cells = _fill_regions(path_only, regions, region_colors)
    if len(grey_cells) < len(regions):
        return None

    expected = _expected_from_metadata(inp, hub_mask, branches, region_colors)
    produced = main(inp)
    if not np.array_equal(produced, expected):
        return None

    for region in regions:
        vals = [int(inp[int(r), int(c)]) for r, c in region["coords"]]
        color_counts = Counter(v for v in vals if v not in {BG, GREY, PATH})
        if len(color_counts) != 1:
            return None
        main_color, main_count = next(iter(color_counts.items()))
        grey_count = sum(v == GREY for v in vals)
        if not (1 <= grey_count <= 4 and main_count > grey_count):
            return None
        if int(expected[region["coords"][0][0], region["coords"][0][1]]) == PATH:
            return None
        if main_color != region_colors[region["id"]]:
            return None

    if not np.all(expected[inp == GREY] == BG):
        return None

    branch_colors = [region_colors[branch["winner_region"]] for branch in branches]
    if require_hub_tie and sorted(Counter(branch_colors).values(), reverse=True)[:2] != [2, 2]:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid noisy path-network input")
