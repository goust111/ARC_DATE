from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labels for a boolean mask."""
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=int)
    next_id = 0
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            labels[r, c] = next_id
            queue = deque([(r, c)])
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        queue.append((nr, nc))
    return labels, next_id


def _graph_components(graph: dict[int, set[int]]) -> list[set[int]]:
    unseen = set(graph)
    comps: list[set[int]] = []
    while unseen:
        start = unseen.pop()
        comp = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for nxt in graph[node]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps


def _dominant_color(counts: Counter[int]) -> int:
    ordered = counts.most_common()
    assert ordered, "expected a non-empty grid"
    if len(ordered) > 1:
        assert ordered[0][1] > ordered[1][1], "background color must be uniquely most frequent"
    return int(ordered[0][0])


def _build_region_graph(region_labels: np.ndarray, wall_mask: np.ndarray) -> dict[int, set[int]]:
    H, W = region_labels.shape
    n_regions = int(region_labels.max())
    graph = {region_id: set() for region_id in range(1, n_regions + 1)}

    for r, c in np.argwhere(wall_mask):
        r = int(r)
        c = int(c)

        if 0 < r < H - 1:
            a = int(region_labels[r - 1, c])
            b = int(region_labels[r + 1, c])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)

        if 0 < c < W - 1:
            a = int(region_labels[r, c - 1])
            b = int(region_labels[r, c + 1])
            if a > 0 and b > 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)

    return graph


def _parse_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    counts = Counter(map(int, g.flat))
    bg = _dominant_color(counts)

    unique_colors = sorted(color for color, freq in counts.items() if freq == 1 and color != bg)
    assert len(unique_colors) == 2, "expected exactly two uniquely colored seed pixels"

    repeated_non_bg = sorted(
        ((freq, color) for color, freq in counts.items() if color != bg and freq > 1),
        reverse=True,
    )
    assert repeated_non_bg, "expected a repeated wall color"
    wall_freq, wall = repeated_non_bg[0]
    if len(repeated_non_bg) > 1:
        assert wall_freq > repeated_non_bg[1][0], "wall color must be the unique largest repeated non-background color"
    wall = int(wall)

    wall_mask = g == wall
    region_labels, n_regions = _label_mask(~wall_mask)
    assert n_regions >= 4, "expected several wall-separated non-wall regions"

    graph = _build_region_graph(region_labels, wall_mask)
    components = _graph_components(graph)
    assert len(components) == 2, "region adjacency graph must split into exactly two components"
    assert all(len(comp) >= 2 for comp in components), "each region component must contain multiple regions"

    seed_infos = []
    for color in unique_colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (1, 2)
        r, c = map(int, coords[0])
        region_id = int(region_labels[r, c])
        assert region_id > 0, "seed must lie inside a non-wall region"
        seed_infos.append({"color": int(color), "coord": (r, c), "region": region_id})
    assert seed_infos[0]["region"] != seed_infos[1]["region"], "the two seeds must occupy different regions"

    covered_seed_regions: set[int] = set()
    seed_components = []
    for comp in components:
        seeds_here = [info for info in seed_infos if info["region"] in comp]
        assert len(seeds_here) == 1, "each graph component must contain exactly one seed"
        seed_components.append((comp, seeds_here[0]))
        covered_seed_regions.add(int(seeds_here[0]["region"]))
    assert covered_seed_regions == {info["region"] for info in seed_infos}, "seed coverage accounting failed"

    return {
        "grid": g,
        "bg": bg,
        "wall": wall,
        "wall_mask": wall_mask,
        "region_labels": region_labels,
        "graph": graph,
        "components": components,
        "seed_infos": seed_infos,
        "seed_components": seed_components,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every wall-separated region by alternating between the two uniquely colored seeds across wall adjacencies.

    Input structure:
    - The grid contains one dominant background color, one wall color, and exactly two singleton seed colors.
    - The wall pixels partition the non-wall cells into multiple 4-connected regions whose adjacency graph has exactly two connected components.
    - Each connected component contains exactly one seed region; repeated-color noise pixels may appear inside any region.

    Rule:
    - Identify the two seeds by frequency uniqueness and ignore all repeated-color noise.
    - Keep all wall pixels unchanged.
    - Treat non-wall cells as regions, connect two regions when they lie on opposite sides of the same one-cell wall separator, then process each graph component independently.
    - In each component, color the seed region with its own seed color and alternate with the other seed color across every wall adjacency until the whole component is filled.

    Key insight:
    - The output is fixed only after combining two cues: frequency uniqueness finds the two anchors, and wall-defined region adjacency determines how their two colors must alternate through two disconnected bipartite components.
    """
    parsed = _parse_input(grid)
    g = parsed["grid"]
    region_labels = parsed["region_labels"]
    graph = parsed["graph"]
    seed_colors = {info["color"] for info in parsed["seed_infos"]}

    out = np.full_like(g, parsed["wall"])
    for comp, seed_info in parsed["seed_components"]:
        anchor_region = int(seed_info["region"])
        anchor_color = int(seed_info["color"])
        other_color = int(next(color for color in seed_colors if color != anchor_color))

        parity = {anchor_region: 0}
        queue = deque([anchor_region])
        while queue:
            region = queue.popleft()
            for nxt in graph[region]:
                nxt_parity = 1 - parity[region]
                if nxt in parity:
                    assert parity[nxt] == nxt_parity, "region graph must be bipartite"
                else:
                    parity[nxt] = nxt_parity
                    queue.append(nxt)

        assert set(parity) == comp, "seed color must propagate through its whole graph component"
        for region_id, bit in parity.items():
            out[region_labels == region_id] = anchor_color if bit == 0 else other_color

    return out


# --------------- generator ---------------

PALETTE = list(range(10))
MAX_GENERATION_ATTEMPTS = 400
MIN_REGION_SPAN = 2
MIN_SECONDARY_SPAN = 5


def _rand_comp(total: int, parts: int, min_value: int) -> list[int]:
    extra = total - parts * min_value
    assert extra >= 0
    pieces = [min_value] * parts
    for _ in range(extra):
        pieces[random.randrange(parts)] += 1
    random.shuffle(pieces)
    return pieces


def _sample_cell_in_box(box: tuple[int, int, int, int], used: set[tuple[int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    preferred = []
    fallback = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in used:
                continue
            blocked = False
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    if (nr, nc) in used:
                        blocked = True
                        break
                if blocked:
                    break
            fallback.append((r, c))
            if not blocked:
                preferred.append((r, c))
    pool = preferred if preferred else fallback
    assert pool, "expected at least one free cell in region box"
    return random.choice(pool)


def _choose_seed_region(region_count: int) -> int:
    if region_count >= 3 and random.random() < 0.7:
        return random.randint(1, region_count - 2)
    return random.randrange(region_count)


def _sample_chamber_spec() -> dict:
    orientation = random.choice(["vertical", "horizontal"])
    region_count = random.choices([2, 3, 4], weights=[3, 4, 3])[0]
    min_height = 3 * region_count - 1 if orientation == "horizontal" else MIN_SECONDARY_SPAN
    min_width = 3 * region_count - 1 if orientation == "vertical" else MIN_SECONDARY_SPAN
    return {
        "orientation": orientation,
        "region_count": region_count,
        "min_h": min_height,
        "min_w": min_width,
    }


def _layout_chambers() -> tuple[tuple[int, int], list[dict]]:
    for _ in range(100):
        specs = [_sample_chamber_spec(), _sample_chamber_spec()]
        layout = random.choice(["side", "stack"])
        if layout == "side":
            interior_h = random.randint(max(spec["min_h"] for spec in specs), 13)
            interior_ws = [random.randint(spec["min_w"], min(spec["min_w"] + 3, 13)) for spec in specs]
            H = interior_h + 2
            W = sum(w + 2 for w in interior_ws)
            if H > 30 or W > 30:
                continue

            rects = []
            c0 = 0
            for spec, iw in zip(specs, interior_ws):
                rect = (0, c0, H - 1, c0 + iw + 1)
                rects.append({**spec, "rect": rect})
                c0 = rect[3] + 1
            return (H, W), rects

        interior_w = random.randint(max(spec["min_w"] for spec in specs), 13)
        interior_hs = [random.randint(spec["min_h"], min(spec["min_h"] + 3, 13)) for spec in specs]
        H = sum(h + 2 for h in interior_hs)
        W = interior_w + 2
        if H > 30 or W > 30:
            continue

        rects = []
        r0 = 0
        for spec, ih in zip(specs, interior_hs):
            rect = (r0, 0, r0 + ih + 1, W - 1)
            rects.append({**spec, "rect": rect})
            r0 = rect[2] + 1
        return (H, W), rects

    raise RuntimeError("failed to sample chamber layout")


def _build_chamber(grid: np.ndarray, wall: int, bg: int, spec: dict) -> dict:
    r0, c0, r1, c1 = spec["rect"]
    grid[r0 + 1 : r1, c0 + 1 : c1] = bg

    region_count = int(spec["region_count"])
    region_boxes: list[tuple[int, int, int, int]] = []

    if spec["orientation"] == "vertical":
        interior_width = c1 - c0 - 1
        widths = _rand_comp(interior_width - (region_count - 1), region_count, MIN_REGION_SPAN)
        cursor = c0 + 1
        for idx, width in enumerate(widths):
            region_boxes.append((r0 + 1, cursor, r1 - 1, cursor + width - 1))
            cursor += width
            if idx < region_count - 1:
                grid[r0 : r1 + 1, cursor] = wall
                cursor += 1
    else:
        interior_height = r1 - r0 - 1
        heights = _rand_comp(interior_height - (region_count - 1), region_count, MIN_REGION_SPAN)
        cursor = r0 + 1
        for idx, height in enumerate(heights):
            region_boxes.append((cursor, c0 + 1, cursor + height - 1, c1 - 1))
            cursor += height
            if idx < region_count - 1:
                grid[cursor, c0 : c1 + 1] = wall
                cursor += 1

    return {
        **spec,
        "region_boxes": region_boxes,
        "seed_region_index": _choose_seed_region(region_count),
    }


def _fill_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _render_expected(shape: tuple[int, int], wall: int, chambers: list[dict], seed_colors: tuple[int, int]) -> np.ndarray:
    expected = np.full(shape, int(wall), dtype=int)
    seed_a, seed_b = map(int, seed_colors)

    for chamber in chambers:
        anchor = int(chamber["seed_color"])
        other = seed_b if anchor == seed_a else seed_a
        seed_idx = int(chamber["seed_region_index"])
        for idx, box in enumerate(chamber["region_boxes"]):
            color = anchor if abs(idx - seed_idx) % 2 == 0 else other
            _fill_box(expected, box, color)
    return expected


def _scatter_noise(
    grid: np.ndarray,
    chambers: list[dict],
    bg: int,
    wall: int,
    seed_colors: tuple[int, int],
    used: set[tuple[int, int]],
) -> None:
    remaining = [color for color in PALETTE if color not in seed_colors and color not in {bg, wall}]
    random.shuffle(remaining)
    noise_color_count = random.randint(1, min(3, len(remaining)))
    noise_colors = remaining[:noise_color_count]

    all_boxes = [(ci, ri, box) for ci, chamber in enumerate(chambers) for ri, box in enumerate(chamber["region_boxes"])]
    mandatory = []
    for ci, chamber in enumerate(chambers):
        target_region = random.randrange(len(chamber["region_boxes"]))
        mandatory.append((ci, target_region, chamber["region_boxes"][target_region]))

    placements: list[tuple[tuple[int, int], int]] = []
    first_color = noise_colors[0]
    for ci, ri, box in mandatory:
        spot = _sample_cell_in_box(box, used | {pos for pos, _color in placements})
        placements.append((spot, first_color))

    for color in noise_colors:
        target_count = random.randint(2, 4)
        already = sum(1 for _spot, cc in placements if cc == color)
        need = target_count - already
        if need <= 0:
            continue
        choices = all_boxes[:]
        random.shuffle(choices)
        for _ci, _ri, box in choices:
            if need == 0:
                break
            spot = _sample_cell_in_box(box, used | {pos for pos, _color in placements})
            if spot not in {pos for pos, _color in placements}:
                placements.append((spot, color))
                need -= 1
        assert need == 0, "failed to place all repeated noise pixels"

    for (r, c), color in placements:
        grid[r, c] = int(color)


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    shape, chamber_specs = _layout_chambers()
    colors = random.sample(PALETTE, random.randint(5, 7))
    bg = int(colors[0])
    wall = int(colors[1])
    seed_colors = (int(colors[2]), int(colors[3]))

    grid = np.full(shape, wall, dtype=int)
    chambers = [_build_chamber(grid, wall, bg, spec) for spec in chamber_specs]

    used_special: set[tuple[int, int]] = set()
    for chamber, seed_color in zip(chambers, seed_colors):
        box = chamber["region_boxes"][chamber["seed_region_index"]]
        spot = _sample_cell_in_box(box, used_special)
        chamber["seed_color"] = int(seed_color)
        chamber["seed_coord"] = spot
        grid[spot] = int(seed_color)
        used_special.add(spot)

    _scatter_noise(grid, chambers, bg, wall, seed_colors, used_special)
    expected = _render_expected(shape, wall, chambers, seed_colors)

    counts = Counter(map(int, grid.flat))
    if counts[bg] <= counts[wall]:
        return None
    if counts[seed_colors[0]] != 1 or counts[seed_colors[1]] != 1:
        return None
    if any(freq == 1 for color, freq in counts.items() if color not in {bg, wall, seed_colors[0], seed_colors[1]}):
        return None

    try:
        parsed = _parse_input(grid)
        actual = main(grid)
    except AssertionError:
        return None

    if parsed["bg"] != bg or parsed["wall"] != wall:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        generated = _generate_attempt()
        if generated is not None:
            return generated[0]
    raise RuntimeError("failed to generate a valid puzzle input")
