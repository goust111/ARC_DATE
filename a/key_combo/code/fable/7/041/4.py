import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
MAX_GENERATION_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], tuple[int, int, int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((coords, (r0, c0, r1, c1)))
    return comps


def _find_legend(grid: np.ndarray) -> tuple[int, list[int]]:
    h, w = grid.shape
    candidates = []
    for col in (0, w - 1):
        rows = np.flatnonzero(grid[:, col] != BG)
        if len(rows) < 2:
            continue
        if int(rows[-1] - rows[0] + 1) != int(len(rows)):
            continue
        if rows[0] != 0 and rows[-1] != h - 1:
            continue
        inner_col = 1 if col == 0 else w - 2
        if np.any(grid[rows, inner_col] != BG):
            continue
        colors = [int(grid[r, col]) for r in rows]
        if len(set(colors)) != len(colors):
            continue
        candidates.append((len(rows), col, rows))
    if not candidates:
        raise AssertionError("legend column not found")
    _len, col, rows = max(candidates)
    return int(col), [int(grid[r, col]) for r in rows[::-1]]


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers = []
    seen_boxes = set()
    for color in sorted(set(map(int, grid.flatten())) - {BG}):
        for _coords, bbox in _components(grid == color):
            r0, c0, r1, c1 = bbox
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            if bbox in seen_boxes:
                continue
            if not np.all(grid[r0, c0 : c1 + 1] == color):
                continue
            if not np.all(grid[r0 : r1 + 1, c0] == color):
                continue
            if not np.all(grid[r0 : r1 + 1, c1] == color):
                continue
            bottom = grid[r1, c0 : c1 + 1] == color
            if not bottom[0] or not bottom[-1]:
                continue
            holes = [c0 + 1 + i for i, ok in enumerate(bottom[1:-1]) if not ok]
            if len(holes) > 1:
                continue
            containers.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "border_color": int(color),
                    "hole_col": None if not holes else int(holes[0]),
                }
            )
            seen_boxes.add(bbox)
    containers.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    if not containers:
        raise AssertionError("no containers found")
    return containers


def _container_particles(grid: np.ndarray, container: dict) -> Counter:
    r0, c0, r1, c1 = container["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    colors = [int(v) for v in interior.flatten() if int(v) != BG]
    return Counter(colors)


def _relay_target(idx: int, containers: list[dict]) -> int:
    memo = {}

    def resolve(i: int) -> int:
        if i in memo:
            return memo[i]
        hole_col = containers[i]["hole_col"]
        if hole_col is None:
            memo[i] = i
            return i
        r0, _c0, r1, _c1 = containers[i]["bbox"]
        candidates = []
        for j, other in enumerate(containers):
            if j == i:
                continue
            or0, oc0, _or1, oc1 = other["bbox"]
            if or0 > r1 and oc0 < hole_col < oc1:
                candidates.append((or0, j))
        if not candidates:
            raise AssertionError(f"leak at container {i} has no receiver")
        memo[i] = resolve(min(candidates)[1])
        return memo[i]

    return resolve(idx)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route particles through leaking rectangular containers and restack them in the intact receivers by legend order.

    Input structure:
    - The background is black.
    - Several single-color hollow rectangles act as containers; some have exactly one missing cell on the bottom edge.
    - Colored particle pixels are scattered inside container interiors, and one edge column holds a vertical color legend.

    Rule:
    - An intact container keeps all particles that ultimately belong to it and repacks them into full horizontal layers from bottom to top.
    - A leaking container keeps nothing: every particle inside it falls through the bottom hole column into the first lower container spanning that column, repeating until an intact container is reached.
    - The legend column gives the sedimentation priority from bottom to top, so the legend's bottom color fills the lowest layers first.

    Key insight:
    - Original particle positions are irrelevant; the solution depends on detecting leak chains, preserving per-color counts, and then rebuilding each terminal container as ordered full-width layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    legend_col, legend_order = _find_legend(g)
    containers = _find_containers(g)

    out = g.copy()
    particle_counts = [_container_particles(g, container) for container in containers]
    totals = {idx: Counter() for idx, container in enumerate(containers) if container["hole_col"] is None}

    for idx, counts in enumerate(particle_counts):
        target = _relay_target(idx, containers)
        totals[target].update(counts)

    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = BG

    legend_set = set(legend_order)
    for idx, counts in totals.items():
        r0, c0, r1, c1 = containers[idx]["bbox"]
        width = c1 - c0 - 1
        row = r1 - 1
        extras = set(counts) - legend_set
        if extras:
            raise AssertionError(f"particles use colors absent from legend: {sorted(extras)}")
        for color in legend_order:
            amount = int(counts.get(color, 0))
            if amount == 0:
                continue
            layers, rem = divmod(amount, width)
            if rem != 0:
                raise AssertionError("particle count does not form full layers")
            if row - layers + 1 < r0 + 1:
                raise AssertionError("too many layers for terminal container")
            for _ in range(layers):
                out[row, c0 + 1 : c1] = color
                row -= 1

    if legend_col not in (0, g.shape[1] - 1):
        raise AssertionError("legend must stay on an outer column")
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    prev = -1
    out = []
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_layer_counts(width: int, max_layers: int, legend_order: list[int]) -> Counter:
    usable = random.randint(2, min(len(legend_order), max_layers))
    used_colors = random.sample(legend_order, usable)
    total_layers = random.randint(usable, max_layers)
    extras = _random_partition(total_layers - usable, usable) if total_layers > usable else [0] * usable
    counts = Counter()
    for color, extra in zip(used_colors, extras):
        counts[color] = (1 + extra) * width
    return counts


def _build_chain(chain_len: int, legend_order: list[int]) -> dict:
    bottom_width = random.randint(3, 5)
    bottom_height = random.randint(3, 5)
    containers = [
        {
            "r0": 0,
            "c0": 0,
            "inner_w": bottom_width,
            "inner_h": bottom_height,
            "hole_col": None,
        }
    ]
    counts = _choose_layer_counts(bottom_width, bottom_height, legend_order)
    for _ in range(chain_len - 1):
        lower = containers[-1]
        inner_w = random.randint(3, 5)
        inner_h = random.randint(2, 4)
        gap = random.randint(1, 3)
        upper_c0_min = lower["c0"] - inner_w + 1
        upper_c0_max = lower["c0"] + lower["inner_w"] - 1
        upper_c0 = random.randint(upper_c0_min, upper_c0_max)
        overlap_cols = [
            col
            for col in range(upper_c0 + 1, upper_c0 + inner_w + 1)
            if lower["c0"] + 1 <= col <= lower["c0"] + lower["inner_w"]
        ]
        hole_col = random.choice(overlap_cols)
        upper_r0 = lower["r0"] - gap - (inner_h + 2)
        containers.append(
            {
                "r0": upper_r0,
                "c0": upper_c0,
                "inner_w": inner_w,
                "inner_h": inner_h,
                "hole_col": hole_col,
            }
        )

    min_r0 = min(container["r0"] for container in containers)
    min_c0 = min(container["c0"] for container in containers)
    for container in containers:
        container["r0"] -= min_r0
        container["c0"] -= min_c0
        if container["hole_col"] is not None:
            container["hole_col"] -= min_c0
        container["r1"] = container["r0"] + container["inner_h"] + 1
        container["c1"] = container["c0"] + container["inner_w"] + 1
    containers.sort(key=lambda item: item["r0"])
    return {
        "containers": containers,
        "counts": counts,
        "height": max(container["r1"] for container in containers) + 1,
        "width": max(container["c1"] for container in containers) + 1,
    }


def _distribute_particles(grid: np.ndarray, containers: list[dict], counts: Counter) -> None:
    particle_colors = []
    for color, amount in counts.items():
        particle_colors.extend([int(color)] * int(amount))
    random.shuffle(particle_colors)

    interior_cells = []
    capacities = []
    for container in containers:
        cells = [
            (r, c)
            for r in range(container["r0"] + 1, container["r1"])
            for c in range(container["c0"] + 1, container["c1"])
        ]
        interior_cells.append(cells)
        capacities.append(len(cells))

    total = len(particle_colors)
    allocation = [0] * len(containers)
    for i in range(len(containers)):
        if total > 0:
            allocation[i] += 1
            total -= 1
    while total > 0:
        valid = [i for i, cap in enumerate(capacities) if allocation[i] < cap]
        pick = random.choice(valid)
        allocation[pick] += 1
        total -= 1

    cells = []
    for i, count in enumerate(allocation):
        cells.extend(random.sample(interior_cells[i], count))
    random.shuffle(cells)

    for (r, c), color in zip(cells, particle_colors):
        grid[r, c] = color


def _draw_container(grid: np.ndarray, container: dict, border_color: int) -> None:
    r0, c0, r1, c1 = container["r0"], container["c0"], container["r1"], container["c1"]
    grid[r0, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    if container["hole_col"] is not None:
        grid[r1, container["hole_col"]] = BG


def _gen_attempt() -> np.ndarray | None:
    legend_len = random.randint(2, 4)
    legend_order = random.sample(PALETTE, legend_len)

    num_chains = random.randint(1, 2)
    chain_lengths = [random.randint(2, 3)]
    for _ in range(num_chains - 1):
        chain_lengths.append(random.randint(1, 3))

    chains = [_build_chain(length, legend_order) for length in chain_lengths]
    total_containers = sum(len(chain["containers"]) for chain in chains)
    border_pool = [c for c in PALETTE if c not in legend_order]
    if total_containers <= len(border_pool):
        border_colors = random.sample(border_pool, total_containers)
    else:
        border_colors = [random.choice(border_pool) for _ in range(total_containers)]

    chain_gap = random.randint(2, 4)
    left_margin = 2
    right_margin = 2
    content_width = sum(chain["width"] for chain in chains) + chain_gap * (len(chains) - 1)
    slack_w = random.randint(0, 3)
    width = content_width + left_margin + right_margin + 1 + slack_w
    if width > 30:
        return None

    content_height = max(max(chain["height"] for chain in chains), legend_len)
    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    slack_h = random.randint(0, 3)
    height = content_height + top_margin + bottom_margin + slack_h
    if height > 30:
        return None

    legend_side = random.choice(["left", "right"])
    legend_anchor = random.choice(["top", "bottom"])
    grid = np.zeros((height, width), dtype=int)

    if legend_side == "left":
        x = 2
        chain_offsets = []
        for chain in chains:
            chain_offsets.append(x)
            x += chain["width"] + chain_gap
        legend_col = 0
    else:
        x = left_margin
        chain_offsets = []
        for chain in chains:
            chain_offsets.append(x)
            x += chain["width"] + chain_gap
        legend_col = width - 1

    if legend_anchor == "top":
        legend_rows = list(range(legend_len))
    else:
        legend_rows = list(range(height - legend_len, height))
    for row, color in zip(legend_rows[::-1], legend_order):
        grid[row, legend_col] = color

    color_iter = iter(border_colors)
    used_particle_colors = set()
    for chain, x0 in zip(chains, chain_offsets):
        y0 = random.randint(top_margin, height - bottom_margin - chain["height"])
        for container in chain["containers"]:
            placed = dict(container)
            placed["r0"] += y0
            placed["r1"] += y0
            placed["c0"] += x0
            placed["c1"] += x0
            if placed["hole_col"] is not None:
                placed["hole_col"] += x0
            container.update(placed)
            _draw_container(grid, container, next(color_iter))
        _distribute_particles(grid, chain["containers"], chain["counts"])
        used_particle_colors.update(chain["counts"])

    if len(used_particle_colors) < 2:
        return None
    if main(grid).shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid funnel-settling input")
