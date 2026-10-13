from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 600


def _components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _border_distance(h: int, w: int, r: int, c: int) -> int:
    return min(r, c, h - 1 - r, w - 1 - c)


def _shell_cells(r: int, c: int, radius: int) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for cc in range(c - radius, c + radius + 1):
        cells.append((r - radius, cc))
        cells.append((r + radius, cc))
    for rr in range(r - radius + 1, r + radius):
        cells.append((rr, c - radius))
        cells.append((rr, c + radius))
    return cells


def _find_separator(grid: np.ndarray) -> int:
    candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    if not candidates:
        raise ValueError("separator column not found")
    if len(candidates) == 1:
        return candidates[0]
    for c in candidates:
        if np.any(grid[:, :c] != BG) and np.any(grid[:, c + 1 :] != BG):
            return c
    return candidates[0]


def _extract_templates(left: np.ndarray) -> Dict[int, List[int]]:
    templates: Dict[int, List[int]] = {}
    for comp in _components(left != BG):
        pts = np.array(comp, dtype=int)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        if h != w or h % 2 == 0 or np.any(patch == BG):
            continue
        center = h // 2
        core = int(patch[center, center])
        seq: List[int] = []
        ok = True
        for radius in range(1, center + 1):
            colors = {
                int(patch[rr, cc])
                for rr in range(h)
                for cc in range(w)
                if max(abs(rr - center), abs(cc - center)) == radius
            }
            if len(colors) != 1:
                ok = False
                break
            seq.append(int(next(iter(colors))))
        if ok and seq:
            templates[core] = seq
    if not templates:
        raise ValueError("template library not found")
    return templates


def _grow_canvas(
    h: int, w: int, seeds: Sequence[Tuple[int, int, int]], templates: Dict[int, List[int]]
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    out = np.full((h, w), BG, dtype=int)
    occupied = {(int(r), int(c)) for r, c, _color in seeds}
    for r, c, color in seeds:
        out[r, c] = int(color)

    ordered = sorted(
        ((int(r), int(c), int(color)) for r, c, color in seeds),
        key=lambda item: (_border_distance(h, w, item[0], item[1]), item[0], item[1]),
    )

    meta: List[Dict[str, int]] = []
    for r, c, color in ordered:
        seq = templates.get(int(color), [])
        max_radius = 0
        stop_code = 0
        if seq:
            radius = 1
            while True:
                ring = _shell_cells(r, c, radius)
                if any(rr < 0 or rr >= h or cc < 0 or cc >= w for rr, cc in ring):
                    stop_code = 1
                    break
                if any((rr, cc) in occupied for rr, cc in ring):
                    stop_code = 2
                    break
                ring_color = int(seq[(radius - 1) % len(seq)])
                for rr, cc in ring:
                    out[rr, cc] = ring_color
                    occupied.add((rr, cc))
                max_radius = radius
                radius += 1
        meta.append(
            {
                "r": int(r),
                "c": int(c),
                "color": int(color),
                "seq_len": int(len(seq)),
                "max_radius": int(max_radius),
                "stop_code": int(stop_code),
            }
        )
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each right-side seed with the left-side template sharing its core color,
    then grow full concentric square shells until the next shell would leave the
    canvas or hit a reserved/grown cell from another seed.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator(g)
    templates = _extract_templates(g[:, :sep])
    canvas = g[:, sep + 1 :]
    seeds = [(int(r), int(c), int(canvas[r, c])) for r, c in np.argwhere(canvas != BG)]
    if not seeds:
        return np.full_like(canvas, BG)
    out, _meta = _grow_canvas(canvas.shape[0], canvas.shape[1], seeds, templates)
    return out


def _make_template_patch(core: int, seq: Sequence[int]) -> np.ndarray:
    radius = len(seq)
    size = 2 * radius + 1
    center = radius
    patch = np.full((size, size), core, dtype=int)
    for r in range(size):
        for c in range(size):
            dist = max(abs(r - center), abs(c - center))
            patch[r, c] = int(core if dist == 0 else seq[dist - 1])
    return patch


def _far_enough(pos: Tuple[int, int], existing: Sequence[Tuple[int, int]], min_sep: int) -> bool:
    return all(max(abs(pos[0] - r), abs(pos[1] - c)) >= min_sep for r, c in existing)


def _random_position(
    h: int,
    w: int,
    existing: Sequence[Tuple[int, int]],
    min_sep: int,
    border_min: int,
    border_max: int | None = None,
) -> Tuple[int, int] | None:
    for _ in range(400):
        r = random.randint(0, h - 1)
        c = random.randint(0, w - 1)
        dist = _border_distance(h, w, r, c)
        if dist < border_min:
            continue
        if border_max is not None and dist > border_max:
            continue
        if _far_enough((r, c), existing, min_sep):
            return (r, c)
    return None


def _random_collision_pair(
    h: int, w: int, existing: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int]] | None:
    for _ in range(400):
        dist = random.randint(4, 5)
        horizontal = random.choice([True, False])
        if horizontal and w >= 3 * dist + 1 and h >= 2 * dist + 1:
            row = random.randint(dist, h - 1 - dist)
            c1 = random.randint(dist, w - 1 - 2 * dist)
            p1 = (row, c1)
            p2 = (row, c1 + dist)
        elif (not horizontal) and h >= 3 * dist + 1 and w >= 2 * dist + 1:
            col = random.randint(dist, w - 1 - dist)
            r1 = random.randint(dist, h - 1 - 2 * dist)
            p1 = (r1, col)
            p2 = (r1 + dist, col)
        else:
            continue
        if not _far_enough(p1, existing, 4) or not _far_enough(p2, existing, 4):
            continue
        return [p1, p2]
    return None


def _layout_seed_positions(mode: str, h: int, w: int) -> List[Tuple[int, int]] | None:
    positions: List[Tuple[int, int]] = []
    if mode == "boundary":
        target = random.randint(3, 4)
        core_pos = _random_position(h, w, positions, 6, border_min=4)
        if core_pos is None:
            return None
        positions.append(core_pos)
        for _ in range(2):
            pos = _random_position(h, w, positions, 6, border_min=1, border_max=2)
            if pos is None:
                return None
            positions.append(pos)
        while len(positions) < target:
            if random.random() < 0.5:
                pos = _random_position(h, w, positions, 6, border_min=3)
            else:
                pos = _random_position(h, w, positions, 6, border_min=1, border_max=2)
            if pos is None:
                return None
            positions.append(pos)
        return positions

    if mode == "collision":
        target = random.randint(3, 5)
        pair = _random_collision_pair(h, w, positions)
        if pair is None:
            return None
        positions.extend(pair)
        pos = _random_position(h, w, positions, 4, border_min=4)
        if pos is None:
            return None
        positions.append(pos)
        while len(positions) < target:
            if random.random() < 0.35:
                pos = _random_position(h, w, positions, 4, border_min=1, border_max=2)
            else:
                pos = _random_position(h, w, positions, 4, border_min=3)
            if pos is None:
                return None
            positions.append(pos)
        return positions

    if mode == "mixed":
        target = random.randint(4, 5)
        pair = _random_collision_pair(h, w, positions)
        if pair is None:
            return None
        positions.extend(pair)
        boundary_pos = _random_position(h, w, positions, 4, border_min=1, border_max=2)
        if boundary_pos is None:
            return None
        positions.append(boundary_pos)
        interior_pos = _random_position(h, w, positions, 4, border_min=4)
        if interior_pos is None:
            return None
        positions.append(interior_pos)
        while len(positions) < target:
            if random.random() < 0.5:
                pos = _random_position(h, w, positions, 4, border_min=3)
            else:
                pos = _random_position(h, w, positions, 4, border_min=1, border_max=2)
            if pos is None:
                return None
            positions.append(pos)
        return positions

    return None


def _accept_layout(mode: str, h: int, w: int, seeds: Sequence[Tuple[int, int, int]], templates: Dict[int, List[int]]) -> bool:
    out, meta = _grow_canvas(h, w, seeds, templates)
    boundary_count = sum(item["stop_code"] == 1 for item in meta)
    collision_count = sum(item["stop_code"] == 2 for item in meta)
    cycled = any(item["max_radius"] > item["seq_len"] for item in meta if item["seq_len"] > 0)
    if not cycled:
        return False
    if np.count_nonzero(out) <= len(seeds) + 8:
        return False
    if np.count_nonzero(out) >= (h * w * 2) // 5:
        return False
    if mode == "boundary" and (boundary_count < 2 or collision_count != 0):
        return False
    if mode == "collision" and collision_count < 1:
        return False
    if mode == "mixed" and (boundary_count < 1 or collision_count < 1):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    left_w = random.randint(9, 13)
    right_w = random.randint(14, 20)
    mode = random.choice(["boundary", "collision", "mixed"])

    template_count = random.randint(2, 4)
    lengths = [random.choice([1, 2]) for _ in range(template_count)]
    if 2 not in lengths:
        lengths[random.randrange(template_count)] = 2
    if template_count >= 2 and 1 not in lengths:
        lengths[random.randrange(template_count)] = 1

    max_size = 2 * max(lengths) + 1
    if max_size > left_w - 2:
        return None

    cores = random.sample(PALETTE, template_count)
    templates: Dict[int, List[int]] = {}
    template_patches: List[Tuple[int, np.ndarray]] = []
    for core, length in zip(cores, lengths):
        seq = random.sample([color for color in PALETTE if color != core], length)
        templates[int(core)] = [int(color) for color in seq]
        template_patches.append((int(core), _make_template_patch(int(core), seq)))
    random.shuffle(template_patches)

    gaps = [random.randint(2, 3) for _ in range(template_count - 1)]
    block_h = sum(patch.shape[0] for _core, patch in template_patches) + sum(gaps)
    if block_h > h - 2:
        return None

    positions = _layout_seed_positions(mode, h, right_w)
    if positions is None:
        return None

    colors = random.sample(cores, min(len(cores), len(positions)))
    while len(colors) < len(positions):
        colors.append(random.choice(cores))
    seeds = [(int(r), int(c), int(color)) for (r, c), color in zip(positions, colors)]
    if len({color for _r, _c, color in seeds}) < 2:
        return None
    if not _accept_layout(mode, h, right_w, seeds, templates):
        return None

    grid = np.full((h, left_w + 1 + right_w), BG, dtype=int)
    grid[:, left_w] = SEP

    row = random.randint(1, h - block_h - 1)
    for idx, (_core, patch) in enumerate(template_patches):
        size = patch.shape[0]
        col = random.randint(1, left_w - size - 1)
        grid[row : row + size, col : col + size] = patch
        row += size
        if idx < len(gaps):
            row += gaps[idx]

    for r, c, color in seeds:
        grid[r, left_w + 1 + c] = int(color)

    expected, _meta = _grow_canvas(h, right_w, seeds, templates)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
