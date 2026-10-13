from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
GREY = 5
DIRS = ("up", "down", "left", "right")
GEN_RETRIES = 500
PLACE_RETRIES = 200
PARTICLE_COLORS = tuple(color for color in range(10) if color not in (BG, GREY))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _u_mask(direction: str, depth: int, opening_width: int) -> np.ndarray:
    if direction == "up":
        mask = np.zeros((depth + 1, opening_width + 2), dtype=bool)
        mask[:, 0] = True
        mask[:, -1] = True
        mask[-1, :] = True
        return mask
    if direction == "down":
        mask = np.zeros((depth + 1, opening_width + 2), dtype=bool)
        mask[:, 0] = True
        mask[:, -1] = True
        mask[0, :] = True
        return mask
    if direction == "left":
        mask = np.zeros((opening_width + 2, depth + 1), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, -1] = True
        return mask
    assert direction == "right"
    mask = np.zeros((opening_width + 2, depth + 1), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    return mask


def _parse_container(coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True

    matches: list[dict] = []
    if 3 <= h <= 5 and 4 <= w <= 6:
        depth = h - 1
        opening_width = w - 2
        if 2 <= depth <= 4 and 2 <= opening_width <= 4:
            for direction in ("up", "down"):
                if np.array_equal(local, _u_mask(direction, depth, opening_width)):
                    matches.append(
                        {
                            "direction": direction,
                            "depth": depth,
                            "opening_width": opening_width,
                        }
                    )
    if 4 <= h <= 6 and 3 <= w <= 5:
        opening_width = h - 2
        depth = w - 1
        if 2 <= depth <= 4 and 2 <= opening_width <= 4:
            for direction in ("left", "right"):
                if np.array_equal(local, _u_mask(direction, depth, opening_width)):
                    matches.append(
                        {
                            "direction": direction,
                            "depth": depth,
                            "opening_width": opening_width,
                        }
                    )
    if len(matches) != 1:
        return None

    info = matches[0]
    info["bbox"] = (r0, c0, r1, c1)

    if info["direction"] in ("up", "down"):
        center_c = (c0 + 1 + c1 - 1) // 2
        if info["direction"] == "up":
            info["stack_cells"] = [(r, center_c) for r in range(r1 - 1, r0 - 1, -1)]
        else:
            info["stack_cells"] = [(r, center_c) for r in range(r0 + 1, r1 + 1)]
    else:
        center_r = (r0 + 1 + r1 - 1) // 2
        if info["direction"] == "left":
            info["stack_cells"] = [(center_r, c) for c in range(c1 - 1, c0 - 1, -1)]
        else:
            info["stack_cells"] = [(center_r, c) for c in range(c0 + 1, c1 + 1)]
    return info


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for comp in _components(grid == GREY):
        parsed = _parse_container(comp)
        assert parsed is not None, "every grey component must be a valid U-shaped container"
        containers.append(parsed)
    containers.sort(key=lambda item: item["bbox"])
    assert containers, "expected at least one grey container"
    return containers


def _in_any_bbox(containers: list[dict], r: int, c: int) -> bool:
    for cont in containers:
        r0, c0, r1, c1 = cont["bbox"]
        if r0 <= r <= r1 and c0 <= c <= c1:
            return True
    return False


def _particle_hit(cont: dict, r: int, c: int) -> tuple[int, int] | None:
    r0, c0, r1, c1 = cont["bbox"]
    direction = cont["direction"]
    if direction == "up":
        if c0 < c < c1 and r < r0:
            return r0 - r, c - (c0 + 1)
        return None
    if direction == "down":
        if c0 < c < c1 and r > r1:
            return r - r1, c - (c0 + 1)
        return None
    if direction == "left":
        if r0 < r < r1 and c < c0:
            return c0 - c, r - (r0 + 1)
        return None
    assert direction == "right"
    if r0 < r < r1 and c > c1:
        return c - c1, r - (r0 + 1)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Capture external particles along each container opening ray and stack them
    from the closed end toward the opening, preserving particle colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    containers = _find_containers(g)

    out = g.copy()
    captured_by_container: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    captured_positions: set[tuple[int, int]] = set()

    for raw_r, raw_c in np.argwhere((g != BG) & (g != GREY)):
        r = int(raw_r)
        c = int(raw_c)
        if _in_any_bbox(containers, r, c):
            continue
        color = int(g[r, c])
        hits: list[tuple[int, int, int]] = []
        for idx, cont in enumerate(containers):
            hit = _particle_hit(cont, r, c)
            if hit is not None:
                dist, ray_idx = hit
                hits.append((dist, ray_idx, idx))
        if not hits:
            continue
        dist, ray_idx, winner = min(hits, key=lambda item: (item[0], item[2]))
        captured_by_container[winner].append((dist, ray_idx, r, c, color))
        captured_positions.add((r, c))

    for r, c in captured_positions:
        out[r, c] = BG

    for idx, cont in enumerate(containers):
        stack_cells = cont["stack_cells"]
        captured = sorted(captured_by_container.get(idx, []), key=lambda item: (item[0], item[1], item[2], item[3]))
        for slot, (_dist, _ray_idx, _r, _c, color) in zip(stack_cells, captured):
            out[slot] = int(color)

    return out


def _container_size(direction: str, depth: int, opening_width: int) -> tuple[int, int]:
    mask = _u_mask(direction, depth, opening_width)
    return int(mask.shape[0]), int(mask.shape[1])


def _draw_container(grid: np.ndarray, cont: dict) -> None:
    direction = cont["direction"]
    depth = cont["depth"]
    opening_width = cont["opening_width"]
    r0, c0, _, _ = cont["bbox"]
    mask = _u_mask(direction, depth, opening_width)
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = GREY


def _expanded_bbox(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _sample_container_placement(
    rng: random.Random,
    shape: tuple[int, int],
    existing: list[dict],
    direction: str,
    depth: int,
    opening_width: int,
) -> dict | None:
    h, w = shape
    box_h, box_w = _container_size(direction, depth, opening_width)

    if direction == "up":
        row_lo, row_hi = 4, h - box_h - 1
        col_lo, col_hi = 1, w - box_w - 1
    elif direction == "down":
        row_lo, row_hi = 1, h - box_h - 4
        col_lo, col_hi = 1, w - box_w - 1
    elif direction == "left":
        row_lo, row_hi = 1, h - box_h - 1
        col_lo, col_hi = 4, w - box_w - 1
    else:
        row_lo, row_hi = 1, h - box_h - 1
        col_lo, col_hi = 1, w - box_w - 4

    if row_lo > row_hi or col_lo > col_hi:
        return None

    for _ in range(PLACE_RETRIES):
        r0 = rng.randint(row_lo, row_hi)
        c0 = rng.randint(col_lo, col_hi)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        padded = _expanded_bbox(box, pad=1, h=h, w=w)
        if any(_boxes_overlap(padded, _expanded_bbox(other["bbox"], pad=1, h=h, w=w)) for other in existing):
            continue
        cont = {
            "direction": direction,
            "depth": depth,
            "opening_width": opening_width,
            "bbox": box,
        }
        if direction in ("up", "down"):
            cont["outward_space"] = r0 if direction == "up" else h - 1 - box[2]
        else:
            cont["outward_space"] = c0 if direction == "left" else w - 1 - box[3]
        return cont
    return None


def _candidate_map(shape: tuple[int, int], containers: list[dict]) -> tuple[dict[int, dict[int, list[tuple[int, int]]]], dict[int, dict[int, list[tuple[int, int]]]], list[tuple[int, int]]]:
    h, w = shape
    owned: dict[int, dict[int, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    contested: dict[int, dict[int, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
    free: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if _in_any_bbox(containers, r, c):
                continue
            hits: list[tuple[int, int, int]] = []
            for idx, cont in enumerate(containers):
                hit = _particle_hit(cont, r, c)
                if hit is None:
                    continue
                dist, ray_idx = hit
                hits.append((dist, ray_idx, idx))
            if not hits:
                free.append((r, c))
                continue
            dist, _ray_idx, winner = min(hits, key=lambda item: (item[0], item[2]))
            owned[winner][dist].append((r, c))
            if len(hits) >= 2:
                best = min(item[0] for item in hits)
                if sum(item[0] == best for item in hits) == 1:
                    contested[winner][dist].append((r, c))
    return owned, contested, free


def _place_particles(
    rng: random.Random,
    grid: np.ndarray,
    containers: list[dict],
) -> bool:
    owned, contested, free = _candidate_map(grid.shape, containers)
    selections: set[tuple[int, int]] = set()

    forced_change = False
    overflow_used = False
    contested_used = False

    for idx, cont in enumerate(containers):
        depth = int(cont["depth"])
        distances = sorted(int(d) for d in owned.get(idx, {}))
        if not distances:
            continue

        max_take = min(len(distances), max(depth + 2, 1))
        if not max_take:
            continue

        if not overflow_used and len(distances) > depth and rng.random() < 0.45:
            take = rng.randint(depth + 1, min(len(distances), depth + 2))
            overflow_used = True
        else:
            low = 0 if forced_change else 1
            take = rng.randint(low, min(depth, len(distances)))
        if take == 0:
            continue

        chosen_dists = rng.sample(distances, take)
        if not contested_used:
            contested_dists = [d for d in chosen_dists if contested.get(idx, {}).get(d)]
            if contested_dists and rng.random() < 0.6:
                d = rng.choice(contested_dists)
                cell = rng.choice(contested[idx][d])
                selections.add(cell)
                contested_used = True
                chosen_dists.remove(d)

        for dist in chosen_dists:
            cell = rng.choice(owned[idx][dist])
            selections.add(cell)
        forced_change = True

    if not selections:
        possible = [(idx, dist) for idx, by_dist in owned.items() for dist in by_dist]
        if not possible:
            return False
        idx, dist = rng.choice(possible)
        selections.add(rng.choice(owned[idx][dist]))

    for r, c in selections:
        grid[r, c] = rng.choice(PARTICLE_COLORS)

    rng.shuffle(free)
    stray_cap = min(len(free), 2 + len(containers))
    stray_count = rng.randint(0, stray_cap)
    for r, c in free[:stray_count]:
        if grid[r, c] != BG:
            continue
        grid[r, c] = rng.choice(PARTICLE_COLORS)

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 26)
    w = rng.randint(18, 26)
    n_containers = rng.randint(1, 4)
    directions = rng.sample(DIRS, n_containers)

    specs: list[dict] = []
    for direction in directions:
        depth = rng.randint(2, 4)
        opening_width = rng.randint(2, 4)
        placed = _sample_container_placement(rng, (h, w), specs, direction, depth, opening_width)
        if placed is None:
            return None
        specs.append(placed)

    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        _draw_container(grid, spec)

    containers = _find_containers(grid)
    if not _place_particles(rng, grid, containers):
        return None

    particle_mask = (grid != BG) & (grid != GREY)
    if int(particle_mask.sum()) < 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int(((out != BG) & (out != GREY)).sum()) < int(particle_mask.sum()):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
