from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_ORDER = ("up", "down", "left", "right")
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return components


def _detect_guard(grid: np.ndarray, comp: dict[str, object]) -> dict[str, object] | None:
    color = int(comp["color"])
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height != width or height not in (3, 5, 7):
        return None

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if int(coords.shape[0]) != int(height * width - 2):
        return None
    if not bool(np.isin(patch, [BG, color]).all()):
        return None
    if int(np.count_nonzero(patch == BG)) != 2:
        return None

    center = height // 2
    if int(patch[center, center]) != BG:
        return None

    openings: list[str] = []
    for name, (dr, dc) in DIRS.items():
        if int(patch[center + dr, center + dc]) == BG:
            openings.append(name)
    if len(openings) != 1:
        return None

    direction = openings[0]
    layer_count = (height - 3) // 2
    dr, dc = DIRS[direction]
    start_r = int(r0 + center + dr * (center + 1))
    start_c = int(c0 + center + dc * (center + 1))

    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "dir_vec": (dr, dc),
        "layers": int(layer_count),
        "ray_len": int(2 * (layer_count + 1)),
        "start": (start_r, start_c),
    }


def _find_guards(grid: np.ndarray) -> list[dict[str, object]]:
    guards: list[dict[str, object]] = []
    for comp in _find_components(grid):
        guard = _detect_guard(grid, comp)
        if guard is not None:
            guards.append(guard)
    guards.sort(key=lambda g: tuple(int(v) for v in g["bbox"]))
    return guards


def _trace_ray(grid: np.ndarray, guard: dict[str, object]) -> list[tuple[int, int]]:
    h, w = grid.shape
    dr, dc = guard["dir_vec"]
    r, c = map(int, guard["start"])
    cells: list[tuple[int, int]] = []
    for _ in range(int(guard["ray_len"])):
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(grid[r, c]) != BG:
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the background cells watched by each hollow guard square.

    Input structure:
    - Background is black.
    - Each guard is a 3x3 monochrome square whose center is black and whose
      open-side midpoint is also black, so the hole points up, down, left, or right.
    - A guard may be wrapped by 0, 1, or 2 tight same-color square fence layers.

    Rule:
    - Detect each guard's viewing direction from the extra black hole beside its center.
    - Count its fence layers N from the size of the concentric square shell.
    - From the exterior cell just beyond that open side, cast a straight ray in the
      facing direction for (N+1)*2 cells, painting only black cells in the guard's color.
    - Stop the ray early if it reaches the grid boundary or a non-black cell.

    Key insight:
    - The inner hole tells you which way the guard faces, while the number of
      concentric same-color fence layers determines the ray length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for guard in _find_guards(g):
        for r, c in _trace_ray(g, guard):
            out[r, c] = int(guard["color"])
    return out


def _make_guard_patch(color: int, layers: int, direction: str) -> np.ndarray:
    size = 3 + 2 * int(layers)
    patch = np.full((size, size), int(color), dtype=int)
    center = size // 2
    patch[center, center] = BG
    dr, dc = DIRS[direction]
    patch[center + dr, center + dc] = BG
    return patch


def _boxes_far_enough(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    margin: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 < br0 - margin
        or br1 < ar0 - margin
        or ac1 < bc0 - margin
        or bc1 < ac0 - margin
    )


def _sample_layers(n_guards: int) -> list[int]:
    if n_guards == 2:
        layers = random.sample([0, 1, 2], 2)
    else:
        layers = [0, 1, 2]
        while len(layers) < n_guards:
            layers.append(random.randint(0, 2))
        random.shuffle(layers)
    return layers


def _sample_directions(n_guards: int) -> list[str]:
    while True:
        directions = [random.choice(DIR_ORDER) for _ in range(n_guards)]
        if n_guards == 1 or len(set(directions)) >= 2:
            return directions


def _sample_position(
    height: int,
    width: int,
    size: int,
    ray_len: int,
    direction: str,
    mode: str,
) -> tuple[int, int] | None:
    if direction in ("up", "down"):
        axis_room = height - size
    else:
        axis_room = width - size

    if axis_room < 1:
        return None

    if mode == "full":
        if axis_room < ray_len:
            return None
        available = random.randint(ray_len, axis_room)
    else:
        max_trunc = min(ray_len - 1, axis_room)
        if max_trunc < 1:
            return None
        available = random.randint(1, max_trunc)

    if direction == "up":
        r0 = available
        c0 = random.randint(0, width - size)
    elif direction == "down":
        r0 = height - size - available
        c0 = random.randint(0, width - size)
    elif direction == "left":
        r0 = random.randint(0, height - size)
        c0 = available
    else:
        r0 = random.randint(0, height - size)
        c0 = width - size - available

    return int(r0), int(c0)


def _render_candidate(
    height: int,
    width: int,
    specs: list[dict[str, object]],
    truncated: bool,
) -> np.ndarray | None:
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["size"]), reverse=True)
    special_idx = random.randrange(len(specs)) if truncated else None
    placements: dict[int, dict[str, object]] = {}

    for idx in order:
        spec = specs[idx]
        mode = "trunc" if idx == special_idx else "full"
        placed = None
        for _ in range(300):
            pos = _sample_position(
                height,
                width,
                int(spec["size"]),
                int(spec["ray_len"]),
                str(spec["direction"]),
                mode,
            )
            if pos is None:
                return None
            r0, c0 = pos
            size = int(spec["size"])
            bbox = (r0, c0, r0 + size - 1, c0 + size - 1)
            if all(_boxes_far_enough(bbox, tuple(map(int, other["bbox"])), margin=1) for other in placements.values()):
                placed = {"bbox": bbox, "r0": r0, "c0": c0}
                break
        if placed is None:
            return None
        placements[idx] = placed

    grid = np.zeros((height, width), dtype=int)
    for idx, placement in placements.items():
        spec = specs[idx]
        patch = _make_guard_patch(int(spec["color"]), int(spec["layers"]), str(spec["direction"]))
        r0 = int(placement["r0"])
        c0 = int(placement["c0"])
        size = patch.shape[0]
        grid[r0 : r0 + size, c0 : c0 + size] = patch
    return grid


def _valid_candidate(grid: np.ndarray, expected: int, truncated: bool) -> bool:
    guards = _find_guards(grid)
    if len(guards) != expected:
        return False

    traces = [_trace_ray(grid, guard) for guard in guards]
    if not any(traces):
        return False

    used: set[tuple[int, int]] = set()
    for trace in traces:
        overlap = used.intersection(trace)
        if overlap:
            return False
        used.update(trace)

    lengths = [len(trace) for trace in traces]
    full_lengths = [int(guard["ray_len"]) for guard in guards]
    if truncated:
        if not any(actual < target for actual, target in zip(lengths, full_lengths)):
            return False
    else:
        if any(actual < target for actual, target in zip(lengths, full_lengths)):
            return False

    out = main(grid)
    if not bool((out != grid).any()):
        return False
    return True


def _generate_candidate() -> np.ndarray | None:
    height = random.randint(15, 30)
    width = random.randint(15, 30)
    n_guards = random.randint(2, 4)
    truncated = random.choice([False, True])

    colors = random.sample(list(range(1, 10)), n_guards)
    layers = _sample_layers(n_guards)
    directions = _sample_directions(n_guards)

    specs = []
    for color, layer, direction in zip(colors, layers, directions):
        specs.append(
            {
                "color": int(color),
                "layers": int(layer),
                "direction": str(direction),
                "size": int(3 + 2 * layer),
                "ray_len": int(2 * (layer + 1)),
            }
        )

    random.shuffle(specs)
    grid = _render_candidate(height, width, specs, truncated)
    if grid is None or not _valid_candidate(grid, n_guards, truncated):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
