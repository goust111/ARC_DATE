from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
ORIENTATIONS = ("up", "down", "left", "right")
COLOR_POOL = list(range(1, 10))
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_for_color(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    starts = np.argwhere(grid == color)
    for sr, sc in starts:
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        coords = np.array(pts, dtype=int)
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        components.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": int(len(pts)),
            }
        )
    return components


def _expected_u_cells(r0: int, c0: int, r1: int, c1: int, orient: str) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if orient == "up":
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
    elif orient == "down":
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
        cells.update((r0, c) for c in range(c0, c1 + 1))
    elif orient == "left":
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
    elif orient == "right":
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
        cells.update((r, c0) for r in range(r0, r1 + 1))
    else:
        raise AssertionError(f"unknown orientation: {orient}")
    return cells


def _match_container(component: dict[str, object]) -> dict[str, int | str] | None:
    coords = component["coords"]
    assert isinstance(coords, np.ndarray)
    cells = {tuple(map(int, pt)) for pt in coords}
    r0, c0, r1, c1 = map(int, component["bbox"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    matches: list[tuple[str, int]] = []

    if height >= 2 and width >= 3:
        if cells == _expected_u_cells(r0, c0, r1, c1, "up"):
            matches.append(("up", height - 1))
        if cells == _expected_u_cells(r0, c0, r1, c1, "down"):
            matches.append(("down", height - 1))
    if height >= 3 and width >= 2:
        if cells == _expected_u_cells(r0, c0, r1, c1, "left"):
            matches.append(("left", width - 1))
        if cells == _expected_u_cells(r0, c0, r1, c1, "right"):
            matches.append(("right", width - 1))

    if len(matches) != 1:
        return None

    orient, depth = matches[0]
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "height": height,
        "width": width,
        "orient": orient,
        "depth": depth,
    }


def _is_aligned_droplet(spec: dict[str, int | str], r: int, c: int) -> bool:
    orient = spec["orient"]
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    r1 = int(spec["r1"])
    c1 = int(spec["c1"])
    if orient == "up":
        return r < r0 and c0 < c < c1
    if orient == "down":
        return r > r1 and c0 < c < c1
    if orient == "left":
        return c < c0 and r0 < r < r1
    assert orient == "right"
    return c > c1 and r0 < r < r1


def _fill_layers(out: np.ndarray, spec: dict[str, int | str], color: int, layers: int) -> None:
    if layers <= 0:
        return
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    r1 = int(spec["r1"])
    c1 = int(spec["c1"])
    orient = str(spec["orient"])
    if orient == "up":
        for i in range(layers):
            out[r1 - 1 - i, c0 + 1 : c1] = color
    elif orient == "down":
        for i in range(layers):
            out[r0 + 1 + i, c0 + 1 : c1] = color
    elif orient == "left":
        for i in range(layers):
            out[r0 + 1 : r1, c1 - 1 - i] = color
    else:
        assert orient == "right"
        for i in range(layers):
            out[r0 + 1 : r1, c0 + 1 + i] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collect aligned same-color droplets into U-shaped containers and fill them from the closed side.

    Input structure:
    - Background is black.
    - Each non-black color forms one U-shaped 1-pixel-thick container and zero or more
      same-color singleton droplets outside the opening.
    - Containers can open up, down, left, or right.

    Rule:
    - Detect each container's opening side and opening span.
    - Count same-color droplets that lie in the opening half-plane and within that span.
    - Remove those counted droplets.
    - Fill that many interior layers, clamped by the container depth.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    colors = sorted(int(v) for v in np.unique(g) if int(v) != BLACK)

    for color in colors:
        components = _components_for_color(g, color)
        container_specs: list[dict[str, int | str]] = []
        droplets: list[tuple[int, int]] = []
        for component in components:
            spec = _match_container(component)
            if spec is not None:
                container_specs.append(spec)
            elif int(component["area"]) == 1:
                coords = component["coords"]
                assert isinstance(coords, np.ndarray)
                droplets.append(tuple(map(int, coords[0])))

        if len(container_specs) != 1:
            continue

        spec = container_specs[0]
        counted = [(r, c) for r, c in droplets if _is_aligned_droplet(spec, r, c)]
        for r, c in counted:
            out[r, c] = BLACK
        _fill_layers(out, spec, color, min(len(counted), int(spec["depth"])))

    return out


def _draw_container(grid: np.ndarray, spec: dict[str, int | str]) -> None:
    color = int(spec["color"])
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    r1 = int(spec["r1"])
    c1 = int(spec["c1"])
    for r, c in _expected_u_cells(r0, c0, r1, c1, str(spec["orient"])):
        grid[r, c] = color


def _reserve_bounds(spec: dict[str, int | str], h: int, w: int) -> tuple[int, int, int, int]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    r1 = int(spec["r1"])
    c1 = int(spec["c1"])
    strip = int(spec["strip"])
    pad = 1
    rr0 = max(0, r0 - pad - (strip if spec["orient"] == "up" else 0))
    rr1 = min(h - 1, r1 + pad + (strip if spec["orient"] == "down" else 0))
    cc0 = max(0, c0 - pad - (strip if spec["orient"] == "left" else 0))
    cc1 = min(w - 1, c1 + pad + (strip if spec["orient"] == "right" else 0))
    return rr0, cc0, rr1, cc1


def _sample_orientations(n: int) -> list[str]:
    if n <= len(ORIENTATIONS) and random.random() < 0.7:
        return random.sample(list(ORIENTATIONS), n)
    return [random.choice(ORIENTATIONS) for _ in range(n)]


def _sample_dimensions(orient: str, h: int, w: int) -> tuple[int, int]:
    if orient in ("up", "down"):
        box_h = random.randint(3, min(7, h - 4))
        box_w = random.randint(4, min(8, w - 4))
    else:
        box_h = random.randint(4, min(8, h - 4))
        box_w = random.randint(3, min(7, w - 4))
    return box_h, box_w


def _place_container(
    h: int,
    w: int,
    reserve: np.ndarray,
    color: int,
    orient: str,
) -> dict[str, int | str] | None:
    for _ in range(80):
        box_h, box_w = _sample_dimensions(orient, h, w)
        strip = random.randint(2, 5)
        rmin = 1 + (strip if orient == "up" else 0)
        cmin = 1 + (strip if orient == "left" else 0)
        rmax = h - box_h - 1 - (strip if orient == "down" else 0)
        cmax = w - box_w - 1 - (strip if orient == "right" else 0)
        if rmin > rmax or cmin > cmax:
            continue
        r0 = random.randint(rmin, rmax)
        c0 = random.randint(cmin, cmax)
        spec: dict[str, int | str] = {
            "color": color,
            "orient": orient,
            "r0": r0,
            "c0": c0,
            "r1": r0 + box_h - 1,
            "c1": c0 + box_w - 1,
            "strip": strip,
            "depth": box_h - 1 if orient in ("up", "down") else box_w - 1,
        }
        rr0, cc0, rr1, cc1 = _reserve_bounds(spec, h, w)
        if reserve[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue
        reserve[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return spec
    return None


def _can_place_same_color(grid: np.ndarray, color: int, r: int, c: int, selected: set[tuple[int, int]] | None = None) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if int(grid[nr, nc]) == color:
                return False
            if selected is not None and (nr, nc) in selected:
                return False
    return True


def _greedy_selection(
    grid: np.ndarray,
    color: int,
    candidates: list[tuple[int, int]],
    tries: int = 40,
) -> list[tuple[int, int]]:
    best: list[tuple[int, int]] = []
    for _ in range(tries):
        order = list(candidates)
        random.shuffle(order)
        picked: list[tuple[int, int]] = []
        picked_set: set[tuple[int, int]] = set()
        for r, c in order:
            if _can_place_same_color(grid, color, r, c, picked_set):
                picked.append((r, c))
                picked_set.add((r, c))
        if len(picked) > len(best):
            best = picked
    return best


def _pick_exact(
    grid: np.ndarray,
    color: int,
    candidates: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    for _ in range(40):
        order = list(candidates)
        random.shuffle(order)
        picked: list[tuple[int, int]] = []
        picked_set: set[tuple[int, int]] = set()
        for r, c in order:
            if _can_place_same_color(grid, color, r, c, picked_set):
                picked.append((r, c))
                picked_set.add((r, c))
                if len(picked) == count:
                    return picked
    return None


def _aligned_candidates(spec: dict[str, int | str], h: int, w: int) -> list[tuple[int, int]]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    r1 = int(spec["r1"])
    c1 = int(spec["c1"])
    strip = int(spec["strip"])
    orient = str(spec["orient"])
    cells: list[tuple[int, int]] = []
    if orient == "up":
        for r in range(max(0, r0 - strip), r0):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif orient == "down":
        for r in range(r1 + 1, min(h, r1 + strip + 1)):
            for c in range(c0 + 1, c1):
                cells.append((r, c))
    elif orient == "left":
        for r in range(r0 + 1, r1):
            for c in range(max(0, c0 - strip), c0):
                cells.append((r, c))
    else:
        for r in range(r0 + 1, r1):
            for c in range(c1 + 1, min(w, c1 + strip + 1)):
                cells.append((r, c))
    return cells


def _noise_candidates(spec: dict[str, int | str], h: int, w: int) -> list[tuple[int, int]]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    r1 = int(spec["r1"])
    c1 = int(spec["c1"])
    margin = max(2, int(spec["strip"]))
    local: list[tuple[int, int]] = []
    fallback: list[tuple[int, int]] = []

    rr0 = max(0, r0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc0 = max(0, c0 - margin)
    cc1 = min(w - 1, c1 + margin)
    for r in range(rr0, rr1 + 1):
        for c in range(cc0, cc1 + 1):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if not _is_aligned_droplet(spec, r, c):
                local.append((r, c))

    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if not _is_aligned_droplet(spec, r, c):
                fallback.append((r, c))

    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    random.shuffle(local)
    random.shuffle(fallback)
    for cell in local + fallback:
        if cell not in seen:
            seen.add(cell)
            ordered.append(cell)
    return ordered


def _sample_aligned_count(depth: int, max_count: int) -> int:
    modes = ["zero"]
    if min(depth - 1, max_count) >= 1:
        modes.append("partial")
    if max_count >= depth:
        modes.append("full")
    mode = random.choice(modes)
    if mode == "zero":
        return 0
    if mode == "partial":
        return random.randint(1, min(depth - 1, max_count))
    return random.randint(depth, min(max_count, depth + 3))


def _build_expected_output(inp: np.ndarray, specs: list[dict[str, int | str]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        color = int(spec["color"])
        aligned = spec["aligned"]
        assert isinstance(aligned, list)
        for r, c in aligned:
            out[r, c] = BLACK
        _fill_layers(out, spec, color, min(len(aligned), int(spec["depth"])))
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    area = h * w
    if area < 220:
        n_containers = random.randint(1, 2)
    elif area < 360:
        n_containers = random.randint(2, 3)
    else:
        n_containers = random.randint(2, 4)

    colors = random.sample(COLOR_POOL, n_containers)
    orientations = _sample_orientations(n_containers)
    reserve = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, int | str]] = []

    for color, orient in zip(colors, orientations):
        spec = _place_container(h, w, reserve, color, orient)
        if spec is None:
            return None
        specs.append(spec)

    inp = np.zeros((h, w), dtype=int)
    for spec in specs:
        _draw_container(inp, spec)

    total_aligned = 0
    total_noise = 0
    for spec in specs:
        color = int(spec["color"])
        aligned_pool = _aligned_candidates(spec, h, w)
        max_aligned_positions = _greedy_selection(inp, color, aligned_pool)
        max_aligned = len(max_aligned_positions)
        target_aligned = _sample_aligned_count(int(spec["depth"]), max_aligned)
        aligned = _pick_exact(inp, color, aligned_pool, target_aligned)
        if aligned is None:
            return None
        for r, c in aligned:
            inp[r, c] = color
        spec["aligned"] = aligned
        total_aligned += len(aligned)

        noise_target = random.randint(0, 2)
        noise_pool = _noise_candidates(spec, h, w)
        noise = _pick_exact(inp, color, noise_pool, noise_target)
        if noise is None:
            return None
        for r, c in noise:
            inp[r, c] = color
        spec["noise"] = noise
        total_noise += len(noise)

    if total_aligned == 0:
        return None

    if total_noise == 0:
        for spec in specs:
            color = int(spec["color"])
            noise_pool = _noise_candidates(spec, h, w)
            extra = _pick_exact(inp, color, noise_pool, 1)
            if extra:
                for r, c in extra:
                    inp[r, c] = color
                spec["noise"] = list(spec["noise"]) + extra  # type: ignore[index]
                total_noise += 1
                break
        if total_noise == 0:
            return None

    expected = _build_expected_output(inp, specs)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
