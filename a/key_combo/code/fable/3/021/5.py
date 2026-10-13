from __future__ import annotations

import random

import numpy as np


BLACK = 0
COLORS = tuple(range(1, 10))
GENERATION_ATTEMPTS = 900

_GEN_CALL_COUNT = 0


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox in the arc_utils convention: inclusive (r0, c0, r1, c1)."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _frame_patch(size: int, color: int) -> np.ndarray:
    patch = np.full((int(size), int(size)), BLACK, dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    return patch


def _is_hollow_square(sub: np.ndarray, color: int) -> bool:
    if sub.shape[0] != sub.shape[1] or sub.shape[0] < 3:
        return False
    expected = np.zeros_like(sub, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(sub == int(color), expected))


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    frames: list[dict] = []
    seen_sizes: set[int] = set()
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BLACK):
        comps = _components4(g == int(color))
        assert len(comps) == 1, "each frame color must form exactly one component"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox_coords(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        assert _is_hollow_square(sub, color), "foreground components must be one-cell hollow square frames"
        size = int(r1 - r0 + 1)
        assert size not in seen_sizes, "frame sizes must be distinct"
        seen_sizes.add(size)
        frames.append(
            {
                "color": int(color),
                "size": size,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )

    assert frames, "expected at least one frame"
    return frames


def _exact_fit_edges(frames: list[dict]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if abs(int(frames[i]["size"]) - int(frames[j]["size"])) == 2:
                edges.append((i, j))
    return edges


def _fit_chain(frames: list[dict]) -> list[dict]:
    edges = _exact_fit_edges(frames)
    assert edges, "expected at least one exact-fit pair"

    adjacency = {i: set() for i in range(len(frames))}
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)

    visited: set[int] = set()
    edge_components: list[set[int]] = []
    for start in range(len(frames)):
        if start in visited or not adjacency[start]:
            continue
        stack = [start]
        comp: set[int] = set()
        visited.add(start)
        while stack:
            node = stack.pop()
            comp.add(node)
            for nxt in adjacency[node]:
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        edge_components.append(comp)

    assert len(edge_components) == 1, "exact-fit pairs must form one unique chain"
    comp = edge_components[0]
    edge_count = sum(1 for i, j in edges if i in comp and j in comp)
    assert edge_count == len(comp) - 1, "exact-fit component must be a simple chain"
    assert all(len(adjacency[i]) <= 2 for i in comp), "no frame may have more than two exact-fit neighbors"

    chain = [frames[i] for i in comp]
    chain.sort(key=lambda f: int(f["size"]), reverse=True)
    sizes = [int(f["size"]) for f in chain]
    assert all(a - b == 2 for a, b in zip(sizes, sizes[1:])), "chain sizes must step by exactly two"
    return chain


def _render_chain(chain: list[dict]) -> np.ndarray:
    largest = int(chain[0]["size"])
    out = np.full((largest, largest), BLACK, dtype=int)
    for frame in chain:
        size = int(frame["size"])
        color = int(frame["color"])
        offset = (largest - size) // 2
        out[offset, offset : offset + size] = color
        out[offset + size - 1, offset : offset + size] = color
        out[offset : offset + size, offset] = color
        out[offset : offset + size, offset + size - 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recompose the unique exact-fit chain of hollow square frames as centered nesting dolls.

    Input structure:
    - The input is a black-background grid with several separated, one-cell-thick hollow
      square frames.
    - Every frame has a distinct color and a distinct side length.
    - Some frame sizes form one chain where each smaller frame's outer side is exactly
      two cells shorter than the next larger frame's side.
    - Other frames are decoys, often visually between chain sizes but not exact fits.

    Rule:
    - Detect every hollow square frame and compare frame side lengths.
    - Link exactly those pairs whose side lengths differ by two, so the smaller outer
      border would touch the larger frame's inner border with no gap.
    - Keep the single linked chain, discard all unlinked decoy frames, and make an output
      canvas the size of the largest chain frame.
    - Draw the chain frames in their original colors, centered on the same point, with
      adjacent layers touching and no empty ring between them.

    Key insight:
    - Nesting is not based on ordinary size order: only a side-length difference of two
      proves that two one-cell-thick hollow frames fit together exactly.
    """
    frames = _find_frames(np.asarray(grid, dtype=int))
    chain = _fit_chain(frames)
    return _render_chain(chain)


def _candidate_decoy_sizes(chain_sizes: list[int]) -> tuple[list[int], list[int]]:
    ascending = sorted(chain_sizes)
    in_between = [small + 1 for small, large in zip(ascending, ascending[1:]) if large - small == 2]

    extras: list[int] = []
    if ascending[0] > 3:
        extras.append(ascending[0] - 1)
    if ascending[-1] < 16:
        extras.append(ascending[-1] + 1)
    for size in range(3, 17):
        if size not in chain_sizes and all(abs(size - s) != 2 for s in chain_sizes):
            extras.append(size)

    in_between = sorted(set(in_between))
    extras = sorted(set(extras) - set(in_between))
    return in_between, extras


def _choose_decoys(chain_sizes: list[int], required: bool, prefer_larger: bool) -> list[int]:
    if not required:
        return []

    in_between, extras = _candidate_decoy_sizes(chain_sizes)
    random.shuffle(in_between)
    random.shuffle(extras)

    decoys: list[int] = []
    if in_between:
        decoys.append(in_between[0])

    if prefer_larger:
        larger = [s for s in extras if s > max(chain_sizes)]
        if larger and all(abs(larger[0] - d) != 2 for d in decoys):
            decoys.append(larger[0])

    target = random.randint(1, min(3, 9 - len(chain_sizes)))
    for size in in_between[1:] + extras:
        if len(decoys) >= target:
            break
        if size in decoys:
            continue
        if any(abs(size - d) == 2 for d in decoys):
            continue
        decoys.append(size)

    return sorted(decoys, reverse=True)


def _choose_specs(require_decoys: bool, prefer_larger_decoy: bool) -> tuple[list[dict], list[dict]]:
    chain_len = random.randint(3, 5)
    smallest = random.randint(3, 7)
    chain_sizes = [smallest + 2 * i for i in range(chain_len)]
    decoy_sizes = _choose_decoys(chain_sizes, require_decoys, prefer_larger_decoy)

    total = len(chain_sizes) + len(decoy_sizes)
    if total > len(COLORS):
        raise ValueError("too many frames for distinct ARC colors")
    colors = random.sample(COLORS, total)

    chain_specs = []
    for size in sorted(chain_sizes, reverse=True):
        chain_specs.append({"size": int(size), "color": int(colors.pop()), "chain": True})

    decoy_specs = []
    for size in decoy_sizes:
        decoy_specs.append({"size": int(size), "color": int(colors.pop()), "chain": False})

    return chain_specs, decoy_specs


def _can_place_footprint(footprint: np.ndarray, size: int, r0: int, c0: int, padding: int) -> bool:
    h, w = footprint.shape
    r0 = int(r0)
    c0 = int(c0)
    size = int(size)
    if r0 < 0 or c0 < 0 or r0 + size > h or c0 + size > w:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(h, r0 + size + padding)
    cc1 = min(w, c0 + size + padding)
    return not bool(footprint[rr0:rr1, cc0:cc1].any())


def _place_specs(chain_specs: list[dict], decoy_specs: list[dict]) -> np.ndarray | None:
    specs = [dict(s) for s in chain_specs + decoy_specs]
    max_side = max(int(s["size"]) for s in specs)
    total_area = sum((int(s["size"]) + 2) ** 2 for s in specs)

    min_dim = max(max_side + 4, 18)
    h = random.randint(min_dim, 30)
    w = random.randint(min_dim, 30)
    if total_area > int(h * w * 0.82):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    footprint = np.zeros((h, w), dtype=bool)
    order = sorted(specs, key=lambda s: int(s["size"]), reverse=True)

    for spec in order:
        size = int(spec["size"])
        color = int(spec["color"])
        placed = False
        for _ in range(180):
            r_min = 1 if h - size >= 2 else 0
            c_min = 1 if w - size >= 2 else 0
            r_max = h - size - r_min
            c_max = w - size - c_min
            if r_max < r_min or c_max < c_min:
                return None
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            if not _can_place_footprint(footprint, size, r0, c0, padding=1):
                continue

            grid[r0 : r0 + size, c0 : c0 + size] = np.where(
                _frame_patch(size, color) != BLACK,
                int(color),
                grid[r0 : r0 + size, c0 : c0 + size],
            )
            footprint[r0 : r0 + size, c0 : c0 + size] = True
            placed = True
            break
        if not placed:
            return None

    return grid


def _semantic_generation_ok(
    grid: np.ndarray,
    chain_specs: list[dict],
    decoy_specs: list[dict],
    expected: np.ndarray,
) -> bool:
    try:
        frames = _find_frames(grid)
        chain = _fit_chain(frames)
        actual = main(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False

    chain_sizes = sorted([int(s["size"]) for s in chain_specs], reverse=True)
    found_chain_sizes = [int(f["size"]) for f in chain]
    if found_chain_sizes != chain_sizes:
        return False

    frame_sizes = sorted(int(f["size"]) for f in frames)
    spec_sizes = sorted(int(s["size"]) for s in chain_specs + decoy_specs)
    if frame_sizes != spec_sizes:
        return False

    output_colors = {int(c) for c in np.unique(expected) if int(c) != BLACK}
    chain_colors = {int(s["color"]) for s in chain_specs}
    decoy_colors = {int(s["color"]) for s in decoy_specs}
    if output_colors != chain_colors or output_colors & decoy_colors:
        return False

    if decoy_specs:
        decoy_sizes = [int(s["size"]) for s in decoy_specs]
        if any(abs(a - b) == 2 for i, a in enumerate(decoy_sizes) for b in decoy_sizes[i + 1 :]):
            return False
        if any(abs(d - c) == 2 for d in decoy_sizes for c in chain_sizes):
            return False
        if not any(
            min(a, b) < d < max(a, b)
            for d in decoy_sizes
            for a, b in zip(chain_sizes, chain_sizes[1:])
        ):
            return False

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    return True


def _gen_attempt(require_decoys: bool, prefer_larger_decoy: bool) -> np.ndarray | None:
    chain_specs, decoy_specs = _choose_specs(require_decoys, prefer_larger_decoy)
    grid = _place_specs(chain_specs, decoy_specs)
    if grid is None:
        return None

    expected = _render_chain(chain_specs)
    if not _semantic_generation_ok(grid, chain_specs, decoy_specs, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    call_index = int(_GEN_CALL_COUNT)
    require_decoys = call_index % 3 != 0
    prefer_larger_decoy = call_index % 5 == 2

    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(require_decoys, prefer_larger_decoy)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid exact-fit nesting input")
