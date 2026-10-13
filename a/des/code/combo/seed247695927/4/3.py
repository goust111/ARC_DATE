import numpy as np


_RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _legend_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates = []
    for coords in _components4(grid != bg):
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or w != 2:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(patch != bg) and coords.shape[0] == h * w:
            candidates.append((h, r0, c0, r1, c1))
    assert len(candidates) == 1, "expected a unique filled 2-column legend"
    _h, r0, c0, r1, c1 = candidates[0]
    return r0, c0, r1, c1


def _apply_rules(color: int, pairs: list[tuple[int, int]]) -> tuple[int, int]:
    current = int(color)
    steps = 0
    for src, dst in pairs:
        if current == src:
            current = dst
            steps += 1
    return current, steps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    r0, c0, r1, c1 = _legend_bbox(g, bg)
    pairs = [(int(g[r, c0]), int(g[r, c0 + 1])) for r in range(r0, r1 + 1)]

    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 2] = True

    out = g.copy()
    target_coords = np.argwhere((g != bg) & (~legend_mask))
    for r, c in target_coords:
        out[r, c] = _apply_rules(int(g[r, c]), pairs)[0]
    return out


_SHAPES = (
    np.array([(0, 0)], dtype=int),
    np.array([(0, 0), (1, 0)], dtype=int),
    np.array([(0, 0), (0, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (2, 0)], dtype=int),
    np.array([(0, 0), (0, 1), (0, 2)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 1), (1, 0), (1, 1)], dtype=int),
    np.array([(0, 0), (1, 0), (1, 1), (2, 1)], dtype=int),
)


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for cc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                blocked[rr, cc] = True


def _sample_pairs(bg: int, rows: int) -> list[tuple[int, int]]:
    palette = [c for c in range(10) if c != bg]
    _RNG.shuffle(palette)
    chain_len = 3 if rows >= 4 and _RNG.random() < 0.5 else 2
    chain_nodes = palette[: chain_len + 1]
    pairs = [(int(chain_nodes[i]), int(chain_nodes[i + 1])) for i in range(chain_len)]
    used = set(pairs)

    while len(pairs) < rows:
        if _RNG.random() < 0.5:
            src = int(chain_nodes[_RNG.integers(0, len(chain_nodes) - 1)])
        else:
            src = int(palette[_RNG.integers(0, len(palette))])
        dst_choices = [c for c in palette if c != src]
        dst = int(dst_choices[_RNG.integers(0, len(dst_choices))])
        pair = (src, dst)
        if pair in used:
            continue
        used.add(pair)
        insert_at = int(_RNG.integers(0, len(pairs) + 1))
        if insert_at <= chain_len:
            insert_at = chain_len
        pairs.insert(insert_at, pair)

    return pairs


def _can_place(shape: np.ndarray, origin: tuple[int, int], blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    rr = origin[0] + shape[:, 0]
    cc = origin[1] + shape[:, 1]
    if rr.min() < 0 or cc.min() < 0 or rr.max() >= h or cc.max() >= w:
        return False
    return not blocked[rr, cc].any()


def _place_shape(grid: np.ndarray, blocked: np.ndarray, shape: np.ndarray, origin: tuple[int, int], color: int) -> None:
    coords = shape + np.array(origin, dtype=int)
    grid[coords[:, 0], coords[:, 1]] = int(color)
    _mark_blocked(blocked, coords)


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(10, 23))
    w = int(_RNG.integers(10, 23))
    bg = int(_RNG.integers(0, 10))
    rows = int(_RNG.integers(3, 7))
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    r0 = int(_RNG.integers(1, h - rows - 1))
    c0 = int(_RNG.integers(1, w - 3))
    pairs = _sample_pairs(bg, rows)
    for i, (src, dst) in enumerate(pairs):
        grid[r0 + i, c0] = src
        grid[r0 + i, c0 + 1] = dst
    legend_coords = np.array([(r, c) for r in range(r0, r0 + rows) for c in (c0, c0 + 1)], dtype=int)
    _mark_blocked(blocked, legend_coords)

    chain_starts = []
    for i, (src, _dst) in enumerate(pairs):
        _, steps = _apply_rules(src, pairs[i:])
        if steps >= 2:
            chain_starts.append(src)
    if not chain_starts:
        return None

    source_colors = [src for src, _dst in pairs]
    dst_colors = [dst for _src, dst in pairs]
    unchanged_colors = [c for c in range(10) if c != bg and c not in source_colors]
    if not unchanged_colors:
        return None

    forced_colors = [
        int(chain_starts[0]),
        int(source_colors[-1]),
        int(unchanged_colors[0]),
    ]
    candidate_colors = forced_colors + source_colors + dst_colors + unchanged_colors

    object_count = int(_RNG.integers(5, 11))
    placed = 0
    for i in range(object_count):
        if i < len(forced_colors):
            color = forced_colors[i]
        else:
            color = int(candidate_colors[_RNG.integers(0, len(candidate_colors))])
        shape = _SHAPES[int(_RNG.integers(0, len(_SHAPES)))]
        tries = 0
        while tries < 80:
            tries += 1
            origin = (
                int(_RNG.integers(0, h - int(shape[:, 0].max()) )),
                int(_RNG.integers(0, w - int(shape[:, 1].max()) )),
            )
            if _can_place(shape, origin, blocked):
                _place_shape(grid, blocked, shape, origin, color)
                placed += 1
                break
        if i < len(forced_colors) and tries >= 80:
            return None

    if placed < 4:
        return None

    bg2 = get_bg(grid)
    try:
        lr0, lc0, lr1, lc1 = _legend_bbox(grid, bg2)
    except AssertionError:
        return None
    if (lr0, lc0, lr1, lc1) != (r0, c0, r0 + rows - 1, c0 + 1):
        return None

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r0 + rows, c0 : c0 + 2] = True
    targets = np.argwhere((grid != bg) & (~legend_mask))
    if targets.shape[0] < 4:
        return None

    multi_step = 0
    changed = 0
    unchanged = 0
    for rr, cc in targets:
        out_color, steps = _apply_rules(int(grid[rr, cc]), pairs)
        if steps >= 2:
            multi_step += 1
        if out_color != int(grid[rr, cc]):
            changed += 1
        else:
            unchanged += 1
    if multi_step == 0 or changed == 0 or unchanged == 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
