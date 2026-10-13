import numpy as np


_RNG = np.random.default_rng()
_NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in _NEIGHBORS8:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_component(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    rel = coords - np.array([r0, c0], dtype=int)
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    colors = sorted({int(grid[r, c]) for r, c in coords})
    return {
        "coords_abs": coords,
        "coords_rel": rel,
        "bbox": (r0, c0, r1, c1),
        "mask": mask,
        "colors": colors,
        "color": colors[0],
    }


def _coords_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    anchor = coords.min(axis=0)
    norm = coords - anchor
    order = np.lexsort((norm[:, 1], norm[:, 0]))
    return tuple((int(norm[i, 0]), int(norm[i, 1])) for i in order)


def _edge_coords(mask: np.ndarray) -> dict[str, np.ndarray]:
    h, w = mask.shape
    top: list[tuple[int, int]] = []
    bottom: list[tuple[int, int]] = []
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if len(rows) == 0:
            continue
        top.append((int(rows[0]), c))
        bottom.append((int(rows[-1]), c))

    left: list[tuple[int, int]] = []
    right: list[tuple[int, int]] = []
    for r in range(h):
        cols = np.flatnonzero(mask[r, :])
        if len(cols) == 0:
            continue
        left.append((r, int(cols[0])))
        right.append((r, int(cols[-1])))

    return {
        "top": np.array(top, dtype=int),
        "bottom": np.array(bottom, dtype=int),
        "left": np.array(left, dtype=int),
        "right": np.array(right, dtype=int),
    }


def _is_single_component(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return len(_components(mask)) == 1


def _edge_data(mask: np.ndarray) -> dict[str, dict]:
    data: dict[str, dict] = {}
    for side, coords in _edge_coords(mask).items():
        if len(coords) == 0:
            raise AssertionError("empty edge")
        if not _is_single_component(coords, mask.shape):
            raise AssertionError("edge shadow must stay connected")
        data[side] = {
            "coords": coords,
            "key": _coords_key(coords),
            "anchor": coords.min(axis=0),
            "length": int(len(coords)),
        }
    return data


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    comps = [_extract_component(g, coords) for coords in _components(g != bg)]
    sources = [comp for comp in comps if len(comp["colors"]) > 1]
    shadows = [comp for comp in comps if len(comp["colors"]) == 1]
    assert sources and shadows and len(sources) == len(shadows)

    candidate_matches: list[dict] = []
    for source_index, source in enumerate(sources):
        edges = _edge_data(source["mask"])
        source["edges"] = edges
        for side, edge in edges.items():
            candidate_matches.append(
                {
                    "source_index": source_index,
                    "side": side,
                    "key": edge["key"],
                    "length": edge["length"],
                }
            )

    out = np.full_like(g, bg)
    used_sources: set[int] = set()
    for shadow in shadows:
        shadow_key = _coords_key(shadow["coords_rel"])
        shadow_len = int(len(shadow["coords_rel"]))
        matches = [
            cand
            for cand in candidate_matches
            if cand["key"] == shadow_key and cand["length"] == shadow_len
        ]
        assert len(matches) == 1, "shadow should match exactly one source edge"
        match = matches[0]
        src_idx = int(match["source_index"])
        assert src_idx not in used_sources, "each source should be used once"
        used_sources.add(src_idx)

        source = sources[src_idx]
        edge = source["edges"][match["side"]]
        shadow_r0, shadow_c0, _, _ = shadow["bbox"]
        tr = int(shadow_r0 - edge["anchor"][0])
        tc = int(shadow_c0 - edge["anchor"][1])
        dest = source["coords_rel"] + np.array([tr, tc], dtype=int)
        out[dest[:, 0], dest[:, 1]] = int(shadow["color"])

    assert len(used_sources) == len(sources)
    return out


GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 29)
PAIR_COUNT_OPTIONS = (2, 3)
SHAPE_H_RANGE = (4, 8)
SHAPE_W_RANGE = (4, 8)
GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACEMENT_PAD = 1
SIDES = ("top", "bottom", "left", "right")


def _random_walk_profile(length: int, low: int, high: int, start: int) -> np.ndarray:
    prof = np.empty(length, dtype=int)
    prof[0] = start
    for i in range(1, length):
        opts = [v for v in (prof[i - 1] - 1, prof[i - 1], prof[i - 1] + 1) if low <= v <= high]
        prof[i] = int(opts[_RNG.integers(len(opts))])
    return prof


def _make_shape_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        top = _random_walk_profile(w, 0, h - 3, int(_RNG.integers(0, h - 2)))
        thickness = _random_walk_profile(w, 2, h, int(_RNG.integers(2, h + 1)))
        thickness = np.minimum(thickness, h - top)
        thickness = np.maximum(thickness, 2)
        bottom = top + thickness - 1
        if np.any(bottom >= h):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for c in range(w):
            mask[top[c] : bottom[c] + 1, c] = True

        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if len(_components(mask)) != 1:
            continue

        try:
            edges = _edge_data(mask)
        except AssertionError:
            continue
        keys = [edge["key"] for edge in edges.values()]
        if len(set(keys)) != 4:
            continue

        varied = (
            not np.all(edges["top"]["coords"][:, 0] == edges["top"]["coords"][0, 0])
            or not np.all(edges["left"]["coords"][:, 1] == edges["left"]["coords"][0, 1])
        )
        if not varied:
            continue
        return mask
    return None


def _paint_source(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    h, w = mask.shape
    patch = np.zeros((h, w), dtype=int)
    mode = int(_RNG.integers(4))
    phase = int(_RNG.integers(0, 7))
    for r, c in np.argwhere(mask):
        if mode == 0:
            idx = (r + c + phase) % len(colors)
        elif mode == 1:
            idx = (2 * r + c + phase) % len(colors)
        elif mode == 2:
            idx = (r + 2 * c + phase) % len(colors)
        else:
            idx = (3 * r + 5 * c + phase) % len(colors)
        patch[r, c] = int(colors[idx])

    used = {int(patch[r, c]) for r, c in np.argwhere(mask)}
    if len(used) < 2:
        coords = np.argwhere(mask)
        patch[coords[0, 0], coords[0, 1]] = int(colors[0])
        patch[coords[-1, 0], coords[-1, 1]] = int(colors[1])
    return patch


def _can_place_rect(
    r0: int,
    c0: int,
    h: int,
    w: int,
    reserved: list[tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
    pad: int,
) -> bool:
    r1, c1 = r0 + h - 1, c0 + w - 1
    if r0 < 0 or c0 < 0 or r1 >= grid_h or c1 >= grid_w:
        return False
    for rr0, cc0, rr1, cc1 in reserved:
        if not (r1 + pad < rr0 or rr1 + pad < r0 or c1 + pad < cc0 or cc1 + pad < c0):
            return False
    return True


def _random_rect_position(
    h: int,
    w: int,
    reserved: list[tuple[int, int, int, int]],
    grid_h: int,
    grid_w: int,
    pad: int,
) -> tuple[int, int] | None:
    candidates = [
        (r, c)
        for r in range(grid_h - h + 1)
        for c in range(grid_w - w + 1)
        if _can_place_rect(r, c, h, w, reserved, grid_h, grid_w, pad)
    ]
    if not candidates:
        return None
    return candidates[int(_RNG.integers(len(candidates)))]


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, pos: tuple[int, int], mask: np.ndarray | None = None) -> None:
    r0, c0 = pos
    h, w = patch.shape
    if mask is None:
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        return
    rr, cc = np.argwhere(mask).T
    grid[r0 + rr, c0 + cc] = patch[rr, cc]


def _build_pair(bg: int, unavailable_lengths: set[int], unavailable_edge_keys: set[tuple[tuple[int, int], ...]]) -> dict | None:
    side = SIDES[int(_RNG.integers(len(SIDES)))]
    palette = [color for color in range(10) if color != bg]
    for _ in range(SHAPE_ATTEMPTS):
        h = int(_RNG.integers(*SHAPE_H_RANGE))
        w = int(_RNG.integers(*SHAPE_W_RANGE))
        mask = _make_shape_mask(h, w)
        if mask is None:
            continue
        try:
            edges = _edge_data(mask)
        except AssertionError:
            continue
        if any(edge["key"] in unavailable_edge_keys for edge in edges.values()):
            continue
        selected_len = edges[side]["length"]
        if selected_len in unavailable_lengths:
            continue
        color_count = int(_RNG.integers(2, 4))
        src_colors = list(_RNG.choice(palette, size=color_count, replace=False))
        source_patch = _paint_source(mask, src_colors)
        return {
            "mask": mask,
            "edges": edges,
            "side": side,
            "source_patch": source_patch,
            "selected_length": selected_len,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    grid_h = int(_RNG.integers(*GRID_H_RANGE))
    grid_w = int(_RNG.integers(*GRID_W_RANGE))
    bg = int(_RNG.integers(10))
    pair_count = int(PAIR_COUNT_OPTIONS[_RNG.integers(len(PAIR_COUNT_OPTIONS))])

    palette = [color for color in range(10) if color != bg]
    if len(palette) < pair_count + 2:
        return None
    shadow_colors = list(_RNG.choice(palette, size=pair_count, replace=False))

    pairs: list[dict] = []
    used_lengths: set[int] = set()
    used_edge_keys: set[tuple[tuple[int, int], ...]] = set()
    for shadow_color in shadow_colors:
        pair = _build_pair(bg, used_lengths, used_edge_keys)
        if pair is None:
            return None
        pair["shadow_color"] = int(shadow_color)
        pairs.append(pair)
        used_lengths.add(int(pair["selected_length"]))
        used_edge_keys.update(edge["key"] for edge in pair["edges"].values())

    reserved: list[tuple[int, int, int, int]] = []
    for pair in pairs:
        h, w = pair["mask"].shape
        source_pos = _random_rect_position(h, w, reserved, grid_h, grid_w, PLACEMENT_PAD)
        if source_pos is None:
            return None
        reserved.append((source_pos[0], source_pos[1], source_pos[0] + h - 1, source_pos[1] + w - 1))

        target_pos = _random_rect_position(h, w, reserved, grid_h, grid_w, PLACEMENT_PAD)
        if target_pos is None:
            return None
        reserved.append((target_pos[0], target_pos[1], target_pos[0] + h - 1, target_pos[1] + w - 1))

        pair["source_pos"] = source_pos
        pair["target_pos"] = target_pos

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    expected = np.full((grid_h, grid_w), bg, dtype=int)

    for pair in pairs:
        _stamp_patch(grid, pair["source_patch"], pair["source_pos"], pair["mask"])
        edge = pair["edges"][pair["side"]]["coords"]
        shadow_patch = np.zeros(pair["mask"].shape, dtype=int)
        shadow_patch[edge[:, 0], edge[:, 1]] = int(pair["shadow_color"])
        shadow_mask = np.zeros(pair["mask"].shape, dtype=bool)
        shadow_mask[edge[:, 0], edge[:, 1]] = True
        _stamp_patch(grid, shadow_patch, pair["target_pos"], shadow_mask)

        full_patch = np.zeros(pair["mask"].shape, dtype=int)
        full_patch[pair["mask"]] = int(pair["shadow_color"])
        _stamp_patch(expected, full_patch, pair["target_pos"], pair["mask"])

    if _background_color(grid) != bg:
        return None
    if len(_components(grid != bg)) != pair_count * 2:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
