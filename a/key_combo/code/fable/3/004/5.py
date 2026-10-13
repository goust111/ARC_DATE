from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLORS = list(range(10))
MAX_GENERATION_ATTEMPTS = 600


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r0, c0 in np.argwhere(mask):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        stack = [(r0, c0)]
        seen[r0, c0] = True
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _mask_from_rel(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
    coords = np.array(cells, dtype=int)
    return _mask_from_coords(coords)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _find_frame(grid: np.ndarray, bg: int) -> dict[str, object]:
    comps: list[dict[str, object]] = []
    for color in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        for coords in _components(grid == int(color)):
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    assert comps, "expected a foreground template frame"
    comps.sort(key=lambda d: int(d["area"]), reverse=True)
    frame = comps[0]
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
    assert int(r1) - int(r0) >= 6 and int(c1) - int(c0) >= 6, "template frame is too small"
    return frame


def _extract_holes(grid: np.ndarray, bg: int, frame: dict[str, object]) -> list[dict[str, object]]:
    fr0, fc0, fr1, fc1 = map(int, frame["bbox"])  # type: ignore[arg-type]
    sub = grid[fr0 : fr1 + 1, fc0 : fc1 + 1]
    h, w = sub.shape
    holes: list[dict[str, object]] = []
    for coords in _components(sub == int(bg)):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1):
            continue
        if np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        mask = _mask_from_coords(coords)
        holes.append(
            {
                "bbox_rel": (r0, c0, r1, c1),
                "bbox_abs": (fr0 + r0, fc0 + c0, fr0 + r1, fc0 + c1),
                "mask": mask,
                "key": _mask_key(mask),
            }
        )
    assert holes, "expected at least one enclosed hole"
    return holes


def _parse_outside_objects(grid: np.ndarray, bg: int, frame: dict[str, object]) -> list[dict[str, object]]:
    fr0, fc0, fr1, fc1 = map(int, frame["bbox"])  # type: ignore[arg-type]
    outside = grid != int(bg)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    prelim: list[dict[str, object]] = []
    for coords in _components(outside):
        vals = [int(grid[int(r), int(c)]) for r, c in coords]
        counts = Counter(vals)
        assert len(counts) == 2, "each outside component must be one sprite plus one connector color"
        singleton_colors = [color for color, count in counts.items() if count == 1]
        assert len(singleton_colors) == 1, "connector must be a single pixel"
        conn_color = int(singleton_colors[0])
        sprite_colors = [color for color in counts if color != conn_color]
        assert len(sprite_colors) == 1
        sprite_color = int(sprite_colors[0])

        conn_index = vals.index(conn_color)
        conn_r, conn_c = map(int, coords[conn_index])
        body_abs = np.array(
            [(int(r), int(c)) for (r, c), v in zip(coords, vals) if int(v) == sprite_color],
            dtype=int,
        )
        assert body_abs.shape[0] >= 2, "sprites must have visible area beyond the connector"

        body_set = {tuple(map(int, p)) for p in body_abs}
        assert any((nr, nc) in body_set for nr, nc in _neighbors4(conn_r, conn_c)), (
            "connector must touch its sprite orthogonally"
        )
        assert len(_components(_mask_from_coords(body_abs))) == 1, "sprite body must be connected"

        body_rel = tuple(sorted((int(r) - conn_r, int(c) - conn_c) for r, c in body_abs))
        all_rel = tuple(sorted(body_rel + ((0, 0),)))
        prelim.append(
            {
                "connector_color": conn_color,
                "sprite_color": sprite_color,
                "body_rel": body_rel,
                "all_rel": all_rel,
                "connector_abs": (conn_r, conn_c),
            }
        )

    by_connector: dict[int, list[dict[str, object]]] = defaultdict(list)
    for obj in prelim:
        by_connector[int(obj["connector_color"])].append(obj)

    pairs: list[dict[str, object]] = []
    for conn_color, objs in by_connector.items():
        assert len(objs) == 2, "each connector color must occur on exactly two sprites"
        pairs.append({"connector_color": int(conn_color), "objects": (objs[0], objs[1])})
    assert pairs, "expected at least one connector-color pair"
    return pairs


def _assembly_options(obj_a: dict[str, object], obj_b: dict[str, object]) -> list[dict[str, object]]:
    options: list[dict[str, object]] = []
    a_cells = list(obj_a["all_rel"])  # type: ignore[arg-type]
    b_cells = list(obj_b["all_rel"])  # type: ignore[arg-type]
    a_color = int(obj_a["sprite_color"])
    b_color = int(obj_b["sprite_color"])

    for dr, dc in DIRS4:
        colored: dict[tuple[int, int], int] = {}
        ok = True
        for rr, cc in a_cells:
            pos = (int(rr), int(cc))
            if pos in colored:
                ok = False
                break
            colored[pos] = a_color
        if not ok:
            continue
        for rr, cc in b_cells:
            pos = (int(rr) + int(dr), int(cc) + int(dc))
            if pos in colored:
                ok = False
                break
            colored[pos] = b_color
        if not ok:
            continue

        rs = [r for r, _c in colored]
        cs = [c for _r, c in colored]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch = np.zeros(mask.shape, dtype=int)
        for (r, c), color in colored.items():
            rr, cc = r - r0, c - c0
            mask[rr, cc] = True
            patch[rr, cc] = int(color)
        options.append({"dir": (int(dr), int(dc)), "mask": mask, "key": _mask_key(mask), "patch": patch})
    return options


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    frame = _find_frame(g, bg)
    holes = _extract_holes(g, bg, frame)
    pairs = _parse_outside_objects(g, bg, frame)
    return bg, frame, holes, pairs


def _unique_hole_matches(
    holes: list[dict[str, object]], pairs: list[dict[str, object]]
) -> tuple[dict[int, dict[str, object]], dict[int, list[dict[str, object]]]]:
    assert len(holes) == len(pairs), "number of connector pairs must equal number of holes"
    matches_by_hole: dict[int, list[tuple[int, dict[str, object]]]] = {i: [] for i in range(len(holes))}
    options_by_pair: dict[int, list[dict[str, object]]] = {}

    for pair_index, pair in enumerate(pairs):
        obj_a, obj_b = pair["objects"]  # type: ignore[misc]
        options = _assembly_options(obj_a, obj_b)
        assert options, "paired sprites must have at least one non-overlapping docking"
        options_by_pair[pair_index] = options
        for opt in options:
            for hole_index, hole in enumerate(holes):
                if opt["key"] == hole["key"]:
                    matches_by_hole[hole_index].append((pair_index, opt))

    chosen: dict[int, dict[str, object]] = {}
    used_pairs: set[int] = set()
    for hole_index, candidates in matches_by_hole.items():
        assert len(candidates) == 1, "each hole must resolve one unique pair and docking direction"
        pair_index, opt = candidates[0]
        assert pair_index not in used_pairs, "one pair cannot fill two holes"
        used_pairs.add(pair_index)
        chosen[hole_index] = opt
    assert len(used_pairs) == len(pairs), "every connector pair must fill one hole"
    return chosen, options_by_pair


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill template holes with compound keys assembled from same-colored connector pixels.

    Input structure:
    - The dominant background surrounds one large monochrome rectangular template frame.
    - Inside the frame are several enclosed background-color holes with distinct compound shapes.
    - Outside the frame are small monochrome sprites; each sprite has one adjacent singleton connector pixel,
      and connector colors occur in pairs.

    Rule:
    - Pair the two outside sprites that share a connector color.
    - For each pair, enumerate translations where the two connector pixels become orthogonally adjacent and
      the two recolored sprite-plus-connector shapes do not overlap.
    - The unique enumerated compound mask that matches a frame hole determines the docking direction.
    - Fill that hole with the compound colors, recoloring each connector pixel as the color of its own sprite,
      then clear all outside sprites and connector pixels to background.

    Key insight:
    - Neither sprite alone is the key: connector color gives the partner, while the hole shape selects the
      only docking direction after assembly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, frame, holes, pairs = _parse_scene(g)
    chosen, _options_by_pair = _unique_hole_matches(holes, pairs)

    out = g.copy()
    fr0, fc0, fr1, fc1 = map(int, frame["bbox"])  # type: ignore[arg-type]
    outside = np.ones(g.shape, dtype=bool)
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    out[outside] = int(bg)

    for hole_index, hole in enumerate(holes):
        r0, c0, _r1, _c1 = map(int, hole["bbox_abs"])  # type: ignore[arg-type]
        mask = hole["mask"]  # type: ignore[assignment]
        patch = chosen[hole_index]["patch"]  # type: ignore[assignment]
        assert isinstance(mask, np.ndarray) and isinstance(patch, np.ndarray)
        assert patch.shape == mask.shape
        target = out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        target[mask] = patch[mask]
    return out


# ------------------------- generator -------------------------


def _random_connected_mask() -> np.ndarray:
    for _ in range(100):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(2, min(6, h * w))
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        mask = _mask_from_rel(tuple(sorted(cells)))
        if int(mask.sum()) >= 2:
            return mask
    raise RuntimeError("failed to build a connected mask")


def _attach_connector(mask: np.ndarray) -> dict[str, object]:
    h, w = mask.shape
    choices: list[tuple[int, int]] = []
    for c in range(w):
        if bool(mask[0, c]):
            choices.append((-1, c))
        if bool(mask[h - 1, c]):
            choices.append((h, c))
    for r in range(h):
        if bool(mask[r, 0]):
            choices.append((r, -1))
        if bool(mask[r, w - 1]):
            choices.append((r, w))
    conn_r, conn_c = random.choice(choices)
    body_rel = tuple(
        sorted((int(r) - conn_r, int(c) - conn_c) for r, c in np.argwhere(mask))
    )
    all_rel = tuple(sorted(body_rel + ((0, 0),)))
    return {"body_rel": body_rel, "all_rel": all_rel}


def _shape_object(shape: dict[str, object], sprite_color: int) -> dict[str, object]:
    return {
        "body_rel": shape["body_rel"],
        "all_rel": shape["all_rel"],
        "sprite_color": int(sprite_color),
    }


def _sample_pair_shape() -> tuple[dict[str, object], dict[str, object], tuple[int, int], set[tuple[tuple[int, int], bytes]]] | None:
    for _ in range(300):
        a_shape = _attach_connector(_random_connected_mask())
        b_shape = _attach_connector(_random_connected_mask())
        obj_a = _shape_object(a_shape, 1)
        obj_b = _shape_object(b_shape, 2)
        options = _assembly_options(obj_a, obj_b)
        if len(options) < 2:
            continue

        key_counts = Counter(opt["key"] for opt in options)
        unique_options = [opt for opt in options if key_counts[opt["key"]] == 1]
        if not unique_options:
            continue
        chosen = random.choice(unique_options)

        single_keys = {
            _mask_key(_mask_from_rel(a_shape["body_rel"])),  # type: ignore[arg-type]
            _mask_key(_mask_from_rel(a_shape["all_rel"])),  # type: ignore[arg-type]
            _mask_key(_mask_from_rel(b_shape["body_rel"])),  # type: ignore[arg-type]
            _mask_key(_mask_from_rel(b_shape["all_rel"])),  # type: ignore[arg-type]
        }
        if chosen["key"] in single_keys:
            continue
        return a_shape, b_shape, chosen["dir"], {opt["key"] for opt in options}  # type: ignore[return-value]
    return None


def _piece_patch(shape: dict[str, object], sprite_color: int, connector_color: int, bg: int) -> tuple[np.ndarray, np.ndarray]:
    colored: dict[tuple[int, int], int] = {}
    for r, c in shape["body_rel"]:  # type: ignore[union-attr]
        colored[(int(r), int(c))] = int(sprite_color)
    colored[(0, 0)] = int(connector_color)

    rs = [r for r, _c in colored]
    cs = [c for _r, c in colored]
    r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    mask = np.zeros(patch.shape, dtype=bool)
    for (r, c), color in colored.items():
        rr, cc = r - r0, c - c0
        patch[rr, cc] = int(color)
        mask[rr, cc] = True
    return patch, mask


def _can_place_mask(occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = mask.shape
    h_grid, w_grid = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > h_grid or c0 + w > w_grid:
        return False
    for dr, dc in np.argwhere(mask):
        rr, cc = int(r0 + dr), int(c0 + dc)
        r_min, r_max = max(0, rr - padding), min(h_grid, rr + padding + 1)
        c_min, c_max = max(0, cc - padding), min(w_grid, cc + padding + 1)
        if bool(occupied[r_min:r_max, c_min:c_max].any()):
            return False
    return True


def _place_holes(
    frame_h: int, frame_w: int, masks: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((frame_h, frame_w), dtype=bool)
    positions: list[tuple[int, int]] = []
    order = list(range(len(masks)))
    random.shuffle(order)
    placed_by_index: dict[int, tuple[int, int]] = {}

    for index in order:
        mask = masks[index]
        mh, mw = mask.shape
        candidates = [
            (r, c)
            for r in range(1, frame_h - mh)
            for c in range(1, frame_w - mw)
            if _can_place_mask(occupied, mask, r, c, padding=1)
        ]
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        occupied[r0 : r0 + mh, c0 : c0 + mw] |= mask
        placed_by_index[index] = (int(r0), int(c0))

    for index in range(len(masks)):
        positions.append(placed_by_index[index])
    return positions


def _place_outside_piece(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    *,
    padding: int = 1,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not _can_place_mask(occupied, mask, r0, c0, padding=padding):
            continue
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        view[mask] = patch[mask]
        occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
        return True
    return False


def _render_expected_fill(
    out: np.ndarray,
    frame_origin: tuple[int, int],
    hole_positions: list[tuple[int, int]],
    pair_infos: list[dict[str, object]],
) -> None:
    fr, fc = frame_origin
    for (hr, hc), info in zip(hole_positions, pair_infos):
        obj_a = info["obj_a"]  # type: ignore[assignment]
        obj_b = info["obj_b"]  # type: ignore[assignment]
        wanted_dir = info["chosen_dir"]
        options = _assembly_options(obj_a, obj_b)  # type: ignore[arg-type]
        chosen = [opt for opt in options if opt["dir"] == wanted_dir]
        assert len(chosen) == 1
        patch = chosen[0]["patch"]
        mask = chosen[0]["mask"]
        target = out[fr + hr : fr + hr + mask.shape[0], fc + hc : fc + hc + mask.shape[1]]
        target[mask] = patch[mask]


def _semantic_generated_ok(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, _frame, holes, pairs = _parse_scene(grid)
        chosen, options_by_pair = _unique_hole_matches(holes, pairs)
        if bg != _background(grid) or len(holes) < 2 or len(pairs) != len(holes):
            return False
        hole_keys = {hole["key"] for hole in holes}
        all_objects = [obj for pair in pairs for obj in pair["objects"]]  # type: ignore[union-attr]
        for obj in all_objects:
            body_key = _mask_key(_mask_from_rel(obj["body_rel"]))  # type: ignore[arg-type]
            single_key = _mask_key(_mask_from_rel(obj["all_rel"]))  # type: ignore[arg-type]
            if body_key in hole_keys or single_key in hole_keys:
                return False
        for pair_index, opts in options_by_pair.items():
            if len(opts) < 2:
                return False
            matching = [opt for opt in opts if any(opt["key"] == hole["key"] for hole in holes)]
            if len(matching) != 1:
                return False
            if chosen and pair_index not in {None}:
                pass
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice([2, 2, 3])
    bg = random.choice(COLORS)
    frame_color = random.choice([c for c in COLORS if c != bg])
    connector_colors = random.sample([c for c in COLORS if c not in (bg, frame_color)], pair_count)
    sprite_palette = [c for c in COLORS if c not in (bg, frame_color) and c not in connector_colors]
    if len(sprite_palette) < 2:
        return None

    pair_infos: list[dict[str, object]] = []
    used_hole_keys: set[tuple[tuple[int, int], bytes]] = set()
    prior_option_keys: set[tuple[tuple[int, int], bytes]] = set()
    for conn_color in connector_colors:
        sampled = None
        for _ in range(200):
            candidate = _sample_pair_shape()
            if candidate is None:
                continue
            a_shape, b_shape, chosen_dir, option_keys = candidate
            obj_a_tmp = _shape_object(a_shape, 1)
            obj_b_tmp = _shape_object(b_shape, 2)
            chosen_key = next(
                opt["key"] for opt in _assembly_options(obj_a_tmp, obj_b_tmp) if opt["dir"] == chosen_dir
            )
            if chosen_key in used_hole_keys:
                continue
            if option_keys & used_hole_keys:
                continue
            if chosen_key in prior_option_keys:
                continue
            sampled = (a_shape, b_shape, chosen_dir, option_keys, chosen_key)
            break
        if sampled is None:
            return None
        a_shape, b_shape, chosen_dir, option_keys, chosen_key = sampled
        color_a, color_b = random.sample(sprite_palette, 2)
        obj_a = _shape_object(a_shape, color_a)
        obj_b = _shape_object(b_shape, color_b)
        pair_infos.append(
            {
                "connector_color": int(conn_color),
                "shape_a": a_shape,
                "shape_b": b_shape,
                "obj_a": obj_a,
                "obj_b": obj_b,
                "color_a": int(color_a),
                "color_b": int(color_b),
                "chosen_dir": chosen_dir,
            }
        )
        used_hole_keys.add(chosen_key)
        prior_option_keys |= option_keys

    hole_masks: list[np.ndarray] = []
    for info in pair_infos:
        options = _assembly_options(info["obj_a"], info["obj_b"])  # type: ignore[arg-type]
        chosen = [opt for opt in options if opt["dir"] == info["chosen_dir"]]
        if len(chosen) != 1:
            return None
        hole_masks.append(chosen[0]["mask"])  # type: ignore[arg-type]

    max_h = max(mask.shape[0] for mask in hole_masks)
    max_w = max(mask.shape[1] for mask in hole_masks)
    frame_h = random.randint(max(11, max_h + 5), 17)
    frame_w = random.randint(max(11, max_w + 5), 19)
    hole_positions = _place_holes(frame_h, frame_w, hole_masks)
    if hole_positions is None:
        return None

    grid_h_min = max(23, frame_h + 8)
    grid_w_min = max(23, frame_w + 8)
    if grid_h_min > 30 or grid_w_min > 30:
        return None
    grid_h = random.randint(grid_h_min, 30)
    grid_w = random.randint(grid_w_min, 30)
    fr = random.randint(2, grid_h - frame_h - 2)
    fc = random.randint(2, grid_w - frame_w - 2)

    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    frame_patch = np.full((frame_h, frame_w), int(frame_color), dtype=int)
    for (hr, hc), mask in zip(hole_positions, hole_masks):
        frame_patch[hr : hr + mask.shape[0], hc : hc + mask.shape[1]][mask] = int(bg)
    inp[fr : fr + frame_h, fc : fc + frame_w] = frame_patch

    expected = inp.copy()
    _render_expected_fill(expected, (fr, fc), hole_positions, pair_infos)

    occupied = np.zeros(inp.shape, dtype=bool)
    occupied[fr : fr + frame_h, fc : fc + frame_w] = True
    placement_order = []
    for info in pair_infos:
        placement_order.append((info["shape_a"], info["color_a"], info["connector_color"]))
        placement_order.append((info["shape_b"], info["color_b"], info["connector_color"]))
    random.shuffle(placement_order)

    for shape, sprite_color, connector_color in placement_order:
        patch, mask = _piece_patch(shape, int(sprite_color), int(connector_color), bg)  # type: ignore[arg-type]
        if not _place_outside_piece(inp, occupied, patch, mask, padding=1):
            return None

    if not _semantic_generated_ok(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid docking-key puzzle after {MAX_GENERATION_ATTEMPTS} attempts")
