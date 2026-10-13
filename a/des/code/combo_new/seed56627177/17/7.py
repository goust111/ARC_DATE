import numpy as np
from collections import Counter, defaultdict, deque


CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
FLAT = ("flat", -1)
LAYOUTS = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (1, 4), (4, 1)]
GEN_RETRIES = 400
PLACE_RETRIES = 200


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            comps.append({"coords": pts, "bbox": (int(r0), int(c0), int(r1), int(c1))})
    return comps


def _extract_patch(grid: np.ndarray, comp: dict, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in comp["coords"]:
        patch[int(r) - int(r0), int(c) - int(c0)] = int(grid[int(r), int(c)])
    return patch


def _infer_piece(patch: np.ndarray, bg: int) -> dict:
    mask = np.asarray(patch, dtype=int) != int(bg)
    ph, pw = mask.shape

    top_out = int(mask[0].sum()) == 1
    bottom_out = int(mask[-1].sum()) == 1
    left_out = int(mask[:, 0].sum()) == 1
    right_out = int(mask[:, -1].sum()) == 1

    core_r0 = 1 if top_out else 0
    core_r1 = ph - 2 if bottom_out else ph - 1
    core_c0 = 1 if left_out else 0
    core_c1 = pw - 2 if right_out else pw - 1
    core_mask = mask[core_r0 : core_r1 + 1, core_c0 : core_c1 + 1]
    core_h, core_w = core_mask.shape

    assert core_h >= 3 and core_w >= 3, "core rectangle must remain visible"
    assert core_mask[1:-1, 1:-1].all(), "only edge cells may be indented"

    def edge_feature(side: str) -> tuple[str, int]:
        if side == "top":
            if top_out:
                cols = np.flatnonzero(mask[0])
                assert len(cols) == 1
                return ("out", int(cols[0] - core_c0))
            holes = np.flatnonzero(~core_mask[0])
        elif side == "bottom":
            if bottom_out:
                cols = np.flatnonzero(mask[-1])
                assert len(cols) == 1
                return ("out", int(cols[0] - core_c0))
            holes = np.flatnonzero(~core_mask[-1])
        elif side == "left":
            if left_out:
                rows = np.flatnonzero(mask[:, 0])
                assert len(rows) == 1
                return ("out", int(rows[0] - core_r0))
            holes = np.flatnonzero(~core_mask[:, 0])
        else:
            if right_out:
                rows = np.flatnonzero(mask[:, -1])
                assert len(rows) == 1
                return ("out", int(rows[0] - core_r0))
            holes = np.flatnonzero(~core_mask[:, -1])

        if len(holes) == 0:
            return FLAT
        assert len(holes) == 1, "each side may have at most one indentation"
        return ("in", int(holes[0]))

    features = {side: edge_feature(side) for side in ("top", "bottom", "left", "right")}

    for side, feature in features.items():
        if side == "top":
            edge = core_mask[0]
        elif side == "bottom":
            edge = core_mask[-1]
        elif side == "left":
            edge = core_mask[:, 0]
        else:
            edge = core_mask[:, -1]
        if feature[0] == "flat":
            assert edge.all()
        elif feature[0] == "in":
            missing = np.flatnonzero(~edge)
            assert len(missing) == 1 and int(missing[0]) == int(feature[1])
        else:
            assert edge.all()

    return {
        "patch": np.asarray(patch, dtype=int),
        "core_origin": (int(core_r0), int(core_c0)),
        "core_shape": (int(core_h), int(core_w)),
        "features": features,
    }


def _side_delta(side: str, core_h: int, core_w: int) -> tuple[int, int]:
    if side == "top":
        return (-int(core_h), 0)
    if side == "bottom":
        return (int(core_h), 0)
    if side == "left":
        return (0, -int(core_w))
    return (0, int(core_w))


def _solve_positions(pieces: list[dict]) -> tuple[dict[int, tuple[int, int]], int, int]:
    shapes = {piece["core_shape"] for piece in pieces}
    assert len(shapes) == 1, "all pieces must share one core rectangle size"
    core_h, core_w = next(iter(shapes))

    feature_lookup: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for idx, piece in enumerate(pieces):
        for side, feature in piece["features"].items():
            kind, offset = feature
            if kind != "flat":
                feature_lookup[(side, kind, int(offset))].append(int(idx))

    adj: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in pieces]
    for idx, piece in enumerate(pieces):
        for side, feature in piece["features"].items():
            kind, offset = feature
            if kind == "flat":
                continue
            mate_side = OPPOSITE[side]
            mate_kind = "in" if kind == "out" else "out"
            mates = [j for j in feature_lookup[(mate_side, mate_kind, int(offset))] if j != idx]
            assert len(mates) == 1, "every lock must have one complementary mate"
            other = int(mates[0])
            adj[idx].append((other, _side_delta(side, core_h, core_w)))

    positions: dict[int, tuple[int, int]] = {0: (0, 0)}
    q = deque([0])
    while q:
        idx = q.popleft()
        base_r, base_c = positions[idx]
        for other, (dr, dc) in adj[idx]:
            target = (int(base_r + dr), int(base_c + dc))
            if other in positions:
                assert positions[other] == target, "inconsistent jigsaw placement"
            else:
                positions[other] = target
                q.append(other)

    assert len(positions) == len(pieces), "lock graph must connect all pieces"
    assert len(set(positions.values())) == len(positions), "different pieces cannot occupy one slot"
    return positions, int(core_h), int(core_w)


def _assemble_pieces(pieces: list[dict], positions: dict[int, tuple[int, int]], bg: int) -> np.ndarray:
    cells = []
    for idx, piece in enumerate(pieces):
        patch = piece["patch"]
        core_r0, core_c0 = piece["core_origin"]
        pr, pc = positions[idx]
        for r, c in np.argwhere(patch != int(bg)):
            ar = int(pr + int(r) - int(core_r0))
            ac = int(pc + int(c) - int(core_c0))
            cells.append((ar, ac, int(patch[int(r), int(c)])))

    min_r = min(r for r, _, _ in cells)
    max_r = max(r for r, _, _ in cells)
    min_c = min(c for _, c, _ in cells)
    max_c = max(c for _, c, _ in cells)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)

    for r, c, value in cells:
        rr = int(r - min_r)
        cc = int(c - min_c)
        current = int(out[rr, cc])
        assert current in (int(bg), int(value)), "assembled pieces may only overlap at complementary holes"
        out[rr, cc] = int(value)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered multicolor jigsaw pieces by matching one-pixel protrusions to complementary one-pixel indentations.

    Input structure:
    - The background is black.
    - Several separated multicolor connected components each form one puzzle piece.
    - Every piece has the same underlying rectangular core, but some sides may
      have a single outward knob or a single missing edge pixel.

    Rule:
    - Detect, for every piece side, whether it is flat, has a one-pixel
      protrusion, or has a one-pixel indentation, together with that feature's
      offset along the side.
    - Match pieces only when one side's protrusion complements another piece's
      opposite-side indentation at the same offset.
    - Translate all pieces without rotating them until all complementary locks
      fit, letting protrusion pixels fill the matched indentations.
    - Return the tightly cropped assembled picture with no extra black border.

    Key insight:
    - The irregular edge cells are the only placement clues: the whole assembly
      is recovered from complementary lock geometry rather than from color ids
      or the pieces' scattered positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    comps = find_components(g, bg)
    assert len(comps) >= 2, "expected multiple puzzle pieces"

    pieces = [_infer_piece(_extract_patch(g, comp, bg), bg) for comp in comps]
    positions, _, _ = _solve_positions(pieces)
    return _assemble_pieces(pieces, positions, bg)


def _make_core_pattern(core_h: int, core_w: int, palette: np.ndarray, piece_index: int, rng: np.random.Generator) -> np.ndarray:
    palette = np.asarray(palette, dtype=int)
    mode = int(rng.integers(4))
    phase = int(rng.integers(len(palette)))
    core = np.zeros((core_h, core_w), dtype=int)

    for r in range(core_h):
        for c in range(core_w):
            if mode == 0:
                idx = (r + phase + piece_index) % len(palette)
            elif mode == 1:
                idx = (c + phase + 2 * piece_index) % len(palette)
            elif mode == 2:
                idx = (r + c + phase + piece_index) % len(palette)
            else:
                idx = ((r // 2) + c + phase + piece_index) % len(palette)
            core[r, c] = int(palette[idx])

    assert len(set(map(int, core.ravel()))) >= 2
    return core


def _build_patch(core: np.ndarray, features: dict[str, tuple[str, int]], bg: int) -> np.ndarray:
    core_h, core_w = core.shape
    top_out = features["top"][0] == "out"
    bottom_out = features["bottom"][0] == "out"
    left_out = features["left"][0] == "out"
    right_out = features["right"][0] == "out"

    patch = np.full(
        (core_h + int(top_out) + int(bottom_out), core_w + int(left_out) + int(right_out)),
        int(bg),
        dtype=int,
    )
    core_r0 = 1 if top_out else 0
    core_c0 = 1 if left_out else 0
    patch[core_r0 : core_r0 + core_h, core_c0 : core_c0 + core_w] = core

    for side, (kind, offset) in features.items():
        if kind == "in":
            if side == "top":
                patch[core_r0, core_c0 + offset] = int(bg)
            elif side == "bottom":
                patch[core_r0 + core_h - 1, core_c0 + offset] = int(bg)
            elif side == "left":
                patch[core_r0 + offset, core_c0] = int(bg)
            else:
                patch[core_r0 + offset, core_c0 + core_w - 1] = int(bg)

    for side, (kind, offset) in features.items():
        if kind != "out":
            continue
        if side == "top":
            color = int(patch[core_r0, core_c0 + offset])
            patch[core_r0 - 1, core_c0 + offset] = color
        elif side == "bottom":
            color = int(patch[core_r0 + core_h - 1, core_c0 + offset])
            patch[core_r0 + core_h, core_c0 + offset] = color
        elif side == "left":
            color = int(patch[core_r0 + offset, core_c0])
            patch[core_r0 + offset, core_c0 - 1] = color
        else:
            color = int(patch[core_r0 + offset, core_c0 + core_w - 1])
            patch[core_r0 + offset, core_c0 + core_w] = color

    return patch


def _layout_features(layout: tuple[int, int], core_h: int, core_w: int, rng: np.random.Generator) -> tuple[list[tuple[int, int]], list[dict[str, tuple[str, int]]]]:
    rows, cols = layout
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    features = [{side: FLAT for side in ("top", "bottom", "left", "right")} for _ in cells]
    index_of = {cell: idx for idx, cell in enumerate(cells)}

    horizontal_edges = [(r, c) for r in range(rows) for c in range(cols - 1)]
    vertical_edges = [(r, c) for r in range(rows - 1) for c in range(cols)]
    assert len(horizontal_edges) <= core_h - 2
    assert len(vertical_edges) <= core_w - 2

    if horizontal_edges:
        offsets = rng.permutation(np.arange(1, core_h - 1))[: len(horizontal_edges)]
        for offset, (r, c) in zip(offsets, horizontal_edges):
            left = index_of[(r, c)]
            right = index_of[(r, c + 1)]
            if int(rng.integers(2)) == 0:
                features[left]["right"] = ("out", int(offset))
                features[right]["left"] = ("in", int(offset))
            else:
                features[left]["right"] = ("in", int(offset))
                features[right]["left"] = ("out", int(offset))

    if vertical_edges:
        offsets = rng.permutation(np.arange(1, core_w - 1))[: len(vertical_edges)]
        for offset, (r, c) in zip(offsets, vertical_edges):
            top = index_of[(r, c)]
            bottom = index_of[(r + 1, c)]
            if int(rng.integers(2)) == 0:
                features[top]["bottom"] = ("out", int(offset))
                features[bottom]["top"] = ("in", int(offset))
            else:
                features[top]["bottom"] = ("in", int(offset))
                features[bottom]["top"] = ("out", int(offset))

    return cells, features


def _truth_positions(cells: list[tuple[int, int]], core_h: int, core_w: int) -> dict[int, tuple[int, int]]:
    return {idx: (int(r * core_h), int(c * core_w)) for idx, (r, c) in enumerate(cells)}


def _scatter_patches(patches: list[np.ndarray], bg: int, rng: np.random.Generator) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    lower_h = max(20, min(30, 2 * max_h + 6))
    lower_w = max(20, min(30, 2 * max_w + 6))

    for _ in range(PLACE_RETRIES):
        grid_h = int(rng.integers(lower_h, 31))
        grid_w = int(rng.integers(lower_w, 31))
        grid = np.full((grid_h, grid_w), int(bg), dtype=int)
        blocked = np.zeros((grid_h, grid_w), dtype=bool)
        order = list(map(int, rng.permutation(len(patches))))
        placed = [None] * len(patches)
        ok = True

        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            success = False
            for _ in range(PLACE_RETRIES):
                r0 = int(rng.integers(0, grid_h - ph + 1))
                c0 = int(rng.integers(0, grid_w - pw + 1))
                rr0 = max(0, r0 - 1)
                rr1 = min(grid_h, r0 + ph + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(grid_w, c0 + pw + 1)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue

                mask = patch != int(bg)
                grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
                blocked[r0 : r0 + ph, c0 : c0 + pw] = True
                placed[idx] = (r0, c0)
                success = True
                break
            if not success:
                ok = False
                break

        if ok:
            return grid
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = 0
    layout = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    core_h = int(rng.integers(5, 8))
    core_w = int(rng.integers(5, 8))
    palette = np.array(rng.choice(np.arange(1, 10), size=3, replace=False), dtype=int)

    cells, feature_specs = _layout_features(layout, core_h, core_w, rng)
    pieces = []
    for idx, features in enumerate(feature_specs):
        core = _make_core_pattern(core_h, core_w, palette, idx, rng)
        patch = _build_patch(core, features, bg)
        pieces.append(
            {
                "patch": patch,
                "core_origin": (1 if features["top"][0] == "out" else 0, 1 if features["left"][0] == "out" else 0),
                "core_shape": (core_h, core_w),
                "features": features,
            }
        )

    expected = _assemble_pieces(pieces, _truth_positions(cells, core_h, core_w), bg)
    assert max(expected.shape) <= 30

    inp = _scatter_patches([piece["patch"] for piece in pieces], bg, rng)
    if inp is None:
        return None
    if max(inp.shape) > 30:
        return None

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid jigsaw-lock ARC sample")
