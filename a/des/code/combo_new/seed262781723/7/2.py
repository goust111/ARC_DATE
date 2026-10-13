import numpy as np


_BG = 0
_FG_COLORS = tuple(range(1, 10))
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_GEN_RETRIES = 400
_GEN_FRAME_RETRIES = 120
_GEN_PLACEMENT_RETRIES = 120
_GEN_FRAGMENT_RETRIES = 80


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _top_left_pixel(coords: np.ndarray) -> tuple[int, int]:
    best = min((int(r), int(c)) for r, c in coords.tolist())
    return best


def _sort_components(comps: list[np.ndarray]) -> list[np.ndarray]:
    return sorted(comps, key=_top_left_pixel)


def _placed_cells(patch: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    cells = set()
    for rr, cc in np.argwhere(patch):
        cells.add((int(r0 + rr), int(c0 + cc)))
    return cells


def _frame_info(coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 5 or w < 5:
        return None

    local = np.zeros((h, w), dtype=bool)
    rel = coords - np.array([r0, c0])
    local[rel[:, 0], rel[:, 1]] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[h - 1, :] = True
    border[:, 0] = True
    border[:, w - 1] = True
    if not np.all(local[border]):
        return None

    extras = np.argwhere(local & ~border)
    if not (1 <= len(extras) <= 2):
        return None

    anchors: list[tuple[int, int]] = []
    for ar_raw, ac_raw in extras:
        ar = int(ar_raw)
        ac = int(ac_raw)
        side_hits = int(ar == 1) + int(ar == h - 2) + int(ac == 1) + int(ac == w - 2)
        if side_hits != 1:
            return None

        same_neighbors = 0
        for dr, dc in _NEIGHBORS4:
            nr = ar + dr
            nc = ac + dc
            if 0 <= nr < h and 0 <= nc < w and local[nr, nc]:
                same_neighbors += 1
        if same_neighbors != 1:
            return None
        anchors.append((r0 + ar, c0 + ac))

    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "anchors": sorted(anchors),
    }


def _frame_and_fragments(g: np.ndarray, color: int) -> tuple[dict, list[np.ndarray]]:
    comps = _components(g == int(color))
    frames = []
    for comp in comps:
        info = _frame_info(comp)
        if info is not None:
            frames.append(info)
    assert len(frames) == 1, "each color must have exactly one framed container"

    frame = frames[0]
    frame_cells = {tuple(map(int, cell)) for cell in frame["coords"].tolist()}
    fragments = []
    for comp in comps:
        comp_cells = {tuple(map(int, cell)) for cell in comp.tolist()}
        if comp_cells == frame_cells:
            continue
        fragments.append(comp)
    fragments = _sort_components(fragments)
    assert fragments, "each frame color must have at least one fragment"
    assert len(fragments) <= len(frame["anchors"]), "fragments must fit the available anchors"
    return frame, fragments


def _stamp_fragment(out: np.ndarray, color: int, coords: np.ndarray, anchor: tuple[int, int]) -> None:
    origin_r, origin_c = _top_left_pixel(coords)
    anchor_r, anchor_c = anchor
    for rr_raw, cc_raw in coords:
        rr = int(rr_raw)
        cc = int(cc_raw)
        nr = int(anchor_r + (rr - origin_r))
        nc = int(anchor_c + (cc - origin_c))
        assert 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]
        assert int(out[nr, nc]) in (_BG, int(color))
        out[nr, nc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Move each scattered fragment into the same-colored framed container at its inward anchor.

    Input structure:
    - The background is black.
    - Several colors each appear as one hollow rectangular frame plus one or two single-pixel inward anchors on the frame's inner wall.
    - Outside the frames, the same colors also form one or two separate 4-connected monochrome fragments.

    Rule:
    - Keep every rectangular frame and its anchor pixels in place.
    - Remove all same-colored fragments that lie outside the frames.
    - For each color independently, sort its external fragments from top to bottom then left to right, sort the frame anchors in the same way, and stamp each fragment unchanged so its own top-left occupied pixel lands on the matching anchor.

    Key insight:
    - Color identifies which frame owns a fragment, while the frame's inward protrusions are the non-obvious placement slots that determine where the copied shapes must start.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    colors = [int(c) for c in np.unique(g) if int(c) != _BG]
    for color in colors:
        frame, fragments = _frame_and_fragments(g, color)

        for comp in fragments:
            out[comp[:, 0], comp[:, 1]] = _BG

        for comp, anchor in zip(fragments, frame["anchors"], strict=False):
            _stamp_fragment(out, color, comp, anchor)

    return out


def _grow_patch(rng: np.random.Generator, height: int, width: int, area: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=bool)
    patch[0, 0] = True
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        seen = set()
        for rr, cc in cells:
            for dr, dc in _NEIGHBORS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells and (nr, nc) not in seen:
                    frontier.append((nr, nc))
                    seen.add((nr, nc))
        if not frontier:
            return np.zeros((0, 0), dtype=bool)
        nr, nc = frontier[int(rng.integers(len(frontier)))]
        patch[nr, nc] = True
        cells.add((nr, nc))
    return patch


def _make_fragment_patch(rng: np.random.Generator) -> np.ndarray:
    for _ in range(_GEN_FRAGMENT_RETRIES):
        area = int(rng.integers(2, 6))
        height = int(rng.integers(1, 4))
        width = int(rng.integers(1, 4))
        if height * width < area:
            continue
        patch = _grow_patch(rng, height, width, area)
        if patch.size == 0:
            continue
        if not patch[0, 0]:
            continue
        if height > 1 and width > 1 and int(patch.sum()) == height * width:
            continue
        return patch
    raise RuntimeError("failed to build fragment patch")


def _patch_candidates(frame_h: int, frame_w: int, patch: np.ndarray) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    candidates: list[tuple[int, int]] = []
    if ph <= frame_h - 2:
        top_stop = min(frame_w - pw, frame_w - 2)
        for c in range(2, top_stop):
            candidates.append((1, int(c)))
    if pw <= frame_w - 2:
        left_stop = min(frame_h - ph, frame_h - 2)
        for r in range(2, left_stop):
            candidates.append((int(r), 1))
    return candidates


def _cells_touch(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    if a & b:
        return True
    for rr, cc in a:
        for dr, dc in _NEIGHBORS4:
            if (rr + dr, cc + dc) in b:
                return True
    return False


def _choose_frame_layout(
    rng: np.random.Generator, patches: list[np.ndarray]
) -> tuple[int, int, list[tuple[np.ndarray, tuple[int, int]]]]:
    max_ph = max(int(p.shape[0]) for p in patches)
    max_pw = max(int(p.shape[1]) for p in patches)

    for _ in range(_GEN_FRAME_RETRIES):
        frame_h = int(rng.integers(max(6, max_ph + 3), 12))
        frame_w = int(rng.integers(max(6, max_pw + 3), 12))

        candidates_per_patch = []
        for patch in patches:
            candidates = _patch_candidates(frame_h, frame_w, patch)
            if not candidates:
                break
            order = rng.permutation(len(candidates))
            candidates_per_patch.append([candidates[int(i)] for i in order])
        else:
            if len(patches) == 1:
                origin = candidates_per_patch[0][0]
                return frame_h, frame_w, [(patches[0], origin)]

            valid_pairs: list[list[tuple[np.ndarray, tuple[int, int]]]] = []
            for origin0 in candidates_per_patch[0]:
                cells0 = _placed_cells(patches[0], origin0[0], origin0[1])
                for origin1 in candidates_per_patch[1]:
                    if origin0 == origin1:
                        continue
                    cells1 = _placed_cells(patches[1], origin1[0], origin1[1])
                    if _cells_touch(cells0, cells1):
                        continue
                    pair = [(patches[0], origin0), (patches[1], origin1)]
                    pair.sort(key=lambda item: item[1])
                    valid_pairs.append(pair)
            if valid_pairs:
                return frame_h, frame_w, valid_pairs[int(rng.integers(len(valid_pairs)))]

    raise RuntimeError("failed to choose frame layout")


def _fits_with_pad(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _mark_with_pad(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_frames(
    rng: np.random.Generator, height: int, width: int, specs: list[dict]
) -> bool:
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["frame_h"] * specs[idx]["frame_w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        fh = int(spec["frame_h"])
        fw = int(spec["frame_w"])
        placed = False
        for _ in range(_GEN_PLACEMENT_RETRIES):
            r0 = int(rng.integers(0, height - fh + 1))
            c0 = int(rng.integers(0, width - fw + 1))
            if not _fits_with_pad(occupied, r0, c0, fh, fw, pad=1):
                continue
            spec["frame_pos"] = (r0, c0)
            _mark_with_pad(occupied, r0, c0, fh, fw, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _draw_frame(grid: np.ndarray, color: int, r0: int, c0: int, h: int, w: int, anchors: list[tuple[int, int]]) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color
    for ar, ac in anchors:
        grid[ar, ac] = color


def _place_external_fragments(
    rng: np.random.Generator, height: int, width: int, specs: list[dict]
) -> bool:
    occupied = np.zeros((height, width), dtype=bool)
    for spec in specs:
        r0, c0 = spec["frame_pos"]
        _mark_with_pad(occupied, r0, c0, spec["frame_h"], spec["frame_w"], pad=1)

    for spec in specs:
        prev_origin: tuple[int, int] | None = None
        placements: list[tuple[np.ndarray, tuple[int, int]]] = []
        for patch, _anchor_local in spec["placements"]:
            ph, pw = patch.shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(height - ph + 1):
                for c0 in range(width - pw + 1):
                    if prev_origin is not None and (r0, c0) <= prev_origin:
                        continue
                    if not _fits_with_pad(occupied, r0, c0, ph, pw, pad=1):
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                return False
            chosen = candidates[int(rng.integers(len(candidates)))]
            placements.append((patch, chosen))
            _mark_with_pad(occupied, chosen[0], chosen[1], ph, pw, pad=1)
            prev_origin = chosen
        spec["external"] = placements
    return True


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, color: int, r0: int, c0: int) -> None:
    for rr, cc in np.argwhere(patch):
        grid[r0 + int(rr), c0 + int(cc)] = int(color)


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    color_count = int(rng.integers(2, 4))
    colors = [int(v) for v in rng.choice(_FG_COLORS, size=color_count, replace=False)]

    if color_count == 2:
        fragment_counts = [1, 2]
    else:
        fragment_counts = [1, 1, 2] if int(rng.integers(2)) == 0 else [1, 2, 2]
    rng.shuffle(fragment_counts)

    specs: list[dict] = []
    for color, fragment_count in zip(colors, fragment_counts, strict=False):
        patches = [_make_fragment_patch(rng) for _ in range(fragment_count)]
        frame_h, frame_w, placements = _choose_frame_layout(rng, patches)
        specs.append(
            {
                "color": int(color),
                "frame_h": int(frame_h),
                "frame_w": int(frame_w),
                "placements": placements,
            }
        )

    if color_count == 2:
        height = int(rng.integers(18, 27))
        width = int(rng.integers(18, 27))
    else:
        height = int(rng.integers(23, 31))
        width = int(rng.integers(23, 31))

    if not _place_frames(rng, height, width, specs):
        raise RuntimeError("failed to place frames")
    if not _place_external_fragments(rng, height, width, specs):
        raise RuntimeError("failed to place fragments")

    inp = np.full((height, width), _BG, dtype=int)
    out = np.full((height, width), _BG, dtype=int)

    for spec in specs:
        color = int(spec["color"])
        frame_r, frame_c = spec["frame_pos"]
        anchors_abs = []
        for patch, anchor_local in spec["placements"]:
            ar = frame_r + int(anchor_local[0])
            ac = frame_c + int(anchor_local[1])
            anchors_abs.append((ar, ac))

        _draw_frame(inp, color, frame_r, frame_c, spec["frame_h"], spec["frame_w"], anchors_abs)
        _draw_frame(out, color, frame_r, frame_c, spec["frame_h"], spec["frame_w"], anchors_abs)

        for (patch, _anchor_local), (_patch2, ext_origin) in zip(spec["placements"], spec["external"], strict=False):
            _stamp_patch(inp, patch, color, ext_origin[0], ext_origin[1])

        for patch, anchor_abs in zip((p for p, _ in spec["placements"]), anchors_abs, strict=False):
            _stamp_patch(out, patch, color, anchor_abs[0], anchor_abs[1])

    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        try:
            inp, out = _build_case(rng)
        except RuntimeError:
            continue
        if np.array_equal(main(inp), out):
            return inp
    raise RuntimeError("failed to generate a valid anchored-container puzzle")
