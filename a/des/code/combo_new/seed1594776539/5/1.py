from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_TEMPLATE_COUNT_RANGE = (2, 4)
GEN_EXTRA_FRAMES_MAX = 2
GEN_GRID_MIN = 18
GEN_GRID_MAX = 30
GEN_PADDING = 1
GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _monochrome_components(grid: np.ndarray, *, skip: set[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    skip = set() if skip is None else set(skip)
    comps: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color in skip:
            continue
        for coords in _components_from_mask(g == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(coords)),
                }
            )
    return comps


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _is_rectangular_frame(grid: np.ndarray, comp: dict) -> bool:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    return int(comp["area"]) == int(border.sum()) and np.array_equal(sub == color, border)


def _inside_frame_interior(comp_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = comp_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    return fr0 < r0 <= r1 < fr1 and fc0 < c0 <= c1 < fc1


def _shape_mask_from_coords(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _c in coords) + 1
    w = max(c for _r, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _subset_offsets(
    template_coords: tuple[tuple[int, int], ...],
    fragment_coords: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    template_set = set(template_coords)
    frag = list(fragment_coords)
    offsets: set[tuple[int, int]] = set()
    for tr, tc in template_coords:
        for fr, fc in frag:
            off = (tr - fr, tc - fc)
            if all((rr + off[0], cc + off[1]) in template_set for rr, cc in frag):
                offsets.add(off)
    return sorted(offsets)


def _frame_border_patch(height: int, width: int, frame_color: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each frame's source template from its internal colored fragment, then complete the framed shape using the grey anchor.

    Input structure:
    - The background is black.
    - Several single-color rectangular hollow frames sit on the grid; each frame has a black interior.
    - Inside every frame there is exactly one connected colored fragment and one grey marker cell.
    - Outside the frames there are separate colored template objects, one per template color; each frame fragment color matches one template color.

    Rule:
    - Detect every rectangular frame and the unique fragment and grey marker inside it.
    - For a frame, find the outside template with the same color as the fragment and locate the unique translation that embeds the fragment shape into the template shape without rotation or reflection.
    - Use the grey cell as the anchor for that matched template position, then paint the entire template inside the frame while keeping the frame border unchanged.
    - Remove all outside template objects to black.

    Key insight:
    - Color identifies which outside template belongs to a frame, the fragment's shape fixes where it sits inside that template, and the grey anchor confirms the exact template placement to complete the missing cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(_dominant_color(g))
    comps = _monochrome_components(g, skip={bg})
    frames = [comp for comp in comps if _is_rectangular_frame(g, comp)]
    assert frames, "expected at least one frame"

    frame_ids = {id(frame) for frame in frames}
    frame_bboxes = [frame["bbox"] for frame in frames]
    template_comps: dict[int, dict] = {}

    for comp in comps:
        if id(comp) in frame_ids:
            continue
        if any(_inside_frame_interior(comp["bbox"], fb) for fb in frame_bboxes):
            continue
        color = int(comp["color"])
        if color == GREY:
            continue
        assert color not in template_comps, f"expected one outside template for color {color}"
        template_comps[color] = comp

    frame_specs: list[dict] = []
    for frame in frames:
        fr0, fc0, fr1, fc1 = frame["bbox"]
        interior = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
        marker_coords = np.argwhere(interior == GREY)
        assert marker_coords.shape == (1, 2), "each frame interior must contain one grey marker"
        marker_r = int(fr0 + 1 + marker_coords[0, 0])
        marker_c = int(fc0 + 1 + marker_coords[0, 1])

        inner_colors = set(map(int, np.unique(interior))) - {bg, GREY}
        assert len(inner_colors) == 1, "each frame interior must contain one fragment color"
        frag_color = inner_colors.pop()

        frag_mask = interior == frag_color
        frag_comps = _components_from_mask(frag_mask)
        assert len(frag_comps) == 1, "fragment must be one connected component"
        frag_local = frag_comps[0]
        frag_abs = frag_local + np.array([fr0 + 1, fc0 + 1], dtype=int)
        frag_bbox = _bbox(frag_abs)
        frame_specs.append(
            {
                "frame_color": int(frame["color"]),
                "frame_bbox": frame["bbox"],
                "frag_color": int(frag_color),
                "frag_norm": _normalize(frag_abs),
                "frag_bbox": frag_bbox,
                "marker": (marker_r, marker_c),
            }
        )

    out = np.full_like(g, bg)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _frame_border_patch(r1 - r0 + 1, c1 - c0 + 1, int(frame["color"]))

    for spec in frame_specs:
        color = int(spec["frag_color"])
        assert color in template_comps, f"missing outside template for color {color}"
        template_coords = _normalize(template_comps[color]["coords"])
        offsets = _subset_offsets(template_coords, spec["frag_norm"])
        assert len(offsets) == 1, "fragment must embed into its template in exactly one way"
        off_r, off_c = offsets[0]

        frag_r0, frag_c0, _frag_r1, _frag_c1 = spec["frag_bbox"]
        marker_r, marker_c = spec["marker"]
        anchor = (marker_r - frag_r0 + off_r, marker_c - frag_c0 + off_c)
        template_set = set(template_coords)
        frag_set = {(r + off_r, c + off_c) for r, c in spec["frag_norm"]}
        assert anchor in template_set, "grey marker must correspond to a template pixel"
        assert anchor not in frag_set, "grey marker must point to a missing template cell"

        place_r0 = marker_r - anchor[0]
        place_c0 = marker_c - anchor[1]
        fr0, fc0, fr1, fc1 = spec["frame_bbox"]

        for tr, tc in template_coords:
            rr = place_r0 + tr
            cc = place_c0 + tc
            assert fr0 < rr < fr1 and fc0 < cc < fc1, "completed template must stay inside the frame interior"
            out[rr, cc] = color

    return out


def _mask_is_connected(mask: np.ndarray) -> bool:
    comps = _components_from_mask(mask)
    return len(comps) == 1 if np.any(mask) else False


def _random_connected_mask() -> np.ndarray:
    area = random.randint(4, 8)
    h = random.randint(2, 5)
    w = random.randint(2, 5)
    if area > h * w:
        return _random_connected_mask()

    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    cells = [seed]
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return _random_connected_mask()
        cell = random.choice(tuple(frontier))
        mask[cell] = True
        cells.append(cell)

    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
    if cropped.shape[0] < 2 or cropped.shape[1] < 2:
        return _random_connected_mask()
    if not (cropped[0, :].any() and cropped[-1, :].any() and cropped[:, 0].any() and cropped[:, -1].any()):
        return _random_connected_mask()
    if _is_rectangular_frame(cropped.astype(int), {"color": 1, "bbox": (0, 0, cropped.shape[0] - 1, cropped.shape[1] - 1), "area": int(cropped.sum())}):
        return _random_connected_mask()
    return cropped


def _connected_subsets(mask: np.ndarray) -> list[dict]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    n = len(coords)
    template_norm = tuple(sorted(coords))
    template_set = set(template_norm)
    out: list[dict] = []

    for bits in range(1 << n):
        size = int(bits.bit_count())
        if size < 2 or size > min(5, n - 1):
            continue
        subset = [coords[i] for i in range(n) if (bits >> i) & 1]
        subset_mask = _shape_mask_from_coords(_normalize(np.array(subset, dtype=int)))
        if not _mask_is_connected(subset_mask):
            continue
        frag_norm = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(subset_mask)))
        offsets = _subset_offsets(template_norm, frag_norm)
        if len(offsets) != 1:
            continue
        off_r, off_c = offsets[0]
        placed = {(r + off_r, c + off_c) for r, c in frag_norm}
        missing = sorted(template_set - placed)
        if not missing:
            continue
        out.append({"frag_norm": frag_norm, "offset": (off_r, off_c), "missing": missing})

    return out


def _render_template(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _render_frame_pair(
    template_mask: np.ndarray,
    template_color: int,
    frame_color: int,
    option: dict,
) -> tuple[np.ndarray, np.ndarray]:
    th, tw = template_mask.shape
    pad_top = random.randint(0, 2)
    pad_bottom = random.randint(0, 2)
    pad_left = random.randint(0, 2)
    pad_right = random.randint(0, 2)
    inner_h = th + pad_top + pad_bottom
    inner_w = tw + pad_left + pad_right

    inp = _frame_border_patch(inner_h + 2, inner_w + 2, frame_color)
    out = _frame_border_patch(inner_h + 2, inner_w + 2, frame_color)

    base_r = pad_top
    base_c = pad_left
    full_coords = [tuple(map(int, rc)) for rc in np.argwhere(template_mask)]
    frag_norm = option["frag_norm"]
    off_r, off_c = option["offset"]
    anchor_r, anchor_c = random.choice(option["missing"])

    for tr, tc in full_coords:
        out[1 + base_r + tr, 1 + base_c + tc] = template_color
    for fr, fc in frag_norm:
        inp[1 + base_r + off_r + fr, 1 + base_c + off_c + fc] = template_color
    inp[1 + base_r + anchor_r, 1 + base_c + anchor_c] = GREY
    return inp, out


def _rect_fits(
    placements: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
    H: int,
    W: int,
    pad: int,
) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for pr0, pc0, ph, pw in placements:
        if not (r0 + h + pad <= pr0 or pr0 + ph + pad <= r0 or c0 + w + pad <= pc0 or pc0 + pw + pad <= c0):
            return False
    return True


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    placements: list[tuple[int, int, int, int]],
    *,
    pad: int,
) -> tuple[int, int] | None:
    H, W = grid.shape
    h, w = patch.shape
    candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates:
        if not _rect_fits(placements, r0, c0, h, w, H, W, pad):
            continue
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        placements.append((r0, c0, h, w))
        return r0, c0
    return None


def _generate_templates(count: int) -> list[dict] | None:
    templates: list[dict] = []
    shape_keys: set[bytes] = set()
    for _ in range(count):
        for _attempt in range(100):
            mask = _random_connected_mask()
            options = _connected_subsets(mask)
            key = mask.astype(np.uint8).tobytes() + str(mask.shape).encode()
            if options and key not in shape_keys:
                shape_keys.add(key)
                templates.append({"mask": mask, "options": options})
                break
        else:
            return None
    return templates


def _build_scene() -> tuple[np.ndarray, np.ndarray] | None:
    template_count = random.randint(*GEN_TEMPLATE_COUNT_RANGE)
    templates = _generate_templates(template_count)
    if templates is None:
        return None

    colors = list(PALETTE)
    random.shuffle(colors)
    template_colors = colors[:template_count]
    max_extra = min(GEN_EXTRA_FRAMES_MAX, len(PALETTE) - 2 * template_count)
    frame_count = template_count + random.randint(0, max_extra)
    frame_colors = colors[template_count : template_count + frame_count]

    assignments = list(range(template_count))
    while len(assignments) < frame_count:
        assignments.append(random.randrange(template_count))
    random.shuffle(assignments)

    frame_pairs = []
    for frame_color, template_index in zip(frame_colors, assignments):
        template = templates[template_index]
        option = random.choice(template["options"])
        frame_pairs.append(
            {
                "template_index": template_index,
                "template_color": template_colors[template_index],
                "frame_color": frame_color,
                "input_patch": None,
                "output_patch": None,
            }
        )
        inp_patch, out_patch = _render_frame_pair(
            template["mask"],
            template_colors[template_index],
            frame_color,
            option,
        )
        frame_pairs[-1]["input_patch"] = inp_patch
        frame_pairs[-1]["output_patch"] = out_patch

    min_side = max(GEN_GRID_MIN, 8 + 3 * len(frame_pairs))
    if min_side > GEN_GRID_MAX:
        return None
    H = random.randint(min_side, GEN_GRID_MAX)
    W = random.randint(min_side, GEN_GRID_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    placements: list[tuple[int, int, int, int]] = []

    frame_order = list(range(len(frame_pairs)))
    random.shuffle(frame_order)
    frame_positions: dict[int, tuple[int, int]] = {}
    for idx in frame_order:
        patch = frame_pairs[idx]["input_patch"]
        pos = _place_patch(inp, patch, placements, pad=GEN_PADDING)
        if pos is None:
            return None
        frame_positions[idx] = pos
        r0, c0 = pos
        oh, ow = frame_pairs[idx]["output_patch"].shape
        out[r0 : r0 + oh, c0 : c0 + ow] = frame_pairs[idx]["output_patch"]

    for template_index, template in enumerate(templates):
        patch = _render_template(template["mask"], template_colors[template_index])
        pos = _place_patch(inp, patch, placements, pad=GEN_PADDING)
        if pos is None:
            return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_scene()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ARC input")
