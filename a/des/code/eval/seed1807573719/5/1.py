import numpy as np
from collections import deque


BG = 0
CORNERS = ("tl", "tr", "bl", "br")
GEN_RETRIES = 400
PLACEMENT_RETRIES = 80


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0)


def _has_hole(mask: np.ndarray) -> bool:
    bg_parts = _components(~mask)
    h, w = mask.shape
    for comp in bg_parts:
        if not np.any(
            (comp[:, 0] == 0)
            | (comp[:, 0] == h - 1)
            | (comp[:, 1] == 0)
            | (comp[:, 1] == w - 1)
        ):
            return True
    return False


def _corner_slices(name: str) -> tuple[int, int]:
    if name == "tl":
        return 0, 0
    if name == "tr":
        return 0, -1
    if name == "bl":
        return -1, 0
    return -1, -1


def _classify_template(grid: np.ndarray, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return None
    if np.any(patch[0] == BG) or np.any(patch[-1] == BG) or np.any(patch[:, 0] == BG) or np.any(patch[:, -1] == BG):
        return None

    corner_vals = {
        "tl": int(patch[0, 0]),
        "tr": int(patch[0, -1]),
        "bl": int(patch[-1, 0]),
        "br": int(patch[-1, -1]),
    }
    colors, counts = np.unique(list(corner_vals.values()), return_counts=True)
    if len(colors) != 2 or sorted(counts.tolist()) != [1, 3]:
        return None
    border_color = int(colors[np.argmax(counts)])
    marker_color = int(colors[np.argmin(counts)])
    marker_corner = next(name for name, value in corner_vals.items() if value == marker_color)

    nonzero = patch[patch != BG]
    obj_colors = set(map(int, np.unique(nonzero)))
    if obj_colors != {border_color, marker_color}:
        return None
    if int(np.count_nonzero(patch == marker_color)) != 1:
        return None

    cavity = patch == BG
    if not np.any(cavity):
        return None
    cavity_parts = _components(cavity)
    if len(cavity_parts) != 1:
        return None
    cavity_mask, (hr0, hc0) = _tight_mask(cavity_parts[0])
    if hr0 == 0 or hc0 == 0 or hr0 + cavity_mask.shape[0] == patch.shape[0] or hc0 + cavity_mask.shape[1] == patch.shape[1]:
        return None

    return {
        "kind": "template",
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "border_color": border_color,
        "marker_color": marker_color,
        "marker_corner": marker_corner,
        "cavity_mask": cavity_mask,
        "cavity_top_left": (hr0, hc0),
        "full_cavity": cavity,
    }


def _classify_fragment(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    vals = patch[patch != BG]
    colors = set(map(int, np.unique(vals)))
    assert len(colors) == 1, "fragment must be monochrome"
    color = next(iter(colors))
    mask, _ = _tight_mask(coords)
    return {
        "kind": "fragment",
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "color": int(color),
        "mask": mask,
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objs = _components(g != BG)
    templates: list[dict] = []
    fragments: dict[int, dict] = {}

    for coords in objs:
        template = _classify_template(g, coords)
        if template is not None:
            templates.append(template)
            continue
        fragment = _classify_fragment(g, coords)
        assert fragment["color"] not in fragments, "fragment colors must be unique"
        fragments[fragment["color"]] = fragment

    assert templates, "expected at least one template"
    assert len(templates) == len(fragments), "template/fragment count mismatch"

    out = g.copy()
    for fragment in fragments.values():
        out[fragment["coords"][:, 0], fragment["coords"][:, 1]] = BG

    for template in templates:
        border_color = template["border_color"]
        marker_color = template["marker_color"]
        assert border_color in fragments, "missing matching fragment"
        fragment = fragments[border_color]

        oriented = None
        for k in range(4):
            rot = np.rot90(fragment["mask"], k)
            if rot.shape == template["cavity_mask"].shape and np.array_equal(rot, template["cavity_mask"]):
                oriented = rot
                break
        assert oriented is not None, "fragment must match cavity under rotation"

        r0, c0, r1, c1 = template["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        full_cavity = template["full_cavity"]
        hole_r0, hole_c0 = template["cavity_top_left"]
        hh, hw = oriented.shape

        if template["marker_corner"] == "tl":
            piece_color = border_color
        elif template["marker_corner"] == "tr":
            piece_color = marker_color
        elif template["marker_corner"] == "bl":
            patch[~full_cavity] = border_color
            piece_color = marker_color
        else:
            patch[~full_cavity] = marker_color
            piece_color = marker_color

        patch[hole_r0 : hole_r0 + hh, hole_c0 : hole_c0 + hw][oriented] = piece_color

    return out


def _random_piece(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        target = int(rng.integers(4, 9))
        cells = {(0, 0)}
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if -3 <= nr <= 3 and -3 <= nc <= 3 and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(frontier[int(rng.integers(len(frontier)))])

        coords = np.array(sorted(cells), dtype=int)
        mask, _ = _tight_mask(coords)
        h, w = mask.shape
        area = int(mask.sum())
        if h < 2 or w < 2 or h > 5 or w > 5:
            continue
        if area == h * w:
            continue
        if _has_hole(mask):
            continue
        return mask
    raise RuntimeError("failed to generate piece")


def _template_input_patch(
    cavity_mask: np.ndarray,
    border_color: int,
    marker_color: int,
    marker_corner: str,
    margins: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    top_extra, bottom_extra, left_extra, right_extra = margins
    ch, cw = cavity_mask.shape
    h = ch + 2 + top_extra + bottom_extra
    w = cw + 2 + left_extra + right_extra
    patch = np.full((h, w), border_color, dtype=int)
    r0 = 1 + top_extra
    c0 = 1 + left_extra
    patch[r0 : r0 + ch, c0 : c0 + cw][cavity_mask] = BG
    cr, cc = _corner_slices(marker_corner)
    patch[cr, cc] = marker_color
    full_cavity = patch == BG
    return patch, full_cavity, (r0, c0)


def _template_output_patch(
    input_patch: np.ndarray,
    full_cavity: np.ndarray,
    cavity_mask: np.ndarray,
    cavity_top_left: tuple[int, int],
    border_color: int,
    marker_color: int,
    marker_corner: str,
) -> np.ndarray:
    out = input_patch.copy()
    if marker_corner == "tl":
        piece_color = border_color
    elif marker_corner == "tr":
        piece_color = marker_color
    elif marker_corner == "bl":
        out[~full_cavity] = border_color
        piece_color = marker_color
    else:
        out[~full_cavity] = marker_color
        piece_color = marker_color
    r0, c0 = cavity_top_left
    ch, cw = cavity_mask.shape
    out[r0 : r0 + ch, c0 : c0 + cw][cavity_mask] = piece_color
    return out


def _fragment_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _place_patches(
    patches: list[np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)
    min_h = max(22, max_h + 8)
    min_w = max(22, max_w + 8)
    h = int(rng.integers(min_h, 33))
    w = int(rng.integers(min_w, 33))
    occ = np.zeros((h, w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    order = list(rng.permutation(len(patches)))

    for idx in order:
        ph, pw = patches[idx].shape
        candidates: list[tuple[int, int]] = []
        for r in range(h - ph + 1):
            for c in range(w - pw + 1):
                rr0 = max(0, r - 1)
                cc0 = max(0, c - 1)
                rr1 = min(h, r + ph + 1)
                cc1 = min(w, c + pw + 1)
                if not occ[rr0:rr1, cc0:cc1].any():
                    candidates.append((r, c))
        if not candidates:
            return None
        pos = candidates[int(rng.integers(len(candidates)))]
        positions[idx] = pos
        r, c = pos
        occ[r : r + ph, c : c + pw] = True

    grid = np.zeros((h, w), dtype=int)
    for patch, pos in zip(patches, positions):
        assert pos is not None
        r, c = pos
        mask = patch != BG
        grid[r : r + patch.shape[0], c : c + patch.shape[1]][mask] = patch[mask]
    return grid, [(int(r), int(c)) for r, c in positions]


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    colors = rng.permutation(np.arange(1, 10))
    border_colors = [int(c) for c in colors[:4]]
    marker_colors = [int(c) for c in colors[4:8]]

    template_inputs: list[np.ndarray] = []
    template_outputs: list[np.ndarray] = []
    fragments: list[np.ndarray] = []

    for i, corner in enumerate(CORNERS):
        base_mask = _random_piece(rng)
        cavity_mask = np.rot90(base_mask, int(rng.integers(4)))
        fragment_mask = np.rot90(base_mask, int(rng.integers(4)))
        margins = tuple(int(rng.integers(0, 3)) for _ in range(4))
        inp_patch, full_cavity, cavity_top_left = _template_input_patch(
            cavity_mask,
            border_colors[i],
            marker_colors[i],
            corner,
            margins,
        )
        out_patch = _template_output_patch(
            inp_patch,
            full_cavity,
            cavity_mask,
            cavity_top_left,
            border_colors[i],
            marker_colors[i],
            corner,
        )
        frag_patch = _fragment_patch(fragment_mask, border_colors[i])
        template_inputs.append(inp_patch)
        template_outputs.append(out_patch)
        fragments.append(frag_patch)

    all_patches = template_inputs + fragments
    placed = _place_patches(all_patches, rng)
    if placed is None:
        return None
    inp, positions = placed
    out = np.zeros_like(inp)

    for patch, pos in zip(template_outputs, positions[:4]):
        r, c = pos
        mask = patch != BG
        out[r : r + patch.shape[0], c : c + patch.shape[1]][mask] = patch[mask]

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        built = _gen_attempt(rng)
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate valid input")
