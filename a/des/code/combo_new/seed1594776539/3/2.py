import random
import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_RETRIES = 400


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
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k) for k in range(4)]


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = [((int(rot.shape[0]), int(rot.shape[1])), rot.tobytes()) for rot in _rotation_variants(mask)]
    return min(variants)


def _rotation_count(mask: np.ndarray) -> int:
    return len({((int(rot.shape[0]), int(rot.shape[1])), rot.tobytes()) for rot in _rotation_variants(mask)})


def _match_rotation(src: np.ndarray, dst: np.ndarray) -> int | None:
    for k, rot in enumerate(_rotation_variants(src)):
        if rot.shape == dst.shape and np.array_equal(rot, dst):
            return k
    return None


def _find_templates(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = grid.shape
    used = np.zeros((h, w), dtype=bool)
    templates: list[dict] = []
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    for color in colors:
        for coords in _components(grid == color):
            mask, bbox = _tight_mask(coords)
            r0, c0, r1, c1 = bbox
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if mask.shape[0] < 3 or mask.shape[1] < 3:
                continue
            if not np.all((patch == color) | (patch == bg)):
                continue
            if not (np.all(patch[0, :] == color) and np.all(patch[-1, :] == color)
                    and np.all(patch[:, 0] == color) and np.all(patch[:, -1] == color)):
                continue
            bg_comps = _components(patch == bg)
            if len(bg_comps) != 1:
                continue
            hole = bg_comps[0]
            if np.any(hole[:, 0] == 0) or np.any(hole[:, 1] == 0) or np.any(hole[:, 0] == patch.shape[0] - 1) or np.any(hole[:, 1] == patch.shape[1] - 1):
                continue
            hole_mask, hole_bbox = _tight_mask(hole)
            hole_global = hole + np.array([[r0, c0]])
            used[coords[:, 0], coords[:, 1]] = True
            templates.append(
                {
                    "color": color,
                    "component_coords": coords,
                    "hole_coords": hole_global,
                    "hole_mask": hole_mask,
                    "bbox": bbox,
                    "hole_bbox": hole_bbox,
                    "area": int(hole_mask.sum()),
                }
            )
    return templates, used


def _find_fragments(grid: np.ndarray, bg: int, template_mask: np.ndarray) -> list[dict]:
    leftovers = (grid != bg) & (~template_mask)
    fragments: list[dict] = []
    for coords in _components(leftovers):
        mask, bbox = _tight_mask(coords)
        vals = grid[coords[:, 0], coords[:, 1]]
        color = int(vals[0])
        fragments.append(
            {
                "color": color,
                "coords": coords,
                "mask": mask,
                "bbox": bbox,
                "area": int(mask.sum()),
            }
        )
    return fragments


def _solve_assignment(templates: list[dict], fragments: list[dict]) -> list[int]:
    assert len(templates) == len(fragments), "template and fragment counts must match"
    candidates: list[list[int]] = []
    for template in templates:
        hits = []
        for j, fragment in enumerate(fragments):
            if fragment["area"] != template["area"]:
                continue
            if _match_rotation(fragment["mask"], template["hole_mask"]) is not None:
                hits.append(j)
        assert hits, "each template must have at least one matching fragment"
        candidates.append(hits)

    order = sorted(range(len(templates)), key=lambda i: (len(candidates[i]), -templates[i]["area"]))
    assignment = [-1] * len(templates)
    used: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        ti = order[pos]
        for fj in candidates[ti]:
            if fj in used:
                continue
            used.add(fj)
            assignment[ti] = fj
            if dfs(pos + 1):
                return True
            used.remove(fj)
            assignment[ti] = -1
        return False

    assert dfs(0), "matching must be one-to-one"
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reinsert scattered fragments into their matching rectangular templates by shape, recoloring them to the template color.

    Input structure:
    - The background is black.
    - Some monochrome rectangular templates appear on the grid; each one has a solid same-color border/fill with exactly one connected black hole strictly inside.
    - Elsewhere, each remaining non-black connected component is a small fragment whose shape matches one template hole up to a 0/90/180/270 degree rotation.

    Rule:
    - Detect every rectangular template and extract its internal black hole shape.
    - Find the unique outside fragment whose shape becomes identical to that hole under one of the four right-angle rotations.
    - Erase the matched fragment from its original location and fill the template hole with the template's own color.

    Key insight:
    - Fragment color is a distractor: matching is done only by connected shape under rotation, while the filled result inherits the template color.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())

    templates, template_mask = _find_templates(g, bg)
    fragments = _find_fragments(g, bg, template_mask)
    assert templates, "expected at least one template"
    assert fragments, "expected at least one fragment"

    assignment = _solve_assignment(templates, fragments)

    out = g.copy()
    for fragment in fragments:
        out[fragment["coords"][:, 0], fragment["coords"][:, 1]] = bg
    for ti, fj in enumerate(assignment):
        template = templates[ti]
        fragment = fragments[fj]
        assert _match_rotation(fragment["mask"], template["hole_mask"]) is not None
        out[template["hole_coords"][:, 0], template["hole_coords"][:, 1]] = int(template["color"])
    return out


def _random_shape() -> np.ndarray | None:
    board_size = 7
    seed = board_size // 2
    for _ in range(80):
        area = random.randint(3, 6)
        cells = {(seed, seed)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))
        coords = np.array(sorted(cells), dtype=int)
        mask, _ = _tight_mask(coords)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if mask.shape[0] > 4 or mask.shape[1] > 4:
            continue
        if int(mask.sum()) == mask.size:
            continue
        if _rotation_count(mask) != 4:
            continue
        return mask
    return None


def _template_patch(hole_mask: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray]:
    hh, hw = hole_mask.shape
    pad_t = random.randint(1, 2)
    pad_b = random.randint(1, 2)
    pad_l = random.randint(1, 2)
    pad_r = random.randint(1, 2)
    patch = np.full((hh + pad_t + pad_b, hw + pad_l + pad_r), color, dtype=int)
    hole_coords = np.argwhere(hole_mask)
    hole_coords = hole_coords + np.array([[pad_t, pad_l]])
    patch[hole_coords[:, 0], hole_coords[:, 1]] = BG
    return patch, hole_coords


def _fragment_patch(hole_mask: np.ndarray, color: int) -> tuple[np.ndarray, np.ndarray, int]:
    rot_k = random.randint(0, 3)
    rot = np.rot90(hole_mask, rot_k)
    patch = np.full(rot.shape, BG, dtype=int)
    coords = np.argwhere(rot)
    patch[coords[:, 0], coords[:, 1]] = color
    return patch, coords, rot_k


def _valid_origins(occupied: np.ndarray, ph: int, pw: int) -> list[tuple[int, int]]:
    h, w = occupied.shape
    origins = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            origins.append((r0, c0))
    random.shuffle(origins)
    return origins


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    ph, pw = patch.shape
    origins = _valid_origins(occupied, ph, pw)
    if not origins:
        return None
    r0, c0 = origins[0]
    mask = patch != BG
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1] = True
    return r0, c0


def _expected_output(inp: np.ndarray, template_infos: list[dict], fragment_infos: list[dict]) -> np.ndarray:
    out = inp.copy()
    for info in fragment_infos:
        coords = info["coords"]
        out[coords[:, 0], coords[:, 1]] = BG
    for info in template_infos:
        coords = info["hole_coords"]
        out[coords[:, 0], coords[:, 1]] = int(info["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    n_templates = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 1])[0]
    colors = random.sample(range(1, 10), 2 * n_templates)
    template_colors = colors[:n_templates]
    fragment_colors = colors[n_templates:]

    hole_masks: list[np.ndarray] = []
    seen_keys: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(n_templates):
        mask = None
        for _ in range(100):
            cand = _random_shape()
            if cand is None:
                continue
            key = _rotation_key(cand)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            mask = cand
            break
        if mask is None:
            return None
        hole_masks.append(mask)

    template_patches = []
    fragment_patches = []
    template_meta = []
    fragment_meta = []
    for hole_mask, tcol, fcol in zip(hole_masks, template_colors, fragment_colors):
        tpatch, local_hole = _template_patch(hole_mask, tcol)
        fpatch, frag_local, _rot = _fragment_patch(hole_mask, fcol)
        template_patches.append((tpatch, local_hole, tcol))
        fragment_patches.append((fpatch, frag_local, fcol))

    all_patches = [patch for patch, _hole, _color in template_patches] + [patch for patch, _frag, _color in fragment_patches]
    padded_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in all_patches)
    max_h = max(int(patch.shape[0]) for patch in all_patches)
    max_w = max(int(patch.shape[1]) for patch in all_patches)

    grid_h_min = max(max_h + 2, int(np.ceil(np.sqrt(padded_area))) - 1)
    grid_h_max = min(30, max(grid_h_min, grid_h_min + 8))
    if grid_h_min > 30:
        return None
    grid_h = random.randint(grid_h_min, grid_h_max)

    grid_w_min = max(max_w + 2, int(np.ceil(padded_area / grid_h)))
    grid_w_max = min(30, max(grid_w_min, grid_w_min + 10))
    if grid_w_min > 30:
        return None
    grid_w = random.randint(grid_w_min, grid_w_max)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    template_order = list(range(n_templates))
    random.shuffle(template_order)
    for idx in sorted(template_order, key=lambda i: -template_patches[i][0].size):
        patch, local_hole, color = template_patches[idx]
        placed = _place_patch(inp, occupied, patch)
        if placed is None:
            return None
        r0, c0 = placed
        template_meta.append(
            {
                "color": color,
                "hole_coords": local_hole + np.array([[r0, c0]]),
            }
        )

    fragment_order = list(range(n_templates))
    random.shuffle(fragment_order)
    for idx in sorted(fragment_order, key=lambda i: -fragment_patches[i][0].size):
        patch, frag_local, color = fragment_patches[idx]
        placed = _place_patch(inp, occupied, patch)
        if placed is None:
            return None
        r0, c0 = placed
        fragment_meta.append(
            {
                "color": color,
                "coords": frag_local + np.array([[r0, c0]]),
            }
        )

    out = _expected_output(inp, template_meta, fragment_meta)
    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_RETRIES} attempts")
