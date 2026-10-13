import numpy as np


BG = 0
TEMPLATE_COLOR = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = tuple(c for c in range(1, 10) if c != TEMPLATE_COLOR)
RNG = np.random.default_rng()
GENERATION_RETRIES = 400


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
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _mask_key(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    arr = mask.astype(np.uint8)
    return int(arr.shape[0]), int(arr.shape[1]), tuple(int(x) for x in arr.ravel())


def _mirror_key(mask: np.ndarray) -> tuple[tuple[int, int, tuple[int, ...]], tuple[int, int, tuple[int, ...]]]:
    return tuple(sorted((_mask_key(mask), _mask_key(np.fliplr(mask)))))


def _has_enclosed_bg(mask: np.ndarray) -> bool:
    h, w = mask.shape
    zeros = ~mask
    seen = np.zeros_like(mask, dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if zeros[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if zeros[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        rr, cc = stack.pop()
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and zeros[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(zeros & ~seen))


def _random_shape(rng: np.random.Generator) -> np.ndarray:
    for _ in range(300):
        target_area = int(rng.integers(4, 9))
        h = int(rng.integers(2, 5))
        w = int(rng.integers(3, 6))
        mask = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        mask[start] = True
        while int(mask.sum()) < min(target_area, h * w - 1):
            frontier = set()
            cells = np.argwhere(mask)
            for rr, cc in cells:
                for dr, dc in DIR4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = list(frontier)[int(rng.integers(0, len(frontier)))]
            mask[nr, nc] = True
        coords = np.argwhere(mask)
        tight, _ = _tight_mask(coords)
        if tight.shape[1] < 2:
            continue
        if int(tight.sum()) < 4 or int(tight.sum()) == tight.size:
            continue
        if np.array_equal(tight, np.fliplr(tight)):
            continue
        if _has_enclosed_bg(tight):
            continue
        return tight
    raise RuntimeError("failed to generate a valid shape")


def _pick_marker_cell(template: np.ndarray, side: str, rng: np.random.Generator) -> tuple[int, int]:
    h, w = template.shape
    if side == "left":
        candidates = [(r, 0) for r in range(1, h - 1)]
    elif side == "right":
        candidates = [(r, w - 1) for r in range(1, h - 1)]
    elif side == "top":
        candidates = [(0, c) for c in range(1, w - 1)]
    else:
        candidates = [(h - 1, c) for c in range(1, w - 1)]
    return candidates[int(rng.integers(0, len(candidates)))]


def _make_template(
    hole_mask: np.ndarray,
    marker_color: int,
    side: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    hh, hw = hole_mask.shape
    top = int(rng.integers(1, 3))
    bottom = int(rng.integers(1, 3))
    left = int(rng.integers(1, 3))
    right = int(rng.integers(1, 3))
    template = np.full((hh + top + bottom, hw + left + right), TEMPLATE_COLOR, dtype=int)
    hole_full = np.zeros_like(template, dtype=bool)
    hole_full[top : top + hh, left : left + hw] = hole_mask
    template[hole_full] = BG
    mr, mc = _pick_marker_cell(template, side, rng)
    template[mr, mc] = marker_color
    return template, hole_full


def _place_patch(grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray, rng: np.random.Generator) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    for _ in range(250):
        r0 = int(rng.integers(0, h - ph + 1))
        c0 = int(rng.integers(0, w - pw + 1))
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc1 = min(w, c0 + pw + 1)
        if np.any(reserve[rr0:rr1, cc0:cc1]):
            continue
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != BG
        if np.any(sub[mask] != BG):
            continue
        sub[mask] = patch[mask]
        reserve[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _solve_matching(templates: list[dict], fragments: list[dict]) -> dict[int, int]:
    candidate_lists = []
    for ti, template in enumerate(templates):
        need_flip = bool(template["flip_required"])
        hole_mask = template["hole_tight"]
        candidates = []
        for fi, fragment in enumerate(fragments):
            frag_mask = fragment["mask"]
            oriented = np.fliplr(frag_mask) if need_flip else frag_mask
            if oriented.shape == hole_mask.shape and np.array_equal(oriented, hole_mask):
                candidates.append(fi)
        candidate_lists.append((ti, candidates))

    candidate_lists.sort(key=lambda item: len(item[1]))
    solution: dict[int, int] | None = None
    used: set[int] = set()

    def backtrack(idx: int, current: dict[int, int]) -> int:
        nonlocal solution
        if idx == len(candidate_lists):
            solution = dict(current)
            return 1
        ti, candidates = candidate_lists[idx]
        count = 0
        for fi in candidates:
            if fi in used:
                continue
            used.add(fi)
            current[ti] = fi
            count += backtrack(idx + 1, current)
            if count > 1:
                return count
            del current[ti]
            used.remove(fi)
        return count

    count = backtrack(0, {})
    if count != 1 or solution is None:
        raise ValueError("template-fragment matching is not uniquely solvable")
    return solution


def _extract_templates(grid: np.ndarray) -> tuple[int, list[dict], set[tuple[int, int]]]:
    non_bg = [int(v) for v in np.unique(grid) if int(v) != BG]
    if not non_bg:
        raise ValueError("grid has no foreground objects")
    counts = {color: int(np.count_nonzero(grid == color)) for color in non_bg}
    template_color = max(counts, key=counts.get)

    templates = []
    marker_cells: set[tuple[int, int]] = set()
    for comp in _components(grid == template_color):
        r0 = int(comp[:, 0].min())
        c0 = int(comp[:, 1].min())
        r1 = int(comp[:, 0].max())
        c1 = int(comp[:, 1].max())
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        marker_coords = np.argwhere((patch != BG) & (patch != template_color))
        if len(marker_coords) != 1:
            raise ValueError("each template must contain exactly one boundary marker")
        mr, mc = map(int, marker_coords[0])
        marker_cells.add((r0 + mr, c0 + mc))
        if mc == 0 or mr == 0:
            flip_required = False
        elif mc == patch.shape[1] - 1 or mr == patch.shape[0] - 1:
            flip_required = True
        else:
            raise ValueError("marker must sit on the template boundary")

        hole_mask = patch == BG
        if not np.any(hole_mask):
            raise ValueError("template must contain a black hole")
        if np.any(hole_mask[0]) or np.any(hole_mask[-1]) or np.any(hole_mask[:, 0]) or np.any(hole_mask[:, -1]):
            raise ValueError("hole must stay inside the template")
        hole_coords = np.argwhere(hole_mask)
        hole_tight, hole_bbox = _tight_mask(hole_coords)
        templates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "hole_mask": hole_mask,
                "hole_tight": hole_tight,
                "hole_bbox_local": hole_bbox,
                "flip_required": flip_required,
            }
        )
    if not templates:
        raise ValueError("no templates found")
    return template_color, templates, marker_cells


def _extract_fragments(grid: np.ndarray, template_color: int, marker_cells: set[tuple[int, int]]) -> list[dict]:
    colored = np.zeros_like(grid, dtype=bool)
    for value in np.unique(grid):
        color = int(value)
        if color in (BG, template_color):
            continue
        coords = np.argwhere(grid == color)
        for r, c in coords:
            if (int(r), int(c)) not in marker_cells:
                colored[int(r), int(c)] = True

    fragments = []
    seen = np.zeros_like(colored, dtype=bool)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not colored[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and colored[nr, nc]
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            mask, bbox = _tight_mask(coords)
            fragments.append({"color": color, "coords": coords, "mask": mask, "bbox": bbox})
    if not fragments:
        raise ValueError("no fragments found")
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered colored fragments into gray template holes, using each template's edge marker to decide whether the fragment should be mirrored first.

    Input structure:
    - The background is black.
    - Several gray rectangular templates appear on the grid; each contains one connected black hole fully enclosed by gray pixels.
    - Each template has a single colored marker pixel on one boundary edge, and the remaining colored objects are scattered monochrome fragments.

    Rule:
    - Match each fragment to the unique template hole whose shape it can fill, allowing at most one horizontal mirror of the fragment.
    - If a template marker lies on the left or top edge, use the fragment as-is; if the marker lies on the right or bottom edge, mirror the fragment horizontally before fitting it.
    - Remove the original scattered fragments, keep the gray templates and markers unchanged, and fill each matched hole with that fragment's color.

    Key insight:
    - Shape matching identifies which fragment belongs to which hole, while the marker location tells whether the fragment's visible orientation should be direct or horizontally mirrored.
    """
    g = np.asarray(grid, dtype=int)
    template_color, templates, marker_cells = _extract_templates(g)
    fragments = _extract_fragments(g, template_color, marker_cells)
    matching = _solve_matching(templates, fragments)

    out = g.copy()
    for fragment in fragments:
        coords = fragment["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for ti, fi in matching.items():
        template = templates[ti]
        fragment = fragments[fi]
        r0, c0, _, _ = template["bbox"]
        hole_cells = np.argwhere(template["hole_mask"])
        out[r0 + hole_cells[:, 0], c0 + hole_cells[:, 1]] = int(fragment["color"])

    return out


def _build_expected(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        frag_coords = item["fragment_coords"]
        out[frag_coords[:, 0], frag_coords[:, 1]] = BG
        hole_cells = item["hole_global"]
        out[hole_cells[:, 0], hole_cells[:, 1]] = int(item["fragment_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng(int(RNG.integers(0, 1 << 32)))
    h = int(rng.integers(20, 31))
    w = int(rng.integers(20, 31))
    grid = np.zeros((h, w), dtype=int)
    reserve = np.zeros_like(grid, dtype=bool)

    n_templates = int(rng.choice([2, 4]))
    paired_mode = bool(rng.integers(0, 2))
    fragment_colors = rng.choice(COLOR_POOL, size=n_templates, replace=False).tolist()
    remaining = [c for c in COLOR_POOL if c not in fragment_colors]
    marker_colors = rng.choice(remaining, size=n_templates, replace=False).tolist()

    shapes: list[np.ndarray] = []
    seen_keys = set()
    base_count = n_templates if not paired_mode else n_templates // 2
    for _ in range(base_count):
        for _ in range(200):
            shape = _random_shape(rng)
            key = _mirror_key(shape)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            shapes.append(shape)
            break
        else:
            return None

    specs = []
    if paired_mode:
        for shape in shapes:
            specs.append({"hole": shape.copy(), "flip_required": False})
            specs.append({"hole": shape.copy(), "flip_required": True})
    else:
        flags = [False] * (n_templates // 2) + [True] * (n_templates // 2)
        rng.shuffle(flags)
        for shape, flag in zip(shapes, flags):
            specs.append({"hole": shape.copy(), "flip_required": bool(flag)})
    rng.shuffle(specs)

    placements = []
    for i, spec in enumerate(specs):
        side = rng.choice(["left", "top"]) if not spec["flip_required"] else rng.choice(["right", "bottom"])
        template_patch, hole_full = _make_template(spec["hole"], int(marker_colors[i]), str(side), rng)
        hole_tight = spec["hole"]
        fragment_mask = np.fliplr(hole_tight) if spec["flip_required"] else hole_tight.copy()
        fragment_patch = np.zeros_like(fragment_mask, dtype=int)
        fragment_patch[fragment_mask] = int(fragment_colors[i])

        template_pos = _place_patch(grid, reserve, template_patch, rng)
        if template_pos is None:
            return None
        fragment_pos = _place_patch(grid, reserve, fragment_patch, rng)
        if fragment_pos is None:
            return None

        tr, tc = template_pos
        fr, fc = fragment_pos
        hole_local = np.argwhere(hole_full)
        frag_local = np.argwhere(fragment_mask)
        placements.append(
            {
                "hole_global": hole_local + np.array([tr, tc]),
                "fragment_coords": frag_local + np.array([fr, fc]),
                "fragment_color": int(fragment_colors[i]),
            }
        )

    expected = _build_expected(grid, placements)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
