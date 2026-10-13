import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
GEN_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.ravel(grid))).most_common(1)[0][0])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _bbox_from_cells(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_cells(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _, _ = _bbox_from_cells(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            visited[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _patch_components(mask: np.ndarray) -> int:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if visited[r, c] or not bool(mask[r, c]):
                continue
            count += 1
            visited[r, c] = True
            q = deque([(r, c)])
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not visited[nr, nc] and bool(mask[nr, nc]):
                        visited[nr, nc] = True
                        q.append((nr, nc))
    return count


def _interior_cells_from_component(
    coords: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], frozenset[tuple[int, int]], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_cells(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    outline = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        outline[r - r0, c - c0] = True

    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if not bool(outline[r, c]) and not bool(outside[r, c]):
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bool(outline[nr, nc]) or bool(outside[nr, nc]):
                continue
            outside[nr, nc] = True
            q.append((nr, nc))

    interior_global = set()
    interior_rel = set()
    for r in range(h):
        for c in range(w):
            if not bool(outline[r, c]) and not bool(outside[r, c]):
                interior_global.add((r0 + r, c0 + c))
                interior_rel.add((r, c))
    return interior_global, frozenset(interior_rel), (r0, c0, r1, c1)


def _select_outline_group(grid: np.ndarray, bg: int):
    grouped: dict[tuple[int, frozenset[tuple[int, int]]], list[dict]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for comp in _components_of_color(grid, color):
            sig = _normalize_cells(comp)
            grouped.setdefault((color, sig), []).append({"color": color, "coords": comp})

    candidates = []
    for (_color, sig), members in grouped.items():
        if len(members) < 2:
            continue
        proto_indices = []
        interior_sig = None
        valid = True
        for idx, member in enumerate(members):
            interior, rel_interior, bbox = _interior_cells_from_component(member["coords"])
            if not interior:
                valid = False
                break
            if interior_sig is None:
                interior_sig = rel_interior
            elif rel_interior != interior_sig:
                valid = False
                break
            member["bbox"] = bbox
            member["coord_set"] = set(member["coords"])
            member["interior"] = interior
            member["interior_rel"] = rel_interior
            if any(int(grid[r, c]) != int(bg) for r, c in interior):
                proto_indices.append(idx)
        if valid and len(proto_indices) == 1:
            candidates.append((len(members), len(sig), len(interior_sig), members, proto_indices[0]))

    assert candidates, "failed to identify the repeated hollow outline group"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, _, _, members, proto_idx = candidates[0]
    return members, proto_idx


def _find_marker_color(grid: np.ndarray, member: dict, bg: int) -> int | None:
    h, w = grid.shape
    candidates = set()
    for r, c in member["coord_set"]:
        for nr, nc in _neighbors4(r, c, h, w):
            if int(grid[nr, nc]) == int(bg):
                continue
            if (nr, nc) in member["coord_set"] or (nr, nc) in member["interior"]:
                continue
            candidates.add((nr, nc))

    if not candidates:
        return None
    singleton = []
    for r, c in candidates:
        color = int(grid[r, c])
        if all(int(grid[nr, nc]) != color for nr, nc in _neighbors4(r, c, h, w)):
            singleton.append((r, c))
    if len(singleton) == 1:
        return int(grid[singleton[0]])
    assert len(candidates) == 1, "expected at most one external marker per outline"
    only = next(iter(candidates))
    return int(grid[only])


def main(grid: np.ndarray) -> np.ndarray:
    """Copy the prototype's internal seed pattern into every congruent hollow outline, recoloring it from each outline's external marker when present.

    Input structure:
    - The background is the most common color.
    - Several monochrome hollow outline components share the same shape and orientation, differing only by translation.
    - Exactly one outline contains non-background seed pixels inside its enclosed interior; other copies have empty interiors.
    - Any outline may have zero or one single-pixel colored marker attached just outside one side of the outline.

    Rule:
    - Identify the repeated outline shape and the unique prototype whose enclosed interior contains the seed pattern.
    - Read the prototype's seed pixels by their positions relative to the outline's bounding box.
    - For each non-prototype copy, stamp the same relative seed positions into its interior.
    - If that copy has an external marker, recolor every stamped seed pixel to the marker's color; otherwise keep the prototype's original seed colors.

    Key insight:
    - The geometry comes from the repeated outline and the prototype's interior offsets, while the external single-pixel marker only controls the copied pattern's color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    members, proto_idx = _select_outline_group(g, bg)
    proto = members[proto_idx]
    pr0, pc0, _, _ = proto["bbox"]

    pattern = []
    for r, c in sorted(proto["interior"]):
        color = int(g[r, c])
        if color != int(bg):
            pattern.append((r - pr0, c - pc0, color))
    assert pattern, "prototype must contain at least one seed pixel"

    out = g.copy()
    for idx, member in enumerate(members):
        if idx == proto_idx:
            continue
        marker_color = _find_marker_color(g, member, bg)
        r0, c0, _, _ = member["bbox"]
        for dr, dc, color in pattern:
            out[r0 + dr, c0 + dc] = color if marker_color is None else marker_color
    return out


def _random_manhattan_path(start: tuple[int, int], end: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = start
    tr, tc = end
    cells = {(r, c)}
    while (r, c) != (tr, tc):
        moves = []
        if r < tr:
            moves.append((1, 0))
        elif r > tr:
            moves.append((-1, 0))
        if c < tc:
            moves.append((0, 1))
        elif c > tc:
            moves.append((0, -1))
        dr, dc = random.choice(moves)
        r += dr
        c += dc
        cells.add((r, c))
    return cells


def _build_core_mask(inner_h: int, inner_w: int) -> np.ndarray:
    anchor = (random.randrange(inner_h), random.randrange(inner_w))
    cells = {anchor}
    edge_targets = [
        (0, random.randrange(inner_w)),
        (inner_h - 1, random.randrange(inner_w)),
        (random.randrange(inner_h), 0),
        (random.randrange(inner_h), inner_w - 1),
    ]
    for target in edge_targets:
        cells |= _random_manhattan_path(anchor, target)

    full_area = inner_h * inner_w
    extra_budget = max(0, full_area - len(cells))
    target_size = random.randint(len(cells), len(cells) + extra_budget)
    while len(cells) < target_size:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < inner_h and 0 <= cc < inner_w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    core = np.zeros((inner_h, inner_w), dtype=bool)
    for r, c in cells:
        core[r, c] = True
    return core


def _outline_from_core(core: np.ndarray) -> np.ndarray:
    inner_h, inner_w = core.shape
    expanded = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
    expanded[1:-1, 1:-1] = core
    outline = np.zeros_like(expanded)
    for r in range(1, inner_h + 1):
        for c in range(1, inner_w + 1):
            if not bool(expanded[r, c]):
                continue
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if not bool(expanded[rr, cc]):
                    outline[rr, cc] = True
    return outline


def _interior_offsets_from_patch(outline: np.ndarray) -> list[tuple[int, int]]:
    h, w = outline.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if not bool(outline[r, c]) and not bool(outside[r, c]):
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bool(outline[nr, nc]) or bool(outside[nr, nc]):
                continue
            outside[nr, nc] = True
            q.append((nr, nc))
    interior = [(r, c) for r in range(h) for c in range(w) if not bool(outline[r, c]) and not bool(outside[r, c])]
    return interior


def _make_outline_patch() -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(120):
        inner_h = random.randint(2, 5)
        inner_w = random.randint(2, 5)
        outline = _outline_from_core(_build_core_mask(inner_h, inner_w))
        if _patch_components(outline) != 1:
            continue
        interior = _interior_offsets_from_patch(outline)
        if 2 <= len(interior) <= 14:
            return outline, interior
    return None


def _marker_offset_choices(outline: np.ndarray) -> list[tuple[int, int]]:
    h, w = outline.shape
    choices = set()
    for c in range(w):
        if bool(outline[0, c]):
            choices.add((-1, c))
        if bool(outline[h - 1, c]):
            choices.add((h, c))
    for r in range(h):
        if bool(outline[r, 0]):
            choices.add((r, -1))
        if bool(outline[r, w - 1]):
            choices.add((r, w))
    return list(choices)


def _choose_target_marker_flags(target_count: int) -> list[bool]:
    if target_count <= 0:
        return []
    if target_count == 1:
        return [random.random() < 0.5]
    for _ in range(30):
        flags = [random.random() < 0.5 for _ in range(target_count)]
        if any(flags) and not all(flags):
            return flags
    return [True] + [False] * (target_count - 1)


def _reserve_with_halo(reserved: np.ndarray, cells: list[tuple[int, int]] | set[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    reserved[rr, cc] = True


def _place_objects(
    shape_cells: set[tuple[int, int]],
    marker_offsets: list[tuple[int, int] | None],
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    reserved = np.zeros((h, w), dtype=bool)
    placements = []
    base_cells = list(shape_cells)
    for marker in marker_offsets:
        occupied = set(base_cells)
        if marker is not None:
            occupied.add(marker)
        min_r = min(r for r, _ in occupied)
        max_r = max(r for r, _ in occupied)
        min_c = min(c for _, c in occupied)
        max_c = max(c for _, c in occupied)

        candidates = []
        for r0 in range(-min_r, h - max_r):
            for c0 in range(-min_c, w - max_c):
                translated = [(r0 + r, c0 + c) for r, c in occupied]
                if any(bool(reserved[rr, cc]) for rr, cc in translated):
                    continue
                candidates.append((r0, c0, translated))
        if not candidates:
            return None
        r0, c0, translated = random.choice(candidates)
        placements.append((r0, c0))
        _reserve_with_halo(reserved, translated)
    return placements


def _stamp_object(grid: np.ndarray, outline: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = outline.shape
    patch = grid[top : top + h, left : left + w]
    patch[outline] = int(color)


def _gen_attempt() -> np.ndarray | None:
    built = _make_outline_patch()
    if built is None:
        return None
    outline, interior = built
    outline_h, outline_w = outline.shape

    obj_count = random.randint(2, 5)
    proto_idx = random.randrange(obj_count)
    target_flags = _choose_target_marker_flags(obj_count - 1)
    marker_present = []
    for idx in range(obj_count):
        if idx == proto_idx:
            marker_present.append(random.random() < 0.5)
        else:
            marker_present.append(target_flags.pop(0))

    bg = random.randrange(10)
    outline_color = random.choice([c for c in range(10) if c != bg])
    available = [c for c in range(10) if c not in {bg, outline_color}]
    if len(available) < 2:
        return None

    seed_count = random.randint(1, min(len(interior), 6))
    seed_offsets = random.sample(interior, seed_count)
    if seed_count == 1:
        seed_palette_size = 1
    else:
        seed_palette_size = random.choice([1, 2, min(3, seed_count)])
    seed_palette = random.sample(available, seed_palette_size)
    seed_pattern = [(r, c, random.choice(seed_palette)) for r, c in seed_offsets]

    marker_choices = _marker_offset_choices(outline)
    if not marker_choices:
        return None
    marker_colors_pool = [c for c in available if c not in seed_palette] or available
    marker_offsets: list[tuple[int, int] | None] = []
    marker_colors: list[int | None] = []
    for present in marker_present:
        if present:
            marker_offsets.append(random.choice(marker_choices))
            marker_colors.append(random.choice(marker_colors_pool))
        else:
            marker_offsets.append(None)
            marker_colors.append(None)

    footprint = {(r, c) for r in range(outline_h) for c in range(outline_w)}
    min_side = max(
        outline_h + 4,
        outline_w + 4,
        int(np.ceil(np.sqrt(obj_count * len(footprint) * 3.0))),
    )
    if min_side > 30:
        return None
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)
    placements = _place_objects(footprint, marker_offsets, h, w)
    if placements is None:
        return None

    inp = np.full((h, w), bg, dtype=int)
    out = inp.copy()
    for idx, (top, left) in enumerate(placements):
        _stamp_object(inp, outline, top, left, outline_color)
        _stamp_object(out, outline, top, left, outline_color)

        marker_offset = marker_offsets[idx]
        marker_color = marker_colors[idx]
        if marker_offset is not None and marker_color is not None:
            mr, mc = marker_offset
            inp[top + mr, left + mc] = marker_color
            out[top + mr, left + mc] = marker_color

        if idx == proto_idx:
            for dr, dc, color in seed_pattern:
                inp[top + dr, left + dc] = color
                out[top + dr, left + dc] = color
            continue

        fill_color = marker_color
        for dr, dc, color in seed_pattern:
            out[top + dr, left + dc] = color if fill_color is None else fill_color

    if _most_common_color(inp) != bg:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
