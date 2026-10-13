import numpy as np
from collections import Counter


BLACK = 0
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
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
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.argwhere(mask))


def _rotation_signatures(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    sigs: list[tuple[tuple[int, int], ...]] = []
    for k in range(4):
        sig = _mask_signature(np.rot90(mask, k))
        if sig not in sigs:
            sigs.append(sig)
    return sigs


def _largest_component(mask: np.ndarray) -> np.ndarray:
    comps = _find_components(mask)
    if not comps:
        raise ValueError("expected at least one component")
    return max(comps, key=len)


def _find_separator_row(grid: np.ndarray) -> int:
    h = int(grid.shape[0])
    rows = [
        r
        for r in range(1, h - 1)
        if int(grid[r, 0]) != BLACK and np.all(grid[r] == grid[r, 0])
    ]
    if not rows:
        raise ValueError("separator row not found")
    valid = [r for r in rows if np.any(grid[:r] != BLACK) and np.any(grid[r + 1 :] != BLACK)]
    if valid:
        rows = valid
    return int(rows[len(rows) // 2])


def _extract_holes(grid: np.ndarray, board_bbox: tuple[int, int, int, int]) -> list[np.ndarray]:
    r0, c0, r1, c1 = board_bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[np.ndarray] = []
    for comp in _find_components(sub == BLACK):
        touches_edge = (
            np.any(comp[:, 0] == 0)
            or np.any(comp[:, 0] == sub.shape[0] - 1)
            or np.any(comp[:, 1] == 0)
            or np.any(comp[:, 1] == sub.shape[1] - 1)
        )
        if touches_edge:
            continue
        hole = comp + np.array([r0, c0], dtype=int)
        holes.append(hole)
    holes.sort(key=lambda pts: _bbox(pts))
    return holes


def _match_fragments_to_holes(hole_masks: list[np.ndarray], fragment_masks: list[np.ndarray]) -> dict[int, int]:
    if len(hole_masks) != len(fragment_masks):
        raise ValueError("hole count and fragment count differ")

    hole_sigs = [_mask_signature(mask) for mask in hole_masks]
    compatibility: list[list[int]] = []
    for frag_mask in fragment_masks:
        sigs = set(_rotation_signatures(frag_mask))
        matches = [i for i, hole_sig in enumerate(hole_sigs) if hole_sig in sigs]
        if not matches:
            raise ValueError("fragment does not match any hole")
        compatibility.append(matches)

    order = sorted(range(len(fragment_masks)), key=lambda idx: len(compatibility[idx]))
    used = [False] * len(hole_masks)
    assignment: dict[int, int] = {}

    def dfs(depth: int) -> bool:
        if depth == len(order):
            return True
        frag_idx = order[depth]
        for hole_idx in compatibility[frag_idx]:
            if used[hole_idx]:
                continue
            used[hole_idx] = True
            assignment[frag_idx] = hole_idx
            if dfs(depth + 1):
                return True
            used[hole_idx] = False
            del assignment[frag_idx]
        return False

    if not dfs(0):
        raise ValueError("could not find a one-to-one hole assignment")
    return assignment


def _restore_border_color(
    grid: np.ndarray,
    hole_set: set[tuple[int, int]],
    hint_color: int,
    cell: tuple[int, int],
    board_bbox: tuple[int, int, int, int],
) -> int:
    r, c = cell
    r0, c0, r1, c1 = board_bbox
    candidates: list[int] = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if not (r0 <= nr <= r1 and c0 <= nc <= c1):
            continue
        if (nr, nc) in hole_set:
            continue
        value = int(grid[nr, nc])
        if value in (BLACK, hint_color):
            continue
        candidates.append(value)
    if not candidates:
        return int(grid[r, c])
    return int(Counter(candidates).most_common(1)[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_row(g)

    board_coords = _largest_component(g[:sep] != BLACK)
    board_bbox = _bbox(board_coords)
    holes = _extract_holes(g, board_bbox)

    fragment_components = [
        comp + np.array([sep + 1, 0], dtype=int)
        for comp in _find_components(g[sep + 1 :] != BLACK)
    ]
    if not holes or not fragment_components:
        raise ValueError("expected holes and fragments")

    hole_infos = []
    hole_masks: list[np.ndarray] = []
    for hole in holes:
        hole_set = {(int(r), int(c)) for r, c in hole}
        border: set[tuple[int, int]] = set()
        for r, c in hole_set:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if (nr, nc) in hole_set:
                    continue
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) != BLACK:
                    border.add((nr, nc))
        hint_values = [int(g[r, c]) for r, c in border]
        if not hint_values:
            raise ValueError("hole border not found")
        hint_color = int(Counter(hint_values).most_common(1)[0][0])
        hole_infos.append(
            {
                "coords": hole,
                "hole_set": hole_set,
                "border": sorted(border),
                "hint_color": hint_color,
            }
        )
        hole_masks.append(_coords_to_mask(hole))

    fragment_masks = [_coords_to_mask(comp) for comp in fragment_components]
    _match_fragments_to_holes(hole_masks, fragment_masks)

    out = g.copy()
    for comp in fragment_components:
        out[comp[:, 0], comp[:, 1]] = BLACK

    for hole_info in hole_infos:
        hole = hole_info["coords"]
        out[hole[:, 0], hole[:, 1]] = int(hole_info["hint_color"])
        for cell in hole_info["border"]:
            out[cell] = _restore_border_color(
                g,
                hole_info["hole_set"],
                int(hole_info["hint_color"]),
                cell,
                board_bbox,
            )

    return out


def _random_partition(rng: np.random.Generator, total: int, parts: int, min_size: int) -> list[int]:
    sizes = [min_size] * parts
    extra = total - parts * min_size
    for _ in range(extra):
        sizes[int(rng.integers(parts))] += 1
    rng.shuffle(sizes)
    return sizes


def _assign_block_colors(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    palette: list[int],
) -> list[list[int]]:
    grid: list[list[int]] = []
    for r in range(rows):
        row: list[int] = []
        for c in range(cols):
            forbidden = set()
            if r > 0:
                forbidden.add(grid[r - 1][c])
            if c > 0:
                forbidden.add(row[c - 1])
            choices = [color for color in palette if color not in forbidden]
            if not choices:
                choices = palette[:]
            row.append(int(rng.choice(choices)))
        grid.append(row)
    return grid


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _signature_to_mask(sig: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in sig) + 1
    w = max(c for _, c in sig) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in sig:
        mask[r, c] = True
    return mask


def _canonical_signature(sig: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_rotation_signatures(_signature_to_mask(sig)))


def _random_shape(
    rng: np.random.Generator,
    max_h: int,
    max_w: int,
) -> tuple[tuple[int, int], ...] | None:
    if max_h < 2 or max_w < 2:
        return None
    max_area = min(7, max_h * max_w - 1)
    if max_area < 3:
        return None
    for _ in range(100):
        target = int(rng.integers(3, max_area + 1))
        cells: set[tuple[int, int]] = {(0, 0)}
        ok = True
        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    candidate = set(cells)
                    candidate.add((nr, nc))
                    sig = _normalize_cells(candidate)
                    mask = _signature_to_mask(sig)
                    if mask.shape[0] <= max_h and mask.shape[1] <= max_w:
                        frontier.append((nr, nc))
            if not frontier:
                ok = False
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if not ok:
            continue
        sig = _normalize_cells(cells)
        mask = _signature_to_mask(sig)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if len(sig) == mask.shape[0] * mask.shape[1]:
            continue
        return sig
    return None


def _place_fragments(
    rng: np.random.Generator,
    grid: np.ndarray,
    sep: int,
    fragment_specs: list[dict[str, object]],
) -> bool:
    bottom_h = int(grid.shape[0] - sep - 1)
    width = int(grid.shape[1])
    occupied = np.zeros((bottom_h, width), dtype=bool)

    order = sorted(range(len(fragment_specs)), key=lambda idx: -int(fragment_specs[idx]["mask"].sum()))
    for idx in order:
        spec = fragment_specs[idx]
        mask = np.asarray(spec["mask"], dtype=bool)
        color = int(spec["color"])
        h, w = map(int, mask.shape)

        row_min = 1 if bottom_h - h >= 2 else 0
        row_max = bottom_h - h - (1 if bottom_h - h >= 2 else 0)
        if row_max < row_min:
            row_min, row_max = 0, bottom_h - h

        col_min = 1 if width - w >= 2 else 0
        col_max = width - w - (1 if width - w >= 2 else 0)
        if col_max < col_min:
            col_min, col_max = 0, width - w

        candidates = [(r, c) for r in range(row_min, row_max + 1) for c in range(col_min, col_max + 1)]
        rng.shuffle(candidates)

        placed = False
        pixels = np.argwhere(mask)
        for top, left in candidates:
            coords = [(int(top + r), int(left + c)) for r, c in pixels]
            blocked = False
            for r, c in coords:
                rr0 = max(0, r - 1)
                rr1 = min(bottom_h - 1, r + 1)
                cc0 = max(0, c - 1)
                cc1 = min(width - 1, c + 1)
                if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    blocked = True
                    break
            if blocked:
                continue
            for r, c in coords:
                occupied[r, c] = True
                grid[sep + 1 + r, c] = color
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    hole_count = int(rng.integers(2, 5))
    hint_colors = list(map(int, rng.choice(COLOR_POOL, size=hole_count, replace=False)))
    remaining = [color for color in COLOR_POOL if color not in hint_colors]

    row_parts = int(rng.choice([2, 2, 3]))
    col_parts_choices = [2, 3] if hole_count <= 4 else [3]
    col_parts = int(rng.choice(col_parts_choices))
    if row_parts * col_parts < hole_count:
        return None

    board_h_low = row_parts * 6
    board_h_high = min(18, board_h_low + 4)
    board_w_low = col_parts * 6
    board_w_high = min(22, board_w_low + 6)
    if board_h_high < board_h_low or board_w_high < board_w_low:
        return None
    board_h = int(rng.integers(board_h_low, board_h_high + 1))
    board_w = int(rng.integers(board_w_low, board_w_high + 1))

    row_sizes = _random_partition(rng, board_h, row_parts, 6)
    col_sizes = _random_partition(rng, board_w, col_parts, 6)

    board_palette_size = int(rng.integers(2, min(3, len(remaining)) + 1))
    board_palette = list(map(int, rng.choice(remaining, size=board_palette_size, replace=False)))
    block_colors = _assign_block_colors(rng, row_parts, col_parts, board_palette)

    board_solved = np.zeros((board_h, board_w), dtype=int)
    blocks: list[dict[str, int]] = []
    r0 = 0
    for br, bh in enumerate(row_sizes):
        c0 = 0
        for bc, bw in enumerate(col_sizes):
            color = int(block_colors[br][bc])
            board_solved[r0 : r0 + bh, c0 : c0 + bw] = color
            blocks.append({"r0": r0, "c0": c0, "h": bh, "w": bw, "color": color})
            c0 += bw
        r0 += bh

    chosen_block_ids = list(map(int, rng.choice(len(blocks), size=hole_count, replace=False)))
    used_signatures: set[tuple[tuple[int, int], ...]] = set()
    hole_specs: list[dict[str, object]] = []

    for hint_color, block_id in zip(hint_colors, chosen_block_ids):
        block = blocks[block_id]
        max_h = int(block["h"]) - 4
        max_w = int(block["w"]) - 4
        shape_sig = None
        for _ in range(120):
            candidate = _random_shape(rng, max_h, max_w)
            if candidate is None:
                continue
            canon = _canonical_signature(candidate)
            if canon in used_signatures:
                continue
            used_signatures.add(canon)
            shape_sig = candidate
            break
        if shape_sig is None:
            return None

        shape_mask = _signature_to_mask(shape_sig)
        sh, sw = map(int, shape_mask.shape)
        hole_r = int(rng.integers(2, int(block["h"]) - sh - 1))
        hole_c = int(rng.integers(2, int(block["w"]) - sw - 1))
        coords = np.array(
            [(int(block["r0"]) + hole_r + r, int(block["c0"]) + hole_c + c) for r, c in shape_sig],
            dtype=int,
        )
        hole_specs.append(
            {
                "hint_color": int(hint_color),
                "shape_sig": shape_sig,
                "coords": coords,
                "block_color": int(block["color"]),
            }
        )

    board_input = board_solved.copy()
    expected_board = board_solved.copy()
    for spec in hole_specs:
        coords = np.asarray(spec["coords"], dtype=int)
        hole_set = {(int(r), int(c)) for r, c in coords}
        ring: set[tuple[int, int]] = set()
        for r, c in hole_set:
            expected_board[r, c] = int(spec["hint_color"])
            board_input[r, c] = BLACK
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < board_h and 0 <= nc < board_w):
                    continue
                if (nr, nc) in hole_set:
                    continue
                ring.add((nr, nc))
        if not ring:
            return None
        ring_colors = {int(board_solved[r, c]) for r, c in ring}
        if ring_colors != {int(spec["block_color"])}:
            return None
        for r, c in ring:
            board_input[r, c] = int(spec["hint_color"])

    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    width = board_w + left_margin + right_margin
    sep = board_h + 2

    fragment_masks: list[np.ndarray] = []
    rotations = [int(rng.integers(4)) for _ in range(hole_count)]
    if all(rot == 0 for rot in rotations):
        rotations[int(rng.integers(hole_count))] = int(rng.integers(1, 4))

    max_fragment_h = 0
    for spec, rot in zip(hole_specs, rotations):
        mask = np.rot90(_signature_to_mask(spec["shape_sig"]), rot)
        fragment_masks.append(mask)
        max_fragment_h = max(max_fragment_h, int(mask.shape[0]))

    bottom_low = max(7, max_fragment_h + 3)
    bottom_high = min(9, 27 - board_h)
    if bottom_high < bottom_low:
        return None
    bottom_h = int(rng.integers(bottom_low, bottom_high + 1))

    grid = np.zeros((sep + 1 + bottom_h, width), dtype=int)
    grid[sep, :] = GREY
    grid[1 : 1 + board_h, left_margin : left_margin + board_w] = board_input

    fragment_specs = []
    for mask in fragment_masks:
        fragment_specs.append(
            {
                "mask": mask,
                "color": int(rng.choice(remaining)),
            }
        )
    if not _place_fragments(rng, grid, sep, fragment_specs):
        return None

    expected = grid.copy()
    expected[sep + 1 :] = BLACK
    expected[1 : 1 + board_h, left_margin : left_margin + board_w] = expected_board
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
