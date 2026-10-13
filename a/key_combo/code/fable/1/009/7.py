from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label as scipy_label


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    lab, n = scipy_label(mask.astype(bool), structure=STRUCT4)
    return [np.argwhere(lab == k) for k in range(1, int(n) + 1)]


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    return coords.size > 0 and len(_components(mask)) == 1


def _piece_from_sprite(sprite: np.ndarray) -> dict:
    sprite = np.asarray(sprite, dtype=int)
    colors = [int(v) for v in sprite[sprite != BG].flatten()]
    counts = Counter(colors)
    body = int(counts.most_common(1)[0][0])
    interfaces = {}
    for r, c in np.argwhere((sprite != BG) & (sprite != body)):
        interfaces[int(sprite[r, c])] = (int(r), int(c))
    mask = sprite != BG
    return {
        "sprite": sprite,
        "mask": mask,
        "body": body,
        "interfaces": interfaces,
        "area": int(mask.sum()),
    }


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    pieces = []
    for coords in _components(g != bg):
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        crop = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), bg, dtype=int)
        for r, c in coords:
            crop[int(r - r0), int(c - c0)] = int(g[int(r), int(c)])
        pieces.append(_piece_from_sprite(crop))
    return pieces


def _stable_keep_pieces(pieces: list[dict]) -> list[dict]:
    active = set(range(len(pieces)))
    while True:
        occurrences: dict[int, list[int]] = defaultdict(list)
        for i in active:
            for color in pieces[i]["interfaces"]:
                occurrences[int(color)].append(int(i))

        bad = set()
        for i in active:
            colors = list(pieces[i]["interfaces"])
            if not colors:
                bad.add(i)
                continue
            for color in colors:
                carriers = sorted(set(occurrences[int(color)]))
                if len(carriers) != 2:
                    bad.add(i)
                    break

        if not bad:
            break
        active -= bad
        if not active:
            break

    return [pieces[i] for i in sorted(active)]


def _piece_variants(piece: dict) -> list[dict]:
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(piece["sprite"], k=k)
        key = (rot.shape, rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        mask = rot != BG
        body = int(piece["body"])
        interfaces = {
            int(rot[r, c]): (int(r), int(c))
            for r, c in np.argwhere((rot != BG) & (rot != body))
        }
        variants.append(
            {
                "mask": mask,
                "coords": [tuple(map(int, rc)) for rc in np.argwhere(mask)],
                "body_sprite": np.where(mask, body, BG).astype(int),
                "interfaces": interfaces,
                "area": int(mask.sum()),
            }
        )
    return variants


def _build_links(pieces: list[dict]) -> tuple[list[list[tuple[int, int]]], dict[int, tuple[int, int]]]:
    occurrences: dict[int, list[int]] = defaultdict(list)
    for i, piece in enumerate(pieces):
        for color in piece["interfaces"]:
            occurrences[int(color)].append(int(i))

    links = [[] for _ in pieces]
    pairs = {}
    for color, carriers in occurrences.items():
        uniq = sorted(set(carriers))
        if len(uniq) == 2:
            a, b = uniq
            links[a].append((b, int(color)))
            links[b].append((a, int(color)))
            pairs[int(color)] = (int(a), int(b))
    return links, pairs


def _render_solution(
    pieces: list[dict],
    variants: list[list[dict]],
    anchor_idx: int,
    placed: dict[int, tuple[int, int, int]],
) -> np.ndarray | None:
    occupied: dict[tuple[int, int], int] = {}
    for piece_idx, (var_idx, r0, c0) in placed.items():
        for dr, dc in variants[piece_idx][var_idx]["coords"]:
            cell = (int(r0 + dr), int(c0 + dc))
            if cell in occupied:
                return None
            occupied[cell] = int(piece_idx)

    rows = [r for r, _c in occupied]
    cols = [c for _r, c in occupied]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    out_h = int(max_r - min_r + 1)
    out_w = int(max_c - min_c + 1)
    if out_h * out_w != len(occupied):
        return None
    if occupied.get((min_r, min_c)) != anchor_idx:
        return None

    out = np.zeros((out_h, out_w), dtype=int)
    for piece_idx, (var_idx, r0, c0) in placed.items():
        body_sprite = variants[piece_idx][var_idx]["body_sprite"]
        body = int(pieces[piece_idx]["body"])
        for dr, dc in variants[piece_idx][var_idx]["coords"]:
            rr = int(r0 + dr - min_r)
            cc = int(c0 + dc - min_c)
            out[rr, cc] = body

    return out


def _enumerate_solutions(pieces: list[dict], limit: int = 2) -> list[np.ndarray]:
    n = len(pieces)
    if n == 0:
        return []

    variants = [_piece_variants(piece) for piece in pieces]
    links, _pairs = _build_links(pieces)
    degrees = [len(piece["interfaces"]) for piece in pieces]
    max_deg = max(degrees)
    if degrees.count(max_deg) != 1:
        return []
    anchor_idx = int(degrees.index(max_deg))

    outputs: list[np.ndarray] = []
    seen = set()

    def recurse(placed: dict[int, tuple[int, int, int]], occupied: dict[tuple[int, int], int]) -> None:
        if len(outputs) >= limit:
            return
        if len(placed) == n:
            out = _render_solution(pieces, variants, anchor_idx, placed)
            if out is None:
                return
            sig = (tuple(out.shape), out.tobytes())
            if sig not in seen:
                seen.add(sig)
                outputs.append(out)
            return

        choice = None
        for piece_idx in range(n):
            if piece_idx in placed:
                continue
            cons = [(nbr, color) for nbr, color in links[piece_idx] if nbr in placed]
            if not cons:
                continue
            rank = (len(cons), -pieces[piece_idx]["area"])
            if choice is None or rank > choice[0]:
                choice = (rank, piece_idx, cons)
        if choice is None:
            return

        _rank, piece_idx, cons = choice
        for var_idx, variant in enumerate(variants[piece_idx]):
            candidate_positions = None
            for nbr, color in cons:
                nbr_var_idx, nbr_r, nbr_c = placed[nbr]
                nbr_variant = variants[nbr][nbr_var_idx]
                if color not in variant["interfaces"] or color not in nbr_variant["interfaces"]:
                    candidate_positions = set()
                    break
                ar, ac = nbr_variant["interfaces"][color]
                br, bc = variant["interfaces"][color]
                pos_set = {
                    (int(nbr_r + ar + dr - br), int(nbr_c + ac + dc - bc))
                    for dr, dc in DIR4
                }
                candidate_positions = pos_set if candidate_positions is None else candidate_positions & pos_set

            if not candidate_positions:
                continue

            for r0, c0 in sorted(candidate_positions):
                coords = [(int(r0 + dr), int(c0 + dc)) for dr, dc in variant["coords"]]
                if any(cell in occupied for cell in coords):
                    continue

                placed[piece_idx] = (int(var_idx), int(r0), int(c0))
                for cell in coords:
                    occupied[cell] = int(piece_idx)

                recurse(placed, occupied)

                for cell in coords:
                    del occupied[cell]
                del placed[piece_idx]

                if len(outputs) >= limit:
                    return

    for anchor_var_idx, anchor_variant in enumerate(variants[anchor_idx]):
        placed = {anchor_idx: (int(anchor_var_idx), 0, 0)}
        occupied = {tuple(map(int, rc)): int(anchor_idx) for rc in anchor_variant["coords"]}
        recurse(placed, occupied)
        if len(outputs) >= limit:
            break

    return outputs


def _pairwise_fit_count(piece_a: dict, piece_b: dict, color: int) -> int:
    variants_a = _piece_variants(piece_a)
    variants_b = _piece_variants(piece_b)
    seen = set()
    for va in variants_a:
        if color not in va["interfaces"]:
            continue
        ar, ac = va["interfaces"][color]
        cells_a = set(va["coords"])
        for vb in variants_b:
            if color not in vb["interfaces"]:
                continue
            br, bc = vb["interfaces"][color]
            for dr, dc in DIR4:
                shift_r = int(ar + dr - br)
                shift_c = int(ac + dc - bc)
                cells_b = {(int(r + shift_r), int(c + shift_c)) for r, c in vb["coords"]}
                if cells_a & cells_b:
                    continue
                all_cells = cells_a | cells_b
                min_r = min(r for r, _ in all_cells)
                min_c = min(c for _r, c in all_cells)
                norm_a = tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells_a))
                norm_b = tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells_b))
                seen.add((norm_a, norm_b))
    return len(seen)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the interface-marked fragments into a solid rectangle, melt the interfaces into body color, and crop the result.

    Input structure:
    - Several disconnected fragments sit on the background.
    - Each usable fragment is mostly one body color with one or more differently colored edge interface pixels.
    - Every usable interface color appears on exactly two different fragments, while bait fragments carry an interface color with no partner.
    - The only usable fragment with two paired interfaces is the corner anchor.

    Rule:
    - Discard every fragment that has an unpaired interface color.
    - Rotate and translate the remaining fragments so same-colored interface pixels become orthogonally adjacent and the union becomes a hole-free solid rectangle.
    - If the same finished rectangle exists under global rotation, keep the orientation where the unique two-interface anchor occupies the rectangle's upper-left corner.
    - Recolor each interface pixel to its fragment's body color and tightly crop the finished rectangle.

    Key insight:
    - A color match only tells which fragments touch; the correct rotations and offsets are fixed by the global solid-rectangle check, and the two-interface anchor fixes the output orientation.
    """
    pieces = _stable_keep_pieces(_extract_pieces(np.asarray(grid, dtype=int)))
    solutions = _enumerate_solutions(pieces, limit=2)
    if len(solutions) != 1:
        raise ValueError(f"expected exactly one assembly, found {len(solutions)}")
    return solutions[0]


def _anchor_mask(height: int, width: int, rng: random.Random) -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] | None:
    for _ in range(80):
        h_a = rng.randint(2, height - 2)
        widths = [rng.randint(max(2, width // 3), width - 2)]
        for _row in range(1, h_a):
            lo = max(1, widths[-1] - 2)
            hi = widths[-1]
            widths.append(rng.randint(lo, hi))

        area = sum(widths)
        if not (height * width * 0.2 <= area <= height * width * 0.55):
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, w in enumerate(widths):
            mask[r, :w] = True

        right_candidates = []
        for r, w in enumerate(widths):
            if w < width:
                right_candidates.append(((r, w - 1), (r, w)))
        bottom_candidates = [((h_a - 1, c), (h_a, c)) for c in range(widths[-1])]
        if not right_candidates or not bottom_candidates:
            continue

        rng.shuffle(right_candidates)
        rng.shuffle(bottom_candidates)
        for a_b in right_candidates:
            for a_c in bottom_candidates:
                if a_b[0] != a_c[0]:
                    return mask, a_b, a_c
    return None


def _grow_two_regions(
    free_mask: np.ndarray,
    seed_b: tuple[int, int],
    seed_c: tuple[int, int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    labels = np.full(free_mask.shape, -1, dtype=int)
    labels[~free_mask] = -2
    labels[seed_b] = 0
    labels[seed_c] = 1
    frontier = {0: set(), 1: set()}
    counts = [1, 1]

    def add_frontier(label: int, cell: tuple[int, int]) -> None:
        r, c = cell
        for dr, dc in DIR4:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < free_mask.shape[0] and 0 <= cc < free_mask.shape[1]:
                if free_mask[rr, cc] and labels[rr, cc] == -1:
                    frontier[label].add((rr, cc))

    add_frontier(0, seed_b)
    add_frontier(1, seed_c)

    while bool((labels == -1).any()):
        available = [label for label in (0, 1) if frontier[label]]
        if not available:
            return None
        if len(available) == 1:
            label = available[0]
        elif counts[0] + 1 < counts[1]:
            label = 0
        elif counts[1] + 1 < counts[0]:
            label = 1
        else:
            label = int(rng.choice(available))

        cell = rng.choice(tuple(frontier[label]))
        frontier[0].discard(cell)
        frontier[1].discard(cell)
        labels[cell] = label
        counts[label] += 1
        add_frontier(label, cell)

    mask_b = labels == 0
    mask_c = labels == 1
    if counts[0] < 3 or counts[1] < 3:
        return None
    if not _mask_connected(mask_b) or not _mask_connected(mask_c):
        return None
    return mask_b, mask_c


def _crop_mask_with_colors(color_grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    sub_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
    out[sub_mask] = color_grid[r0 : r1 + 1, c0 : c1 + 1][sub_mask]
    return out


def _decoy_sprite(rng: random.Random, body: int, interface: int) -> np.ndarray:
    shapes = [
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    ]
    mask = shapes[rng.randrange(len(shapes))]
    sprite = np.where(mask, body, BG).astype(int)
    boundary = []
    for r, c in np.argwhere(mask):
        for dr, dc in DIR4:
            rr = int(r + dr)
            cc = int(c + dc)
            if rr < 0 or cc < 0 or rr >= mask.shape[0] or cc >= mask.shape[1] or not mask[rr, cc]:
                boundary.append((int(r), int(c)))
                break
    ir, ic = rng.choice(boundary)
    sprite[ir, ic] = int(interface)
    return sprite


def _place_sprites(sprites: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    max_h = max(sprite.shape[0] for sprite in sprites)
    max_w = max(sprite.shape[1] for sprite in sprites)
    min_h = max(12, max_h + 2)
    min_w = max(12, max_w + 2)

    for _ in range(80):
        height = rng.randint(min_h, 30)
        width = rng.randint(min_w, 30)
        grid = np.zeros((height, width), dtype=int)
        blocked = np.zeros((height, width), dtype=bool)

        order = list(range(len(sprites)))
        rng.shuffle(order)
        ok = True

        for idx in order:
            sprite = sprites[idx]
            sh, sw = sprite.shape
            placed = False
            for _ in range(200):
                r0 = rng.randint(0, height - sh)
                c0 = rng.randint(0, width - sw)
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(height, r0 + sh + 1)
                cc1 = min(width, c0 + sw + 1)
                if blocked[rr0:rr1, cc0:cc1].any():
                    continue
                mask = sprite != BG
                sub = grid[r0 : r0 + sh, c0 : c0 + sw]
                sub[mask] = sprite[mask]
                blocked[rr0:rr1, cc0:cc1] = True
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return np.rot90(grid, k=rng.randrange(4))
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(6, 10)
    width = rng.randint(7, 12)
    anchor_info = _anchor_mask(height, width, rng)
    if anchor_info is None:
        return None
    mask_a, pair_ab, pair_ac = anchor_info

    free_mask = ~mask_a
    seed_b = pair_ab[1]
    seed_c = pair_ac[1]
    bc_masks = _grow_two_regions(free_mask, seed_b, seed_c, rng)
    if bc_masks is None:
        return None
    mask_b, mask_c = bc_masks

    if not (_mask_connected(mask_a) and _mask_connected(mask_b) and _mask_connected(mask_c)):
        return None

    palette = list(range(1, 10))
    rng.shuffle(palette)
    use_decoy = rng.random() < 0.7
    if use_decoy:
        if len(palette) < 7:
            return None
        body_a, body_b, body_c, color_ab, color_ac, dec_body, dec_iface = palette[:7]
    else:
        body_a, body_b, body_c, color_ab, color_ac = palette[:5]
        dec_body = dec_iface = None

    owner = np.full((height, width), -1, dtype=int)
    owner[mask_a] = 0
    owner[mask_b] = 1
    owner[mask_c] = 2

    color_grid = np.zeros((height, width), dtype=int)
    color_grid[mask_a] = int(body_a)
    color_grid[mask_b] = int(body_b)
    color_grid[mask_c] = int(body_c)

    (a1r, a1c), (b1r, b1c) = pair_ab
    (a2r, a2c), (c1r, c1c) = pair_ac
    color_grid[a1r, a1c] = int(color_ab)
    color_grid[b1r, b1c] = int(color_ab)
    color_grid[a2r, a2c] = int(color_ac)
    color_grid[c1r, c1c] = int(color_ac)

    sprites = [
        _crop_mask_with_colors(color_grid, owner == 0),
        _crop_mask_with_colors(color_grid, owner == 1),
        _crop_mask_with_colors(color_grid, owner == 2),
    ]

    if use_decoy:
        sprites.append(_decoy_sprite(rng, int(dec_body), int(dec_iface)))

    sprites = [np.rot90(sprite, k=rng.randrange(4)) for sprite in sprites]
    grid = _place_sprites(sprites, rng)
    if grid is None:
        return None

    pieces = _stable_keep_pieces(_extract_pieces(grid))
    if len(pieces) != 3:
        return None

    degrees = sorted(len(piece["interfaces"]) for piece in pieces)
    if degrees != [1, 1, 2]:
        return None

    links, pairs = _build_links(pieces)
    if len(pairs) != 2:
        return None

    solutions = _enumerate_solutions(pieces, limit=3)
    if len(solutions) != 1:
        return None

    anchor_idx = max(range(len(pieces)), key=lambda i: len(pieces[i]["interfaces"]))
    ambiguous = False
    for nbr, color in links[anchor_idx]:
        if _pairwise_fit_count(pieces[anchor_idx], pieces[nbr], color) > 1:
            ambiguous = True
            break
    if not ambiguous:
        return None

    out = solutions[0]
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid interface-assembly puzzle")
