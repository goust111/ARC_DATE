import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
NON_BG_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bool(mask: np.ndarray) -> np.ndarray:
    b = bbox(mask)
    if b is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = b
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(grid: np.ndarray, bg: int):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return components


def mask_components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def enclosed_background(object_mask: np.ndarray) -> np.ndarray:
    h, w = object_mask.shape
    bg_mask = ~object_mask
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        rr, cc = q.popleft()
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w and bg_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return bg_mask & ~outside


def rotation_keys(mask: np.ndarray):
    keys = []
    for k in range(4):
        rot = np.rot90(mask, k)
        keys.append((rot.shape, tuple(rot.astype(np.uint8).ravel())))
    return keys


def canonical_shape(mask: np.ndarray):
    return min(rotation_keys(mask))


def distinct_rotations(mask: np.ndarray) -> int:
    return len(set(rotation_keys(mask)))


def colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def random_silhouette(min_side: int = 2, max_side: int = 4, max_dim: int = 8) -> np.ndarray:
    for _ in range(200):
        base_h = random.randint(min_side, max_side)
        base_w = random.randint(min_side, max_side)
        canvas = np.zeros((base_h + 8, base_w + 8), dtype=bool)
        top = left = 4
        canvas[top : top + base_h, left : left + base_w] = True

        for _ in range(random.randint(1, 3)):
            side = random.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                span = random.randint(1, base_w)
                start = left + random.randint(0, base_w - span)
                depth = random.randint(1, 2)
                if side == "top":
                    canvas[top - depth : top, start : start + span] = True
                else:
                    canvas[top + base_h : top + base_h + depth, start : start + span] = True
            else:
                span = random.randint(1, base_h)
                start = top + random.randint(0, base_h - span)
                depth = random.randint(1, 2)
                if side == "left":
                    canvas[start : start + span, left - depth : left] = True
                else:
                    canvas[start : start + span, left + base_w : left + base_w + depth] = True

        mask = crop_bool(canvas)
        if max(mask.shape) > max_dim:
            continue
        if distinct_rotations(mask) != 4:
            continue
        return np.rot90(mask, random.randint(0, 3))
    raise RuntimeError("failed to generate asymmetric silhouette")


def make_parent_piece(child_silhouette: np.ndarray, max_dim: int = 14):
    for _ in range(250):
        rot_k = random.randint(0, 3)
        hole = np.rot90(child_silhouette, rot_k)
        hole_h, hole_w = hole.shape

        margin_top = random.randint(1, 2)
        margin_bottom = random.randint(1, 2)
        margin_left = random.randint(1, 2)
        margin_right = random.randint(1, 2)

        base_h = hole_h + margin_top + margin_bottom
        base_w = hole_w + margin_left + margin_right
        canvas = np.zeros((base_h + 10, base_w + 10), dtype=bool)
        base_r = base_c = 5
        canvas[base_r : base_r + base_h, base_c : base_c + base_w] = True

        for _ in range(random.randint(1, 3)):
            side = random.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                span = random.randint(1, base_w)
                start = base_c + random.randint(0, base_w - span)
                depth = random.randint(1, 2)
                if side == "top":
                    canvas[base_r - depth : base_r, start : start + span] = True
                else:
                    canvas[base_r + base_h : base_r + base_h + depth, start : start + span] = True
            else:
                span = random.randint(1, base_h)
                start = base_r + random.randint(0, base_h - span)
                depth = random.randint(1, 2)
                if side == "left":
                    canvas[start : start + span, base_c - depth : base_c] = True
                else:
                    canvas[start : start + span, base_c + base_w : base_c + base_w + depth] = True

        sil_box = bbox(canvas)
        if sil_box is None:
            continue
        r0, c0, r1, c1 = sil_box
        silhouette = canvas[r0 : r1 + 1, c0 : c1 + 1].copy()
        if max(silhouette.shape) > max_dim:
            continue
        if distinct_rotations(silhouette) != 4:
            continue

        hole_r = base_r + margin_top - r0
        hole_c = base_c + margin_left - c0
        occupied = silhouette.copy()
        sub = occupied[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w]
        sub[hole] = False
        occupied[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = sub

        if len(mask_components(occupied)) != 1:
            continue

        hole_mask = enclosed_background(occupied)
        hole_parts = mask_components(hole_mask)
        if len(hole_parts) != 1:
            continue
        hole_crop = crop_bool(hole_mask)
        if not np.array_equal(hole_crop, hole):
            continue
        return occupied, silhouette
    raise RuntimeError("failed to wrap child silhouette")


def intersects(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def scatter_patches(patches: list[np.ndarray]) -> np.ndarray:
    order = sorted(range(len(patches)), key=lambda i: -(patches[i] != 0).sum())
    max_h = max(p.shape[0] for p in patches)
    max_w = max(p.shape[1] for p in patches)

    for padding in (2, 1):
        for _ in range(300):
            height = random.randint(max(max_h + 2, 18), 30)
            width = random.randint(max(max_w + 2, 18), 30)
            grid = np.zeros((height, width), dtype=int)
            boxes = []
            ok = True
            for idx in order:
                patch = patches[idx]
                ph, pw = patch.shape
                spots = []
                row_min = 1
                row_max = height - ph - 1
                col_min = 1
                col_max = width - pw - 1
                if row_max < row_min or col_max < col_min:
                    ok = False
                    break
                for _ in range(250):
                    r = random.randint(row_min, row_max)
                    c = random.randint(col_min, col_max)
                    cand = (r - padding, c - padding, r + ph - 1 + padding, c + pw - 1 + padding)
                    if cand[0] < 0 or cand[1] < 0 or cand[2] >= height or cand[3] >= width:
                        continue
                    if any(intersects(cand, prev) for prev in boxes):
                        continue
                    spots.append((r, c))
                    if len(spots) >= 8:
                        break
                if not spots:
                    ok = False
                    break
                r, c = random.choice(spots)
                sub = grid[r : r + ph, c : c + pw]
                mask = patch != 0
                sub[mask] = patch[mask]
                grid[r : r + ph, c : c + pw] = sub
                boxes.append(cand)
            if ok:
                return grid
    raise RuntimeError("failed to scatter patches")


def build_chain():
    n_pieces = random.randint(2, 4)
    silhouettes = []
    occupied_masks = []
    signatures = set()

    inner = random_silhouette(min_side=2, max_side=4, max_dim=7)
    silhouettes.append(inner)
    occupied_masks.append(inner.copy())
    signatures.add(canonical_shape(inner))

    for _ in range(n_pieces - 1):
        for _ in range(120):
            parent_occ, parent_sil = make_parent_piece(silhouettes[-1], max_dim=14)
            sig = canonical_shape(parent_sil)
            if sig in signatures:
                continue
            silhouettes.append(parent_sil)
            occupied_masks.append(parent_occ)
            signatures.add(sig)
            break
        else:
            raise RuntimeError("failed to extend nesting chain")

    return silhouettes, occupied_masks, signatures


def maybe_make_distractor(forbidden_signatures):
    if random.random() < 0.45:
        return None
    for _ in range(120):
        mask = random_silhouette(min_side=2, max_side=5, max_dim=8)
        sig = canonical_shape(mask)
        if sig not in forbidden_signatures:
            return mask.copy()
    raise RuntimeError("failed to generate distractor")


def main(grid: np.ndarray) -> np.ndarray:
    """Nest the scattered pieces by matching each hole with the rotated outline of the next inner piece.

    Input structure:
    - The grid has a black background and several separated colored pieces.
    - Hollow pieces contain exactly one enclosed black hole; one solid piece is the innermost filler.
    - A solid distractor may also appear, but it matches no hole.

    Rule:
    - Extract each piece as a connected monochromatic component.
    - For every hollow piece, compute its enclosed hole shape and match it against another piece's outer silhouette under 0/90/180/270 degree rotation.
    - Start from the unique hollow piece whose silhouette fits no other hole, place the matched child into its hole with the required rotation, and repeat until the solid innermost piece is inserted.
    - Ignore any unmatched solid distractor and return the tight crop of the completed assembly.

    Key insight:
    - The assembly chain is determined by negative space: each hole encodes the silhouette of the next piece, while the outermost hollow piece is the only hollow component with no incoming match.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(g, bg=bg)
    assert components, "expected at least one foreground piece"

    pieces = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        color = int(comp["color"])
        object_mask = patch == color
        hole_mask = enclosed_background(object_mask)
        filled_mask = object_mask | hole_mask
        piece = {
            "patch": patch,
            "silhouette": crop_bool(filled_mask),
            "has_hole": bool(hole_mask.any()),
        }
        if piece["has_hole"]:
            hb = bbox(hole_mask)
            assert hb is not None
            hr0, hc0, hr1, hc1 = hb
            piece["hole_bbox"] = (hr0, hc0, hr1, hc1)
            piece["hole_shape"] = hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1].copy()
        pieces.append(piece)

    n = len(pieces)
    incoming = [None] * n
    outgoing = [None] * n

    for parent_idx, piece in enumerate(pieces):
        if not piece["has_hole"]:
            continue
        matches = []
        for child_idx, child in enumerate(pieces):
            if child_idx == parent_idx:
                continue
            for rot_k in range(4):
                if np.array_equal(np.rot90(child["silhouette"], rot_k), piece["hole_shape"]):
                    matches.append((child_idx, rot_k))
        assert len(matches) == 1, "each hole must match exactly one other piece silhouette"
        child_idx, rot_k = matches[0]
        assert incoming[child_idx] is None, "each piece may fit at most one parent hole"
        incoming[child_idx] = parent_idx
        outgoing[parent_idx] = (child_idx, rot_k)

    roots = [idx for idx, piece in enumerate(pieces) if piece["has_hole"] and incoming[idx] is None]
    assert len(roots) == 1, "expected exactly one outermost hollow piece"
    root = roots[0]

    memo = {}

    def assemble(piece_idx: int) -> np.ndarray:
        if piece_idx in memo:
            return memo[piece_idx].copy()

        current = pieces[piece_idx]["patch"].copy()
        link = outgoing[piece_idx]
        if link is not None:
            child_idx, rot_k = link
            child_patch = np.rot90(assemble(child_idx), rot_k)
            hr0, hc0, hr1, hc1 = pieces[piece_idx]["hole_bbox"]
            assert child_patch.shape == (hr1 - hr0 + 1, hc1 - hc0 + 1)
            sub = current[hr0 : hr1 + 1, hc0 : hc1 + 1]
            child_mask = child_patch != bg
            sub[child_mask] = child_patch[child_mask]
            current[hr0 : hr1 + 1, hc0 : hc1 + 1] = sub

        memo[piece_idx] = current
        return current.copy()

    assembled = assemble(root)
    fg_box = bbox(assembled != bg)
    assert fg_box is not None
    r0, c0, r1, c1 = fg_box
    return assembled[r0 : r1 + 1, c0 : c1 + 1]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            silhouettes, occupied_masks, signatures = build_chain()
            distractor = maybe_make_distractor(signatures)

            piece_masks = []
            colors = random.sample(NON_BG_COLORS, len(occupied_masks) + (1 if distractor is not None else 0))
            for mask, color in zip(occupied_masks, colors):
                piece_masks.append(colorize(np.rot90(mask, random.randint(0, 3)), color))

            if distractor is not None:
                piece_masks.append(colorize(np.rot90(distractor, random.randint(0, 3)), colors[-1]))

            random.shuffle(piece_masks)
            grid = scatter_patches(piece_masks)

            out = main(grid)
            if max(grid.shape) <= 30 and max(out.shape) <= 30:
                return grid
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("failed to generate valid negative-space nesting input")
