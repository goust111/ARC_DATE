from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 9)

TARGET_H_RANGE = (5, 8)
TARGET_W_RANGE = (5, 9)
PIECE_COUNT_RANGE = (3, 5)
MIN_PIECE_AREA = 4
GENERATION_RETRIES = 500
CANVAS_MIN = 20
CANVAS_MAX = 30


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _components(mask.astype(bool))
    return len(comps) == 1


def _mask_has_hole(mask: np.ndarray) -> bool:
    inv = ~mask.astype(bool)
    if not inv.any():
        return False
    h, w = inv.shape
    seen = np.zeros_like(inv, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if inv[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if inv[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and inv[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))
    return bool(np.any(inv & ~seen))


def _mask_is_rectangle(mask: np.ndarray) -> bool:
    return bool(mask.size) and bool(mask.all())


def _build_piece(mask: np.ndarray, color: int) -> dict:
    coords = np.argwhere(mask)
    rel = [(int(r), int(c)) for r, c in coords]
    return {
        "color": int(color),
        "mask": mask.astype(bool),
        "coords": rel,
        "area": int(mask.sum()),
    }


def _normalize_pieces(pieces: list[dict]) -> list[dict]:
    return sorted(
        pieces,
        key=lambda p: (-int(p["area"]), -int(p["mask"].shape[0]), -int(p["mask"].shape[1]), int(p["color"])),
    )


def _extract_target_shape(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    azure_comps = _components(g == AZURE)
    candidates: list[tuple[int, int]] = []
    for comp in azure_comps:
        coords = np.array(comp, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if not np.all(g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == AZURE):
            continue
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        er0, ec0, er1, ec1 = int(r0) - 1, int(c0) - 1, int(r1) + 1, int(c1) + 1
        block = g[er0 : er1 + 1, ec0 : ec1 + 1]
        ring = np.zeros_like(block, dtype=bool)
        ring[0, :] = True
        ring[-1, :] = True
        ring[:, 0] = True
        ring[:, -1] = True
        if not np.all(block[ring] == GREY):
            continue
        if not np.all(np.isin(block, (GREY, AZURE))):
            continue
        candidates.append((int(r1 - r0 + 1), int(c1 - c0 + 1)))
    assert len(candidates) == 1, "expected a unique grey-framed azure size legend"
    return candidates[0]


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    pieces: list[dict] = []
    blocked = {BLACK, GREY, AZURE}
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in blocked or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            for pr, pc in coords:
                mask[int(pr - r0), int(pc - c0)] = True
            pieces.append(_build_piece(mask, color))
    assert pieces, "expected at least one colored fragment"
    return _normalize_pieces(pieces)


def _prepare_placements(pieces: list[dict], target_shape: tuple[int, int]):
    h, w = target_shape
    if sum(int(piece["area"]) for piece in pieces) != h * w:
        return None

    placements: list[tuple[int, int, int, int]] = []
    covers: list[list[int]] = [[] for _ in range(h * w)]
    for pi, piece in enumerate(pieces):
        ph, pw = piece["mask"].shape
        if ph > h or pw > w:
            return None
        rel = piece["coords"]
        for r0 in range(h - ph + 1):
            for c0 in range(w - pw + 1):
                bitmask = 0
                for dr, dc in rel:
                    bitmask |= 1 << ((r0 + dr) * w + (c0 + dc))
                pid = len(placements)
                placements.append((pi, int(r0), int(c0), bitmask))
                for dr, dc in rel:
                    covers[(r0 + dr) * w + (c0 + dc)].append(pid)
    return placements, covers


def _solve_once(pieces: list[dict], target_shape: tuple[int, int]) -> dict[int, tuple[int, int]] | None:
    prepared = _prepare_placements(pieces, target_shape)
    if prepared is None:
        return None
    h, w = target_shape
    placements, covers = prepared
    n = len(pieces)
    full_cells = (1 << (h * w)) - 1
    full_pieces = (1 << n) - 1
    chosen: dict[int, tuple[int, int]] = {}

    def dfs(used: int, occupied: int) -> bool:
        if used == full_pieces:
            return occupied == full_cells

        best: list[int] | None = None
        for cell in range(h * w):
            if (occupied >> cell) & 1:
                continue
            valid: list[int] = []
            for pid in covers[cell]:
                piece_idx, r0, c0, pmask = placements[pid]
                if (used >> piece_idx) & 1:
                    continue
                if pmask & occupied:
                    continue
                valid.append(pid)
            if not valid:
                return False
            if best is None or len(valid) < len(best):
                best = valid
                if len(best) == 1:
                    break

        assert best is not None
        for pid in best:
            piece_idx, r0, c0, pmask = placements[pid]
            chosen[piece_idx] = (int(r0), int(c0))
            if dfs(used | (1 << piece_idx), occupied | pmask):
                return True
            del chosen[piece_idx]
        return False

    return chosen.copy() if dfs(0, 0) else None


def _count_solutions(pieces: list[dict], target_shape: tuple[int, int], limit: int = 2) -> int:
    prepared = _prepare_placements(pieces, target_shape)
    if prepared is None:
        return 0
    h, w = target_shape
    placements, covers = prepared
    n = len(pieces)
    full_cells = (1 << (h * w)) - 1
    full_pieces = (1 << n) - 1
    memo: dict[tuple[int, int], int] = {}

    def dfs(used: int, occupied: int) -> int:
        key = (used, occupied)
        if key in memo:
            return memo[key]
        if used == full_pieces:
            return 1 if occupied == full_cells else 0

        best: list[int] | None = None
        for cell in range(h * w):
            if (occupied >> cell) & 1:
                continue
            valid: list[int] = []
            for pid in covers[cell]:
                piece_idx, _r0, _c0, pmask = placements[pid]
                if (used >> piece_idx) & 1:
                    continue
                if pmask & occupied:
                    continue
                valid.append(pid)
            if not valid:
                memo[key] = 0
                return 0
            if best is None or len(valid) < len(best):
                best = valid
                if len(best) == 1:
                    break

        total = 0
        assert best is not None
        for pid in best:
            piece_idx, _r0, _c0, pmask = placements[pid]
            total += dfs(used | (1 << piece_idx), occupied | pmask)
            if total >= limit:
                total = limit
                break
        memo[key] = total
        return total

    return dfs(0, 0)


def _render_solution(pieces: list[dict], target_shape: tuple[int, int], placements: dict[int, tuple[int, int]]) -> np.ndarray:
    h, w = target_shape
    out = np.full((h, w), BLACK, dtype=int)
    for pi, piece in enumerate(pieces):
        r0, c0 = placements[pi]
        for dr, dc in piece["coords"]:
            out[r0 + dr, c0 + dc] = int(piece["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered monochrome jigsaw fragments into the compact rectangle specified by the grey-framed azure legend.

    Input structure:
    - The background is black.
    - A small grey rectangular frame in one corner encloses a solid azure rectangle whose size equals the final puzzle size.
    - All other colored 4-connected components are fragments that may only be translated.

    Rule:
    - Read the target rectangle size from the azure legend.
    - Translate every fragment, without rotation or reflection, so the fragments exactly tile that rectangle.
    - Return only the assembled compact rectangle.
    """
    g = np.asarray(grid, dtype=int)
    target_shape = _extract_target_shape(g)
    pieces = _extract_pieces(g)
    placement = _solve_once(pieces, target_shape)
    assert placement is not None, "fragments must tile the hinted rectangle"
    return _render_solution(pieces, target_shape, placement)


def _piece_mask_from_labels(labels: np.ndarray, pid: int) -> np.ndarray:
    return _crop_mask(labels == pid)


def _piece_valid(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    if int(mask.sum()) < MIN_PIECE_AREA:
        return False
    if mask.shape[0] < 2 or mask.shape[1] < 2:
        return False
    if not _mask_connected(mask):
        return False
    if _mask_has_hole(mask):
        return False
    return True


def _labels_valid(labels: np.ndarray) -> bool:
    n = int(labels.max()) + 1
    masks = [_piece_mask_from_labels(labels, pid) for pid in range(n)]
    if not all(_piece_valid(mask) for mask in masks):
        return False
    non_rect = sum(not _mask_is_rectangle(mask) for mask in masks)
    need_non_rect = 1 if n <= 3 else 2
    return non_rect >= need_non_rect


def _split_rectangles(h: int, w: int, n_pieces: int) -> np.ndarray | None:
    rects = [(0, 0, h, w)]
    while len(rects) < n_pieces:
        candidates: list[tuple[int, list[str]]] = []
        for idx, (r0, c0, r1, c1) in enumerate(rects):
            opts: list[str] = []
            if r1 - r0 >= 4:
                opts.append("h")
            if c1 - c0 >= 4:
                opts.append("v")
            if opts:
                candidates.append((idx, opts))
        if not candidates:
            return None
        idx, opts = random.choice(candidates)
        r0, c0, r1, c1 = rects.pop(idx)
        orient = random.choice(opts)
        if orient == "h":
            cut = random.randint(r0 + 2, r1 - 2)
            rects.append((r0, c0, cut, c1))
            rects.append((cut, c0, r1, c1))
        else:
            cut = random.randint(c0 + 2, c1 - 2)
            rects.append((r0, c0, r1, cut))
            rects.append((r0, cut, r1, c1))

    labels = np.full((h, w), -1, dtype=int)
    for pid, (r0, c0, r1, c1) in enumerate(rects):
        labels[r0:r1, c0:c1] = pid
    return labels


def _boundary_segments(labels: np.ndarray) -> list[dict]:
    h, w = labels.shape
    segs: list[dict] = []
    for c in range(w - 1):
        r = 0
        while r < h:
            left = int(labels[r, c])
            right = int(labels[r, c + 1])
            if left == right:
                r += 1
                continue
            r0 = r
            while (
                r + 1 < h
                and int(labels[r + 1, c]) == left
                and int(labels[r + 1, c + 1]) == right
                and left != right
            ):
                r += 1
            segs.append({"kind": "v", "c": int(c), "r0": int(r0), "r1": int(r), "a": left, "b": right})
            r += 1
    for r in range(h - 1):
        c = 0
        while c < w:
            top = int(labels[r, c])
            bottom = int(labels[r + 1, c])
            if top == bottom:
                c += 1
                continue
            c0 = c
            while (
                c + 1 < w
                and int(labels[r, c + 1]) == top
                and int(labels[r + 1, c + 1]) == bottom
                and top != bottom
            ):
                c += 1
            segs.append({"kind": "h", "r": int(r), "c0": int(c0), "c1": int(c), "a": top, "b": bottom})
            c += 1
    return segs


def _transfer_cells(labels: np.ndarray, seg: dict) -> bool:
    trial = labels.copy()
    if seg["kind"] == "v":
        length = int(seg["r1"] - seg["r0"] + 1)
        run = random.randint(1, min(3, length))
        start = random.randint(int(seg["r0"]), int(seg["r1"]) - run + 1)
        donor = random.choice(("left", "right"))
        if donor == "left":
            donor_id = int(seg["a"])
            recv_id = int(seg["b"])
            cells = [(r, int(seg["c"])) for r in range(start, start + run)]
        else:
            donor_id = int(seg["b"])
            recv_id = int(seg["a"])
            cells = [(r, int(seg["c"]) + 1) for r in range(start, start + run)]
    else:
        length = int(seg["c1"] - seg["c0"] + 1)
        run = random.randint(1, min(3, length))
        start = random.randint(int(seg["c0"]), int(seg["c1"]) - run + 1)
        donor = random.choice(("top", "bottom"))
        if donor == "top":
            donor_id = int(seg["a"])
            recv_id = int(seg["b"])
            cells = [(int(seg["r"]), c) for c in range(start, start + run)]
        else:
            donor_id = int(seg["b"])
            recv_id = int(seg["a"])
            cells = [(int(seg["r"]) + 1, c) for c in range(start, start + run)]

    for r, c in cells:
        if int(trial[r, c]) != donor_id:
            return False
        trial[r, c] = recv_id

    donor_mask = _piece_mask_from_labels(trial, donor_id)
    recv_mask = _piece_mask_from_labels(trial, recv_id)
    if not _piece_valid(donor_mask) or not _piece_valid(recv_mask):
        return False

    labels[:, :] = trial
    return True


def _make_partition(h: int, w: int, n_pieces: int) -> np.ndarray | None:
    labels = _split_rectangles(h, w, n_pieces)
    if labels is None:
        return None

    desired_ops = random.randint(max(1, n_pieces - 2), n_pieces + 1)
    successes = 0
    for _ in range(60):
        segs = _boundary_segments(labels)
        if not segs:
            break
        random.shuffle(segs)
        changed = False
        for seg in segs:
            if _transfer_cells(labels, seg):
                successes += 1
                changed = True
                break
        if successes >= desired_ops:
            break
        if not changed:
            break

    if successes == 0 or not _labels_valid(labels):
        return None
    return labels


def _labels_to_output(labels: np.ndarray, colors: list[int]) -> np.ndarray:
    out = np.full(labels.shape, BLACK, dtype=int)
    for pid, color in enumerate(colors):
        out[labels == pid] = int(color)
    return out


def _legend_patch(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    patch = np.full((h + 2, w + 2), GREY, dtype=int)
    patch[1 : h + 1, 1 : w + 1] = AZURE
    return patch


def _piece_patches(labels: np.ndarray, colors: list[int]) -> list[np.ndarray]:
    patches: list[np.ndarray] = []
    for pid, color in enumerate(colors):
        mask = _piece_mask_from_labels(labels, pid)
        patch = np.full(mask.shape, BLACK, dtype=int)
        patch[mask] = int(color)
        patches.append(patch)
    return patches


def _corner_origin(corner: str, gh: int, gw: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, gw - pw
    if corner == "BL":
        return gh - ph, 0
    return gh - ph, gw - pw


def _can_place_bbox(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    bh, bw = blocked.shape
    if r0 < 1 or c0 < 1 or r0 + h >= bh - 1 or c0 + w >= bw - 1:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(bh, r0 + h + 1)
    cc1 = min(bw, c0 + w + 1)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_bbox(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    bh, bw = blocked.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(bh, r0 + h + 1)
    cc1 = min(bw, c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BLACK
    block = grid[r0 : r0 + h, c0 : c0 + w]
    block[mask] = patch[mask]


def _scatter_input(patches: list[np.ndarray], target_shape: tuple[int, int]) -> np.ndarray | None:
    legend = _legend_patch(target_shape)
    total_bbox_area = int(sum(p.shape[0] * p.shape[1] for p in patches) + legend.shape[0] * legend.shape[1])
    min_side = max(
        CANVAS_MIN,
        legend.shape[0] + 2,
        legend.shape[1] + 2,
        int(max(p.shape[0] for p in patches)) + 4,
        int(max(p.shape[1] for p in patches)) + 4,
    )
    for _ in range(40):
        gh = random.randint(min_side, CANVAS_MAX)
        gw = random.randint(min_side, CANVAS_MAX)
        if gh * gw < total_bbox_area * 2:
            continue
        grid = np.full((gh, gw), BLACK, dtype=int)
        blocked = np.zeros((gh, gw), dtype=bool)

        corner = random.choice(("TL", "TR", "BL", "BR"))
        lr, lc = _corner_origin(corner, gh, gw, legend.shape[0], legend.shape[1])
        _paste(grid, legend, lr, lc)
        _mark_bbox(blocked, lr, lc, legend.shape[0], legend.shape[1])

        ok = True
        for patch in sorted(patches, key=lambda p: (-p.shape[0] * p.shape[1], -p.shape[0], -p.shape[1])):
            ph, pw = patch.shape
            candidates: list[tuple[int, int]] = []
            for r0 in range(1, gh - ph - 1):
                for c0 in range(1, gw - pw - 1):
                    if _can_place_bbox(blocked, r0, c0, ph, pw):
                        candidates.append((int(r0), int(c0)))
            if not candidates:
                ok = False
                break
            pr, pc = random.choice(candidates)
            _paste(grid, patch, pr, pc)
            _mark_bbox(blocked, pr, pc, ph, pw)
        if ok:
            return grid
    return None


def _attempt_case() -> tuple[np.ndarray, np.ndarray] | None:
    target_h = random.randint(*TARGET_H_RANGE)
    target_w = random.randint(*TARGET_W_RANGE)
    max_pieces = min(PIECE_COUNT_RANGE[1], len(PIECE_COLORS), (target_h * target_w) // MIN_PIECE_AREA)
    if max_pieces < PIECE_COUNT_RANGE[0]:
        return None
    n_pieces = random.randint(PIECE_COUNT_RANGE[0], max_pieces)

    labels = _make_partition(target_h, target_w, n_pieces)
    if labels is None:
        return None

    colors = random.sample(list(PIECE_COLORS), n_pieces)
    out = _labels_to_output(labels, colors)
    piece_masks = [_piece_mask_from_labels(labels, pid) for pid in range(n_pieces)]
    pieces = _normalize_pieces([_build_piece(mask, color) for mask, color in zip(piece_masks, colors)])
    if _count_solutions(pieces, (target_h, target_w), limit=2) != 1:
        return None

    patches = _piece_patches(labels, colors)
    inp = _scatter_input(patches, (target_h, target_w))
    if inp is None:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        candidate = _attempt_case()
        if candidate is None:
            continue
        inp, out = candidate
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError("failed to generate a valid jigsaw-assembly input")
