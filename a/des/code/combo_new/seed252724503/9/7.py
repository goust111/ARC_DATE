from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
FG = 8
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _pad_square(mask: np.ndarray, side: int) -> np.ndarray:
    out = np.zeros((side, side), dtype=bool)
    r0 = (side - mask.shape[0]) // 2
    c0 = (side - mask.shape[1]) // 2
    out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]] = mask
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align all rotated copies of a motif, find the unique modified copy, and mark its disagreements with the consensus in red.

    Input structure:
    - The background is black.
    - Several disconnected copies of one monochrome motif appear on the grid, each inside its own same-size bounding box.
    - The copies are rotations of one base motif, except that exactly one copy has a few added or removed pixels.

    Rule:
    - Extract every copy and compare them only after trying all four rotations to align them to a common orientation.
    - Build the majority-vote consensus mask in that common orientation.
    - Identify the unique copy whose aligned mask differs from the consensus the most.
    - Return only that copy's bounding box, keep its original foreground pixels in the original color, and recolor every disagreement position with red.

    Key insight:
    - The anomaly is invisible until rotation equivalence is normalized away; only after orientation alignment can the single tampered copy be isolated.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _components(g == FG)
    raw_masks = [_crop_mask(comp) for comp in comps]
    side = max(max(mask.shape) for mask in raw_masks)
    masks = [_pad_square(mask, side) for mask in raw_masks]

    best = None
    for ref_idx, ref in enumerate(masks):
        aligned = []
        rotations = []
        for mask in masks:
            opts = [np.rot90(mask, k) for k in range(4)]
            k = min(range(4), key=lambda idx: int(np.sum(opts[idx] != ref)))
            aligned.append(opts[k])
            rotations.append(k)
        stack = np.stack(aligned)
        consensus = np.sum(stack, axis=0) >= (len(aligned) / 2)
        scores = [int(np.sum(mask != consensus)) for mask in aligned]
        order = sorted(scores, reverse=True)
        gap = order[0] - order[1]
        candidate = (gap, -sum(scores), ref_idx, aligned, rotations, consensus, scores)
        if best is None or candidate > best:
            best = candidate

    assert best is not None
    _gap, _neg_total, _ref_idx, aligned, rotations, consensus, scores = best
    anomaly_idx = max(range(len(scores)), key=lambda idx: scores[idx])
    anomaly_orig = masks[anomaly_idx]
    consensus_orig = np.rot90(consensus, -rotations[anomaly_idx])
    diff = anomaly_orig != consensus_orig
    out = np.full(anomaly_orig.shape, BG, dtype=int)
    out[anomaly_orig] = FG
    out[diff] = RED
    return out


def _random_base_mask(size: int) -> np.ndarray | None:
    for _ in range(60):
        mask = np.zeros((size, size), dtype=bool)
        r = c = size // 2
        mask[r, c] = True
        target = random.randint(size + 1, size * size // 2)
        while int(mask.sum()) < target:
            frontier = []
            for rr, cc in np.argwhere(mask):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                        frontier.append((nr, nc))
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        return mask
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        size = random.randint(5, 7)
        base = _random_base_mask(size)
        if base is None:
            continue
        count = random.randint(4, 6)
        h = random.randint(max(18, size * 2), 30)
        w = random.randint(max(18, size * 2), 30)
        grid = np.full((h, w), BG, dtype=int)
        slots = []
        for r in range(1, h - size, size + 1):
            for c in range(1, w - size, size + 1):
                slots.append((r, c))
        random.shuffle(slots)
        if len(slots) < count:
            continue
        anomaly_idx = random.randrange(count)
        ok = True
        for i in range(count):
            mask = np.rot90(base, random.randrange(4))
            if i == anomaly_idx:
                mod = mask.copy()
                candidates = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
                random.shuffle(candidates)
                changed = 0
                for r, c in candidates:
                    mod[r, c] = ~mod[r, c]
                    if not np.array_equal(mod, mask):
                        changed += 1
                    if changed == random.randint(1, 2):
                        break
                mask = mod
            r0, c0 = slots[i]
            if np.any(grid[r0 : r0 + size, c0 : c0 + size] != BG):
                ok = False
                break
            grid[r0 : r0 + size, c0 : c0 + size][mask] = FG
        if ok:
            return grid
    raise RuntimeError("failed to generate a valid instance")
