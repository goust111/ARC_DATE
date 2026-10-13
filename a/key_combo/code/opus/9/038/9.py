from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLUE = 1
RED = 2
BLACK = 0
MAX_GENERATION_ATTEMPTS = 600

_SEEN_OUTPUT_SIGNATURES: set[bytes] = set()
_GEN_CALL_COUNT = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _side_contacts(red_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> list[str]:
    r0, c0, r1, c1 = bbox
    h, w = red_mask.shape
    contacts: list[str] = []
    if r0 > 0 and np.any(red_mask[r0 - 1, c0 : c1 + 1]):
        contacts.append("top")
    if r1 + 1 < h and np.any(red_mask[r1 + 1, c0 : c1 + 1]):
        contacts.append("bottom")
    if c0 > 0 and np.any(red_mask[r0 : r1 + 1, c0 - 1]):
        contacts.append("left")
    if c1 + 1 < w and np.any(red_mask[r0 : r1 + 1, c1 + 1]):
        contacts.append("right")
    return contacts


def _mirror_from_side(motif: np.ndarray, side: str) -> np.ndarray:
    if side in {"left", "right"}:
        return np.fliplr(motif)
    if side in {"top", "bottom"}:
        return np.flipud(motif)
    raise ValueError(f"unknown side: {side}")


def _stamp_blocks(mask: np.ndarray, stamp: np.ndarray) -> np.ndarray:
    n = int(mask.shape[0])
    out = np.zeros((n * n, n * n), dtype=int)
    for r, c in np.argwhere(mask == BLUE):
        rs = int(r) * n
        cs = int(c) * n
        out[rs : rs + n, cs : cs + n] = np.maximum(out[rs : rs + n, cs : cs + n], stamp)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the original blue motif as a block-placement mask and its marker-directed mirror as the stamped content.

    Input structure:
    - A black grid contains one compact blue motif whose bounding box is N x N with N in {2,3}.
    - One or two red pixels sit immediately outside exactly one side of that blue motif.
    - The full canvas is N^2 x N^2, while only the blue motif and red markers are foreground.

    Rule:
    - Crop the blue motif from its bounding box, ignoring the red pixels.
    - Determine which side the red marker touches: left/right means mirror the motif left-right, top/bottom means mirror it up-down.
    - Create an N^2 x N^2 black output split into N x N blocks of size N x N.
    - For every blue pixel of the original motif at local position (r,c), paste the mirrored motif into output block (r,c); keep all other blocks black and drop red.

    Key insight:
    - The blue motif is split into two roles: its original foreground positions say where to place copies, while the red marker changes what is copied by forcing a mirrored version of the motif.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"

    blue_mask = g == BLUE
    red_mask = g == RED
    assert blue_mask.any(), "expected a blue motif"
    assert red_mask.any(), "expected red marker pixels"
    assert np.all((g == BLACK) | blue_mask | red_mask), "unexpected colors present"

    motif_bbox = _bbox(blue_mask)
    assert motif_bbox is not None
    r0, c0, r1, c1 = motif_bbox
    n = r1 - r0 + 1
    assert n == c1 - c0 + 1 and n in {2, 3}, "expected a 2x2 or 3x3 motif bbox"
    assert g.shape == (n * n, n * n), "expected the full grid size to be motif-size squared"

    contacts = _side_contacts(red_mask, motif_bbox)
    assert len(contacts) == 1, "red markers must touch exactly one side of the motif"
    side = contacts[0]

    motif = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    motif[motif == RED] = BLACK
    assert np.all((motif == BLACK) | (motif == BLUE))

    mirrored = _mirror_from_side(motif, side)
    return _stamp_blocks(motif, mirrored)


def _full_bbox(mask: np.ndarray) -> bool:
    return (
        np.any(mask[0, :])
        and np.any(mask[-1, :])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    )


def _nondegenerate_mask(mask: np.ndarray) -> bool:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    n = int(mask.shape[0])
    count = int(mask.sum())
    if count < 2 or count >= n * n:
        return False
    if not _full_bbox(mask):
        return False
    if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
        return False
    if n == 3:
        if not all(0 < int(mask[r, :].sum()) < n for r in range(n)):
            return False
        if not all(0 < int(mask[:, c].sum()) < n for c in range(n)):
            return False
    return True


def _random_mask(n: int) -> np.ndarray:
    for _ in range(200):
        mask = np.random.rand(n, n) < random.uniform(0.35, 0.7)
        if _nondegenerate_mask(mask):
            return mask
    raise RuntimeError("failed to sample a valid motif mask")


def _side_cycle() -> Iterable[str]:
    global _GEN_CALL_COUNT
    while True:
        base = ["right", "bottom", "left", "top"]
        side = base[_GEN_CALL_COUNT % len(base)]
        _GEN_CALL_COUNT += 1
        yield side


_SIDE_STREAM = _side_cycle()


def _candidate_blocks(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    n = int(mask.shape[0])
    blocks = []
    for br, bc in np.argwhere(mask):
        br = int(br)
        bc = int(bc)
        if side == "top" and br == 0:
            continue
        if side == "bottom" and br == n - 1:
            continue
        if side == "left" and bc == 0:
            continue
        if side == "right" and bc == n - 1:
            continue
        blocks.append((br, bc))
    return blocks


def _marker_cells(bbox: tuple[int, int, int, int], side: str, count: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        cells = [(r0 - 1, c) for c in range(c0, c1 + 1)]
    elif side == "bottom":
        cells = [(r1 + 1, c) for c in range(c0, c1 + 1)]
    elif side == "left":
        cells = [(r, c0 - 1) for r in range(r0, r1 + 1)]
    elif side == "right":
        cells = [(r, c1 + 1) for r in range(r0, r1 + 1)]
    else:
        raise ValueError(f"unknown side: {side}")

    if count == 1:
        return [random.choice(cells)]
    if len(cells) == 2:
        return cells
    start = random.randint(0, len(cells) - 2)
    return cells[start : start + 2]


def _semantic_checks(inp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    blue_bbox = _bbox(inp == BLUE)
    if blue_bbox is None:
        return False
    r0, c0, r1, c1 = blue_bbox
    n = r1 - r0 + 1
    motif = inp[r0 : r1 + 1, c0 : c1 + 1]
    contacts = _side_contacts(inp == RED, blue_bbox)
    if len(contacts) != 1:
        return False
    mirrored = _mirror_from_side(np.where(motif == BLUE, BLUE, BLACK), contacts[0])
    if np.array_equal(motif == BLUE, mirrored == BLUE):
        return False

    original_self = _stamp_blocks(np.where(motif == BLUE, BLUE, BLACK), np.where(motif == BLUE, BLUE, BLACK))
    if np.array_equal(out, original_self):
        return False

    if int(np.count_nonzero(out == BLUE)) <= int(np.count_nonzero(motif == BLUE)):
        return False
    if out.shape != (n * n, n * n):
        return False
    if len(_side_contacts(inp == RED, blue_bbox)) != 1:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    n = random.choice([2, 3])
    side = next(_SIDE_STREAM)
    mask = _random_mask(n)
    candidates = _candidate_blocks(mask, side)
    if not candidates:
        return None

    br, bc = random.choice(candidates)
    grid = np.zeros((n * n, n * n), dtype=int)
    rs = br * n
    cs = bc * n
    motif = np.where(mask, BLUE, BLACK).astype(int)
    grid[rs : rs + n, cs : cs + n] = motif

    marker_count = random.choice([1, 2])
    for r, c in _marker_cells((rs, cs, rs + n - 1, cs + n - 1), side, marker_count):
        grid[r, c] = RED

    if not _semantic_checks(grid):
        return None

    out = main(grid)
    sig = out.tobytes() + str(out.shape).encode()
    if sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a unique valid input")
