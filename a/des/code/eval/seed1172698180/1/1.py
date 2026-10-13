from __future__ import annotations

from collections import deque
import random

import numpy as np


_BG = 0
_DIRS = ("top", "bottom", "left", "right")
_OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _components(mask: np.ndarray) -> list[dict[str, object]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            out.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _is_solid_rectangle(component: dict[str, object]) -> bool:
    r0, c0, r1, c1 = component["bbox"]  # type: ignore[misc]
    area = component["area"]  # type: ignore[assignment]
    return int(area) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)


def _opening_side(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    assert sub.shape[0] >= 4 and sub.shape[1] >= 4, "envelope must have a nonempty interior"

    ring = np.zeros_like(sub, dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True

    assert np.all(sub <= ring), "envelope pixels must lie on the rectangle perimeter"
    assert not np.any(sub[1:-1, 1:-1]), "envelope interior must stay empty"
    assert sub[0, 0] and sub[0, -1] and sub[-1, 0] and sub[-1, -1], "corners must stay closed"

    side_lines = {
        "top": sub[0, 1:-1],
        "bottom": sub[-1, 1:-1],
        "left": sub[1:-1, 0],
        "right": sub[1:-1, -1],
    }
    gap_sides = [side for side, line in side_lines.items() if not np.all(line)]
    assert len(gap_sides) == 1, "expected exactly one open side"

    gap_side = gap_sides[0]
    gap_idx = np.flatnonzero(~side_lines[gap_side])
    assert gap_idx.size >= 1, "opening must contain at least one missing cell"
    assert int(gap_idx[-1] - gap_idx[0] + 1) == int(gap_idx.size), "opening must be contiguous"
    return gap_side


def _paper_shape(envelope_bbox: tuple[int, int, int, int], opening: str) -> tuple[tuple[int, int], str]:
    r0, c0, r1, c1 = envelope_bbox
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    hinge = _OPPOSITE[opening]
    if hinge in ("top", "bottom"):
        assert inner_h % 2 == 0 and inner_h >= 2
        return (inner_h // 2, inner_w), hinge
    assert inner_w % 2 == 0 and inner_w >= 2
    return (inner_h, inner_w // 2), hinge


def _paint_folded_paper(
    out: np.ndarray, color: int, envelope_bbox: tuple[int, int, int, int], opening: str
) -> None:
    r0, c0, r1, c1 = envelope_bbox
    ir0, ic0 = r0 + 1, c0 + 1
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    _, hinge = _paper_shape(envelope_bbox, opening)

    if hinge == "top":
        base = (slice(ir0, ir0 + inner_h // 2), slice(ic0, ic0 + inner_w))
        mirror = (slice(ir0 + inner_h // 2, ir0 + inner_h), slice(ic0, ic0 + inner_w))
    elif hinge == "bottom":
        base = (slice(ir0 + inner_h // 2, ir0 + inner_h), slice(ic0, ic0 + inner_w))
        mirror = (slice(ir0, ir0 + inner_h // 2), slice(ic0, ic0 + inner_w))
    elif hinge == "left":
        base = (slice(ir0, ir0 + inner_h), slice(ic0, ic0 + inner_w // 2))
        mirror = (slice(ir0, ir0 + inner_h), slice(ic0 + inner_w // 2, ic0 + inner_w))
    else:
        base = (slice(ir0, ir0 + inner_h), slice(ic0 + inner_w // 2, ic0 + inner_w))
        mirror = (slice(ir0, ir0 + inner_h), slice(ic0, ic0 + inner_w // 2))

    out[base] = int(color)
    out[mirror] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """Pair each paper with its same-colored envelope, fold it once, and insert it into the envelope."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != _BG):
        comps = _components(g == color)
        assert len(comps) == 2, "each color must contain one envelope and one paper"

        paper_candidates = [comp for comp in comps if _is_solid_rectangle(comp)]
        assert len(paper_candidates) == 1, "expected exactly one solid paper rectangle per color"
        paper = paper_candidates[0]
        envelope = comps[0] if comps[1] is paper else comps[1]

        opening = _opening_side(g == color, envelope["bbox"])  # type: ignore[arg-type]
        expected_shape, _ = _paper_shape(envelope["bbox"], opening)  # type: ignore[arg-type]
        pr0, pc0, pr1, pc1 = paper["bbox"]  # type: ignore[misc]
        actual_shape = (int(pr1 - pr0 + 1), int(pc1 - pc0 + 1))
        assert actual_shape == expected_shape, "paper dimensions must match the envelope interior"

        paper_coords = paper["coords"]  # type: ignore[assignment]
        out[paper_coords[:, 0], paper_coords[:, 1]] = _BG
        _paint_folded_paper(out, color, envelope["bbox"], opening)  # type: ignore[arg-type]

    return out


# --------------- generator ---------------

_GRID_HEIGHT_RANGE = (18, 30)
_GRID_WIDTH_RANGE = (18, 30)
_ENVELOPE_COUNT_RANGE = (2, 4)
_COLOR_POOL = tuple(range(1, 10))
_GEN_ATTEMPTS = 300
_PLACEMENT_MARGIN = 1


def _make_envelope_patch(inner_h: int, inner_w: int, opening: str) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = 1
    patch[-1, :] = 1
    patch[:, 0] = 1
    patch[:, -1] = 1

    gap_extent = inner_w if opening in ("top", "bottom") else inner_h
    gap_len = random.randint(1, gap_extent)
    gap_start = random.randint(0, gap_extent - gap_len)

    if opening == "top":
        patch[0, 1 + gap_start : 1 + gap_start + gap_len] = 0
    elif opening == "bottom":
        patch[-1, 1 + gap_start : 1 + gap_start + gap_len] = 0
    elif opening == "left":
        patch[1 + gap_start : 1 + gap_start + gap_len, 0] = 0
    else:
        patch[1 + gap_start : 1 + gap_start + gap_len, -1] = 0

    return patch


def _place_bbox(
    blocked: np.ndarray, patch_shape: tuple[int, int]
) -> tuple[int, int] | None:
    h, w = blocked.shape
    ph, pw = patch_shape
    spots: list[tuple[int, int]] = []
    for r0 in range(_PLACEMENT_MARGIN, h - ph - _PLACEMENT_MARGIN + 1):
        for c0 in range(_PLACEMENT_MARGIN, w - pw - _PLACEMENT_MARGIN + 1):
            r1 = r0 + ph - 1
            c1 = c0 + pw - 1
            if blocked[
                r0 - _PLACEMENT_MARGIN : r1 + _PLACEMENT_MARGIN + 1,
                c0 - _PLACEMENT_MARGIN : c1 + _PLACEMENT_MARGIN + 1,
            ].any():
                continue
            spots.append((r0, c0))
    if not spots:
        return None
    return random.choice(spots)


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, color: int) -> None:
    mask = patch.astype(bool)
    view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    view[mask] = int(color)


def _reserve_bbox(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    blocked[
        r0 - _PLACEMENT_MARGIN : r1 + _PLACEMENT_MARGIN + 1,
        c0 - _PLACEMENT_MARGIN : c1 + _PLACEMENT_MARGIN + 1,
    ] = True


def _sample_specs() -> list[dict[str, object]]:
    n = random.randint(*_ENVELOPE_COUNT_RANGE)
    directions = random.sample(_DIRS, n)
    random.shuffle(directions)
    specs: list[dict[str, object]] = []

    for opening in directions:
        if opening in ("top", "bottom"):
            inner_h = 2 * random.randint(1, 4)
            inner_w = random.randint(2, 8)
            paper_shape = (inner_h // 2, inner_w)
        else:
            inner_h = random.randint(2, 8)
            inner_w = 2 * random.randint(1, 4)
            paper_shape = (inner_h, inner_w // 2)
        specs.append(
            {
                "opening": opening,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "paper_shape": paper_shape,
                "envelope_patch": _make_envelope_patch(inner_h, inner_w, opening),
            }
        )
    return specs


def _attempt_generate() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*_GRID_HEIGHT_RANGE)
    w = random.randint(*_GRID_WIDTH_RANGE)
    specs = _sample_specs()
    if len(specs) > len(_COLOR_POOL):
        return None

    colors = random.sample(_COLOR_POOL, len(specs))
    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for color, spec in zip(colors, specs, strict=False):
        envelope_patch = spec["envelope_patch"]  # type: ignore[assignment]
        env_pos = _place_bbox(blocked, envelope_patch.shape)
        if env_pos is None:
            return None
        er0, ec0 = env_pos
        er1 = er0 + envelope_patch.shape[0] - 1
        ec1 = ec0 + envelope_patch.shape[1] - 1
        _paint_patch(grid, envelope_patch, er0, ec0, color)
        _paint_patch(expected, envelope_patch, er0, ec0, color)
        _reserve_bbox(blocked, (er0, ec0, er1, ec1))

        paper_h, paper_w = spec["paper_shape"]  # type: ignore[misc]
        paper_patch = np.ones((int(paper_h), int(paper_w)), dtype=int)
        paper_pos = _place_bbox(blocked, paper_patch.shape)
        if paper_pos is None:
            return None
        pr0, pc0 = paper_pos
        pr1 = pr0 + paper_patch.shape[0] - 1
        pc1 = pc0 + paper_patch.shape[1] - 1
        _paint_patch(grid, paper_patch, pr0, pc0, color)
        _reserve_bbox(blocked, (pr0, pc0, pr1, pc1))

        expected[pr0 : pr1 + 1, pc0 : pc1 + 1] = _BG
        _paint_folded_paper(
            expected,
            int(color),
            (er0, ec0, er1, ec1),
            spec["opening"],  # type: ignore[arg-type]
        )

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError("failed to generate a valid envelope-folding instance")
